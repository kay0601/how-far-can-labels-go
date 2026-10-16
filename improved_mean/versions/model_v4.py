"""Hierarchical smoothed empirical mean predictor.

Improves on the official empirical mean baseline, which returns the raw mean
of acquired labels (0 or 1 after a single label) and 0.5 without evidence.

Estimate for a target (subject, benchmark, item), using only ``labeled``:

1. Global prior: labels on *other* benchmarks, shrunk toward ``PRIOR_MEAN``
   with strength ``GLOBAL_K``.
2. Benchmark prior: labels from *other* subjects on the same benchmark, shrunk
   toward the global prior with strength ``BENCH_K``. The target subject's own
   labels are excluded so they are not counted twice (the scoring draw usually
   holds a single subject per benchmark).
3. Subject shift: the benchmark prior is moved in logit space by
   ``BETA * ability``. Ability comes from ``ability_table.json``, fitted offline
   on the public training data (subject_prior.py): a per-model lookup when the
   model name was seen, else a ridge fit on release date and reasoning effort.
4. Pair estimate: labels of this subject on this benchmark, shrunk toward the
   shifted prior with strength ``ALPHA`` (Beta-Binomial posterior mean).
5. Item adjustment: labels other subjects received on the *same item* shift the
   estimate by their mean residual against those subjects' own pair estimates,
   shrunk with strength ``ITEM_K`` and scaled by ``ITEM_GAMMA``.
6. Group adjustment: labels (any subject, including the target) on items of
   the same benchmark sharing the target's ``item_features`` group shift the
   estimate by their mean residual, shrunk with ``GROUP_K`` and scaled by
   ``GROUP_GAMMA``. The group drops numeric-valued fields such as
   ``problem_idx=3``, so it names e.g. a language, website, paper or contest.

No state is kept between calls; only the ability table is loaded at import.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path

PARAMS = {
    "PRIOR_MEAN": 0.4,
    "GLOBAL_K": 100.0,
    "BENCH_K": 3.0,
    "ALPHA": 3.0,
    "BETA": 0.55,
    "ITEM_K": 1.0,
    "ITEM_GAMMA": 1.0,
    "GROUP_K": 5.0,
    "GROUP_GAMMA": 0.75,
    "CLIP": 0.01,
}

_SUBJECT_KEYS = (
    "normalized_name", "provider", "release_date", "access_date", "harness",
    "reasoning_effort", "harness_version", "subject_features_extra",
)


_EFFORT = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}
_TABLE_PATH = Path(__file__).with_name("ability_table.json")
TABLE = json.loads(_TABLE_PATH.read_text()) if _TABLE_PATH.exists() else None


def _ability(table, s: tuple) -> float:
    """Relative logit ability of subject key ``s`` (0 when unknown)."""
    if not table:
        return 0.0
    name, release, effort, extra = s[0], s[2], s[5], s[7]
    if name:
        key = name.lower()
    else:
        m = re.search(r"source_model_name=([^;]+)", extra)
        key = m.group(1).lower() if m else ""
    if key in table["names"]:
        return table["names"][key]
    try:
        d = date.fromisoformat(release[:10])
        year, year_missing = d.year + d.timetuple().tm_yday / 365 - table["year_mean"], 0.0
    except ValueError:
        year, year_missing = 0.0, 1.0
    e = _EFFORT.get(effort.strip().lower())
    x = (1.0, year, year_missing, float(e or 0), 0.0 if e is not None else 1.0)
    return sum(w * v for w, v in zip(table["ridge"], x))


def _subject_key(subject: dict) -> tuple:
    return tuple(subject.get(k, "") for k in _SUBJECT_KEYS)


def _item_key(item: dict) -> tuple:
    return (item.get("benchmark_id", ""), item.get("item_content", ""), item.get("item_features", ""))


def _group_key(item: dict) -> tuple:
    parts = []
    for part in item.get("item_features", "").split(";"):
        key, _, value = part.partition("=")
        value = value.strip()
        if value and not value.replace(".", "", 1).isdigit() and key.strip() != "prompt_source":
            parts.append(part.strip())
    return (item.get("benchmark_id", ""), ";".join(sorted(parts))) if parts else None


def _summarize(labeled):
    total = [0, 0]
    bench = {}   # benchmark -> [successes, count]
    pair = {}    # (subject, benchmark) -> [successes, count]
    obs = {}     # item_key -> [(subject, benchmark, label)]
    groups = {}  # group_key -> [(subject, benchmark, label)]
    for (subject, item), label in labeled:
        y = int(label)
        s = _subject_key(subject)
        b = item.get("benchmark_id", "")
        total[0] += y
        total[1] += 1
        bench.setdefault(b, [0, 0])
        bench[b][0] += y
        bench[b][1] += 1
        pair.setdefault((s, b), [0, 0])
        pair[(s, b)][0] += y
        pair[(s, b)][1] += 1
        obs.setdefault(_item_key(item), []).append((s, b, y))
        g = _group_key(item)
        if g is not None:
            groups.setdefault(g, []).append((s, b, y))
    return total, bench, pair, obs, groups


def _pair_estimate(total, bench, pair, s, b, p):
    bs, bn = bench.get(b, (0, 0))
    ps, pn = pair.get((s, b), (0, 0))
    g = (total[0] - bs + p["GLOBAL_K"] * p["PRIOR_MEAN"]) / (total[1] - bn + p["GLOBAL_K"])
    prior = (bs - ps + p["BENCH_K"] * g) / (bn - pn + p["BENCH_K"])
    if p["BETA"]:
        prior = min(0.99, max(0.01, prior))
        shift = p["BETA"] * _ability(p.get("_TABLE", TABLE), s)
        prior = 1.0 / (1.0 + math.exp(-(math.log(prior / (1 - prior)) + shift)))
    return (ps + p["ALPHA"] * prior) / (pn + p["ALPHA"])


def predict_with(params: dict, input: list, labeled: list | None = None, _summary=None) -> float:
    subject, item = input
    b = item.get("benchmark_id", "")
    s = _subject_key(subject)
    total, bench, pair, obs, groups = _summary if _summary is not None else _summarize(labeled or [])
    est = _pair_estimate(total, bench, pair, s, b, params)

    if params["ITEM_GAMMA"] > 0:
        resid, n = 0.0, 0
        for os_, ob, y in obs.get(_item_key(item), ()):
            if os_ != s:
                resid += y - _pair_estimate(total, bench, pair, os_, ob, params)
                n += 1
        if n:
            est += params["ITEM_GAMMA"] * resid / (n + params["ITEM_K"])

    g = _group_key(item) if params["GROUP_GAMMA"] > 0 else None
    if g is not None and g in groups:
        resid = sum(y - _pair_estimate(total, bench, pair, os_, ob, params) for os_, ob, y in groups[g])
        est += params["GROUP_GAMMA"] * resid / (len(groups[g]) + params["GROUP_K"])

    c = params["CLIP"]
    return float(min(1.0 - c, max(c, est)))


def predict(input: list, labeled: list | None = None) -> float:
    return predict_with(PARAMS, input, labeled)
