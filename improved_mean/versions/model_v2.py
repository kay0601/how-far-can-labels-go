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
3. Pair estimate: labels of this subject on this benchmark, shrunk toward the
   benchmark prior with strength ``ALPHA`` (Beta-Binomial posterior mean).
4. Item adjustment: labels other subjects received on the *same item* shift the
   estimate by their mean residual against those subjects' own pair estimates,
   shrunk with strength ``ITEM_K`` and scaled by ``ITEM_GAMMA``.

No state is kept between calls; nothing is loaded at import.
"""

from __future__ import annotations

PARAMS = {
    "PRIOR_MEAN": 0.35,
    "GLOBAL_K": 100.0,
    "BENCH_K": 3.0,
    "ALPHA": 3.0,
    "ITEM_K": 1.0,
    "ITEM_GAMMA": 1.0,
    "CLIP": 0.01,
}

_SUBJECT_KEYS = (
    "normalized_name", "provider", "release_date", "access_date", "harness",
    "reasoning_effort", "harness_version", "subject_features_extra",
)


def _subject_key(subject: dict) -> tuple:
    return tuple(subject.get(k, "") for k in _SUBJECT_KEYS)


def _item_key(item: dict) -> tuple:
    return (item.get("benchmark_id", ""), item.get("item_content", ""), item.get("item_features", ""))


def _summarize(labeled):
    total = [0, 0]
    bench = {}   # benchmark -> [successes, count]
    pair = {}    # (subject, benchmark) -> [successes, count]
    obs = {}     # item_key -> [(subject, benchmark, label)]
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
    return total, bench, pair, obs


def _pair_estimate(total, bench, pair, s, b, p):
    bs, bn = bench.get(b, (0, 0))
    ps, pn = pair.get((s, b), (0, 0))
    g = (total[0] - bs + p["GLOBAL_K"] * p["PRIOR_MEAN"]) / (total[1] - bn + p["GLOBAL_K"])
    prior = (bs - ps + p["BENCH_K"] * g) / (bn - pn + p["BENCH_K"])
    return (ps + p["ALPHA"] * prior) / (pn + p["ALPHA"])


def predict_with(params: dict, input: list, labeled: list | None = None, _summary=None) -> float:
    subject, item = input
    b = item.get("benchmark_id", "")
    s = _subject_key(subject)
    total, bench, pair, obs = _summary if _summary is not None else _summarize(labeled or [])
    est = _pair_estimate(total, bench, pair, s, b, params)

    if params["ITEM_GAMMA"] > 0:
        resid, n = 0.0, 0
        for os_, ob, y in obs.get(_item_key(item), ()):
            if os_ != s:
                resid += y - _pair_estimate(total, bench, pair, os_, ob, params)
                n += 1
        if n:
            est += params["ITEM_GAMMA"] * resid / (n + params["ITEM_K"])

    c = params["CLIP"]
    return float(min(1.0 - c, max(c, est)))


def predict(input: list, labeled: list | None = None) -> float:
    return predict_with(PARAMS, input, labeled)
