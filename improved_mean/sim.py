"""Offline simulation of the PAIEC streaming evaluation on public training data.

Mimics the README protocol: eligible (subject, benchmark) groups have >= 80
distinct items; a draw adds complete groups (least-represented benchmark and
subject first, random ties) under a 1,000 distinct-item limit; each group's
items are split 50/50 (odd extra to acquisition); random acquisition reveals
one recorded response per item; the fixed evaluation pool is scored at budgets
0, 1, 3, 7, 15, 31 with weights 0.1, 0.2, 0.2, 0.2, 0.2, 0.1.

Group scores only interact within a benchmark, so results are also reported
per benchmark for leave-one-benchmark-out tuning.
"""

from __future__ import annotations

import glob
import math
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model  # noqa: E402
import paths  # noqa: E402

DATA_DIRS = paths.DATA_DIRS  # matharena's ~60 non-binary rows are dropped below
SKIP_BENCHES = {"mmdocrag", "embed"}  # continuous scores; embedding files only
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
DRAW_LIMIT = 1000
SUBJECT_FIELDS = model._SUBJECT_KEYS


ITEM_FEATURES = {}  # (source bench, item_id) -> item_features string, filled by load_groups


def load_groups():
    """Return {(bench, subject_tuple): {item_id: [labels...]}} for eligible groups."""
    groups = {}
    for bdir in sorted(d for root in DATA_DIRS for d in glob.glob(f"{root}/*/")):
        bench = os.path.basename(bdir.rstrip("/"))
        if bench in SKIP_BENCHES:
            continue
        resp = pd.read_parquet(bdir + "response.parquet")
        subj = pd.read_parquet(bdir + "subjects.parquet").fillna("")
        items_df = pd.read_parquet(bdir + "items.parquet")
        for iid, feat in zip(items_df.item_id, items_df.item_features):
            ITEM_FEATURES[(bench, iid)] = "" if feat is None or pd.isna(feat) else str(feat)
        stuple = {
            row.subject_id: tuple(str(getattr(row, f)) for f in SUBJECT_FIELDS)
            for row in subj.itertuples()
        }
        resp = resp[resp.response.isin([0.0, 1.0])]
        for sid, df in resp.groupby("subject_id"):
            items = df.groupby("item_id").response.apply(lambda x: [int(v) for v in x]).to_dict()
            if len(items) >= 80:
                groups[(bench, stuple[sid])] = items
    return groups


def make_draw(groups, rng):
    keys = [k for k in groups if len(groups[k]) <= DRAW_LIMIT]
    rng.shuffle(keys)
    bcount, scount, used = {}, {}, {}
    chosen, pending = [], set(keys)
    total = 0
    while pending:
        k = min(pending, key=lambda g: (bcount.get(g[0], 0), scount.get(g[1], 0), keys.index(g)))
        pending.discard(k)
        new = set(groups[k]) - used.get(k[0], set())
        if total + len(new) > DRAW_LIMIT:
            continue
        total += len(new)
        used.setdefault(k[0], set()).update(new)
        bcount[k[0]] = bcount.get(k[0], 0) + 1
        scount[k[1]] = scount.get(k[1], 0) + 1
        chosen.append(k)
    draw = []
    for k in chosen:
        items = sorted(groups[k])
        rng.shuffle(items)
        n_acq = (len(items) + 1) // 2
        acq = [(i, rng.choice(groups[k][i])) for i in items[:n_acq]]
        ev = [(i, groups[k][i]) for i in items[n_acq:]]
        draw.append((k, acq, ev))
    return draw


def make_realistic_draw(groups, rng, share_prob=0.15):
    """Test-like draw, matching the first real scoring report: about 8 groups,
    nearly always one subject per benchmark, 88-230 distinct items per group.

    Each group becomes its own pseudo-benchmark ``<bench>#<n>`` holding a random
    item subset; with ``share_prob`` a second subject of the same source
    benchmark joins that pseudo-benchmark on the shared items.
    """
    by_bench = {}
    for k in groups:
        by_bench.setdefault(k[0], []).append(k)
    draw, total, n = [], 0, 0
    while True:
        bench = rng.choice(sorted(by_bench))
        k1 = rng.choice(by_bench[bench])
        size = rng.randint(88, 230)
        if total + size > DRAW_LIMIT:
            break
        items = sorted(groups[k1])
        rng.shuffle(items)
        items = items[:size]
        members = [k1]
        if rng.random() < share_prob:
            others = [k for k in by_bench[bench] if k != k1
                      and sum(i in groups[k] for i in items) >= 80]
            if others:
                members.append(rng.choice(others))
        pseudo = f"{bench}#{n}"
        n += 1
        total += len(items)
        for k in members:
            own = [i for i in items if i in groups[k]]
            rng.shuffle(own)
            n_acq = (len(own) + 1) // 2
            acq = [(i, rng.choice(groups[k][i])) for i in own[:n_acq]]
            ev = [(i, groups[k][i]) for i in own[n_acq:]]
            draw.append(((pseudo, k[1]), acq, ev))
    return draw


def _inp(bench, subject, item_id):
    return [
        dict(zip(SUBJECT_FIELDS, subject)),
        {"item_content": str(item_id),
         "item_features": ITEM_FEATURES.get((bench.split("#")[0], item_id), ""),
         "interactors": "", "benchmark_id": bench},
    ]


def score_draw(draw, predict_fn, summarize=None):
    summarize = summarize or model._summarize
    """predict_fn(input, labeled, summary) -> p. Returns [(bench, alc, [brier per budget])]."""
    out = {k: [] for k, _, _ in draw}
    for b in BUDGETS:
        labeled = [
            [_inp(k[0], k[1], i), y] for k, acq, _ in draw for i, y in acq[:b]
        ]
        summary = summarize(labeled)
        for k, _, ev in draw:
            se, n = 0.0, 0
            for i, ys in ev:
                p = predict_fn(_inp(k[0], k[1], i), labeled, summary)
                se += sum((p - y) ** 2 for y in ys)
                n += len(ys)
            out[k].append(se / n)
    return [(k[0].split("#")[0], sum(w * s for w, s in zip(WEIGHTS, v)), v) for k, v in out.items()]


def stream_acquire(draw, hook):
    """Replay the streaming protocol with an acquisition hook.

    Each pair's acquisition pool arrives in its fixed random order; the hook
    sees the candidate, all labels acquired so far (every pair), and the
    README context, and skipped items are gone for good. Caps are reached in
    order (1, 3, 7, 15, 31) for all pairs before any pair goes further.
    Returns a draw with each pair's acquisition list replaced by the labels
    actually acquired, in order, so score_draw's prefixes are the checkpoints.
    """
    pos = {k: 0 for k, _, _ in draw}
    got = {k: [] for k, _, _ in draw}
    labeled = []
    for cap in BUDGETS[1:]:
        for k, acq, _ in draw:
            while len(got[k]) < cap and pos[k] < len(acq):
                i, y = acq[pos[k]]
                context = {
                    "subject_id": str(k[1]), "benchmark_id": k[0],
                    "labels_remaining": BUDGETS[-1] - len(got[k]), "labels_acquired": len(got[k]),
                    "max_labels": BUDGETS[-1], "items_remaining": len(acq) - pos[k],
                }
                pos[k] += 1
                inp = _inp(k[0], k[1], i)
                if hook(inp, labeled=[[[dict(a), dict(b)], v] for (a, b), v in labeled], context=context) is True:
                    got[k].append((i, y))
                    labeled.append((inp, y))
    return [(k, got[k], ev) for k, _, ev in draw]


def official_mean(input, labeled, summary):
    subject, item = input
    s = model._subject_key(subject)
    ps, pn = summary[2].get((s, item["benchmark_id"]), (0, 0))
    return ps / pn if pn else 0.5


def constant_half(input, labeled, summary):
    return 0.5


def with_params(params):
    return lambda input, labeled, summary: model.predict_with(params, input, labeled, summary)


def evaluate(draws, fn, summarize=None):
    rows = [r for d in draws for r in score_draw(d, fn, summarize)]
    df = pd.DataFrame([(b, a, *v) for b, a, v in rows], columns=["bench", "alc", *[f"b{x}" for x in BUDGETS]])
    return df


def build_draws(n, seed=0, realistic=True):
    groups = load_groups()
    rng = random.Random(seed)
    maker = make_realistic_draw if realistic else make_draw
    return [maker(groups, rng) for _ in range(n)]


if __name__ == "__main__":
    draws = build_draws(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
    sizes = [len(d) for d in draws]
    print(f"{len(draws)} draws, groups/draw mean={np.mean(sizes):.1f}")
    print(pd.Series([k[0] for d in draws for k, _, _ in d]).value_counts().to_string())
    for name, fn in [("constant 0.5", constant_half), ("official mean", official_mean),
                     ("improved v1 (default)", with_params(model.PARAMS))]:
        df = evaluate(draws, fn)
        print(f"\n== {name}: ALC={df.alc.mean():.4f}")
        print(df.groupby("bench").mean().round(4).to_string())


def with_params_lobo(params, path="versions/ability_tables_lobo.json"):
    """Like with_params, but each source benchmark uses the ability table fitted
    without it, so subject shifts never see the benchmark they are scored on."""
    import json
    tables = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), path)))
    per_bench = {b: dict(params, _TABLE=t) for b, t in tables.items()}

    def fn(input, labeled, summary):
        p = per_bench[input[1]["benchmark_id"].split("#")[0]]
        return model.predict_with(p, input, labeled, summary)
    return fn
