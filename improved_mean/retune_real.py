"""Leave-one-report-out check of label-only changes on all real pairs (bench_prior_value.py).

Scores v5's prior parameters (as retune60.py) and experiment 6's mixture prior
(versions/model_mixture_prior.py; MIX_W = 0 is v5) on the proxy pairs rebuilt from
every ALC report. A setting is picked on all reports but one and scored on the
held-out report, so the gain estimate never sees the pairs it is scored on.

    <venv python> retune_real.py
"""

from __future__ import annotations

import importlib.util
import itertools
import random
import statistics as st
from multiprocessing import Pool

import pandas as pd

import bench_prior_value as bp
import model

_spec = importlib.util.spec_from_file_location("model_mixture", "versions/model_mixture_prior.py")
mixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mixture)

FAMILIES = {
    "v5 prior": (model, {"PRIOR_MEAN": [0.3, 0.35, 0.4], "ALPHA": [2.0, 3.0, 4.0, 6.0], "BENCH_K": [1.0, 3.0, 6.0]}),
    "mixture": (mixture, {"MIX_W": [0.0, 0.1, 0.2, 0.3, 0.4], "MIX_MEAN": [0.02, 0.05, 0.1], "MIX_K": [5.0, 10.0, 30.0]}),
}


def run(task):
    family, values = task
    mod, grid = FAMILIES[family]
    params = dict(mod.PARAMS, **dict(zip(grid, values)))
    per_pair = bp.pair_alcs(lambda sid, b: params, mod=mod)
    return {"family": family, **dict(zip(grid, values)), "all": st.mean(x[4] for x in per_pair),
            **{f"r{sid}": sum(x[4] for x in per_pair if x[0] == sid) for sid in bp.pairs},
            **{f"n{sid}": sum(x[0] == sid for x in per_pair) for sid in bp.pairs}}


def loro_change(sub, base, reports):
    """Leave-one-report-out change vs v5 per pair; a held-out report's copies never take part in its selection."""
    num = den = 0.0
    for k in reports:
        others = [r for r in reports if r != k]
        if not others:
            continue
        pick = sub.loc[(sum(sub[f"r{r}"] for r in others) / sum(sub[f"n{r}"] for r in others)).idxmin()]
        num += pick[f"r{k}"] - base[f"r{k}"]
        den += base[f"n{k}"]
    return num / den


if __name__ == "__main__":
    tasks = [(f, v) for f, (_, grid) in FAMILIES.items() for v in itertools.product(*grid.values())]
    with Pool(64) as pool:
        res = pd.DataFrame(pool.map(run, tasks))
    reports = list(bp.pairs)
    n_total = sum(res[f"n{r}"].iloc[0] for r in reports)
    base = res[(res.family == "v5 prior") & (res.PRIOR_MEAN == 0.35) & (res.ALPHA == 3.0) & (res.BENCH_K == 3.0)].iloc[0]
    print(f"{n_total} pairs in {len(reports)} reports; v5 proxy ALC {base['all']:.4f}")
    for family, (_, grid) in FAMILIES.items():
        sub = res[res.family == family]
        best = sub.loc[sub["all"].idxmin()]
        print(f"\n{family}: best on all pairs {best['all']:.4f} with {dict((g, best[g]) for g in grid)}")
        gain = 0.0
        for k in reports:  # pick on the other reports, score the held-out one
            others = [r for r in reports if r != k]
            pick = sub.loc[(sum(sub[f"r{r}"] for r in others) / sum(sub[f"n{r}"] for r in others)).idxmin()]
            gain += pick[f"r{k}"] - base[f"r{k}"]
            print(f"  held-out {k}: {pick[f'r{k}'] / pick[f'n{k}']:.4f} (v5 {base[f'r{k}'] / base[f'n{k}']:.4f}) with {dict((g, pick[g]) for g in grid)}")
        print(f"  leave-one-report-out change vs v5: {gain / n_total:+.4f} ALC per pair")
        rng = random.Random(0)  # resample whole reports; selection is redone inside each resample
        boot = sorted(loro_change(sub, base, rng.choices(reports, k=len(reports))) for _ in range(2000))
        print(f"  95% interval over reports: [{boot[50]:+.4f}, {boot[1949]:+.4f}]")
