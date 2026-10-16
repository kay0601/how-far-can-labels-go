"""Grid search scored on two views, to avoid overfitting either one:

- sim: realistic test-like draws from the public training data, per source
  benchmark (leave-one-benchmark-out selection is reported);
- proxy: the eight real scoring pairs reconstructed in proxy.py.
"""

from __future__ import annotations

import itertools
import sys
from multiprocessing import Pool

import pandas as pd

import model
import proxy
import sim

GRID = {
    "PRIOR_MEAN": [0.35, 0.4, 0.45, 0.5],
    "ALPHA": [2.0, 3.0, 4.0],
    "MIX_W": [0.0, 0.1, 0.2, 0.3, 0.4],
    "MIX_MEAN": [0.01, 0.03, 0.06],
    "MIX_K": [5.0, 10.0, 30.0],
}

DRAWS = sim.build_draws(int(sys.argv[1]) if len(sys.argv) > 1 else 150, seed=1)


def run(values):
    params = dict(model.PARAMS, **dict(zip(GRID, values)))
    df = sim.evaluate(DRAWS, sim.with_params_lobo(params))
    proxy_alc, _ = proxy.evaluate(params, reps=200)
    return {**params, **df.groupby("bench").alc.mean().to_dict(), "sim": df.alc.mean(), "proxy": proxy_alc}


if __name__ == "__main__":
    combos = list(itertools.product(*GRID.values()))
    with Pool(90) as pool:
        res = pd.DataFrame(pool.map(run, combos))
    benches = [c for c in res.columns if c not in model.PARAMS and c not in ("sim", "proxy")]
    res["both"] = (res["sim"] + res["proxy"]) / 2
    res.to_csv("tune_results.csv", index=False)
    cols = list(GRID) + ["sim", "proxy", "both"]
    for metric in ("sim", "proxy", "both"):
        print(f"\nTop 5 by {metric}:")
        print(res.sort_values(metric)[cols].head(5).round(4).to_string(index=False))

    print("\nsim LOBO: tune on the other source benchmarks, score the held-out one")
    lobo = []
    for b in benches:
        others = [o for o in benches if o != b]
        best = res.loc[res[others].mean(axis=1).idxmin()]
        lobo.append(best[b])
        print(f"  held-out {b}: ALC={best[b]:.4f} with", {k: float(best[k]) for k in GRID})
    print(f"  LOBO mean ALC={sum(lobo) / len(lobo):.4f}")

    print("\nBest per ALPHA by both:")
    print(res.loc[res.groupby("ALPHA")["both"].idxmin()][cols].round(4).to_string(index=False))

    print("\nBest per MIX_W by both:")
    print(res.loc[res.groupby("MIX_W")["both"].idxmin()][cols + benches].round(4).to_string(index=False))
    print("\nBest per MIX_W by proxy / by sim:")
    print(res.loc[res.groupby("MIX_W")["proxy"].idxmin()][cols].round(4).to_string(index=False))
    print(res.loc[res.groupby("MIX_W")["sim"].idxmin()][cols].round(4).to_string(index=False))
