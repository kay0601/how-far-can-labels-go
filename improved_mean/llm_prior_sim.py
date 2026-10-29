"""Offline check of a benchmark-level LLM prior for v5 (candidate v7).

At test time the predictor would rate every item it is asked about with the anchored
Qwen3.8-27B prompt (llm_anchor.py) and keep, per benchmark, a running mean of those
ratings across calls. That mean, shrunk toward 0.35 with strength K0, replaces v5's
PRIOR_MEAN. This script simulates it on the four training benchmarks with several
subjects per item, using the cached anchored ratings (120 items per benchmark, rated
with anchors from the other three benchmarks only):

- each pair's evaluation items get ratings drawn from its benchmark's 120 cached ones;
- the running mean restarts at every budget (as if workers restart at each checkpoint)
  and grows item by item in the order the evaluation items are scored;
- K0 and ALPHA are chosen leave-one-benchmark-out (LOBO).

Upper reference: PRIOR_MEAN set to the benchmark's true mean rate on the rated items.

    <venv python> llm_prior_sim.py [n_draws]      # no GPU needed
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import llm_anchor
import model
import paths
import sim

MODEL_NAME = "Qwen/Qwen3.8-27B"
PRIOR = 0.35
BENCHES = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")


def cached_ratings():
    z = np.load(llm_anchor.cache_path(MODEL_NAME), allow_pickle=True)
    probs = z["probs"] / z["probs"].sum(1, keepdims=True)
    p = np.clip(((probs * np.arange(10)).sum(1) + 0.5) / 10, 0.01, 0.99)
    out = {}
    for key, rating in zip(z["keys"], p):
        out.setdefault(str(key).split("/")[0], []).append(float(rating))
    return out


class RunningPrior:
    """v5 whose PRIOR_MEAN is a shrunk running mean of LLM ratings for the item's benchmark.

    ``persist``: ratings are kept across budgets (cached on disk at test time) and budget 0
    keeps v5's constant prediction, so budget-0 feedback still identifies each pair's rate;
    from budget 1 on, the mean covers every item rated so far. Otherwise the running mean
    restarts at every budget and is used from budget 0."""

    def __init__(self, ratings, k0, alpha, seed=0, persist=False):
        self.ratings, self.k0, self.alpha, self.persist = ratings, k0, alpha, persist
        self.rng = random.Random(seed)
        self.summary_id, self.sums, self.seen = None, {}, {}
        self.had_labels = False

    def __call__(self, inp, labeled, summary):
        if self.persist:
            if not labeled and self.had_labels:  # budget 0 again: a new draw starts
                self.sums, self.seen = {}, {}
            self.had_labels = bool(labeled)
            bench = inp[1]["benchmark_id"]
            key = (bench, inp[1]["item_content"])
            if key not in self.seen:
                self.seen[key] = self.rng.choice(self.ratings[bench.split("#")[0]])
                total, n = self.sums.get(bench, (0.0, 0))
                self.sums[bench] = (total + self.seen[key], n + 1)
            if not labeled:
                return model.predict_with(model.PARAMS, inp, labeled, summary)
            total, n = self.sums[bench]
            mu = (total + self.k0 * PRIOR) / (n + self.k0)
            return model.predict_with(dict(model.PARAMS, PRIOR_MEAN=mu, ALPHA=self.alpha), inp, labeled, summary)
        if id(summary) != self.summary_id:  # new budget: start the running means afresh
            self.summary_id, self.sums = id(summary), {}
        bench = inp[1]["benchmark_id"]
        source = bench.split("#")[0]
        total, n = self.sums.get(bench, (0.0, 0))
        total, n = total + self.rng.choice(self.ratings[source]), n + 1
        self.sums[bench] = (total, n)
        mu = (total + self.k0 * PRIOR) / (n + self.k0)
        return model.predict_with(dict(model.PARAMS, PRIOR_MEAN=mu, ALPHA=self.alpha), inp, labeled, summary)


def known_prior(rates, alpha):
    return lambda inp, labeled, summary: model.predict_with(
        dict(model.PARAMS, PRIOR_MEAN=rates[inp[1]["benchmark_id"].split("#")[0]], ALPHA=alpha), inp, labeled, summary)


def main(n_draws=400):
    ratings = cached_ratings()
    fig = Path(paths.RESULTS) / "figure_data.json"  # written by figure_data.py
    actual = {b: v["actual"] for b, v in json.loads(fig.read_text())["anchored"][MODEL_NAME].items()}
    draws = sim.build_draws(n_draws, seed=2024)
    draws = [[g for g in d if g[0][0].split("#")[0] in BENCHES] for d in draws]
    draws = [d for d in draws if d]

    def per_bench(fn):
        df = sim.evaluate(draws, fn)
        return df.groupby("bench").alc.mean(), df.groupby("bench")[[f"b{b}" for b in sim.BUDGETS]].mean()

    base, base_budgets = per_bench(sim.with_params(model.PARAMS))
    print("v5 per benchmark:", base.round(4).to_dict())
    rows = {}
    for k0 in (0, 2, 5, 10, 20, 50):
        for alpha in (3.0, 6.0):
            alc, budgets = per_bench(RunningPrior(ratings, k0, alpha))
            rows[(k0, alpha)] = alc - base
            print(f"K0 {k0:>2} ALPHA {alpha:.0f}: change {dict((alc - base).round(4))}  mean {(alc - base).mean():+.4f}"
                  f"  b0 {budgets['b0'].mean() - base_budgets['b0'].mean():+.4f}")
    grid = pd.DataFrame(rows).T  # rows: settings, columns: benchmarks
    for k0 in (2, 5, 10):
        alc, budgets = per_bench(RunningPrior(ratings, k0, 3.0, persist=True))
        print(f"persist, budget 0 kept, K0 {k0:>2} ALPHA 3: change {dict((alc - base).round(4))}  mean {(alc - base).mean():+.4f}"
              f"  b1 {budgets['b1'].mean() - base_budgets['b1'].mean():+.4f}")
    lobo = {}
    for b in grid.columns:
        best = grid.drop(columns=b).mean(axis=1).idxmin()
        lobo[b] = (best, grid.loc[best, b])
    print("\nLOBO selection: " + ", ".join(f"{b}: K0 {s[0]} ALPHA {s[1]:.0f} -> {d:+.4f}" for b, (s, d) in lobo.items()))
    print(f"LOBO mean change {np.mean([d for _, d in lobo.values()]):+.4f}")

    for alpha in (3.0, 6.0):
        alc, _ = per_bench(known_prior(actual, alpha))
        print(f"reference, true benchmark rate as PRIOR_MEAN (ALPHA {alpha:.0f}): {dict((alc - base).round(4))}"
              f"  mean {(alc - base).mean():+.4f}")
    return grid, lobo


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
