"""How well can a predictor that sees only the target pair's own labels do? (report, Section 5)

The bound knows the distribution of real pair success rates: its prior for a pair is
the empirical set of the *other* pairs' rates, and its prediction is the posterior
mean given the pair's own k successes in n labels. v5 also pools other subjects'
labels on the same benchmark, so it can beat this bound; the oracle knows every rate.

    <venv python> label_only_bound.py
"""

from __future__ import annotations

import random
import statistics as st

import bench_prior_value as bp
import model
import proxy
import sim


def posterior_mean(k, n, rates):
    w = [r ** k * (1 - r) ** (n - k) for r in rates]
    total = sum(w)
    return sum(wi * r for wi, r in zip(w, rates)) / total if total > 0 else st.mean(rates)


def scores(draws, reps=2000, seed=0):
    """Expected Brier ALC of the own-label bound, v5 and the oracle over ``draws``, each a list of
    (subject, benchmark, rate). Each draw is simulated on its own, as v5 pools labels within a draw only;
    the bound's prior is the rates of all other pairs."""
    rng = random.Random(seed)
    every = [p for draw in draws for p in draw]
    bound = v5 = 0.0
    for _ in range(reps):
        for draw in draws:
            acq = {s: [int(rng.random() < r) for _ in range(31)] for s, _, r in draw}
            for weight, budget in zip(sim.WEIGHTS, sim.BUDGETS):
                labeled = [[proxy._inp(s, b, i), y] for s, b, _ in draw for i, y in enumerate(acq[s][:budget])]
                summary = model._summarize(labeled)
                for s, b, r in draw:
                    others = [x[2] for x in every if x[:2] != (s, b)]
                    p_bound = posterior_mean(sum(acq[s][:budget]), budget, others)
                    p_v5 = model.predict_with(model.PARAMS, proxy._inp(s, b, "eval"), labeled, summary)
                    base = r * (1 - r)
                    bound += weight * (base + (p_bound - r) ** 2) / (reps * len(every))
                    v5 += weight * (base + (p_v5 - r) ** 2) / (reps * len(every))
    oracle = st.mean(r * (1 - r) for _, _, r in every)
    return bound, v5, oracle


if __name__ == "__main__":
    sets = {
        "26 pairs (reports of v1, v4, v5)": [proxy.DRAWS[name] for name in ("report1", "report2", "report3")],
        "69 pairs (all eight ALC reports)": list(bp.pairs.values()),
    }
    for name, pairs in sets.items():
        bound, v5, oracle = scores(pairs)
        print(f"{name}: own-label bound {bound:.4f}, v5 {v5:.4f}, oracle {oracle:.4f}")
