"""HBS+ItemLLM on real feedback (report, Section 6.3 and Appendix F). Prints aggregates only.

1. breakdown: the real excess of HBS+ItemLLM over the proxy's expectation for HBS on its draw,
   per budget: the mean over pairs with its label-noise SD, the number of pairs worse than
   expected, the share of the two largest, and each pair's excess as a z-score.
2. synthetic: synthetic item scores on the same nine pairs, with their recovered rates and
   evaluation sizes. Each item gets an outcome and a score
       z = sigma * (rho * u + sqrt(1 - rho^2) * e),
   where u is the outcome standardized within the pair and e is standard normal. Optionally a
   share q of items gets an uninformative score of spread s_out instead, or each benchmark is
   uninformative (rho = 0) with probability w, or the first revealed item of one subject on the
   benchmark with the largest real loss gets the score `first`. The shift of HBS+ItemLLM (gamma = 0.5, centered
   on the labeled items of the benchmark plus the target) is compared with HBS on the same
   labels: mean change in Brier score per budget over 300 label sequences.
3. training: spread and outcome correlation of the deployed probe scores on the four training
   benchmarks, from the leave-one-benchmark-out probe (needs scikit-learn and the Qwen3-32B
   feature cache of llm_difficulty.py).

    <venv python> itemllm_checks.py [breakdown|synthetic|training ...]   # default: all
"""

from __future__ import annotations

import contextlib
import io
import math
import random
import re
import statistics as st
import sys
from multiprocessing import Pool

with contextlib.redirect_stdout(io.StringIO()):
    import bench_prior_value as bp
    import model
    import paths
    import proxy
    import sim
    import v7_feedback as vf

SID = "960385"  # HBS+ItemLLM
GAMMA = 0.5
SYNTHETIC = [  # name, sigma, rho, q, s_out, w, first
    ("training-like", 1, 0.2, 0, 0, 0, None),
    ("training-like, one extreme early score", 1, 0.2, 0, 0, 0, 12),
    ("uninformative", 3, 0.0, 0, 0, 0, None),
    ("5% extreme, uninformative", 1, 0.2, 0.05, 20, 0, None),
    ("uninformative on a quarter of benchmarks", 6, 0.6, 0, 0, 0.25, None),
    ("wide, informative", 4, 0.45, 0, 0, 0, None),
    ("wider, more informative", 8, 0.7, 0, 0, 0, None),
    ("very wide, strongly informative", 12, 0.8, 0, 0, 0, None),
    ("very wide, strongly informative", 16, 0.85, 0, 0, 0, None),
]


def real_and_expected(reps=2000, seed=0):
    """Real Brier per (budget, subject) of HBS+ItemLLM, and the proxy's HBS samples for the same keys."""
    pairs, real, cur = bp.pairs[SID], {}, None
    for line in open(f"{paths.FEEDBACK}/{SID}_scoring_stdout.txt"):
        m = re.match(r"Label budget: (\d+)", line)
        if m:
            cur = int(m.group(1))
        m = re.match(r"(subject_\d+)\s+(benchmark_\d+)\s+\d+\s+([\d.]+)\s+([\d.]+)", line)
        if m and cur is not None:
            real[(cur, m.group(1))] = float(m.group(3))
    rng, samples = random.Random(seed), {(t, s): [] for t in sim.BUDGETS for s, _, _ in pairs}
    for _ in range(reps):
        acq = {s: [int(rng.random() < r) for _ in range(31)] for s, _, r in pairs}
        for t in sim.BUDGETS:
            labeled = [[proxy._inp(s, b, i), y] for s, b, _ in pairs for i, y in enumerate(acq[s][:t])]
            summary = model._summarize(labeled)
            for s, b, r in pairs:
                p = model.predict_with(model.PARAMS, proxy._inp(s, b, "eval"), labeled, summary)
                samples[(t, s)].append(r * (1 - r) + (p - r) ** 2)
    return pairs, real, samples


def loss_benchmark():
    """The benchmark shared by two pairs of the draw whose pairs lose most at budget 1."""
    pairs, real, samples = real_and_expected(reps=500)
    shared = {b for _, b, _ in pairs if sum(b2 == b for _, b2, _ in pairs) > 1}
    loss = {b: sum(real[(1, s)] - st.mean(samples[(1, s)]) for s, b2, _ in pairs if b2 == b) for b in shared}
    return max(loss, key=loss.get)


def breakdown():
    pairs, real, samples = real_and_expected()
    bench = {s: b for s, b, _ in pairs}
    for t in sim.BUDGETS[1:]:
        ex = {s: real[(t, s)] - st.mean(samples[(t, s)]) for s, _, _ in pairs}
        z = sorted((ex[s] / max(1e-9, st.pstdev(samples[(t, s)])) for s in ex), reverse=True)
        top = sorted(ex.values(), reverse=True)
        sd = math.sqrt(sum(st.pvariance(samples[(t, s)]) for s in ex)) / len(ex)
        share = f"{100 * (top[0] + top[1]) / sum(top):.0f}%" if sum(top) > 0.005 else "-"
        largest = sorted(ex, key=ex.get, reverse=True)[:2]
        print(f"budget {t:>2}: mean excess {st.mean(top):+.3f} (label-noise SD {sd:.3f}), worse than expected {sum(e > 0 for e in top)} of {len(top)}, "
              f"two largest {share}{' (same benchmark)' if bench[largest[0]] == bench[largest[1]] else ''}; z " + " ".join(f"{x:+.1f}" for x in z))


def synthetic_run(job, reps=300, seed=0):
    (name, sigma, rho, q, s_out, w, first), target = job
    pairs, counts = bp.pairs[SID], vf.eval_counts(SID)
    rng = random.Random(seed)

    def score(y, r, rho_s):
        if rng.random() < q:
            return s_out * rng.gauss(0, 1)
        sd = math.sqrt(max(r * (1 - r), 1e-6))
        return sigma * (rho_s * (y - r) / sd + math.sqrt(1 - rho_s * rho_s) * rng.gauss(0, 1))

    excess = dict.fromkeys(sim.BUDGETS, 0.0)
    for _ in range(reps):
        uninformative = {b: rng.random() < w for _, b, _ in pairs}
        rho_of = {s: 0.0 if uninformative[b] else rho for s, b, _ in pairs}
        acq = {s: [int(rng.random() < r) for _ in range(31)] for s, _, r in pairs}
        lab = {s: [score(y, r, rho_of[s]) for y in acq[s]] for s, _, r in pairs}
        if first is not None:
            lab[next(s for s, b, _ in pairs if b == target)][0] = first
        ev = {}
        for s, b, r in pairs:
            n = counts[(s, b)]
            ys = [1] * round(r * n) + [0] * (n - round(r * n))
            rng.shuffle(ys)
            ev[s] = [(y, score(y, r, rho_of[s])) for y in ys]
        for t in sim.BUDGETS:
            labeled = [[proxy._inp(s, b, i), y] for s, b, _ in pairs for i, y in enumerate(acq[s][:t])]
            summary = model._summarize(labeled)
            total = 0.0
            for s, b, r in pairs:
                same = [x for s2, b2, _ in pairs if b2 == b for x in lab[s2][:t]]
                if not same:  # no labels on the benchmark: no shift
                    continue
                p = model.predict_with(model.PARAMS, proxy._inp(s, b, "eval"), labeled, summary)
                qq = min(1 - 1e-4, max(1e-4, p))
                base, acc = math.log(qq / (1 - qq)), 0.0
                for y, z in ev[s]:
                    c = (sum(same) + z) / (len(same) + 1)
                    shifted = 1 / (1 + math.exp(-(base + GAMMA * (z - c))))
                    shifted = min(1 - model.PARAMS["CLIP"], max(model.PARAMS["CLIP"], shifted))
                    acc += (shifted - y) ** 2 - (p - y) ** 2
                total += acc / len(ev[s])
            excess[t] += total / len(pairs) / reps
    return job[0], excess


def synthetic():
    target = loss_benchmark()
    with Pool(len(SYNTHETIC)) as pool:
        for (name, sigma, rho, q, s_out, w, first), excess in pool.map(synthetic_run, [(c, target) for c in SYNTHETIC]):
            extra = (f", {100 * q:.0f}% of items with spread {s_out:g} and rho 0" if q else "") + \
                    (f", rho 0 on each benchmark with probability {w:g}" if w else "") + \
                    (f", score {first:g} at the first revealed item of the benchmark with the largest loss" if first is not None else "")
            print(f"{name} (sigma {sigma:g}, rho {rho:g}{extra}): " + "  ".join(f"b{t} {excess[t]:+.3f}" for t in sim.BUDGETS[1:]))


def training():
    import numpy as np
    import pandas as pd
    from llm_difficulty import _bench_dirs, load_eval_data, lobo_probe

    df, z, y, testable = load_eval_data("Qwen/Qwen3-32B")
    pred = lobo_probe(df, z["last_64"].astype(np.float32), y, testable, 1e4)
    scores = pred / pred[df.bench.isin(testable).values].std()  # the standardization of HBS+ItemLLM
    spread, corr = [], []
    for root, b in _bench_dirs():
        if b not in testable:
            continue
        te = (df.bench == b).values
        zb = pd.Series(scores[te], index=df.item_id[te].values)
        resp = pd.read_parquet(f"{root}/{b}/response.parquet", columns=["subject_id", "item_id", "response"])
        resp = resp[resp.response.isin([0.0, 1.0]) & resp.item_id.isin(zb.index)]
        rhos = [np.corrcoef(zb.loc[g.item_id].values, g.response.values)[0, 1]
                for _, g in resp.groupby("subject_id") if len(g) >= 20 and 0 < g.response.mean() < 1]
        spread.append(zb.std())
        corr.append(np.mean(rhos))
    print("training benchmarks: within-benchmark SD of the scores " + ", ".join(f"{v:.2f}" for v in spread)
          + "; mean within-subject correlation with the outcomes " + ", ".join(f"{v:+.2f}" for v in corr))


if __name__ == "__main__":
    for name in sys.argv[1:] or ["breakdown", "synthetic", "training"]:
        {"breakdown": breakdown, "synthetic": synthetic, "training": training}[name]()
