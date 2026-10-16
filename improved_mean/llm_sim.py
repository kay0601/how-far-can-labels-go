"""Simulated Brier ALC of v5 plus an item-level shift from LLM difficulty (llm_difficulty.py).

The shift moves v5's estimate in logit space:
    p = sigmoid(logit(p_v5) + GAMMA * (s_i - c)),
s_i = the item's score, standardized on the *other* benchmarks' items, and c =
mean score of the labeled items on the same benchmark plus the current one,
shrunk toward 0 with strength K_C. K_C = 0 only reorders items within a
benchmark; a large K_C also lets the LLM move a whole benchmark at budget 0.

Draws follow the offline simulator of the report (Section 5.4; 400 draws, seed 2024).
swe_rebench pairs and unscored items get no shift; ALC is reported for the four
benchmarks with enough subjects per item, and GAMMA / K_C are also picked
leave-one-benchmark-out.

    <venv python> llm_sim.py [n_draws] [model] [keys_from_model]

keys_from_model restricts the scored items to another model's cache, so two models
compare on equal coverage (unscored items get no shift, which dilutes the gain).
"""

from __future__ import annotations

import itertools
import math
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import model
import sim
from llm_difficulty import cache_path, load_eval_data, lobo_probe

GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
K_CS = (0.0, 3.0, 10.0, 1e9)
LLM = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B"
KEYS_FROM = sys.argv[3] if len(sys.argv) > 3 else None
PROBES = {  # features and ridge alpha, the best LOBO setting per model (llm_difficulty.py eval)
    "Qwen/Qwen3-8B": {"probe last_27": ("last_27", 1e4), "probe mean_mid": ("mean_mid", 1e4)},
    "Qwen/Qwen3-32B": {"probe last_64": ("last_64", 1e4)},
}[LLM]

DRAWS = sim.build_draws(int(sys.argv[1]) if len(sys.argv) > 1 else 400, seed=2024)


def _standardized_scores():
    """{kind: {(bench, item_id): standardized score}} for the testable benchmarks."""
    df, z, y, testable = load_eval_data(LLM)
    if KEYS_FROM:
        keep = np.isin(df.key.values, np.load(cache_path(KEYS_FROM), allow_pickle=True)["keys"])
        df, z = df[keep].reset_index(drop=True), {f: v[keep] for f, v in z.items()}
        y = (df["diff"] - df.groupby("bench")["diff"].transform("mean")).values
    raw = {"zero-shot": z["score"]}
    raw.update({k: lobo_probe(df, z[f].astype(np.float32), y, testable, lam) for k, (f, lam) in PROBES.items()})
    out = {}
    for kind, v in raw.items():
        d = {}
        for b in testable:
            te, tr = (df.bench == b).values, (df.bench.isin(testable) & (df.bench != b)).values
            zs = (v[te] - v[tr].mean()) / v[tr].std()
            d.update({(b, i): float(s) for i, s in zip(df.item_id[te], zs)})
        out[kind] = d
    return out, testable


SCORES, TESTABLE = _standardized_scores()


def _key(item):
    return item["benchmark_id"].split("#")[0], item["item_content"]


def make_fn(kind, gamma, k_c, params=model.PARAMS):
    S = SCORES[kind]

    def summarize(labeled):
        sums = {}
        for (_, item), _ in labeled:
            s = S.get(_key(item))
            if s is not None:
                a = sums.setdefault(item["benchmark_id"], [0.0, 0])
                a[0] += s
                a[1] += 1
        return model._summarize(labeled), sums

    def fn(input, labeled, summary):
        base, sums = summary
        p = model.predict_with(params, input, labeled, base)
        s = S.get(_key(input[1]))
        if s is None or gamma == 0:
            return p
        tot, n = sums.get(input[1]["benchmark_id"], (0.0, 0))
        c = (tot + s) / (n + 1 + k_c)
        p = min(1 - 1e-4, max(1e-4, p))
        q = 1.0 / (1.0 + math.exp(-(math.log(p / (1 - p)) + gamma * (s - c))))
        return min(1 - params["CLIP"], max(params["CLIP"], q))

    return fn, summarize


def run(cfg):
    kind, gamma, k_c = cfg
    fn, summarize = make_fn(kind, gamma, k_c)
    df = sim.evaluate(DRAWS, fn, summarize)
    df = df[df.bench.isin(TESTABLE)]
    per = df.groupby("bench").mean(numeric_only=True)
    return {"kind": kind, "gamma": gamma, "k_c": k_c, **per.alc.to_dict(), "mean": per.alc.mean(),
            **{f"b{b}": df[f"b{b}"].mean() for b in sim.BUDGETS}}


if __name__ == "__main__":
    cfgs = [("zero-shot", 0.0, 0.0)] + [(k, g, c) for k, g, c in itertools.product(SCORES, GAMMAS[1:], K_CS)]
    with Pool(min(len(cfgs), 64)) as pool:
        res = pd.DataFrame(pool.map(run, cfgs))
    base = res.iloc[0]
    pd.set_option("display.width", 250)
    print(f"{LLM}{' on items of ' + KEYS_FROM if KEYS_FROM else ''}: {sum(map(len, SCORES.values())) // len(SCORES)} scored items")
    print(f"{len(DRAWS)} draws; v5 alone: mean ALC over {TESTABLE} = {base['mean']:.4f}")
    print(res.sort_values("mean").round(4).to_string(index=False))
    for kind in SCORES:
        sub = res[(res.kind == kind) | (res.gamma == 0)]
        lobo = []
        for b in TESTABLE:
            others = [o for o in TESTABLE if o != b]
            best = sub.loc[sub[others].mean(axis=1).idxmin()]
            lobo.append(best[b])
            print(f"  {kind} held-out {b}: ALC {best[b]:.4f} (v5 {base[b]:.4f}) with gamma={best.gamma}, k_c={best.k_c:g}")
        print(f"  {kind} LOBO mean ALC {np.mean(lobo):.4f} vs v5 {base['mean']:.4f} (diff {np.mean(lobo) - base['mean']:+.4f})")
