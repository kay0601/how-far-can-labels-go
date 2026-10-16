"""Numbers behind the report's figures, written to ../report/figures/figure_data.json.

- gap_by_budget: mean Brier per budget on the 69 proxy pairs for v5, for v5 with each
  benchmark's mean success rate known in advance (estimated from the other reports,
  ALPHA 6), for HBS+Prior (v7: PRIOR_MEAN from the LLM benchmark priors measured by the
  probe submission, read from its private scoring log; the budget-0 running mean is
  modelled as in v7_feedback.running_b0), and for the oracle that knows every pair's rate.
  Only these aggregate curves are saved, not the per-benchmark priors.
- anchored: per held-out training benchmark, the actual success rate and the mean
  prediction of the anchored LLM prompts (llm_anchor.py caches; Qwen3-8B, Qwen3.5-9B, Qwen3.8-27B).
- v6_by_budget: real minus proxy Brier per budget for v1, v5 and v6 (proxy_validation.py).

    <venv python> figure_data.py
"""

from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd

import contextlib
import io

import bench_prior_value as bp
import llm_anchor
import model
import proxy
import proxy_validation as pv
import sim
import v7_feedback

PROBE = "978024"  # the v7 probe submission
OUT = Path(__file__).resolve().parent.parent / "report" / "figures" / "figure_data.json"


def gap_by_budget(reps=1000, seed=0):
    known = lambda sid, b: dict(model.PARAMS, PRIOR_MEAN=bp.loro_mean(b, sid), ALPHA=6.0)
    with contextlib.redirect_stdout(io.StringIO()):
        priors = v7_feedback.probe_priors(PROBE)
    llm = lambda sid, b: dict(model.PARAMS, PRIOR_MEAN=priors[b])
    rng = random.Random(seed)
    out = {"v5": [0.0] * 6, "benchmark known": [0.0] * 6, "HBS+Prior": [0.0] * 6}
    n = sum(len(ps) for ps in bp.pairs.values())
    for _ in range(reps):
        for sid, ps in bp.pairs.items():
            acq = {s: [int(rng.random() < r) for _ in range(31)] for s, _, r in ps}
            for bi, budget in enumerate(sim.BUDGETS):
                labeled = [[proxy._inp(s, b, i), y] for s, b, _ in ps for i, y in enumerate(acq[s][:budget])]
                summary = model._summarize(labeled)
                for s, b, r in ps:
                    for name, params in (("v5", model.PARAMS), ("benchmark known", known(sid, b)),
                                         ("HBS+Prior", llm(sid, b))):
                        p = model.predict_with(params, proxy._inp(s, b, "eval"), labeled, summary)
                        out[name][bi] += (r * (1 - r) + (p - r) ** 2) / (reps * n)
    # budget 0 of HBS+Prior: the prior grows item by item (running mean of ratings), not the full mean
    counts_probe = v7_feedback.eval_counts(PROBE)
    n_rated = {}
    for (_, b), k in counts_probe.items():
        n_rated[b] = n_rated.get(b, 0) + k
    out["HBS+Prior"][0] = st.mean(v7_feedback.running_b0(r, priors[b], v7_feedback.eval_counts(sid)[(s, b)], n_rated[b], 0.205)
                                  for sid, ps in bp.pairs.items() for s, b, r in ps)
    out["oracle"] = [st.mean(r * (1 - r) for ps in bp.pairs.values() for _, _, r in ps)] * 6
    out["alc"] = {k: sum(w * v for w, v in zip(sim.WEIGHTS, vals)) for k, vals in out.items()}
    return out


def anchored():
    rows = {}
    df = llm_anchor.item_rates().set_index("key")
    resp = []
    for r, b in llm_anchor._bench_dirs():
        if b in llm_anchor.BENCHES:
            x = pd.read_parquet(f"{r}/{b}/response.parquet", columns=["item_id", "response"])
            x = x[x.response.isin([0.0, 1.0])]
            resp.append(x.assign(key=b + "/" + x.item_id.astype(str)))
    resp = pd.concat(resp, ignore_index=True)
    for name in ("Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B"):
        z = np.load(llm_anchor.cache_path(name), allow_pickle=True)
        probs = z["probs"] / z["probs"].sum(1, keepdims=True)
        p = pd.Series(np.clip(((probs * np.arange(10)).sum(1) + 0.5) / 10, 0.01, 0.99), index=z["keys"])
        g = resp[resp.key.isin(p.index)].copy()
        g["p"] = p.loc[g.key].values
        g["bench"] = g.key.str.split("/").str[0]
        rows[name] = {b: {"actual": float(x.response.mean()), "predicted": float(x.p.mean())} for b, x in g.groupby("bench")}
    return rows


def v6_by_budget():
    v1 = pv._load("versions/model_v1.py", "model_v1")
    out = {}
    for name, sid, mod, params, pairs in (("v1", "951748", v1, v1.PARAMS, proxy.DRAWS["report1"]),
                                          ("v5", "952533", model, model.PARAMS, proxy.DRAWS["report3"]),
                                          ("v6", "960385", model, model.PARAMS, bp.pairs["960385"])):
        real, exp = pv.real_budgets(sid), pv.proxy_budgets(mod, params, pairs)
        out[name] = [real[b] - exp[b] for b in sim.BUDGETS]
    return out


if __name__ == "__main__":
    data = {"budgets": list(sim.BUDGETS), "gap_by_budget": gap_by_budget(), "anchored": anchored(),
            "v6_by_budget": v6_by_budget()}
    OUT.write_text(json.dumps(data, indent=1))
    print(json.dumps(data["gap_by_budget"]["alc"], indent=1))
    print({k: [round(v, 3) for v in vals] for k, vals in data["gap_by_budget"].items() if k != "alc"})
    print(json.dumps(data["anchored"], indent=1))
    print({k: [round(v, 3) for v in vals] for k, vals in data["v6_by_budget"].items()})
