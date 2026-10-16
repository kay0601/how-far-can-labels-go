"""End-to-end check of a staged v7 submission on a local GPU, through the official worker pool.

Takes one realistic subject-benchmark pair from each training benchmark with several
subjects per item (real item text), runs the staged model.py in the official prediction
worker (tools/prediction_workers.py: one process, 120 s per call, restarted per budget as
on the platform) at budgets 0, 1 and 7, and checks that
- every call returns within its time limit;
- budget 0 equals v5 exactly (so budget-0 feedback still identifies each pair's rate), or, with
  prior_at_budget_0 (the final submission), v5 with the running mean of the ratings so far;
- later budgets equal v5 with PRIOR_MEAN = (sum of the benchmark's ratings + K0 * 0.35) / (n + K0),
  recomputed from the ratings the run cached on disk;
- the ratings agree with the offline anchored ratings (nine anchors instead of twelve).

    CUDA_VISIBLE_DEVICES=<idle gpu> HF_HUB_OFFLINE=1 <gpu env python> test_v7.py [stage dir]
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402
sys.path.insert(0, paths.TOOLS)
from prediction_workers import parallel_predictions  # noqa: E402

import llm_anchor  # noqa: E402
import model as v5  # noqa: E402
import sim  # noqa: E402
from item_text_difficulty import load_items  # noqa: E402

STAGE = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "dist" / "v7_stage_strict"
BUDGETS = (0, 1, 7)
N_EVAL = 12
K0 = 5.0
BENCHES = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")


def main():
    items = load_items()
    content = {(b, i): c or "" for b, i, c in zip(items.bench, items.item_id, items.content)}
    pairs, seen = [], set()
    for seed in range(11, 60):
        for k, acq, ev in sim.build_draws(1, seed=seed)[0]:
            src = k[0].split("#")[0]
            if src in BENCHES and src not in seen and len(acq) >= max(BUDGETS):
                pairs.append((k, acq, ev[:N_EVAL]))
                seen.add(src)
        if len(seen) == len(BENCHES):
            break
    print("pairs:", [k[0] for k, _, _ in pairs], flush=True)
    for folder in (STAGE / ".v7_ratings", Path("/tmp") / "paiec_v7_ratings"):
        shutil.rmtree(folder, ignore_errors=True)

    def inp(k, item_id):
        src = k[0].split("#")[0]
        return [dict(zip(sim.SUBJECT_FIELDS, k[1])),
                {"item_content": content[(src, item_id)], "item_features": sim.ITEM_FEATURES.get((src, item_id), ""),
                 "interactors": "", "benchmark_id": k[0]}]

    results = {}
    for budget in BUDGETS:
        labeled = [[inp(k, i), y] for k, acq, _ in pairs for i, y in acq[:budget]]
        inputs = [inp(k, i) for k, _, ev in pairs for i, _ in ev]
        t0, got = time.time(), [None] * len(inputs)
        for index, value in parallel_predictions(str(STAGE / "model.py"), inputs, labeled, concurrency=1,
                                                 deadline=time.monotonic() + 3600, per_call_timeout=120):
            got[index] = value
        results[budget] = (inputs, labeled, got, time.time() - t0)
        missing = sum(g is None for g in got)
        print(f"budget {budget}: {len(inputs)} calls in {time.time() - t0:.0f} s (incl. model load), "
              f"{missing} without a prediction", flush=True)

    ratings, by_key = {}, {}
    for path in (STAGE / ".v7_ratings").glob("*.json"):
        for key, (bench, rating) in json.loads(path.read_text()).items():
            ratings.setdefault(bench, []).append(rating)
            by_key[key] = rating
    config = json.loads((STAGE / "config.json").read_text())
    print("ratings per benchmark:", {b: (len(r), round(float(np.mean(r)), 3)) for b, r in ratings.items()})

    for budget, (inputs, labeled, got, _) in results.items():
        got = np.array(got, dtype=float)
        base = np.array([v5.predict(x, labeled) for x in inputs])
        if budget == 0 and config.get("prior_at_budget_0"):
            sums, want = {}, []
            for x in inputs:  # one worker scores the inputs in order; the running mean includes this item
                bench, text = x[1]["benchmark_id"], x[1]["item_content"]
                key = hashlib.sha1(f"{bench}\0{text}".encode("utf-8", "replace")).hexdigest()
                total, n = sums.get(bench, (0.0, 0))
                if key not in [k for k in sums.get(bench + "/keys", [])]:
                    total, n = total + by_key[key], n + 1
                    sums[bench + "/keys"] = sums.get(bench + "/keys", []) + [key]
                sums[bench] = (total, n)
                prior = (total + K0 * v5.PARAMS["PRIOR_MEAN"]) / (n + K0)
                want.append(v5.predict_with(dict(v5.PARAMS, PRIOR_MEAN=prior), x, labeled))
            want = np.array(want)
        elif budget == 0:
            want = base
        else:
            prior = {b: (sum(r) + K0 * v5.PARAMS["PRIOR_MEAN"]) / (len(r) + K0) for b, r in ratings.items()}
            want = np.array([v5.predict_with(dict(v5.PARAMS, PRIOR_MEAN=prior[x[1]["benchmark_id"]]), x, labeled)
                             for x in inputs])
        print(f"budget {budget}: |v7 - expected| max {np.nanmax(np.abs(got - want)):.6f}; "
              f"|v7 - v5| mean {np.nanmean(np.abs(got - base)):.4f}", flush=True)

    # agreement with the offline ratings (anchors from the other three benchmarks only)
    z = np.load(llm_anchor.cache_path("Qwen/Qwen3.8-27B"), allow_pickle=True)
    probs = z["probs"] / z["probs"].sum(1, keepdims=True)
    offline = dict(zip(z["keys"], np.clip(((probs * np.arange(10)).sum(1) + 0.5) / 10, 0.01, 0.99)))
    actual = {src: [] for src in BENCHES}
    for k, _, ev in pairs:
        actual[k[0].split("#")[0]] += [float(np.mean(ys)) for _, ys in ev]
    print("offline mean rating vs actual rate on these items:",
          {b: (round(float(np.mean([v for kk, v in offline.items() if kk.startswith(b + "/")])), 3),
               round(float(np.mean(a)), 3)) for b, a in actual.items()})


if __name__ == "__main__":
    main()
