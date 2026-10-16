"""End-to-end check of a staged v6 submission on a local GPU, through the official worker pool.

Takes a few realistic subject-benchmark pairs from the public training data (real
item text), runs the staged model.py in the official prediction worker
(tools/prediction_workers.py: one process, 120 s per call, restarted per budget as
on the platform) at budgets 0, 1 and 7, and compares every prediction with the
offline formula: v5 + GAMMA * (s_i - c), scores from the cached Qwen3-32B features.

    CUDA_VISIBLE_DEVICES=<idle gpu> HF_HUB_OFFLINE=1 <gpu env python> test_v6.py [stage dir]
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402
sys.path.insert(0, paths.TOOLS)
from prediction_workers import parallel_predictions  # noqa: E402

import model as v5  # noqa: E402
import sim  # noqa: E402
from item_text_difficulty import load_items  # noqa: E402
from llm_difficulty import cache_path  # noqa: E402

STAGE = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "dist" / "v6_stage_strict"
BUDGETS = (0, 1, 7)
N_PAIRS, N_EVAL = 3, 12
GAMMA = 0.5
BENCHES = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")


def main():
    items = load_items()
    content = {(b, i): c or "" for b, i, c in zip(items.bench, items.item_id, items.content)}
    z = np.load(cache_path("Qwen/Qwen3-32B"), allow_pickle=True)
    probe = np.load(STAGE / "probe.npz")
    s = (((z["last_64"].astype(np.float32) - probe["mean"]) / probe["scale"]) @ probe["coef"] + probe["intercept"]) / probe["sd"]
    score = {content[tuple(k.split("/", 1))]: v for k, v in zip(z["keys"], s) if tuple(k.split("/", 1)) in content}

    pairs, seen = [], set()
    for k, acq, ev in sim.build_draws(1, seed=11)[0]:
        src = k[0].split("#")[0]
        if src in BENCHES and src not in seen and len(acq) >= max(BUDGETS):
            pairs.append((k, acq, ev[:N_EVAL]))
            seen.add(src)
    pairs = pairs[:N_PAIRS]
    print("pairs:", [k[0] for k, _, _ in pairs], flush=True)

    def inp(k, item_id):
        src = k[0].split("#")[0]
        return [dict(zip(sim.SUBJECT_FIELDS, k[1])),
                {"item_content": content[(src, item_id)], "item_features": sim.ITEM_FEATURES.get((src, item_id), ""),
                 "interactors": "", "benchmark_id": k[0]}]

    def expected(x, labeled):
        p = v5.predict(x, labeled)
        same = [it["item_content"] for (_, it), _ in labeled if it["benchmark_id"] == x[1]["benchmark_id"]]
        if not same:
            return p
        vals = [score[c] for c in same] + [score[x[1]["item_content"]]]
        q = min(1 - 1e-4, max(1e-4, p))
        shifted = 1 / (1 + math.exp(-(math.log(q / (1 - q)) + GAMMA * (vals[-1] - sum(vals) / len(vals)))))
        return min(1 - v5.PARAMS["CLIP"], max(v5.PARAMS["CLIP"], shifted))

    for budget in BUDGETS:
        labeled = [[inp(k, i), y] for k, acq, _ in pairs for i, y in acq[:budget]]
        inputs = [inp(k, i) for k, _, ev in pairs for i, _ in ev]
        t0, got = time.time(), [None] * len(inputs)
        for index, value in parallel_predictions(str(STAGE / "model.py"), inputs, labeled, concurrency=1,
                                                 deadline=time.monotonic() + 3600, per_call_timeout=120):
            got[index] = value
        want = np.array([expected(x, labeled) for x in inputs])
        base = np.array([v5.predict(x, labeled) for x in inputs])
        got = np.array(got)
        print(f"budget {budget}: {len(inputs)} calls in {time.time() - t0:.0f} s (incl. model load); "
              f"|v6 - expected| max {np.abs(got - want).max():.4f} mean {np.abs(got - want).mean():.4f}; "
              f"|v6 - v5| mean {np.abs(got - base).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
