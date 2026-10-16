"""Build the v7 submission ZIPs: v7/ files, v5's model.py as v5.py, and the anchors.

The anchors are the anchored prompt's examples (llm_anchor.py): the items at the 20th,
50th and 80th percentiles of success rate of each of the four training benchmarks with
several subjects per item, truncated with the Qwen3.8 tokenizer exactly as offline and
shuffled into one fixed order. Offline, each held-out benchmark saw the nine anchors of
the other three; a test benchmark sees all twelve.

    <venv python> build_v7.py      # CPU only (tokenizer)
        dist/improved_mean_v7_final.zip   final submission: prior from budget 0, falls back to v5
        dist/improved_mean_v7_strict.zip  prior from budget 1, raises when the LLM cannot load
        dist/improved_mean_v7.zip         prior from budget 1, falls back to plain v5
        dist/improved_mean_v7_probe.zip   measurement submission: budget 1 returns the prior itself
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

import llm_anchor
from llm_difficulty import truncate

HERE = Path(__file__).resolve().parent
LLM = "Qwen/Qwen3.8-27B"
FILES = ("model.py", "models.txt", "requirements.txt", "prediction_config.json")


def write_anchors(path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LLM)
    anchors, _ = llm_anchor.anchors_and_targets(llm_anchor.item_rates())
    shots = [a for b in llm_anchor.BENCHES for a in anchors[b]]
    np.random.default_rng(0).shuffle(shots)
    rows = [{"content": truncate(a.content if isinstance(a.content, str) else "", tok,
                                 llm_anchor.ANCHOR_TOKENS, llm_anchor.ANCHOR_HEAD)[0],
             "pct": 100 * float(a.rate), "digit": min(9, int(a.rate * 10))} for a in shots]
    path.write_text(json.dumps(rows, indent=1))
    print(f"anchors: {len(rows)} items, rates {[round(r['pct']) for r in rows]}")


def build(stage, require_llm, out, probe=False, prior_at_budget_0=False):
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for name in FILES:
        shutil.copy(HERE / "v7" / name, stage / name)
    shutil.copy(HERE / "model.py", stage / "v5.py")
    shutil.copy(HERE / "dist" / "v7_anchors.json", stage / "anchors.json")
    config = {"require_llm": require_llm, "probe": probe, "prior_at_budget_0": prior_at_budget_0}
    (stage / "config.json").write_text(json.dumps(config) + "\n")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(stage.iterdir()):
            zf.write(path, path.name)
    print(f"{out.name}: {sorted(p.name for p in stage.iterdir())}")


if __name__ == "__main__":
    (HERE / "dist").mkdir(exist_ok=True)
    write_anchors(HERE / "dist" / "v7_anchors.json")
    build(HERE / "dist" / "v7_stage_final", False, HERE / "dist" / "improved_mean_v7_final.zip", prior_at_budget_0=True)
    build(HERE / "dist" / "v7_stage_strict", True, HERE / "dist" / "improved_mean_v7_strict.zip")
    build(HERE / "dist" / "v7_stage", False, HERE / "dist" / "improved_mean_v7.zip")
    build(HERE / "dist" / "v7_stage_probe", True, HERE / "dist" / "improved_mean_v7_probe.zip", probe=True)
