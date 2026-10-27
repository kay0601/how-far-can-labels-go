"""Build the v8 submission ZIP: v7 with the settings chosen on the real-feedback proxy.

v8/model.py is v7/model.py with two config keys: "prior_stretch" spreads the LLM prior
around 0.35, and "params" overrides v5's parameters. The anchors are those of v7
(dist/v7_anchors.json, written by build_v7.py), so the prompt is unchanged.

    <venv python> build_v8.py      # CPU only; run build_v7.py first for the anchors
        dist/improved_mean_v8_final.zip   final submission: prior from budget 0, stretch 1.5,
                                          ALPHA 4, GLOBAL_K 1e9; falls back to v5
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ("model.py", "models.txt", "requirements.txt", "prediction_config.json")
FINAL = {"require_llm": False, "probe": False, "prior_at_budget_0": True,
         "prior_stretch": 1.5, "params": {"ALPHA": 4.0, "GLOBAL_K": 1e9}}


def build(stage, config, out):
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for name in FILES:
        shutil.copy(HERE / "v8" / name, stage / name)
    shutil.copy(HERE / "model.py", stage / "v5.py")
    shutil.copy(HERE / "dist" / "v7_anchors.json", stage / "anchors.json")
    (stage / "config.json").write_text(json.dumps(config) + "\n")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(stage.iterdir()):
            zf.write(path, path.name)
    print(f"{out.name}: {sorted(p.name for p in stage.iterdir())}")


if __name__ == "__main__":
    build(HERE / "dist" / "v8_stage_final", FINAL, HERE / "dist" / "improved_mean_v8_final.zip")
