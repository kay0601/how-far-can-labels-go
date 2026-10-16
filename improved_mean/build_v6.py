"""Build the v6 submission ZIPs: v6/ files, v5's model.py as v5.py, and the probe.

The probe is the ridge of experiment 11 refitted on all training items (Qwen3-32B
last-layer features from llm_difficulty.py); its scores are divided by the spread
of its leave-one-benchmark-out predictions, the scale llm_sim.py tuned GAMMA on.

    <venv python> build_v6.py
        dist/improved_mean_v6_strict.zip  raises when the LLM cannot load (to check it runs)
        dist/improved_mean_v6.zip         falls back to plain v5 instead (final candidate)
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from llm_difficulty import load_eval_data, lobo_probe

HERE = Path(__file__).resolve().parent
LLM, LAYER, ALPHA = "Qwen/Qwen3-32B", "last_64", 1e4
FILES = ("model.py", "models.txt", "requirements.txt", "prediction_config.json")


def fit_probe(path):
    df, z, y, testable = load_eval_data(LLM)
    X = z[LAYER].astype(np.float32)
    scaler = StandardScaler().fit(X)
    ridge = Ridge(alpha=ALPHA).fit(scaler.transform(X), y)
    lobo = lobo_probe(df, X, y, testable, ALPHA)
    sd = float(lobo[df.bench.isin(testable).values].std())
    np.savez(path, mean=scaler.mean_, scale=scaler.scale_, coef=ridge.coef_, intercept=ridge.intercept_, sd=sd)
    print(f"probe: {X.shape[0]} items, {X.shape[1]} features, score sd {sd:.4f}")


def build(stage, require_llm, out):
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for name in FILES:
        shutil.copy(HERE / "v6" / name, stage / name)
    shutil.copy(HERE / "model.py", stage / "v5.py")
    shutil.copy(HERE / "dist" / "v6_probe.npz", stage / "probe.npz")
    (stage / "config.json").write_text(json.dumps({"require_llm": require_llm}) + "\n")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(stage.iterdir()):
            zf.write(path, path.name)
    print(f"{out.name}: {sorted(p.name for p in stage.iterdir())}")


if __name__ == "__main__":
    (HERE / "dist").mkdir(exist_ok=True)
    fit_probe(HERE / "dist" / "v6_probe.npz")
    build(HERE / "dist" / "v6_stage_strict", True, HERE / "dist" / "improved_mean_v6_strict.zip")
    build(HERE / "dist" / "v6_stage", False, HERE / "dist" / "improved_mean_v6.zip")
