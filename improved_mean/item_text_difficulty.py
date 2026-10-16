"""Feasibility: does item text predict item difficulty on an *unseen* benchmark?

Item difficulty = mean over subjects of (label - subject's rate on that
benchmark). Items are embedded with a small local HF model (the runtime cannot
call external embedding APIs). A ridge regression is trained on all other
benchmarks and scored on the held-out one by Spearman correlation.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

import paths

MODEL = "BAAI/bge-small-en-v1.5"
ROOTS = paths.DATA_DIRS
CACHE = os.path.expanduser("~/AIMS/improved_mean/cache")
MAX_TOKENS = 256


def load_items():
    rows = []
    for root in ROOTS:
        for d in sorted(glob.glob(f"{root}/*/")):
            bench = os.path.basename(d.rstrip("/"))
            if bench in ("mmdocrag", "embed"):
                continue
            items = pd.read_parquet(d + "items.parquet")[["item_id", "content"]]
            resp = pd.read_parquet(d + "response.parquet")
            resp = resp[resp.response.isin([0.0, 1.0])]
            resp["res"] = resp.response - resp.groupby("subject_id").response.transform("mean")
            diff = resp.groupby("item_id").agg(diff=("res", "mean"), n_subj=("subject_id", "nunique"))
            df = items.merge(diff, left_on="item_id", right_index=True)
            df["bench"] = bench
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def embed(texts, name=MODEL, batch=64):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(os.cpu_count() or 8)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
            cls = model(**enc).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(cls, dim=-1).numpy())
            if i // batch % 20 == 0:
                print(f"  embedded {i + len(texts[i:i + batch])}/{len(texts)}", flush=True)
    return np.vstack(out)


def get_embeddings(df):
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{MODEL.replace('/', '_')}_{MAX_TOKENS}.npz"
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        cached = dict(zip(z["keys"], z["emb"]))
        if all(f"{b}/{i}" in cached for b, i in zip(df.bench, df.item_id)):
            return np.vstack([cached[f"{b}/{i}"] for b, i in zip(df.bench, df.item_id)])
    emb = embed(df.content.fillna("").tolist())
    np.savez(path, keys=np.array([f"{b}/{i}" for b, i in zip(df.bench, df.item_id)]), emb=emb)
    return emb


def main():
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge

    df = load_items()
    print(df.groupby("bench").agg(items=("diff", "size"), subjects=("n_subj", "median")))
    X = get_embeddings(df)
    # benchmarks with a single subject give only noisy outcomes; use them for training only
    testable = [b for b in df.bench.unique() if df[df.bench == b].n_subj.median() >= 3]
    for lam in (1.0, 10.0, 100.0):
        rhos = []
        for held in testable:
            tr, te = df.bench != held, df.bench == held
            y = df["diff"] - df.groupby("bench")["diff"].transform("mean")  # difficulty is relative within a benchmark
            m = Ridge(alpha=lam).fit(X[tr], y[tr])
            rhos.append(spearmanr(m.predict(X[te]), y[te]).statistic)
        print(f"ridge alpha={lam:>5}: LOBO spearman " + ", ".join(f"{b}={r:.3f}" for b, r in zip(testable, rhos)) + f" | mean={np.mean(rhos):.3f}")
    # upper reference: within-benchmark 5-fold (text -> difficulty learnable at all?)
    from sklearn.model_selection import cross_val_predict
    for b in testable:
        te = (df.bench == b).values
        pred = cross_val_predict(Ridge(alpha=10.0), X[te], df["diff"][te], cv=5)
        print(f"within-benchmark CV spearman {b}: {spearmanr(pred, df['diff'][te]).statistic:.3f}")


if __name__ == "__main__":
    main()
