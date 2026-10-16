"""Does subject metadata predict a subject's success rate on an unseen benchmark?

For each held-out benchmark, subject ability is learned from the other
benchmarks only (leave-one-benchmark-out):

    logit(rate[subject, bench]) = bench_effect + ability[subject]

Ability for a held-out pair comes from (a) a name lookup when the model was
seen on other benchmarks, else (b) a ridge regression on metadata features
(release date, reasoning effort, provider). We report the rank correlation
between predicted ability and the actual held-out rate, and the budget-0 Brier
of a prior shifted by predicted ability versus the constant prior.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

import paths

ROOTS = paths.DATA_DIRS
SKIP = {"mmdocrag"}
EFFORT = {"": np.nan, "minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def model_name(row):
    """Normalized model identity; matharena often leaves normalized_name blank."""
    if row.normalized_name:
        return row.normalized_name.lower()
    m = re.search(r"source_model_name=([^;]+)", row.subject_features_extra)
    return m.group(1).lower() if m else ""


def load_pairs():
    rows = []
    for root in ROOTS:
        for d in sorted(glob.glob(f"{root}/*/")):
            bench = os.path.basename(d.rstrip("/"))
            if bench in SKIP:
                continue
            resp = pd.read_parquet(d + "response.parquet")
            resp = resp[resp.response.isin([0.0, 1.0])]
            subj = pd.read_parquet(d + "subjects.parquet").fillna("").set_index("subject_id")
            g = resp.groupby("subject_id").agg(rate=("response", "mean"), n_items=("item_id", "nunique"))
            g = g[g.n_items >= 80].join(subj)
            g["bench"] = bench
            rows.append(g.reset_index())
    df = pd.concat(rows, ignore_index=True)
    df["name"] = [model_name(r) for r in df.itertuples()]
    df["year"] = pd.to_datetime(df.release_date, errors="coerce").map(
        lambda t: t.year + t.dayofyear / 365 if pd.notna(t) else np.nan)
    df["effort"] = df.reasoning_effort.map(EFFORT)
    df["y"] = logit(df.rate.values)
    return df


def fit_abilities(train):
    """Alternating least squares for bench effects and subject abilities."""
    ability = pd.Series(0.0, index=train.name.unique())
    for _ in range(50):
        bench_eff = (train.y - train.name.map(ability)).groupby(train.bench).mean()
        resid = train.y - train.bench.map(bench_eff)
        ability = resid.groupby(train.name).mean()
        ability -= ability.mean()
    return bench_eff, ability


def features(df, year_mean):
    x = pd.DataFrame(index=df.index)
    x["year"] = df.year.fillna(year_mean) - year_mean
    x["year_missing"] = df.year.isna().astype(float)
    x["effort"] = df.effort.fillna(0)
    x["effort_missing"] = df.effort.isna().astype(float)
    return x.values


def ridge(x, y, lam=1.0):
    x1 = np.c_[np.ones(len(x)), x]
    reg = lam * np.eye(x1.shape[1])
    reg[0, 0] = 0
    return np.linalg.solve(x1.T @ x1 + reg, x1.T @ y)


def main():
    df = load_pairs()
    print(df.groupby("bench").agg(pairs=("rate", "size"), mean_rate=("rate", "mean"), sd_logit=("y", "std")).round(3))
    between = df.groupby("bench").y.mean().var()
    within = df.groupby("bench").y.var().mean()
    print(f"\nlogit variance: between benchmarks={between:.2f}, within benchmark (between subjects)={within:.2f}")

    out = []
    for held in sorted(df.bench.unique()):
        train, test = df[df.bench != held], df[df.bench == held].copy()
        _, ability = fit_abilities(train)
        # metadata regression for subjects never seen on another benchmark
        tr = train.assign(ab=train.name.map(ability))
        ym = tr.year.mean()
        w = ridge(features(tr, ym), tr.ab.values)
        meta = np.c_[np.ones(len(test)), features(test, ym)] @ w
        seen = test.name.map(ability)
        test["pred"] = seen.fillna(pd.Series(meta, index=test.index))
        test["seen"] = seen.notna()
        rho = test[["pred", "rate"]].corr(method="spearman").iloc[0, 1]
        rho_seen = test[test.seen][["pred", "rate"]].corr(method="spearman").iloc[0, 1] if test.seen.sum() > 2 else np.nan
        out.append((held, len(test), int(test.seen.sum()), rho, rho_seen))
    print("\nLOBO rank correlation between predicted ability and held-out success rate")
    print(pd.DataFrame(out, columns=["held_out", "pairs", "name_seen", "spearman_all", "spearman_seen"]).round(3).to_string(index=False))




FEATURES = ["year", "year_missing", "effort", "effort_missing"]


def build_table(df):
    """Ability lookup + metadata fallback, as stored in ability_table.json."""
    _, ability = fit_abilities(df)
    tr = df.assign(ab=df.name.map(ability))
    ym = float(tr.year.mean())
    w = ridge(features(tr, ym), tr.ab.values)
    return {"names": {k: round(float(v), 4) for k, v in ability.items() if k},
            "ridge": [float(x) for x in w], "year_mean": ym}


def build_all():
    """Full table for the submission plus one leave-one-benchmark-out table per benchmark."""
    import json
    df = load_pairs()
    with open("ability_table.json", "w") as f:
        json.dump(build_table(df), f, indent=0, sort_keys=True)
    lobo = {b: build_table(df[df.bench != b]) for b in df.bench.unique()}
    with open("versions/ability_tables_lobo.json", "w") as f:
        json.dump(lobo, f)
    print("wrote ability_table.json and versions/ability_tables_lobo.json")


if __name__ == "__main__":
    import sys
    build_all() if "--build" in sys.argv else main()
