"""Robustness of the real-feedback proxy (report, Section 5 and Appendix E).

1. Pair-level validity: for v1, v4 and v5, each pair's real Brier ALC (from the scoring
   log) against the proxy's expectation for that pair, as z-scores under label noise alone.
   Only z-scores are saved, so no pair's success rate can be read off the output.
2. Root choice: the 43 pairs of the five reports whose budget-0 prediction is unknown (the
   four early reports and HBS+Ability's, whose prediction depends on the subject) have r and
   1 - r told apart by the benchmark's rates in other reports. The same rule is first checked
   on the 26 pairs whose root is known (reports that predict 0.35 at budget 0). The
   comparisons of proxy_baselines.py are then repeated (a) on the known pairs only, (b) with
   every inferred root flipped, (c) with each inferred root flipped at random with the rule's
   upper 95% error rate on the known pairs and (d) with each inferred root flipped with its
   probability of being wrong if rates scatter normally around the benchmark's mean over the
   known pairs, with the known pairs' within-benchmark standard deviation.

    <venv python> proxy_robustness.py      # about 35 minutes on one CPU
"""

from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

import math

from scipy.stats import beta

import bench_prior_value as bp
import model
from feedback_logs import pair_alcs
import proxy
import proxy_baselines as pb
import proxy_validation as pv
import sim

OUT = Path(__file__).resolve().parent.parent / "report" / "figures" / "robustness.json"
AMBIGUOUS = bp.INFERRED  # reports whose budget-0 prediction is unknown, so their roots are inferred
COMPARED = ["v5 (HBS)", "official empirical mean", "v1", "benchmark pooling removed",
            "own-label Bayes predictor", "benchmark rate known (diagnostic)"]


def real_pair_alc(sid):
    """Per-pair Brier ALC from the 'ALC summary' table of a scoring log."""
    return pair_alcs(sid)


def pair_samples(mod, params, pairs, reps=2000, seed=0):
    """Proxy Brier ALC of each pair for each of ``reps`` label sequences (as pv.alc_samples)."""
    rng, out = random.Random(seed), {(s, b): [] for s, b, _ in pairs}
    for _ in range(reps):
        acq = {s: [int(rng.random() < r) for _ in range(31)] for s, _, r in pairs}
        alc = dict.fromkeys(out, 0.0)
        for weight, budget in zip(sim.WEIGHTS, sim.BUDGETS):
            labeled = [[proxy._inp(s, b, i), y] for s, b, _ in pairs for i, y in enumerate(acq[s][:budget])]
            summary = mod._summarize(labeled)
            for s, b, r in pairs:
                p = mod.predict_with(params, proxy._inp(s, b, "eval"), labeled, summary)
                alc[(s, b)] += weight * (r * (1 - r) + (p - r) ** 2)
        for k, v in alc.items():
            out[k].append(v)
    return out


def pair_validity():
    v1 = pv._load("versions/model_v1.py", "model_v1")
    v4 = pv._load("versions/model_v4.py", "model_v4")
    rows = [("v1", "951748", v1, v1.PARAMS, proxy.DRAWS["report1"]),
            ("v4", "951853", v4, dict(v4.PARAMS, _TABLE=proxy.v4_table(v4.PARAMS)), proxy.DRAWS["report2"]),
            ("v5", "952533", model, model.PARAMS, proxy.DRAWS["report3"])]
    zs, expected, real, fixed = {}, [], [], []
    for name, sid, mod, params, pairs in rows:
        samples, observed = pair_samples(mod, params, pairs), real_pair_alc(sid)
        zs[name] = []
        for key, alcs in samples.items():
            mu, sd = st.mean(alcs), st.pstdev(alcs)
            expected.append(mu)
            real.append(observed[key])
            if sd < 1e-9:  # r = 0 or 1: every label sequence is the same, so there is no noise to scale by
                fixed.append(observed[key] - mu)
            else:
                zs[name].append((observed[key] - mu) / sd)
    every = [z for v in zs.values() for z in v]
    print(f"pair-level validity: {len(every)} pairs with label noise, mean z {st.mean(every):+.2f}, "
          f"SD of z {st.pstdev(every):.2f}, |z| <= 2 for {sum(abs(z) <= 2 for z in every)}, "
          f"max |z| {max(map(abs, every)):.2f}; {len(fixed)} noiseless pairs, real minus proxy "
          f"{', '.join(f'{d:+.4f}' for d in fixed)}; Pearson(expected, real) over all {len(real)} pairs "
          f"{st.correlation(expected, real):.3f}")
    return {"z": zs, "noiseless_diffs": fixed}


def root_rule_check():
    """Apply the root rule to the pairs whose root is known; return the upper 95% error rate."""
    known = [sid for sid in bp.pairs if sid not in AMBIGUOUS]
    wrong = total = within = 0
    for sid in known:
        for _, b, r in bp.pairs[sid]:
            ref = [r2 for sid2 in known if sid2 != sid for _, b2, r2 in bp.pairs[sid2] if b2 == b]
            if not ref:
                continue
            lo = (1 - math.sqrt(max(0.0, 1 - 4 * r * (1 - r)))) / 2
            pick = min((lo, 1 - lo), key=lambda x: abs(x - st.mean(ref)))
            total += 1
            wrong += abs(pick - r) > 1e-9
            within += abs(pick - r) <= 0.1
    upper = float(beta.ppf(0.975, wrong + 1, total - wrong))
    print(f"root rule on {total} known pairs: {total - wrong} correct, {within} within 0.1; upper 95% error rate {upper:.3f}")
    return {"pairs": total, "correct": total - wrong, "within_0.1": within, "upper_error": upper}


def flipped(flip):
    """bp.pairs with r -> 1 - r for the ambiguous pairs where ``flip(sid, s)`` is true."""
    return {sid: [(s, b, 1 - r if sid in AMBIGUOUS and flip(sid, s) else r) for s, b, r in ps]
            for sid, ps in bp.pairs.items()}


def comparisons(pairs, reps):
    methods = {m: pb.METHODS[m] for m in COMPARED}
    res = pb.per_report(pairs, methods, reps=reps)
    size = sum(len(ps) for ps in pairs.values())
    alc = {m: sum(res[m].values()) / size for m in methods}
    oracle = sum(r * (1 - r) for ps in pairs.values() for _, _, r in ps) / size
    ref = "v5 (HBS)"
    diffs = {m: alc[m] - alc[ref] for m in methods if m != ref}
    share = (alc[ref] - alc["benchmark rate known (diagnostic)"]) / (alc[ref] - oracle)
    return diffs, share


def flip_probabilities():
    """Probability that each inferred root is wrong under a normal model of rates around the
    benchmark's mean over the known pairs, with their within-benchmark standard deviation."""
    by_b = {}
    for sid, ps in bp.pairs.items():
        if sid not in AMBIGUOUS:
            for _, b, r in ps:
                by_b.setdefault(b, []).append(r)
    resid = [r - st.mean(v) for v in by_b.values() if len(v) > 1 for r in v]
    sd = math.sqrt(sum(x * x for x in resid) / max(1, len(resid) - len(by_b)))
    ref = {b: st.mean(v) for b, v in by_b.items()}
    prob = {}
    for sid in AMBIGUOUS:
        for s, b, r in bp.pairs[sid]:
            near = math.exp(-0.5 * ((r - ref[b]) / sd) ** 2)
            far = math.exp(-0.5 * ((1 - r - ref[b]) / sd) ** 2)
            prob[(sid, s)] = far / (near + far)
    return sd, prob


def root_sensitivity(n_random=40, seed=2):
    out = {"rule": root_rule_check()}
    flip_p = out["rule"]["upper_error"]
    known = {sid: ps for sid, ps in bp.pairs.items() if sid not in AMBIGUOUS}
    n_known = sum(len(ps) for ps in known.values())
    n_inferred = sum(len(bp.pairs[sid]) for sid in AMBIGUOUS)
    for name, pairs, reps in ((f"{n_known} known pairs", known, 2000),
                              (f"{n_inferred} roots flipped", flipped(lambda sid, s: True), 2000)):
        diffs, share = comparisons(pairs, reps)
        out[name] = {"diffs": diffs, "share": share}
        print(f"{name}: " + ", ".join(f"{m} {d:+.4f}" for m, d in diffs.items()) + f"; share {share:.2f}")
    rng, runs = random.Random(seed), []
    for _ in range(n_random):
        coin = {(sid, s): rng.random() < flip_p for sid in AMBIGUOUS for s, _, _ in bp.pairs[sid]}
        runs.append(comparisons(flipped(lambda sid, s: coin[(sid, s)]), reps=300))
    expected_sign = {"official empirical mean": 1, "v1": 1, "benchmark pooling removed": 1,
                     "own-label Bayes predictor": 1, "benchmark rate known (diagnostic)": -1}
    kept = sum(all(d[m] * sgn > 0 for m, sgn in expected_sign.items()) for d, _ in runs)
    print(f"random flips (p = {flip_p:.3f}): every sign kept in {kept} of {n_random} assignments")
    for m in expected_sign:
        vals = [d[m] for d, _ in runs]
        print(f"  {m}: {min(vals):+.4f} to {max(vals):+.4f}")
    shares = [s for _, s in runs]
    print(f"  share: {min(shares):.2f} to {max(shares):.2f}")
    out["random"] = {"flip_probability": flip_p, "assignments": n_random, "signs_kept": kept,
                     "ranges": {m: [min(d[m] for d, _ in runs), max(d[m] for d, _ in runs)] for m in expected_sign},
                     "share": [min(shares), max(shares)]}
    sd, prob = flip_probabilities()
    rng, runs = random.Random(seed + 1), []
    for _ in range(n_random):
        coin = {k: rng.random() < p for k, p in prob.items()}
        runs.append(comparisons(flipped(lambda sid, s: coin[(sid, s)]), reps=300))
    kept = sum(all(d[m] * sgn > 0 for m, sgn in expected_sign.items()) for d, _ in runs)
    print(f"weighted flips (SD {sd:.3f}; {sum(prob.values()):.1f} expected flips of {len(prob)}, "
          f"{sum(p > 0.1 for p in prob.values())} pairs above 0.1): every sign kept in {kept} of {n_random} assignments")
    for m in expected_sign:
        vals = [d[m] for d, _ in runs]
        print(f"  {m}: {min(vals):+.4f} to {max(vals):+.4f}")
    shares = [s for _, s in runs]
    print(f"  share: {min(shares):.2f} to {max(shares):.2f}")
    out["weighted"] = {"sd": sd, "expected_flips": sum(prob.values()), "assignments": n_random, "signs_kept": kept,
                       "ranges": {m: [min(d[m] for d, _ in runs), max(d[m] for d, _ in runs)] for m in expected_sign},
                       "share": [min(shares), max(shares)]}
    return out


if __name__ == "__main__":
    data = {"pair_z": pair_validity(), "roots": root_sensitivity()}
    OUT.write_text(json.dumps(data, indent=1))
    print("wrote", OUT)
