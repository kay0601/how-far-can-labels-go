"""Read a v7 scoring log against the proxy expectation for v5 (HBS) on the same pairs.

v7's budget-0 prediction is v5's constant 0.35, so each pair's success rate follows from
its budget-0 Brier score and ECE (report, Proposition 1). The proxy then gives v5's
expected Brier per budget on exactly these pairs, with the spread due to label noise.
v7 helps if its real scores at budgets 1 to 7 fall clearly below that expectation, after
allowing for the proxy's usual pessimism toward v5 (real minus proxy -0.006 to -0.010).

    <venv python> v7_feedback.py <codabench submission id>   # log in feedback/<id>_scoring_stdout.txt
    <venv python> v7_feedback.py --probe <id>                 # a v7_probe submission
    <venv python> v7_feedback.py --final <id>                 # did the LLM run in a v7_final submission?

A probe submission returns the benchmark prior itself at budget 1 and the prior plus 0.1
(at most 0.99) at budget 3, constant within each benchmark. Each pair's budget-1 ECE gives
|prior - rate|, and its budget-3 ECE tells which of the two candidates is the prior. With those priors, the proxy compares v7 with v5 on all
pairs of all reports (benchmark identifiers are stable across reports), with 95%
intervals from resampling whole reports.
"""

from __future__ import annotations

import random
import re
import statistics as st
import sys

import bench_prior_value as bp
import model
import proxy_baselines as pb
import proxy_validation as pv
import sim

PRIOR = 0.35
PROBE_SHIFT = 0.1
V5_OFFSET = (-0.0070, -0.0103, -0.0063)  # real minus proxy of v1, v4 and v5 on their own draws (Table 2)


def recovered_pairs(sid):
    out = []
    for s, b, brier, ece in bp.b0_rows(sid):
        v = max(0.0, brier - ece * ece)
        r = min((PRIOR + ece, PRIOR - ece), key=lambda x: abs(x * (1 - x) - v) if 0 <= x <= 1 else 9)
        out.append((s, b, min(1.0, max(0.0, r))))
    return out


def budget_rows(sid, budget):
    rows, cur = [], None
    for line in open(f"{bp.SUBS}/{sid}_scoring_stdout.txt"):
        m = re.match(r"Label budget: (\d+)", line)
        if m:
            cur = int(m.group(1))
        m = re.match(r"(subject_\d+)\s+(benchmark_\d+)\s+\d+\s+([\d.]+)\s+([\d.]+)", line)
        if m and cur == budget:
            rows.append((m.group(1), m.group(2), float(m.group(3)), float(m.group(4))))
    return rows


def probe_priors(sid):
    """Benchmark priors recovered from a probe report: budget 1 gives |prior - rate|, budget 3
    (prior plus PROBE_SHIFT, at most 0.99) picks the side."""
    rate = {(s, b): r for s, b, r in recovered_pairs(sid)}
    ece3 = {(s, b): ece for s, b, _, ece in budget_rows(sid, 3)}
    shifted = lambda m: min(0.99, m + PROBE_SHIFT)
    votes = {}
    for s, b, brier, ece in budget_rows(sid, 1):
        r = rate[(s, b)]
        resid = brier - (r * (1 - r) + ece * ece)  # zero when the prediction is constant within the pair
        options = [m for m in (r - ece, r + ece) if 0.0 <= m <= 1.0]
        fit = sorted((abs(abs(shifted(m) - r) - ece3[(s, b)]), m) for m in options)
        votes.setdefault(b, []).append(fit[0][1])
        print(f"  {b}: rate {r:.3f}, |prior - rate| {ece:.3f}, candidates {[round(m, 3) for m in options]}, "
              f"budget-3 misfit {[round(f, 4) for f, _ in fit]}, constancy residual {resid:+.5f}")
    return {b: st.mean(v) for b, v in votes.items()}


def compare_v7(priors, extra=None):
    pairs = dict(bp.pairs)
    if extra:
        pairs.update(extra)
    pairs = {sid: [p for p in ps if p[1] in priors] for sid, ps in pairs.items()}
    pairs = {sid: ps for sid, ps in pairs.items() if ps}
    methods = {"v5 (HBS)": pb.METHODS["v5 (HBS)"],
               "v7": ("hbs_after_0", lambda sid, b, loro: dict(model.PARAMS, PRIOR_MEAN=priors[b])),
               "v7 from budget 0": ("hbs", lambda sid, b, loro: dict(model.PARAMS, PRIOR_MEAN=priors[b]))}
    res = pb.per_report(pairs, methods, reps=1000)
    size = {sid: len(ps) for sid, ps in pairs.items()}
    sids = sorted(pairs)
    alc = lambda m, ss: sum(res[m][s] for s in ss) / sum(size[s] for s in ss)
    rng = random.Random(1)
    boots = [[rng.choice(sids) for _ in sids] for _ in range(4000)]
    print(f"{sum(size.values())} pairs in {len(sids)} reports on {len(priors)} benchmarks with a recovered prior")
    for m in methods:
        d = sorted(alc(m, bs) - alc("v5 (HBS)", bs) for bs in boots)
        print(f"  {m:<18} {alc(m, sids):.4f}  change {alc(m, sids) - alc('v5 (HBS)', sids):+.4f} [{d[100]:+.4f}, {d[3899]:+.4f}]")


def eval_counts(sid):
    """Evaluation items per pair, from the budget-0 table of a scoring log."""
    out, cur = {}, None
    for line in open(f"{bp.SUBS}/{sid}_scoring_stdout.txt"):
        m = re.match(r"Label budget: (\d+)", line)
        if m:
            cur = int(m.group(1))
        m = re.match(r"(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+[\d.]+\s+[\d.]+", line)
        if m and cur == 0:
            out[(m.group(1), m.group(2))] = int(m.group(3))
    return out


def running_b0(r, prior, n_items, n_rated, sigma, k0=5.0):
    """Expected budget-0 Brier of a pair when the prior is a running mean of item ratings.

    The k-th item scored sees k ratings (mean mu, spread sigma), shrunk toward 0.35 with k0;
    mu is the unshrunk mean behind the measured prior (prior used n_rated ratings)."""
    mu = (prior * (n_rated + k0) - k0 * PRIOR) / n_rated
    total = 0.0
    for k in range(1, n_items + 1):
        mean = (k * mu + k0 * PRIOR) / (k + k0)
        var = k * sigma ** 2 / (k + k0) ** 2
        total += (mean - r) ** 2 + var
    return r * (1 - r) + total / n_items


def final_estimates(priors, probe_sid, sigma=0.205, k0s=(5.0, 10.0, 20.0)):
    """v7 with the prior from budget 1 (A) and from budget 0 (B), on the 69 pairs of the eight
    earlier reports and on the 78 pairs with the probe report, with a realistic budget-0 running mean."""
    counts_probe = eval_counts(probe_sid)
    n_rated = {}
    for (s, b), n in counts_probe.items():
        n_rated[b] = n_rated.get(b, 0) + n
    for label, extra in (("69 pairs, eight earlier reports", None), ("78 pairs, with the probe report", {probe_sid: recovered_pairs(probe_sid)})):
        pairs = dict(bp.pairs)
        if extra:
            pairs.update(extra)
        pairs = {sid: [p for p in ps if p[1] in priors] for sid, ps in pairs.items()}
        pairs = {sid: ps for sid, ps in pairs.items() if ps}
        methods = {"v5 (HBS)": pb.METHODS["v5 (HBS)"],
                   "A: prior from budget 1": ("hbs_after_0", lambda sid, b, loro: dict(model.PARAMS, PRIOR_MEAN=priors[b])),
                   "B: prior from budget 0 (full mean)": ("hbs", lambda sid, b, loro: dict(model.PARAMS, PRIOR_MEAN=priors[b]))}
        res = pb.per_report(pairs, methods, reps=1000)
        for k0 in k0s:  # B with the running mean at budget 0: adjust the budget-0 term of each pair
            name = f"B: prior from budget 0 (running mean, K0 {k0:.0f} at budget 0)"
            res[name] = {}
            for sid, ps in pairs.items():
                counts = eval_counts(sid)
                adj = sum(0.1 * (running_b0(r, priors[b], counts[(s, b)], n_rated[b], sigma, k0)
                                 - (r * (1 - r) + (priors[b] - r) ** 2)) for s, b, r in ps)
                res[name][sid] = res["B: prior from budget 0 (full mean)"][sid] + adj
        size = {sid: len(ps) for sid, ps in pairs.items()}
        sids = sorted(pairs)
        alc = lambda m, ss: sum(res[m][s] for s in ss) / sum(size[s] for s in ss)
        rng = random.Random(1)
        boots = [[rng.choice(sids) for _ in sids] for _ in range(4000)]
        print(f"{label}: {sum(size.values())} pairs in {len(sids)} reports")
        for m in res:
            d = sorted(alc(m, bs) - alc("v5 (HBS)", bs) for bs in boots)
            print(f"  {m:<58} {alc(m, sids):.4f}  change {alc(m, sids) - alc('v5 (HBS)', sids):+.4f} [{d[100]:+.4f}, {d[3899]:+.4f}]")


def llm_ran(sid):
    """For a v7_final report: if the LLM failed and the submission fell back to v5, every pair's
    budget-0 prediction is the constant 0.35, so Brier = r(1 - r) + ECE^2 with r = 0.35 -/+ ECE.
    A running-mean prior varies within each pair and breaks this identity."""
    resid = []
    for s, b, brier, ece in bp.b0_rows(sid):
        best = min(abs(brier - (r * (1 - r) + ece * ece)) for r in (PRIOR - ece, PRIOR + ece) if 0 <= r <= 1)
        resid.append(best)
        print(f"  {s} {b}: budget-0 Brier {brier:.4f}, ECE {ece:.4f}, distance from a constant 0.35 {best:.5f}")
    off = sum(r > 1e-4 for r in resid)
    print(f"{off} of {len(resid)} pairs are inconsistent with a constant 0.35 at budget 0: "
          + ("the LLM prior ran" if off >= len(resid) / 2 else "the submission probably fell back to v5"))


def main(sid):
    pairs = recovered_pairs(sid)
    real = pv.real_budgets(sid)
    expected = pv.proxy_budgets(model, model.PARAMS, pairs)
    alcs = pv.alc_samples(model, model.PARAMS, pairs)
    mu, sd = st.mean(alcs), st.pstdev(alcs)
    real_alc = sum(w * real[b] for w, b in zip(sim.WEIGHTS, sim.BUDGETS))
    print(f"{len(pairs)} pairs; oracle limit {st.mean(r * (1 - r) for _, _, r in pairs):.4f}")
    print("real minus v5's proxy expectation per budget: "
          + "  ".join(f"b{b} {real[b] - expected[b]:+.4f}" for b in sim.BUDGETS))
    offset = st.mean(V5_OFFSET)
    print(f"Brier ALC: real {real_alc:.4f}, v5 expected {mu:.4f} (label-noise SD {sd:.4f}); "
          f"real - expected {real_alc - mu:+.4f} (z {(real_alc - mu) / sd:+.1f}); "
          f"after v5's usual offset {offset:+.4f}: {real_alc - mu - offset:+.4f} (z {(real_alc - mu - offset) / sd:+.1f})")


if __name__ == "__main__":
    if sys.argv[1] == "--final":
        llm_ran(sys.argv[2])
    elif sys.argv[1] == "--probe":
        sid = sys.argv[2]
        print(f"probe report {sid}:")
        priors = probe_priors(sid)
        print("recovered priors:", {b: round(m, 3) for b, m in priors.items()})
        final_estimates(priors, sid)
    else:
        main(sys.argv[1])
