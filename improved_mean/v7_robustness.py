"""Robustness of the HBS+Prior estimate (report, Section 6.4 and Appendix E).

Uses the benchmark priors recovered from the probe submission (v7_feedback.probe_priors)
and the 69 proxy pairs of the eight earlier reports. Only aggregates are printed.

1. Prior uncertainty: each prior is redrawn from its sampling distribution (mean rating
   of n items with spread SIGMA), and the change in Brier ALC is recomputed.
2. K0: the shrinkage of the prior toward 0.35, from 0 to 20.
3. Item order: the budget-0 running mean depends on the order in which items are scored;
   for fixed ratings, the order is permuted at random.
4. Per benchmark: on how many of the seven benchmarks the prior helps.
6. Known roots: the change on the pairs whose root is known (bp.INFERRED excluded), overall
   and per report.
5. The selected submission's own formative run (978186): each pair's rate follows from
   budgets 1 and 3, where the prediction is constant within five of the nine pairs and nearly
   constant within the others (only the item and group corrections vary), so the run can be compared
   with the proxy expectations of HBS and HBS+Prior on its own pairs.

    <venv python> v7_robustness.py      # about 30 minutes on one CPU
"""

from __future__ import annotations

import contextlib
import io
import math
import random
import statistics as st

import bench_prior_value as bp
import model
import proxy
import proxy_baselines as pb
import proxy_validation as pv
import sim
import v7_feedback as vf

PROBE, FINAL = "978024", "978186"
SIGMA, K0 = 0.205, 5.0
with contextlib.redirect_stdout(io.StringIO()):
    PRIORS = vf.probe_priors(PROBE)
N_RATED = {}
for (_, b), n in vf.eval_counts(PROBE).items():
    N_RATED[b] = N_RATED.get(b, 0) + n
RAW = {b: (m * (N_RATED[b] + K0) - K0 * vf.PRIOR) / N_RATED[b] for b, m in PRIORS.items()}  # unshrunk mean ratings


def shrunk(raw, k0):
    return {b: (N_RATED[b] * raw[b] + k0 * vf.PRIOR) / (N_RATED[b] + k0) for b in raw}


def alc_with(priors, pairs=None, reps=500, k0=K0, b0_term=None):
    """Proxy Brier ALC of v5 and of HBS+Prior (prior from budget 0, running mean at budget 0)."""
    pairs = pairs or bp.pairs
    methods = {"v5": pb.METHODS["v5 (HBS)"],
               "prior": ("hbs", lambda sid, b, loro: dict(model.PARAMS, PRIOR_MEAN=priors[b]))}
    res = pb.per_report(pairs, methods, reps=reps)
    size = sum(len(ps) for ps in pairs.values())
    v5 = sum(res["v5"].values()) / size
    full = sum(res["prior"].values()) / size
    adj = 0.0
    for sid, ps in pairs.items():
        counts = vf.eval_counts(sid)
        for s, b, r in ps:
            full_b0 = r * (1 - r) + (priors[b] - r) ** 2
            run_b0 = b0_term(sid, s, b, r) if b0_term else vf.running_b0(r, priors[b], counts[(s, b)], N_RATED[b], SIGMA, k0)
            adj += 0.1 * (run_b0 - full_b0)
    return v5, full + adj / size


def prior_uncertainty(draws=20, seed=0):
    rng = random.Random(seed)
    gains = []
    for _ in range(draws):
        raw = {b: min(0.99, max(0.01, rng.gauss(m, SIGMA / math.sqrt(N_RATED[b])))) for b, m in RAW.items()}
        v5, v7 = alc_with(shrunk(raw, K0), reps=300)
        gains.append(v7 - v5)
    gains.sort()
    print(f"prior uncertainty ({draws} redraws): change {st.mean(gains):+.4f}, SD {st.pstdev(gains):.4f}, "
          f"range {gains[0]:+.4f} to {gains[-1]:+.4f}")


def k0_sensitivity():
    for k0 in (0.0, 2.0, 5.0, 10.0, 20.0):
        v5, v7 = alc_with(shrunk(RAW, k0), k0=k0)
        print(f"K0 {k0:>4.0f}: change {v7 - v5:+.4f}")


def order_sensitivity(orders=200, seed=1):
    """Budget-0 term for fixed ratings in random orders (ratings drawn once per benchmark and report)."""
    rng = random.Random(seed)
    ratings = {(sid, b): [min(0.99, max(0.01, rng.gauss(RAW[b], SIGMA))) for _ in range(N_RATED[b])]
               for sid, ps in bp.pairs.items() for _, b, _ in ps}
    v5, base = alc_with(PRIORS, reps=500)
    size = sum(len(ps) for ps in bp.pairs.values())
    deltas = []
    for _ in range(orders):
        extra = 0.0
        for sid, ps in bp.pairs.items():
            counts = vf.eval_counts(sid)
            for s, b, r in ps:
                seq = ratings[(sid, b)][:]
                rng.shuffle(seq)
                n = counts[(s, b)]
                total, b0 = 0.0, 0.0
                for k in range(1, n + 1):
                    total += seq[(k - 1) % len(seq)]
                    b0 += ((total + K0 * vf.PRIOR) / (k + K0) - r) ** 2
                run = r * (1 - r) + b0 / n
                extra += 0.1 * (run - vf.running_b0(r, PRIORS[b], n, N_RATED[b], SIGMA, K0))
        deltas.append(base + extra / size - v5)
    deltas.sort()
    print(f"item order ({orders} orders of fixed ratings): change {st.mean(deltas):+.4f}, SD {st.pstdev(deltas):.4f}, "
          f"range {deltas[0]:+.4f} to {deltas[-1]:+.4f}")


def per_benchmark(reps=500, seed=0):
    """Change in Brier ALC per benchmark, from per-pair expectations (counts only are printed)."""
    rng = random.Random(seed)
    per = {}
    for sid, draw in sorted(bp.pairs.items()):
        counts = vf.eval_counts(sid)
        acc = {(s, b): [0.0, 0.0] for s, b, _ in draw}
        for _ in range(reps):
            acq = {s: [int(rng.random() < r) for _ in range(31)] for s, _, r in draw}
            for weight, budget in zip(sim.WEIGHTS, sim.BUDGETS):
                labeled = [[proxy._inp(s, b, i), y] for s, b, _ in draw for i, y in enumerate(acq[s][:budget])]
                summary = model._summarize(labeled)
                for s, b, r in draw:
                    inp = proxy._inp(s, b, "eval")
                    p5 = model.predict_with(model.PARAMS, inp, labeled, summary)
                    p7 = model.predict_with(dict(model.PARAMS, PRIOR_MEAN=PRIORS[b]), inp, labeled, summary)
                    if budget == 0:
                        b7 = vf.running_b0(r, PRIORS[b], counts[(s, b)], N_RATED[b], SIGMA, K0)
                    else:
                        b7 = r * (1 - r) + (p7 - r) ** 2
                    acc[(s, b)][0] += weight * (r * (1 - r) + (p5 - r) ** 2) / reps
                    acc[(s, b)][1] += weight * b7 / reps
        for (s, b), (a5, a7) in acc.items():
            per.setdefault(b, []).append(a7 - a5)
    better = sum(st.mean(v) < 0 for v in per.values())
    print(f"per benchmark: the prior lowers Brier ALC on {better} of {len(per)} benchmarks; "
          f"pairs improved {sum(d < 0 for v in per.values() for d in v)} of {sum(len(v) for v in per.values())}")


def known_roots(reps=2000):
    """HBS+Prior against HBS on the pairs whose root is known, overall and per report."""
    known = {sid: ps for sid, ps in bp.pairs.items() if sid not in bp.INFERRED}
    v5, v7 = alc_with(PRIORS, pairs=known, reps=reps)
    per = []
    for sid, ps in known.items():
        a, b = alc_with(PRIORS, pairs={sid: ps}, reps=reps)
        per.append(b - a)
    print(f"{sum(len(ps) for ps in known.values())} pairs with known roots ({len(known)} reports): change "
          f"{v7 - v5:+.4f}; per report {min(per):+.4f} to {max(per):+.4f}")


def final_run():
    """The selected submission's own draw: rates from budgets 1 and 3 (constant predictions)."""
    rows1 = {(s, b): (br, e) for s, b, br, e in vf.budget_rows(FINAL, 1)}
    rows3 = {(s, b): (br, e) for s, b, br, e in vf.budget_rows(FINAL, 3)}
    means = {b: st.mean(r for _, r in v) for b, v in bp.bench_rates.items()}
    pairs, resid = [], []
    for (s, b), (br1, e1) in rows1.items():
        v1 = max(0.0, br1 - e1 * e1)
        v3 = max(0.0, rows3[(s, b)][0] - rows3[(s, b)][1] ** 2)
        resid.append(abs(v1 - v3))
        lo = (1 - math.sqrt(max(0.0, 1 - 4 * v1))) / 2
        r = min((lo, 1 - lo), key=lambda x: abs(x - means.get(b, 0.35)))
        pairs.append((s, b, r))
    print(f"{FINAL}: r(1 - r) from budget 1 and from budget 3 agree to within {max(resid):.4f} (median {st.median(resid):.4f})")
    real = pv.real_budgets(FINAL)
    real_alc = sum(w * real[t] for w, t in zip(sim.WEIGHTS, sim.BUDGETS))
    oracle = st.mean(r * (1 - r) for _, _, r in pairs)
    alcs = pv.alc_samples(model, model.PARAMS, pairs)
    mu5, sd = st.mean(alcs), st.pstdev(alcs)
    v5, v7 = alc_with(PRIORS, pairs={FINAL: pairs}, reps=2000)
    print(f"{FINAL}: real Brier ALC {real_alc:.4f}, oracle limit {oracle:.4f}; proxy expectation HBS {mu5:.4f} "
          f"(label-noise SD {sd:.4f}), HBS+Prior {v7:.4f}; real - HBS {real_alc - mu5:+.4f} (z {(real_alc - mu5) / sd:+.1f}), "
          f"real - HBS+Prior {real_alc - v7:+.4f} (z {(real_alc - v7) / sd:+.1f})")
    # per budget; budget 0 involves no labels, so HBS's budget-0 Brier on these pairs is exact
    e5 = pv.proxy_budgets(model, model.PARAMS, pairs)

    class WithPrior:
        _summarize = staticmethod(model._summarize)

        @staticmethod
        def predict_with(params, inp, labeled, summary):
            return model.predict_with(dict(params, PRIOR_MEAN=PRIORS[inp[1]["benchmark_id"]]), inp, labeled, summary)

    e7 = pv.proxy_budgets(WithPrior, model.PARAMS, pairs)
    counts = vf.eval_counts(FINAL)
    e7[0] = st.mean(vf.running_b0(r, PRIORS[b], counts[(s, b)], N_RATED[b], SIGMA, K0) for s, b, r in pairs)
    for t in sim.BUDGETS:
        print(f"  budget {t:>2}: real {real[t]:.4f}, HBS expected {e5[t]:.4f} ({real[t] - e5[t]:+.4f}), "
              f"HBS+Prior expected {e7[t]:.4f} ({real[t] - e7[t]:+.4f})")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["known"]:
        known_roots()
        raise SystemExit
    final_run()
    known_roots()
    per_benchmark()
    k0_sensitivity()
    order_sensitivity()
    prior_uncertainty()
