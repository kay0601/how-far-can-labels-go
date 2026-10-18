"""Further checks of Appendix E in the report. Prints aggregates only.

1. Benchmark-difficulty diagnostic with alpha = 3 (the value of HBS), alpha = 6, and alpha = 3
   with g_b fixed at the benchmark's rate (GLOBAL_K -> infinity): change from HBS with 95%
   intervals over reports, and the share of HBS's gap to the oracle, on all 69 pairs and on
   the 26 pairs with known roots.
2. Pessimism of the proxy: labels drawn without replacement from a pool of |E_j| items with
   the pair's success rate, against independent draws, for the three retained submissions.
3. Accuracy of the LLM priors across the seven test benchmarks: Kendall's tau, mean absolute
   error and the slope of the priors on the benchmarks' mean rates.
4. Benchmark-level uncertainty of HBS+Prior: leave-one-benchmark-out jackknife.
5. Placement of the LLM prior: in the global rate g_b, as in HBS+Prior, against g_b fixed at the
   prior (GLOBAL_K -> infinity). Both predict the prior at budget 0, where the full prior is used.

    <venv python> appendix_checks.py      # about 25 minutes on one CPU
"""

from __future__ import annotations

import contextlib
import io
import math
import random
import statistics as st

from scipy.stats import kendalltau

with contextlib.redirect_stdout(io.StringIO()):
    import bench_prior_value as bp
    import model
    import proxy
    import proxy_baselines as pb
    import proxy_validation as pv
    import sim
    import v7_robustness as vr
vf = vr.vf

DIAGNOSTICS = {
    "alpha 3": ("hbs", lambda sid, b, loro: dict(model.PARAMS, PRIOR_MEAN=loro(b, sid))),
    "alpha 6": pb.METHODS["benchmark rate known (diagnostic)"],
    "alpha 3, g_b fixed": ("hbs", lambda sid, b, loro: dict(model.PARAMS, PRIOR_MEAN=loro(b, sid), GLOBAL_K=1e9)),
}


def diagnostic(pairs, label, boots=4000):
    methods = {"HBS": pb.METHODS["v5 (HBS)"], **DIAGNOSTICS}
    res = pb.per_report(pairs, methods)
    size = {sid: len(ps) for sid, ps in pairs.items()}
    orc = {sid: sum(r * (1 - r) for _, _, r in ps) for sid, ps in pairs.items()}
    sids = sorted(pairs)
    alc = lambda m, ss: sum(res[m][s] for s in ss) / sum(size[s] for s in ss)
    oracle = lambda ss: sum(orc[s] for s in ss) / sum(size[s] for s in ss)
    share = lambda m, ss: (alc("HBS", ss) - alc(m, ss)) / (alc("HBS", ss) - oracle(ss))
    rng = random.Random(1)
    draws = [[rng.choice(sids) for _ in sids] for _ in range(boots)]
    lo, hi = int(0.025 * boots), int(0.975 * boots) - 1
    print(f"{label}: HBS {alc('HBS', sids):.4f}, oracle {oracle(sids):.4f}")
    for m in DIAGNOSTICS:
        d = sorted(alc(m, ds) - alc("HBS", ds) for ds in draws)
        s = sorted(share(m, ds) for ds in draws)
        print(f"  {m:<20} {alc(m, sids):.4f}, change {alc(m, sids) - alc('HBS', sids):+.4f} [{d[lo]:+.4f}, {d[hi]:+.4f}], "
              f"share {share(m, sids):.2f} [{s[lo]:.2f}, {s[hi]:.2f}]")


def finite_pool(reps=2000, seed=0):
    v1 = pv._load("versions/model_v1.py", "model_v1")
    v4 = pv._load("versions/model_v4.py", "model_v4")
    rows = [("smoothed mean", "951748", v1, v1.PARAMS, proxy.DRAWS["report1"]),
            ("HBS+Ability", "951853", v4, dict(v4.PARAMS, _TABLE=proxy.v4_table(v4.PARAMS)), proxy.DRAWS["report2"]),
            ("HBS", "952533", model, model.PARAMS, proxy.DRAWS["report3"])]
    for name, sid, mod, params, pairs in rows:
        sizes = vf.eval_counts(sid)
        rng, out = random.Random(seed), []
        for _ in range(reps):
            acq = {}
            for s, b, r in pairs:
                n = sizes[(s, b)]
                pool = [1] * round(r * n) + [0] * (n - round(r * n))
                rng.shuffle(pool)
                acq[s] = pool[:31]
            alc = 0.0
            for weight, budget in zip(sim.WEIGHTS, sim.BUDGETS):
                labeled = [[proxy._inp(s, b, i), y] for s, b, _ in pairs for i, y in enumerate(acq[s][:budget])]
                summary = mod._summarize(labeled)
                alc += weight * st.mean(r * (1 - r) + (mod.predict_with(params, proxy._inp(s, b, "eval"), labeled, summary) - r) ** 2
                                        for s, b, r in pairs)
            out.append(alc)
        bern = pv.alc_samples(mod, params, pairs)
        print(f"{name}: finite pool {st.mean(out):.4f} vs independent {st.mean(bern):.4f} ({st.mean(out) - st.mean(bern):+.4f})")


def prior_accuracy():
    rbar = {b: st.mean(r for _, r in v) for b, v in bp.bench_rates.items()}
    bs = sorted(vr.PRIORS)
    x, y = [rbar[b] for b in bs], [vr.PRIORS[b] for b in bs]
    print(f"{len(bs)} test benchmarks: Kendall tau {kendalltau(x, y)[0]:.2f}; mean absolute error {st.mean(abs(a - b) for a, b in zip(x, y)):.3f} "
          f"(constant 0.35: {st.mean(abs(a - 0.35) for a in x):.3f}); slope of prior on mean rate {st.linear_regression(x, y).slope:.2f}; "
          f"correlation {st.correlation(x, y):.2f}")


def benchmark_jackknife(reps=500, seed=0):
    rng, deltas = random.Random(seed), []
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
                    p7 = model.predict_with(dict(model.PARAMS, PRIOR_MEAN=vr.PRIORS[b]), inp, labeled, summary)
                    b7 = (vf.running_b0(r, vr.PRIORS[b], counts[(s, b)], vr.N_RATED[b], vr.SIGMA, vr.K0) if budget == 0
                          else r * (1 - r) + (p7 - r) ** 2)
                    acc[(s, b)][0] += weight * (r * (1 - r) + (p5 - r) ** 2) / reps
                    acc[(s, b)][1] += weight * b7 / reps
        deltas += [(b, a7 - a5) for (s, b), (a5, a7) in acc.items()]
    benches = sorted({b for b, _ in deltas})
    loo = [st.mean(d for b2, d in deltas if b2 != b) for b in benches]
    se = math.sqrt((len(benches) - 1) / len(benches) * sum((v - st.mean(loo)) ** 2 for v in loo))
    print(f"HBS+Prior change {st.mean(d for _, d in deltas):+.4f}; leaving out one benchmark {min(loo):+.4f} to {max(loo):+.4f}; "
          f"jackknife SE over {len(benches)} benchmarks {se:.4f}")


def prior_placement(reps=2000):
    with_prior = lambda **extra: ("hbs", lambda sid, b, loro: dict(model.PARAMS, PRIOR_MEAN=vr.PRIORS[b], **extra))
    methods = {"HBS": pb.METHODS["v5 (HBS)"], "in g_b": with_prior(), "g_b fixed": with_prior(GLOBAL_K=1e9)}
    res = pb.per_report(bp.pairs, methods, reps=reps)
    size = sum(len(ps) for ps in bp.pairs.values())
    alc = {m: sum(res[m].values()) / size for m in methods}
    print(f"LLM prior in g_b {alc['in g_b'] - alc['HBS']:+.4f}, g_b fixed at the prior {alc['g_b fixed'] - alc['HBS']:+.4f} "
          f"(difference {alc['g_b fixed'] - alc['in g_b']:+.4f})")


if __name__ == "__main__":
    diagnostic(bp.pairs, "69 pairs")
    diagnostic({sid: ps for sid, ps in bp.pairs.items() if sid not in bp.INFERRED}, "26 known pairs")
    finite_pool()
    prior_accuracy()
    benchmark_jackknife()
    prior_placement()
