"""Baselines and ablations of v5 (HBS) on the real-feedback proxy (report, Table 3 and Section 6).

For each of the 69 pairs in the eight formative reports, the expected Brier ALC of every
method is computed under the proxy (labels drawn as Bernoulli(r), each report simulated
as its own draw). Differences to v5 get 95% intervals from resampling whole reports, and
are repeated on the six reports that did not inform v5's design (all but those of v1 and
v4). The share of v5's gap to the oracle that knowing each benchmark's rate would close
gets the same interval.

    <venv python> proxy_baselines.py
"""

from __future__ import annotations

import random

import bench_prior_value as bp
import model
import proxy
import proxy_validation as pv
import sim
from label_only_bound import posterior_mean

REPS = 2000
DESIGN_REPORTS = ("951748", "951853")  # v1's and v4's reports, seen while designing v5
REPORTS = sorted(bp.pairs)
v1 = pv._load("versions/model_v1.py", "model_v1")


def _params(**change):
    return lambda sid, b, loro: dict(model.PARAMS, **change)


def loro_means(pairs):
    """Benchmark mean rate from the *other* reports (0.35 when there is none), as bp.loro_mean."""
    rates = {}
    for sid, ps in pairs.items():
        for _, b, r in ps:
            rates.setdefault(b, []).append((sid, r))

    def mean(b, sid):
        rs = [r for other, r in rates.get(b, []) if other != sid]
        return sum(rs) / len(rs) if rs else 0.35
    return mean


# name -> (kind, setting); "hbs" settings map (report, benchmark, loro means) to v5 parameters
METHODS = {
    "v5 (HBS)": ("hbs", _params()),
    "official empirical mean": ("empirical", None),
    "v1": ("v1", None),
    "benchmark pooling removed": ("hbs", _params(BENCH_K=1e9)),
    "alpha 1": ("hbs", _params(ALPHA=1.0)),
    "alpha 10": ("hbs", _params(ALPHA=10.0)),
    "own-label Bayes predictor": ("own", None),
    "benchmark rate known (diagnostic)": ("hbs", lambda sid, b, loro: dict(model.PARAMS, PRIOR_MEAN=loro(b, sid), ALPHA=6.0)),
}


def _predict(kind, setting, sid, s, b, labeled, summary, own, budget, every, loro):
    if kind == "hbs":
        return model.predict_with(setting(sid, b, loro), proxy._inp(s, b, "eval"), labeled, summary)
    if kind == "hbs_after_0":  # v5 at budget 0, ``setting``'s parameters once labels exist (v7)
        params = model.PARAMS if budget == 0 else setting(sid, b, loro)
        return model.predict_with(params, proxy._inp(s, b, "eval"), labeled, summary)
    if kind == "v1":
        return v1.predict_with(v1.PARAMS, proxy._inp(s, b, "eval"), labeled)
    if kind == "empirical":  # the organizers' baseline: mean of the pair's labels, 0.5 without any
        return sum(own[:budget]) / budget if budget else 0.5
    others = [x[2] for x in every if x[:2] != (s, b)]
    return posterior_mean(sum(own[:budget]), budget, others)


def per_report(pairs=None, methods=None, reps=REPS, seed=0):
    """Summed expected Brier ALC over each report's pairs, per method. ``pairs`` maps a report
    to its (subject, benchmark, rate) pairs (default: the rates recovered in bench_prior_value)."""
    pairs = pairs or bp.pairs
    methods = methods or METHODS
    every = [p for sid in sorted(pairs) for p in pairs[sid]]
    loro = loro_means(pairs)
    rng = random.Random(seed)
    out = {m: {} for m in methods}
    for sid in sorted(pairs):
        draw = pairs[sid]
        acc = dict.fromkeys(methods, 0.0)
        for _ in range(reps):
            acq = {s: [int(rng.random() < r) for _ in range(31)] for s, _, r in draw}
            for weight, budget in zip(sim.WEIGHTS, sim.BUDGETS):
                labeled = [[proxy._inp(s, b, i), y] for s, b, _ in draw for i, y in enumerate(acq[s][:budget])]
                summary = model._summarize(labeled)
                for s, b, r in draw:
                    for name, (kind, setting) in methods.items():
                        p = _predict(kind, setting, sid, s, b, labeled, summary, acq[s], budget, every, loro)
                        acc[name] += weight * (r * (1 - r) + (p - r) ** 2) / reps
        for name in methods:
            out[name][sid] = acc[name]
    return out


def main():
    res = per_report()
    size = {sid: len(bp.pairs[sid]) for sid in REPORTS}
    oracle = {sid: sum(r * (1 - r) for _, _, r in bp.pairs[sid]) for sid in REPORTS}

    def alc(name, sids):
        table = oracle if name == "oracle" else res[name]
        return sum(table[s] for s in sids) / sum(size[s] for s in sids)

    rng = random.Random(1)
    boots = [[rng.choice(REPORTS) for _ in REPORTS] for _ in range(4000)]
    held = [s for s in REPORTS if s not in DESIGN_REPORTS]
    ref = "v5 (HBS)"
    print(f"{'method':<36}{'ALC':>8}{'diff':>10}   95% interval        held-out 6 reports")
    for name in METHODS:
        diffs = sorted(alc(name, bs) - alc(ref, bs) for bs in boots)
        print(f"{name:<36}{alc(name, REPORTS):>8.4f}{alc(name, REPORTS) - alc(ref, REPORTS):>+10.4f}"
              f"   [{diffs[100]:+.4f}, {diffs[3899]:+.4f}]   {alc(name, held) - alc(ref, held):+.4f}")
    print(f"{'oracle':<36}{alc('oracle', REPORTS):>8.4f}")

    known = "benchmark rate known (diagnostic)"
    share = lambda sids: (alc(ref, sids) - alc(known, sids)) / (alc(ref, sids) - alc("oracle", sids))
    shares = sorted(share(bs) for bs in boots)
    print(f"\nshare of v5's gap to the oracle closed by knowing benchmark rates: {share(REPORTS):.2f}"
          f" [{shares[100]:.2f}, {shares[3899]:.2f}]; held-out 6 reports {share(held):.2f}")
    ability_shift_cost()


def ability_shift_cost():
    """Cost of v4's subject-ability shift on v4's own draw: v4 against v4 with the shift off
    (both with v4's PRIOR_MEAN 0.4), and v4 against v5 (report, Section 6.1)."""
    import statistics as st
    v4 = pv._load("versions/model_v4.py", "model_v4")
    pairs = proxy.DRAWS["report2"]
    full = pv.alc_samples(v4, dict(v4.PARAMS, _TABLE=proxy.v4_table(v4.PARAMS)), pairs)
    off = pv.alc_samples(v4, dict(v4.PARAMS, BETA=0.0), pairs)
    v5 = pv.alc_samples(model, model.PARAMS, pairs)
    diff = lambda a, b: (st.mean(x - y for x, y in zip(a, b)), st.pstdev([x - y for x, y in zip(a, b)]))
    print("\nv4's draw: ability shift on minus off %+.4f (SD %.4f); v4 minus v5 %+.4f (SD %.4f)"
          % (*diff(full, off), *diff(full, v5)))


if __name__ == "__main__":
    main()
