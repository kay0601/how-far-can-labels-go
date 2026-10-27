"""HBS+Prior with the settings of the final submission (v8) on the real-feedback proxy
(report, Sections 4 and 6.4, Appendix E). Prints aggregates only.

v8 differs from v7 in three settings: the LLM prior is spread around 0.35 by a factor of
1.5 and clipped to [0.05, 0.95], the global rate is fixed at the prior (GLOBAL_K = 1e9),
and ALPHA = 4. At budget 0 the prior is the running mean of the ratings so far, as in v7;
its expected Brier score is integrated over the rating noise, including the clipping.

1. compare: HBS, v7 and v8 on the 69 pairs, with 95% intervals from resampling reports,
   and the share of the benchmark-difficulty diagnostic that v8 recovers.
2. grid: alpha, GLOBAL_K, stretch and K0 chosen leave-one-report-out, against v7.
3. jackknife: leaving out one benchmark at a time; benchmarks and pairs improved.
4. known: the 26 pairs whose root is known, overall and per report.
5. redraw: each measured prior redrawn from its sampling distribution.
6. budgets: mean Brier per budget of v8 (Figure 2) and its Brier ALC (Figure 1).
7. final SID: a formative run of v8 against the proxy's expectations on its own pairs.
8. lobo: the settings chosen leave-one-benchmark-out (K0 = 5), each held-out benchmark scored with
   the setting that is best on the other six, against HBS and v7.

    <venv python> v8_checks.py [compare grid jackknife known redraw budgets lobo] [final SID]
"""

from __future__ import annotations

import contextlib
import io
import itertools
import math
import random
import statistics as st
import sys
from multiprocessing import Pool

import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import bench_prior_value as bp
    import model
    import proxy
    import proxy_baselines as pb
    import proxy_validation as pv
    import sim
    import v7_feedback as vf
    import v7_robustness as vr

PRIORS, N_RATED, RAW, SIGMA, K0 = vr.PRIORS, vr.N_RATED, vr.RAW, vr.SIGMA, vr.K0
CENTER = vf.PRIOR
V8 = {"stretch": 1.5, "ALPHA": 4.0, "GLOBAL_K": 1e9, "K0": 5.0}
V7 = {"stretch": 1.0, "ALPHA": 3.0, "GLOBAL_K": 100.0, "K0": 5.0}
NODES, WEIGHTS = np.polynomial.hermite_e.hermegauss(48)
WEIGHTS = WEIGHTS / WEIGHTS.sum()


def stretch(m, s):
    return m if s == 1.0 else min(0.95, max(0.05, CENTER + s * (m - CENTER)))


def params(prior, cfg):
    return dict(model.PARAMS, PRIOR_MEAN=stretch(prior, cfg["stretch"]), ALPHA=cfg["ALPHA"], GLOBAL_K=cfg["GLOBAL_K"])


def running_b0(r, raw, n_items, cfg, sigma=SIGMA):
    """Expected budget-0 Brier of a pair: the k-th item sees the K0-shrunk mean of k ratings
    (mean raw, spread sigma), spread and clipped as in v8."""
    k0, s = cfg["K0"], cfg["stretch"]
    total = 0.0
    for k in range(1, n_items + 1):
        mean = (k * raw + k0 * CENTER) / (k + k0)
        sd = math.sqrt(k) * sigma / (k + k0)
        m = mean + sd * NODES
        pred = m if s == 1.0 else np.clip(CENTER + s * (m - CENTER), 0.05, 0.95)
        total += float(WEIGHTS @ (pred - r) ** 2)
    return r * (1 - r) + total / n_items


def alc_by_report(cfgs, pairs=None, reps=1000, raw=None, priors=None):
    """Summed proxy Brier ALC per report for HBS and for each named HBS+Prior setting."""
    pairs, raw, priors = pairs or bp.pairs, raw or RAW, priors or PRIORS
    methods = {"HBS": pb.METHODS["v5 (HBS)"]}
    for name, cfg in cfgs.items():
        methods[name] = ("hbs", (lambda cfg: lambda sid, b, loro: params(priors[b], cfg))(cfg))
    res = pb.per_report(pairs, methods, reps=reps)
    for name, cfg in cfgs.items():
        for sid, ps in pairs.items():
            counts = vf.eval_counts(sid)
            for s, b, r in ps:
                full = r * (1 - r) + (stretch(priors[b], cfg["stretch"]) - r) ** 2
                res[name][sid] += 0.1 * (running_b0(r, raw[b], counts[(s, b)], cfg) - full)
    return res


def _alc(res, name, sids):
    return sum(res[name][s] for s in sids) / sum(len(bp.pairs[s]) for s in sids)


def compare():
    res = alc_by_report({"v7": V7, "v8": V8})
    sids = sorted(bp.pairs)
    rng = random.Random(1)
    boots = [[rng.choice(sids) for _ in sids] for _ in range(4000)]
    hbs = _alc(res, "HBS", sids)
    print(f"69 pairs: HBS {hbs:.4f}, v7 {_alc(res, 'v7', sids):.4f}, v8 {_alc(res, 'v8', sids):.4f}")
    for a, b in (("v7", "HBS"), ("v8", "HBS"), ("v8", "v7")):
        d = sorted(_alc(res, a, bs) - _alc(res, b, bs) for bs in boots)
        print(f"  {a} - {b}: {_alc(res, a, sids) - _alc(res, b, sids):+.4f} [{d[100]:+.4f}, {d[3899]:+.4f}]")
    for alpha, diag in (("6", -0.0186), ("3", -0.0166)):
        print(f"  v8 recovers {(_alc(res, 'v8', sids) - hbs) / diag:.0%} of the diagnostic with alpha {alpha} ({diag:+.4f})")


GRID = [{"stretch": s, "ALPHA": a, "GLOBAL_K": g, "K0": k}
        for a, g, s, k in itertools.product((2.0, 3.0, 4.0, 5.0, 6.0, 8.0), (100.0, 1e9), (1.0, 1.25, 1.5, 1.75, 2.0), (0.0, 2.0, 5.0))]


def _grid_part(i, n=6):
    cfgs = {f"g{j}": cfg for j, cfg in enumerate(GRID) if j % n == i}
    return alc_by_report(cfgs)


def grid():
    with Pool(6) as pool:
        parts = pool.map(_grid_part, range(6))
    res = {}
    for part in parts:
        res.update(part)
    sids = sorted(bp.pairs)
    names = [f"g{j}" for j in range(len(GRID))]
    v7 = next(f"g{j}" for j, c in enumerate(GRID) if c == V7)
    best = min(names, key=lambda c: _alc(res, c, sids))
    print(f"grid of {len(GRID)} settings; best in sample {GRID[int(best[1:])]}: {_alc(res, best, sids) - _alc(res, v7, sids):+.4f} against v7")
    diff, picks = {}, []
    for held in sids:
        others = [s for s in sids if s != held]
        pick = min(names, key=lambda c: _alc(res, c, others))
        picks.append(GRID[int(pick[1:])])
        diff[held] = res[pick][held] - res[v7][held]
    size = {s: len(bp.pairs[s]) for s in sids}
    rng = random.Random(1)
    boots = sorted(sum(diff[s] for s in bs) / sum(size[s] for s in bs) for bs in ([rng.choice(sids) for _ in sids] for _ in range(4000)))
    print(f"leave-one-report-out choice against v7: {sum(diff.values()) / sum(size.values()):+.4f} [{boots[100]:+.4f}, {boots[3899]:+.4f}]; "
          f"chosen settings: {sorted({tuple(sorted(p.items())) for p in picks})}")


def _per_pair(cfg, reps=600, seed=0):
    rng, acc = random.Random(seed), {}
    for sid, draw in sorted(bp.pairs.items()):
        counts = vf.eval_counts(sid)
        a = {(s, b): [0.0, 0.0, 0.0] for s, b, _ in draw}
        for _ in range(reps):
            acq = {s: [int(rng.random() < r) for _ in range(31)] for s, _, r in draw}
            for weight, budget in zip(sim.WEIGHTS, sim.BUDGETS):
                labeled = [[proxy._inp(s, b, i), y] for s, b, _ in draw for i, y in enumerate(acq[s][:budget])]
                summary = model._summarize(labeled)
                for s, b, r in draw:
                    inp = proxy._inp(s, b, "eval")
                    vals = [r * (1 - r) + (model.predict_with(model.PARAMS, inp, labeled, summary) - r) ** 2]
                    for c in (V7, cfg):
                        if budget == 0:
                            vals.append(running_b0(r, RAW[b], counts[(s, b)], c))
                        else:
                            vals.append(r * (1 - r) + (model.predict_with(params(PRIORS[b], c), inp, labeled, summary) - r) ** 2)
                    for i, v in enumerate(vals):
                        a[(s, b)][i] += weight * v / reps
        for (s, b), v in a.items():
            acc[(sid, s, b)] = v
    return acc


def _lobo_part(i, n=6, reps=600, seed=0):
    """Weighted Brier per pair (summed over budgets) of HBS and of every K0 = 5 grid setting."""
    cfgs = [c for c in GRID if c["K0"] == 5.0][i::n]
    rng, acc = random.Random(seed), {}
    for sid, draw in sorted(bp.pairs.items()):
        counts = vf.eval_counts(sid)
        a = {(s, b): [0.0] * (len(cfgs) + 1) for s, b, _ in draw}
        for s, b, r in draw:
            for ci, c in enumerate(cfgs):
                a[(s, b)][ci + 1] += 0.1 * running_b0(r, RAW[b], counts[(s, b)], c)
            a[(s, b)][0] += 0.1 * (r * (1 - r) + (model.PARAMS["PRIOR_MEAN"] - r) ** 2)
        for _ in range(reps):
            acq = {s: [int(rng.random() < r) for _ in range(31)] for s, _, r in draw}
            for weight, budget in zip(sim.WEIGHTS[1:], sim.BUDGETS[1:]):
                labeled = [[proxy._inp(s, b, i), y] for s, b, _ in draw for i, y in enumerate(acq[s][:budget])]
                summary = model._summarize(labeled)
                for s, b, r in draw:
                    inp = proxy._inp(s, b, "eval")
                    ps = [model.predict_with(model.PARAMS, inp, labeled, summary)]
                    ps += [model.predict_with(params(PRIORS[b], c), inp, labeled, summary) for c in cfgs]
                    for ci, q in enumerate(ps):
                        a[(s, b)][ci] += weight * (r * (1 - r) + (q - r) ** 2) / reps
        for (s, b), v in a.items():
            acc[(sid, s, b)] = v
    return cfgs, acc


def lobo():
    with Pool(6) as pool:
        parts = pool.map(_lobo_part, range(6))
    cfgs, acc = [], {k: [] for k in parts[0][1]}
    for cs, part in parts:
        cfgs += cs
        for k, v in part.items():
            acc[k] += v[1:]
    hbs = {k: v[0] for k, v in parts[0][1].items()}
    v7, v8 = cfgs.index(V7), cfgs.index(V8)
    benches = sorted({b for _, _, b in acc})
    chosen, out = [], {}
    for h in benches:
        rest = [k for k in acc if k[2] != h]
        pick = min(range(len(cfgs)), key=lambda c: sum(acc[k][c] for k in rest))
        chosen.append(cfgs[pick])
        out.update({k: acc[k][pick] for k in acc if k[2] == h})
    for label, base in (("HBS", hbs), ("v7", {k: v[v7] for k, v in acc.items()})):
        d = {k: out[k] - base[k] for k in acc}
        fixed = {k: acc[k][v8] - base[k] for k in acc}
        per_b = [st.mean(x for k, x in d.items() if k[2] == h) for h in benches]
        print(f"leave-one-benchmark-out choice - {label}: {st.mean(d.values()):+.4f} (fixed v8 {st.mean(fixed.values()):+.4f}); "
              f"better on {sum(x < 0 for x in per_b)} of {len(benches)} benchmarks")
    print("chosen settings: " + "; ".join(f"stretch {c['stretch']:g}, alpha {c['ALPHA']:g}, GLOBAL_K {c['GLOBAL_K']:g}" for c in chosen))


def jackknife():
    acc = _per_pair(V8)
    benches = sorted({b for _, _, b in acc})
    for label, i, j in (("v8 - HBS", 2, 0), ("v8 - v7", 2, 1)):
        d = {k: v[i] - v[j] for k, v in acc.items()}
        loo = [st.mean(x for (sid, s, b), x in d.items() if b != h) for h in benches]
        se = math.sqrt((len(loo) - 1) / len(loo) * sum((x - st.mean(loo)) ** 2 for x in loo))
        per_b = [st.mean(x for (sid, s, b), x in d.items() if b == h) for h in benches]
        print(f"{label}: {st.mean(d.values()):+.4f}; leaving out one benchmark {min(loo):+.4f} to {max(loo):+.4f}; "
              f"jackknife SE {se:.4f}; better on {sum(x < 0 for x in per_b)} of {len(benches)} benchmarks and "
              f"{sum(x < 0 for x in d.values())} of {len(d)} pairs")


def known(reps=2000):
    pairs = {sid: ps for sid, ps in bp.pairs.items() if sid not in bp.INFERRED}
    res = alc_by_report({"v8": V8}, pairs=pairs, reps=reps)
    sids = sorted(pairs)
    per = [_alc_known(res, s, pairs) for s in sids]
    tot = sum(res["v8"][s] - res["HBS"][s] for s in sids) / sum(len(pairs[s]) for s in sids)
    print(f"{sum(len(p) for p in pairs.values())} pairs with known roots: v8 - HBS {tot:+.4f}; per report {min(per):+.4f} to {max(per):+.4f}")


def _alc_known(res, sid, pairs):
    return (res["v8"][sid] - res["HBS"][sid]) / len(pairs[sid])


def redraw(draws=20, seed=0):
    rng, gains = random.Random(seed), []
    sids = sorted(bp.pairs)
    for _ in range(draws):
        raw = {b: min(0.99, max(0.01, rng.gauss(m, SIGMA / math.sqrt(N_RATED[b])))) for b, m in RAW.items()}
        priors = vr.shrunk(raw, K0)
        res = alc_by_report({"v8": V8}, reps=300, raw=raw, priors=priors)
        gains.append(_alc(res, "v8", sids) - _alc(res, "HBS", sids))
    gains.sort()
    print(f"priors redrawn {draws} times: v8 - HBS {st.mean(gains):+.4f}, SD {st.pstdev(gains):.4f}, range {gains[0]:+.4f} to {gains[-1]:+.4f}")


def budgets(reps=1000, seed=0):
    rng, n = random.Random(seed), sum(len(ps) for ps in bp.pairs.values())
    out = [0.0] * len(sim.BUDGETS)
    for _ in range(reps):
        for sid, ps in bp.pairs.items():
            acq = {s: [int(rng.random() < r) for _ in range(31)] for s, _, r in ps}
            for bi, budget in enumerate(sim.BUDGETS[1:], start=1):
                labeled = [[proxy._inp(s, b, i), y] for s, b, _ in ps for i, y in enumerate(acq[s][:budget])]
                summary = model._summarize(labeled)
                for s, b, r in ps:
                    p = model.predict_with(params(PRIORS[b], V8), proxy._inp(s, b, "eval"), labeled, summary)
                    out[bi] += (r * (1 - r) + (p - r) ** 2) / (reps * n)
    out[0] = st.mean(running_b0(r, RAW[b], vf.eval_counts(sid)[(s, b)], V8) for sid, ps in bp.pairs.items() for s, b, r in ps)
    print("v8 mean Brier per budget: " + " ".join(f"{x:.4f}" for x in out)
          + f"; Brier ALC {sum(w * x for w, x in zip(sim.WEIGHTS, out)):.4f}")
    return out


def final(sid):
    """A formative run of v8: rates from budgets 1 and 3, compared with the proxy's expectations."""
    rows1 = {(s, b): (br, e) for s, b, br, e in vf.budget_rows(sid, 1)}
    rows3 = {(s, b): (br, e) for s, b, br, e in vf.budget_rows(sid, 3)}
    means = {b: st.mean(r for _, r in v) for b, v in bp.bench_rates.items()}
    pairs, resid = [], []
    for (s, b), (br1, e1) in rows1.items():
        v1, v3 = max(0.0, br1 - e1 * e1), max(0.0, rows3[(s, b)][0] - rows3[(s, b)][1] ** 2)
        resid.append(abs(v1 - v3))
        lo = (1 - math.sqrt(max(0.0, 1 - 4 * v1))) / 2
        pairs.append((s, b, min((lo, 1 - lo), key=lambda x: abs(x - means.get(b, CENTER)))))
    print(f"{sid}: {len(pairs)} pairs; r(1 - r) from budgets 1 and 3 agree to within {max(resid):.4f} (median {st.median(resid):.4f})")
    real = pv.real_budgets(sid)
    real_alc = sum(w * real[t] for w, t in zip(sim.WEIGHTS, sim.BUDGETS))
    res = alc_by_report({"v7": V7, "v8": V8}, pairs={sid: pairs}, reps=2000)
    sd = st.pstdev(pv.alc_samples(model, model.PARAMS, pairs))
    e = {k: res[k][sid] / len(pairs) for k in ("HBS", "v7", "v8")}
    print(f"{sid}: real Brier ALC {real_alc:.4f}, oracle limit {st.mean(r * (1 - r) for _, _, r in pairs):.4f}; expected HBS {e['HBS']:.4f}, "
          f"v7 {e['v7']:.4f}, v8 {e['v8']:.4f} (label-noise SD {sd:.4f}); real - v8 {real_alc - e['v8']:+.4f}")
    e5 = pv.proxy_budgets(model, model.PARAMS, pairs)

    class WithPrior:
        _summarize = staticmethod(model._summarize)

        @staticmethod
        def predict_with(p, inp, labeled, summary):
            return model.predict_with(params(PRIORS[inp[1]["benchmark_id"]], V8), inp, labeled, summary)

    e8 = pv.proxy_budgets(WithPrior, model.PARAMS, pairs)
    counts = vf.eval_counts(sid)
    e8[0] = st.mean(running_b0(r, RAW[b], counts[(s, b)], V8) for s, b, r in pairs)
    for t in sim.BUDGETS:
        print(f"  budget {t:>2}: real {real[t]:.4f}, HBS expected {e5[t]:.4f}, v8 expected {e8[t]:.4f}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["compare", "grid", "jackknife", "known", "redraw", "budgets"]
    if args[0] == "final":
        final(args[1])
    else:
        for name in args:
            {"compare": compare, "grid": grid, "jackknife": jackknife, "known": known, "redraw": redraw, "budgets": budgets,
             "lobo": lobo}[name]()
