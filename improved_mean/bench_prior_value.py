"""The 69 pairs of the real-feedback proxy, and how much of HBS's gap to the oracle closes
if each benchmark's mean success rate were known before any label (report, Section 6.2).

Reads the eight scoring logs listed in REPORTS from paths.FEEDBACK. Each pair's rate follows
from its budget-0 Brier score and ECE (Proposition 1). For reports whose budget-0 prediction
is unknown (INFERRED), r and 1 - r are told apart by the benchmark's rates in the reports
whose prediction is known; HBS+Ability's rates come from proxy.py, which applies the same rule.
"""
import math, os, random, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model, proxy, sim
from feedback_logs import SUBS, b0_rows  # noqa: F401  (SUBS is used by other scripts)

REPORTS = {"951748": 0.35, "952533": 0.35, "960385": 0.35,  # budget-0 prediction 0.35 (HBS+ItemLLM equals HBS there)
           "951853": None, "928061": None, "930136": None, "930165": None, "931142": None}
INFERRED = [sid for sid, p in REPORTS.items() if p is None]  # budget-0 prediction unknown: roots inferred

v4_rate = {s: r for s, _, r in proxy.DRAWS["report2"]}
pairs, known = {}, {}
for sid, p in REPORTS.items():          # pass 1: reports whose budget-0 prediction is known
    if sid == "951853":
        pairs[sid] = [(s, b, v4_rate[s]) for s, b, _, _ in b0_rows(sid)]
    elif p is not None:
        out = []
        for s, b, brier, ece in b0_rows(sid):
            v = max(0.0, brier - ece * ece)
            r = min((p + ece, p - ece), key=lambda x: abs(x * (1 - x) - v) if 0 <= x <= 1 else 9)
            out.append((s, b, min(1, max(0, r))))
        pairs[sid] = out
    for s, b, r in pairs.get(sid, []):
        known.setdefault(b, []).append(r)
for sid, p in REPORTS.items():          # pass 2: other reports, r vs 1-r picked by the benchmark's known rates
    if sid in pairs: continue
    out = []
    for s, b, brier, ece in b0_rows(sid):
        v = min(0.25, max(0.0, brier - ece * ece))
        lo = (1 - math.sqrt(1 - 4 * v)) / 2
        ref = st.mean(known[b])
        out.append((s, b, min((lo, 1 - lo), key=lambda x: abs(x - ref))))
    pairs[sid] = out

bench_rates = {}
for sid, ps in pairs.items():
    for s, b, r in ps: bench_rates.setdefault(b, []).append((sid, r))

def loro_mean(b, sid):  # benchmark mean from the *other* reports only
    rs = [r for s2, r in bench_rates[b] if s2 != sid]
    return st.mean(rs) if rs else 0.35

def pair_alcs(param_fn, reps=300, seed=0, mod=model):
    out = []
    for sid, ps in pairs.items():
        rng = random.Random(seed)
        acc = {s: 0.0 for s, _, _ in ps}
        for _ in range(reps):
            acq = {s: [int(rng.random() < r) for _ in range(31)] for s, _, r in ps}
            for bi, budget in enumerate(sim.BUDGETS):
                labeled = [[proxy._inp(s, b, i), y] for s, b, _ in ps for i, y in enumerate(acq[s][:budget])]
                summary = mod._summarize(labeled)
                for s, b, r in ps:
                    p = mod.predict_with(param_fn(sid, b), proxy._inp(s, b, "eval"), labeled, summary)
                    acc[s] += sim.WEIGHTS[bi] * (r * (1 - r) + (p - r) ** 2) / reps
        out += [(sid, s, b, r, acc[s]) for s, b, r in ps]
    return out

if __name__ == "__main__":
    variants = {"v5 (PRIOR 0.35)": lambda sid, b: model.PARAMS}
    for a in (3.0, 6.0, 12.0):
        variants[f"bench prior known, ALPHA {a:g}"] = (lambda a: lambda sid, b: dict(model.PARAMS, PRIOR_MEAN=loro_mean(b, sid), ALPHA=a))(a)
    limit = [r * (1 - r) for ps in pairs.values() for _, _, r in ps]
    print(f"\n{len(limit)} pairs; limit (pair rate known) mean {st.mean(limit):.4f}")
    for name, fn in variants.items():
        vals = [x[4] for x in pair_alcs(fn)]
        print(f"{name:<32} mean ALC {st.mean(vals):.4f} | gap to limit {st.mean(vals) - st.mean(limit):+.4f}")
