"""Proxy of real scoring draws, reconstructed from a team's own scoring logs (report, Section 5).

At budget 0, a prediction p that is constant within a pair gives Brier = r(1 - r) + (p - r)^2
and ECE = |p - r| (Proposition 1), so each pair's success rate r follows from its feedback:

- report1 (smoothed mean, 951748) and report3 (HBS, 952533) predict p = 0.35 at budget 0,
  so r = (Brier - 0.35^2) / (1 - 2 * 0.35).
- report2 (HBS+Ability, 951853) predicts a subject-specific p that is not known, so r and p
  are both inferred: r is the root of r(1 - r) = Brier - ECE^2 closer to the benchmark's
  mean rate in the reports that predict 0.35 (951748, 952533, 960385), and p = r + ECE
  (r - ECE if that exceeds 1), with r rounded to three decimals first. V4_B0 keeps p so that the
  proxy can be checked against the real score of HBS+Ability.

Acquired labels are simulated as Bernoulli(r); the expected evaluation Brier of a constant
prediction p is r(1 - r) + (p - r)^2. Item-level variation, subject metadata and item
groups are not represented.
"""

from __future__ import annotations

import math
import random
import statistics as st

import model as current_model
import sim
from feedback_logs import b0_rows, pair_alcs

CONSTANT = 0.35
KNOWN = ("951748", "952533", "960385")  # budget-0 prediction 0.35 (HBS+ItemLLM equals HBS there)


def _rate(brier, p=CONSTANT):
    return min(1.0, max(0.0, (brier - p * p) / (1 - 2 * p)))


def _inferred(sid):
    """(subject, benchmark, r, p) for a report whose budget-0 prediction is unknown."""
    rates = {}
    for known in KNOWN:
        for _, b, brier, _ in b0_rows(known):
            rates.setdefault(b, []).append(_rate(brier))
    out = []
    for s, b, brier, ece in b0_rows(sid):
        v = max(0.0, brier - ece * ece)
        lo = (1 - math.sqrt(max(0.0, 1 - 4 * v))) / 2
        ref = st.mean(rates[b]) if b in rates else CONSTANT
        r = round(min((lo, 1 - lo), key=lambda x: abs(x - ref)), 3)
        p = r + ece if r + ece <= 1 else r - ece
        out.append((s, b, r, round(p, 3)))
    return out


_REPORT2 = _inferred("951853")
DRAWS = {
    "report1": [(s, b, _rate(brier)) for s, b, brier, _ in b0_rows("951748")],
    "report2": [(s, b, r) for s, b, r, _ in _REPORT2],
    "report3": [(s, b, _rate(brier)) for s, b, brier, _ in b0_rows("952533")],
}
V4_B0 = {s: p for s, _, _, p in _REPORT2}
REAL = {key: round(st.mean(pair_alcs(sid).values()), 6)
        for key, sid in ((("v1", "report1"), "951748"), (("v4", "report2"), "951853"), (("v5", "report3"), "952533"))}


def _inp(subject, bench, idx):
    subj = {k: "" for k in current_model._SUBJECT_KEYS}
    subj["normalized_name"] = subject
    return [subj, {"item_content": f"{subject}/{idx}", "item_features": "", "interactors": "", "benchmark_id": bench}]


def evaluate(params, module=current_model, reps=400, seed=0, draws=("report1", "report2", "report3")):
    """Mean Brier ALC and mean Brier per budget, averaged over the given report draws."""
    alc_total, budget_total = 0.0, [0.0] * len(sim.BUDGETS)
    for name in draws:
        pairs = DRAWS[name]
        rng = random.Random(seed)
        for _ in range(reps):
            acq = {s: [int(rng.random() < r) for _ in range(31)] for s, _, r in pairs}
            for bi, budget in enumerate(sim.BUDGETS):
                labeled = [[_inp(s, b, i), y] for s, b, _ in pairs for i, y in enumerate(acq[s][:budget])]
                summary = module._summarize(labeled)
                for s, b, r in pairs:
                    p = module.predict_with(params, _inp(s, b, "eval"), labeled, summary)
                    brier = r * (1 - r) + (p - r) ** 2
                    w = 1.0 / (len(pairs) * reps * len(draws))
                    budget_total[bi] += brier * w
                    alc_total += sim.WEIGHTS[bi] * brier * w
    return alc_total, budget_total


def v4_table(params):
    """Ability table that makes the current model reproduce v4's budget-0 predictions
    for report-2 subjects (used only to validate the proxy against the real v4 score)."""
    import math
    lg = lambda p: math.log(p / (1 - p))
    return {"names": {s.lower(): (lg(p) - lg(params["PRIOR_MEAN"])) / params["BETA"] for s, p in V4_B0.items()},
            "ridge": [0.0] * 5, "year_mean": 2025.0}


if __name__ == "__main__":
    import importlib.util

    spec = importlib.util.spec_from_file_location("model_v1", "versions/model_v1.py")
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    spec = importlib.util.spec_from_file_location("model_v4", "versions/model_v4.py")
    v4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4)
    alc, per = evaluate(v1.PARAMS, v1, draws=("report1",))
    print(f"v1 on report1: proxy={alc:.4f} real={REAL[('v1', 'report1')]:.4f} budgets", [round(x, 3) for x in per])
    v4p = dict(v4.PARAMS, _TABLE=v4_table(v4.PARAMS))
    alc, per = evaluate(v4p, v4, draws=("report2",))
    print(f"v4 on report2: proxy={alc:.4f} real={REAL[('v4', 'report2')]:.4f} budgets", [round(x, 3) for x in per])
    spec = importlib.util.spec_from_file_location("model_v5", "versions/model_v5.py")
    v5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v5)
    alc, per = evaluate(v5.PARAMS, v5, draws=("report3",))
    print(f"v5 on report3: proxy={alc:.4f} real={REAL[('v5', 'report3')]:.4f} budgets", [round(x, 3) for x in per])
    print("\nSame 26 pairs for every version (reports 1-3):")
    for name, mod in (("v1", v1), ("v5", v5)):
        alc, per = evaluate(mod.PARAMS, mod)
        print(f"  {name}: proxy={alc:.4f} budgets", [round(x, 3) for x in per])
