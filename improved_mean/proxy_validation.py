"""Does the real-feedback proxy predict real formative scores? (report, Section 4)

For each submitted version, the proxy's expected Brier ALC on that submission's own
draw is compared with the real score; resampling the acquired labels gives the
spread expected from label luck alone. v6 is compared with v5's proxy (v6 equals v5
whenever its item shift is inactive), so its row measures v6's own effect.

    <venv python> proxy_validation.py
"""

from __future__ import annotations

import importlib.util
import random
import statistics as st

import bench_prior_value as bp
import model
import proxy
import sim


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def real_budgets(sid):
    """Mean real Brier per budget in a scoring log (feedback/<id>_scoring_stdout.txt)."""
    import re
    rows, cur = {}, None
    for line in open(f"{bp.SUBS}/{sid}_scoring_stdout.txt"):
        m = re.match(r"Label budget: (\d+)", line)
        if m:
            cur = int(m.group(1))
        m = re.match(r"(subject_\d+)\s+(benchmark_\d+)\s+\d+\s+([\d.]+)\s+([\d.]+)", line)
        if m and cur is not None:
            rows.setdefault(cur, []).append(float(m.group(3)))
    return {b: st.mean(v) for b, v in rows.items()}


def proxy_budgets(mod, params, pairs, reps=2000, seed=0):
    """Proxy expectation of the mean Brier per budget."""
    rng, out = random.Random(seed), {b: 0.0 for b in sim.BUDGETS}
    for _ in range(reps):
        acq = {s: [int(rng.random() < r) for _ in range(31)] for s, _, r in pairs}
        for budget in sim.BUDGETS:
            labeled = [[proxy._inp(s, b, i), y] for s, b, _ in pairs for i, y in enumerate(acq[s][:budget])]
            summary = mod._summarize(labeled)
            out[budget] += st.mean(r * (1 - r) + (mod.predict_with(params, proxy._inp(s, b, "eval"), labeled, summary) - r) ** 2
                                   for s, b, r in pairs) / reps
    return out


def alc_samples(mod, params, pairs, reps=2000, seed=0):
    """Proxy Brier ALC (mean over pairs) for each of ``reps`` resampled label sequences."""
    rng, out = random.Random(seed), []
    for _ in range(reps):
        acq = {s: [int(rng.random() < r) for _ in range(31)] for s, _, r in pairs}
        alc = 0.0
        for weight, budget in zip(sim.WEIGHTS, sim.BUDGETS):
            labeled = [[proxy._inp(s, b, i), y] for s, b, _ in pairs for i, y in enumerate(acq[s][:budget])]
            summary = mod._summarize(labeled)
            alc += weight * st.mean(r * (1 - r) + (mod.predict_with(params, proxy._inp(s, b, "eval"), labeled, summary) - r) ** 2
                                    for s, b, r in pairs)
        out.append(alc)
    return out


if __name__ == "__main__":
    v1 = _load("versions/model_v1.py", "model_v1")
    v4 = _load("versions/model_v4.py", "model_v4")
    rows = [  # version, module, params, pairs of its own draw, real Brier ALC
        ("v1", v1, v1.PARAMS, proxy.DRAWS["report1"], 0.182910),
        ("v4", v4, dict(v4.PARAMS, _TABLE=proxy.v4_table(v4.PARAMS)), proxy.DRAWS["report2"], 0.163863),
        ("v5", model, model.PARAMS, proxy.DRAWS["report3"], 0.188314),
        ("v6 (vs v5 proxy)", model, model.PARAMS, bp.pairs["960385"], 0.157293),
    ]
    print(f"{'version':<18}{'real':>8}{'proxy':>8}{'label-luck SD':>15}{'real-proxy':>12}{'z':>7}")
    for name, mod, params, pairs, real in rows:
        alcs = alc_samples(mod, params, pairs)
        mu, sd = st.mean(alcs), st.pstdev(alcs)
        print(f"{name:<18}{real:>8.4f}{mu:>8.4f}{sd:>15.4f}{real - mu:>+12.4f}{(real - mu) / sd:>+7.1f}")

    print("\nreal minus proxy, mean Brier per budget (report, Figure 5b)")
    for name, sid, mod, params, pairs in (("v1 (v1 proxy)", "951748", v1, v1.PARAMS, proxy.DRAWS["report1"]),
                                          ("v5 (v5 proxy)", "952533", model, model.PARAMS, proxy.DRAWS["report3"]),
                                          ("v6 (v5 proxy)", "960385", model, model.PARAMS, bp.pairs["960385"])):
        real, exp = real_budgets(sid), proxy_budgets(mod, params, pairs)
        print(f"{name:<16}" + "  ".join(f"b{b} {real[b] - exp[b]:+.3f}" for b in sim.BUDGETS))
