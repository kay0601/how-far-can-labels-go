"""Simulator rows of Table 2 in the report: group-stratified label acquisition and removing
the group correction, each compared with HBS pair by pair on the same simulated draws.

- stratified: replays the streaming protocol (sim.stream_acquire) with the hook of
  versions/labeling_stratified.py, against a hook that takes every candidate, which
  reproduces the default random order;
- group: HBS with the group correction switched off (GROUP_GAMMA = 0).

    <venv python> sim_ablations.py [n_draws] [stratified|group]   # default: 400 draws, seed 2024, both
"""

from __future__ import annotations

import importlib.util
import os
import statistics as st
import sys

import model
import sim

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("labeling_stratified", os.path.join(HERE, "versions", "labeling_stratified.py"))
strat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strat)


def take_all(inp, labeled=None, context=None):
    return True


def report(name, base, new):
    diffs = [b - a for a, b in zip(base, new)]
    print(f"{name}: {len(diffs)} pairs, HBS {st.mean(base):.4f}, variant {st.mean(new):.4f}, "
          f"change {st.mean(diffs):+.4f} (SE {st.stdev(diffs) / len(diffs) ** 0.5:.4f})", flush=True)


def stratified(draws):
    fn = sim.with_params(model.PARAMS)
    base, new = [], []
    for draw in draws:
        base += [alc for _, alc, _ in sim.score_draw(sim.stream_acquire(draw, take_all), fn)]
        new += [alc for _, alc, _ in sim.score_draw(sim.stream_acquire(draw, strat.acquisition_function), fn)]
    report("stratified acquisition", base, new)


def group(draws):
    base, new = [], []
    for draw in draws:
        base += [alc for _, alc, _ in sim.score_draw(draw, sim.with_params(model.PARAMS))]
        new += [alc for _, alc, _ in sim.score_draw(draw, sim.with_params(dict(model.PARAMS, GROUP_GAMMA=0.0)))]
    report("group correction removed", base, new)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    which = sys.argv[2:] or ["stratified", "group"]
    draws = sim.build_draws(n, seed=2024)
    for name in which:
        {"stratified": stratified, "group": group}[name](draws)
