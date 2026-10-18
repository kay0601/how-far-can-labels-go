"""Simulator rows of Table 2 in the report: group-stratified label acquisition and removing
the group correction, each compared with HBS pair by pair on the same simulated draws, and the
corrections removed in Section 4 and Appendix E.

- stratified: replays the streaming protocol (sim.stream_acquire) with the hook of
  versions/labeling_stratified.py, against a hook that takes every candidate, which
  reproduces the default random order;
- group: HBS with the group correction switched off (GROUP_GAMMA = 0);
- item: HBS with the item correction switched off (ITEM_GAMMA = 0);
- corrections: both switched off.

    <venv python> sim_ablations.py [n_draws] [stratified|group|item|corrections]   # default: 400 draws, seed 2024, all
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


def removed(name, **change):
    def run(draws):
        base, new = [], []
        for draw in draws:
            base += [alc for _, alc, _ in sim.score_draw(draw, sim.with_params(model.PARAMS))]
            new += [alc for _, alc, _ in sim.score_draw(draw, sim.with_params(dict(model.PARAMS, **change)))]
        report(name, base, new)
    return run


ABLATIONS = {
    "stratified": stratified,
    "group": removed("group correction removed", GROUP_GAMMA=0.0),
    "item": removed("item correction removed", ITEM_GAMMA=0.0),
    "corrections": removed("item and group corrections removed", ITEM_GAMMA=0.0, GROUP_GAMMA=0.0),
}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    which = sys.argv[2:] or list(ABLATIONS)
    draws = sim.build_draws(n, seed=2024)
    for name in which:
        ABLATIONS[name](draws)
