"""Readers for a team's own Codabench scoring logs (formative feedback).

Save each submission's "Scoring output" as <submission id>_scoring_stdout.txt in the folder
given by paths.FEEDBACK. The logs are private to each team, so ours are not released.
"""

import os
import re

import paths

SUBS = paths.FEEDBACK
_ROW = re.compile(r"(subject_\d+)\s+(benchmark_\d+)\s+\d+\s+([\d.]+)\s+([\d.]+)")


def b0_rows(sid):
    """(subject, benchmark, budget-0 Brier, budget-0 ECE) for every pair of a scoring log."""
    rows, cur = [], None
    for line in open(os.path.join(SUBS, f"{sid}_scoring_stdout.txt")):
        m = re.match(r"Label budget: (\d+)", line)
        if m:
            cur = int(m.group(1))
        m = _ROW.match(line)
        if m and cur == 0:
            rows.append((m.group(1), m.group(2), float(m.group(3)), float(m.group(4))))
    return rows


def pair_alcs(sid):
    """{(subject, benchmark): Brier ALC} from the 'ALC summary' table of a scoring log."""
    text = open(os.path.join(SUBS, f"{sid}_scoring_stdout.txt")).read()
    text = text.split("ALC summary")[1].split("Label budget:")[0]
    rows = (_ROW.match(line) for line in text.splitlines())
    return {(m.group(1), m.group(2)): float(m.group(3)) for m in rows if m}
