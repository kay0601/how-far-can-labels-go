"""Group-stratified streaming acquisition (experimental; report, Section 6.1 and Table 2).

Spread a pair's labels evenly over item groups (``item_features`` without
numeric fields, e.g. language, website, paper or contest), which feeds the
predictor's group adjustment and stratifies its pair-mean estimate.

A candidate is revealed when its group has no more labels than the least
sampled group so far (a new group always qualifies), or when every remaining
candidate is needed to fill the remaining budget. Items without a group are
always revealed, which reduces to the default random order. The predictor is
never called (no ``prediction`` argument), and no state is kept between calls.
"""

from __future__ import annotations

_SUBJECT_KEYS = (
    "normalized_name", "provider", "release_date", "access_date", "harness",
    "reasoning_effort", "harness_version", "subject_features_extra",
)


def _group(item: dict):
    parts = []
    for part in item.get("item_features", "").split(";"):
        key, _, value = part.partition("=")
        value = value.strip()
        if value and not value.replace(".", "", 1).isdigit() and key.strip() != "prompt_source":
            parts.append(part.strip())
    return ";".join(sorted(parts)) or None


def acquisition_function(input, *, labeled=None, context=None) -> bool:
    if context is None:  # legacy one-argument evaluators rank by priority; keep random order
        return 0.0
    remaining = int(context.get("labels_remaining", 0))
    if remaining <= 0:
        return False
    if int(context.get("items_remaining", 0)) <= remaining:
        return True

    subject, item = input
    group = _group(item)
    if group is None:
        return True
    key = tuple(subject.get(k, "") for k in _SUBJECT_KEYS)
    bench = item.get("benchmark_id", "")
    counts = {}
    for (s, i), _ in labeled or ():
        if i.get("benchmark_id", "") == bench and tuple(s.get(k, "") for k in _SUBJECT_KEYS) == key:
            g = _group(i)
            if g is not None:
                counts[g] = counts.get(g, 0) + 1
    return counts.get(group, 0) <= min(counts.values(), default=0)
