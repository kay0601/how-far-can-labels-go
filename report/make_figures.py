"""Figures for the technical report (figures/*.pdf, with .png previews).

Numbers come from ../improved_mean/figure_data.py (figures/figure_data.json) and from
our eight formative reports (DRAWS below). Method names follow the report's Appendix A.

    <python with matplotlib> make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
DATA = json.loads((FIG / "figure_data.json").read_text())
ACCENT, GREY, LIGHT, RED, PRIOR = "#1f6fd1", "0.55", "0.80", "#c0392b", "#e08214"
# NeurIPS accepts only Type 1 or embedded TrueType fonts: fonttype 42 embeds TrueType
# (matplotlib's default, 3, embeds Type 3). Liberation Serif matches the paper's Times.
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"],
                     "font.size": 8, "mathtext.fontset": "stix", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

DRAWS = [  # label, oracle limit, achieved Brier ALC (Codabench IDs in the report's Appendix A)
    ("HBS+ItemLLM", 0.0994, 0.1573),
    ("HBS+Ability", 0.1218, 0.1639),
    ("Smoothed mean", 0.1509, 0.1829),
    ("Early 2", 0.1572, 0.1895),
    ("Early 4", 0.1612, 0.1978),
    ("Early 3", 0.1616, 0.2797),
    ("HBS", 0.1674, 0.1883),
    ("Early 1", 0.1832, 0.2154),
]
SHORT = {"matharena": "math-\narena", "multi_swebench": "multi_\nswebench",
         "real_webagents": "real_\nwebagents", "researchcodebench": "research-\ncodebench"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def teaser():
    """Figure 1: the proxy (left) and what limits label-only prediction (right)."""
    fig = plt.figure(figsize=(5.5, 2.45))
    left = fig.add_axes([0.0, 0.0, 0.50, 1.0])
    left.set_axis_off()
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.text(0.03, 0.95, "(a) A real-feedback proxy", fontsize=9, fontweight="bold", va="top")
    steps = [
        ("Each submission is scored on a new random draw:\nscores mix method quality with draw difficulty", "0.95"),
        ("At budget 0 a pair's predictions are constant,\nso $r(1-r) = \\mathrm{Brier} - \\mathrm{ECE}^2$", "0.95"),
        ("Eight reports yield 69 real pairs\nwith known success rates $r$", "0.95"),
        ("Score any label-only predictor\non the same pairs", "#e8f0fb"),
    ]
    y, h = 0.80, 0.135
    for i, (text, face) in enumerate(steps):
        top = y - i * 0.195
        left.add_patch(FancyBboxPatch((0.05, top - h), 0.88, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                      fc=face, ec=GREY, lw=0.6))
        left.text(0.49, top - h / 2, text, ha="center", va="center", fontsize=7.2)
        if i < len(steps) - 1:
            left.annotate("", xy=(0.49, top - h - 0.055), xytext=(0.49, top - h - 0.004),
                          arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.8))

    gap = DATA["gap_by_budget"]["alc"]
    ax = fig.add_axes([0.735, 0.30, 0.245, 0.58])
    bars = [("Constant 0.5", 0.25, LIGHT), ("HBS", gap["v5"], ACCENT),
            ("HBS+Prior\n(final submission)", gap["HBS+Prior"], PRIOR),
            ("HBS, benchmark\nrate known", gap["benchmark known"], GREY),
            ("Oracle\n(pair rate known)", gap["oracle"], GREY)]
    ys = list(range(len(bars)))[::-1]
    for yv, (name, val, color) in zip(ys, bars):
        ax.barh(yv, val, color=color, height=0.6)
        ax.text(val + 0.003, yv, f"{val:.3f}", va="center", fontsize=8, fontweight="bold" if color == PRIOR else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars], fontsize=7.6)
    ax.set_xlim(0, 0.30)
    ax.set_xlabel("Brier ALC on 69 real pairs", fontsize=7.6)
    ax.tick_params(axis="both", labelsize=7.4, length=2)
    fig.text(0.53, 0.95, "(b) What limits label-only prediction", fontsize=9, fontweight="bold", va="top")
    share = (gap["v5"] - gap["benchmark known"]) / (gap["v5"] - gap["oracle"])
    closed = (gap["v5"] - gap["HBS+Prior"]) / (gap["v5"] - gap["benchmark known"])
    fig.text(0.535, 0.135, f"Unknown benchmark difficulty accounts for {share:.0%} of the\n"
             f"gap to the oracle; the LLM benchmark prior closes {closed:.0%} of it.", fontsize=7.4, va="top", color="0.25")
    save(fig, "teaser")


def draw_limits():
    """Oracle limit and achieved score of each scored draw."""
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    for i, (label, limit, score) in enumerate(DRAWS):
        y = len(DRAWS) - 1 - i
        ax.plot([limit, score], [y, y], color=LIGHT, lw=1.5, zorder=1)
        ax.scatter([limit], [y], s=22, color=GREY, zorder=2, label="oracle limit of the draw" if i == 0 else None)
        ax.scatter([score], [y], s=22, color=ACCENT, zorder=3, label="achieved Brier ALC" if i == 0 else None)
    ax.set_yticks(range(len(DRAWS)))
    ax.set_yticklabels([d[0] for d in reversed(DRAWS)])
    for tick in ax.get_yticklabels():
        if tick.get_text() == "HBS":
            tick.set_fontweight("bold")
    ax.set_xlim(0.08, 0.30)
    ax.set_xticks([0.10, 0.15, 0.20, 0.25, 0.30])
    ax.set_xlabel("Brier ALC (lower is better); rows sorted by oracle limit")
    ax.grid(axis="x", color="0.92", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save(fig, "draw_limits")


def gap_by_budget():
    """Mean Brier per budget: where v5 loses against knowing benchmark difficulty."""
    g, budgets = DATA["gap_by_budget"], DATA["budgets"]
    x = list(range(len(budgets)))
    fig, ax = plt.subplots(figsize=(3.3, 2.0))  # shown at about this size, so 7-8 pt text stays legible
    ax.fill_between(x, g["benchmark known"], g["v5"], color=ACCENT, alpha=0.12, lw=0)
    ax.plot(x, g["v5"], color=ACCENT, marker="o", ms=3.5, lw=1.4, label="HBS")
    ax.plot(x, g["HBS+Prior"], color=PRIOR, marker="s", ms=3.2, lw=1.3, label="HBS+Prior (final submission)")
    ax.plot(x, g["benchmark known"], color=GREY, marker="o", ms=3.5, lw=1.2, ls="--",
            label="HBS, benchmark rate known (diagnostic)")
    ax.plot(x, g["oracle"], color=LIGHT, lw=1.2, label="Oracle (pair rate known)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Label budget (labels per pair)", fontsize=7.5)
    ax.set_ylabel("Mean Brier, 69 real pairs", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.text(0.12, 0.1695, "unknown benchmark\ndifficulty", fontsize=7, color=ACCENT, va="center")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1.6)
    ax.grid(axis="y", color="0.92", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "gap_by_budget")


def text_difficulty():
    """(a) Anchored LLM ratings by benchmark; (b) HBS+ItemLLM against the proxy by budget."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.35), gridspec_kw={"width_ratios": [1.25, 1]})
    rows8 = DATA["anchored"]["Qwen/Qwen3-8B"]
    order = sorted(rows8, key=lambda bench: rows8[bench]["actual"])
    xs = list(range(len(order)))
    a.axhline(0.35, color=GREY, lw=0.8, ls=":", label="HBS prior")
    models = [("Qwen/Qwen3-8B", "o", "#9ecae1", "Qwen3-8B"),
              ("Qwen/Qwen3.5-9B", "^", "#4292c6", "Qwen3.5-9B"),
              ("Qwen/Qwen3.8-27B", "s", "#08306b", "Qwen3.8-27B")]
    models = [m for m in models if m[0] in DATA["anchored"]]
    series = [("actual", "D", "black", "actual", None)] + [(k, mk, c, lab, "predicted") for k, mk, c, lab in models]
    for i, bench in enumerate(order):
        vals = [rows8[bench]["actual"]] + [DATA["anchored"][m[0]][bench]["predicted"] for m in models]
        a.plot([i, i], [min(vals), max(vals)], color="0.88", lw=3, zorder=1, solid_capstyle="round")
    for key, mk, color, label, field in series:
        ys = [rows8[bench]["actual"] if field is None else DATA["anchored"][key][bench]["predicted"] for bench in order]
        a.scatter(xs, ys, marker=mk, s=26 if mk != "D" else 22, color=color, label=label, zorder=4 if mk == "D" else 3,
                  edgecolors="white", linewidths=0.4)
    a.set_xticks(xs)
    a.set_xticklabels([SHORT[bench] for bench in order], fontsize=6.2)
    a.set_xlim(-0.5, len(xs) - 0.5)
    a.set_ylim(0, 1.0)
    a.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    a.set_ylabel("Success rate")
    a.set_xlabel("Held-out benchmark (easier to the right)")
    a.set_title("(a) Anchored LLM ratings vs actual", fontsize=8.5)
    a.legend(frameon=False, fontsize=6.3, loc="upper left", ncol=3, columnspacing=0.8, handletextpad=0.3)

    budgets, v6 = DATA["budgets"], DATA["v6_by_budget"]
    x = list(range(len(budgets)))
    b.axhline(0, color=GREY, lw=0.8)
    b.axhspan(-0.021, 0.021, color="0.94", lw=0)
    b.text(len(x) - 1, -0.0195, "label noise, 1 SD", fontsize=6, color=GREY, ha="right", va="bottom")
    for name, color, lw, label in (("v1", LIGHT, 1.2, "Smoothed mean"), ("v5", GREY, 1.2, "HBS"),
                                   ("v6", RED, 1.6, "HBS+ItemLLM, vs. HBS")):
        b.plot(x, v6[name], color=color, marker="o", ms=3, lw=lw, label=label)
    b.set_xticks(x)
    b.set_xticklabels([str(v) for v in budgets])
    b.set_xlabel("Label budget")
    b.set_ylabel("Real minus proxy Brier")
    b.set_title("(b) HBS+ItemLLM on its real draw", fontsize=8.5)
    b.legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    save(fig, "text_difficulty")


def pair_validity():
    """Appendix: per-pair z-scores of real against proxy-expected Brier ALC (proxy_robustness.py)."""
    data = json.loads((FIG / "robustness.json").read_text())["pair_z"]["z"]
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    ax.axhspan(-2, 2, color="0.94", lw=0)
    ax.axhline(0, color=GREY, lw=0.8)
    names = list(data)
    for i, name in enumerate(names):
        zs = sorted(data[name])
        xs = [i + 0.22 * ((k % 5) - 2) / 2 for k in range(len(zs))]
        ax.scatter(xs, zs, s=16, color=ACCENT, edgecolors="white", linewidths=0.4, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [sum(zs) / len(zs)] * 2, color="black", lw=1.2, zorder=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels({"v1": "Smoothed mean", "v4": "HBS+Ability", "v5": "HBS"}.get(n, n) for n in names)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylabel("$z$ = (real $-$ proxy) / SD", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.text(len(names) - 0.45, 1.85, "$\\pm 2$ SD", fontsize=6.5, color=GREY, ha="right", va="top")
    fig.tight_layout()
    save(fig, "pair_validity")


if __name__ == "__main__":
    teaser()
    draw_limits()
    gap_by_budget()
    text_difficulty()
    if (FIG / "robustness.json").exists():
        pair_validity()
    print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))
