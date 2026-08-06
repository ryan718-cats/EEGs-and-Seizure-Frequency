#!/usr/bin/env python3
"""Aim1 Interaction model — subgroup AUROC figures (lightSpike Youden).

1) Prior available vs unavailable
2) Focal vs Generalized

Point + DeLong 95% CI; no titles. Style matches Aim2 subgroup figs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent
OUT_DIR = (
    ML_DIR
    / "outputs"
    / "Combined"
    / "aim1_model5_interaction_poor1.0mo_matched_prior_youden_grid_lightSpike"
)
PAPER_OUT = ML_DIR.parent / "output"
MODEL = "Model_interaction"

C_A = np.array([0.36, 0.45, 0.66])
C_B = np.array([0.75, 0.51, 0.33])
FONT = 15


def p_fmt(p: float) -> str:
    if not np.isfinite(p):
        return "p=NaN"
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return f"p={p:.3f}"
    if p < 0.1:
        return f"p={p:.2f}"
    return f"p={p:.2f}"


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_two(
    ax,
    row_a: pd.Series,
    row_b: pd.Series,
    labels: tuple[str, str],
    p: float,
    delta: float,
    y_min: float = 0.30,
) -> None:
    xs = [0, 1]
    aucs = [float(row_a["auroc"]), float(row_b["auroc"])]
    lo = [float(row_a["delong_ci_lo"]), float(row_b["delong_ci_lo"])]
    hi = [float(row_a["delong_ci_hi"]), float(row_b["delong_ci_hi"])]
    cols = [C_A, C_B]
    for x, auc, c, lo_i, hi_i in zip(xs, aucs, cols, lo, hi):
        ax.errorbar(
            x,
            auc,
            yerr=[[auc - lo_i], [hi_i - auc]],
            fmt="o",
            markersize=11,
            color=c,
            ecolor=c,
            elinewidth=2.2,
            capsize=7,
            capthick=2.0,
            zorder=3,
        )

    xticklabels = [
        f"{labels[0]}\nAUC = {aucs[0]:.2f}",
        f"{labels[1]}\nAUC = {aucs[1]:.2f}",
    ]
    ax.set_xticks(xs)
    ax.set_xticklabels(xticklabels, fontsize=FONT - 1)
    ax.set_xlim(-0.55, 1.55)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    y_bar = max(hi) + 0.04
    y_bar = max(y_bar, 0.90)
    tick = 0.018
    ax.plot([0, 0, 1, 1], [y_bar - tick, y_bar, y_bar, y_bar - tick], color="0.45", lw=1.4)
    star = stars(p)
    ann = f"{star + ' ' if star else ''}{p_fmt(p)}, Δ = {delta:.2f}"
    ax.text(
        0.5,
        y_bar + 0.012,
        ann.strip(),
        ha="center",
        va="bottom",
        fontsize=FONT,
        color="0.35",
        fontweight="bold" if star else "normal",
    )
    ax.set_ylim(y_min, max(1.05, y_bar + 0.10))
    ax.set_ylabel("Classification AUC", fontsize=FONT + 1)
    ax.tick_params(axis="y", labelsize=FONT - 1)


def save_fig(fig, stem: str, lines: list[str]) -> None:
    out_png = OUT_DIR / f"{stem}.png"
    paper_png = PAPER_OUT / f"{stem}_lightSpike.png"
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paper_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    txt = OUT_DIR / f"{stem}.txt"
    txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved {out_png}")
    print(f"Saved {paper_png}")


def fig_prior() -> None:
    metrics = pd.read_csv(OUT_DIR / "bagged_metrics_by_prior_info.csv")
    delong = pd.read_csv(OUT_DIR / "prior_info_available_vs_unavailable_delong.csv")
    avail = metrics[(metrics.model == MODEL) & (metrics.prior_info == "available")].iloc[0]
    unavail = metrics[(metrics.model == MODEL) & (metrics.prior_info == "unavailable")].iloc[0]
    drow = delong[delong.model == MODEL].iloc[0]
    # delta = available − unavailable
    delta = float(drow["delta"])
    p = float(drow["p"])

    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    plot_two(
        ax, avail, unavail,
        ("Prior info available", "No prior info"),
        p=p, delta=delta, y_min=0.30,
    )
    sig = stars(p) or "ns"
    lines = [
        "Aim1 Interaction — subgroup AUROC by prior info (lightSpike)",
        f"  available   N={int(avail.n)} AUC={avail.auroc:.3f} "
        f"95%CI=[{avail.delong_ci_lo:.3f},{avail.delong_ci_hi:.3f}]",
        f"  unavailable N={int(unavail.n)} AUC={unavail.auroc:.3f} "
        f"95%CI=[{unavail.delong_ci_lo:.3f},{unavail.delong_ci_hi:.3f}]",
        f"  DeLong Δ(avail−unavail)={delta:+.3f} p={p:.4g} {sig}",
        "",
    ]
    save_fig(fig, "FigAim1_subgroup_auroc_by_prior_interaction", lines)


def fig_epilepsy_type() -> None:
    metrics = pd.read_csv(OUT_DIR / "bagged_metrics_by_epilepsy_type.csv")
    delong = pd.read_csv(OUT_DIR / "focal_vs_generalized_delong.csv")
    foc = metrics[(metrics.model == MODEL) & (metrics.epilepsy_type == "Focal")].iloc[0]
    gen = metrics[(metrics.model == MODEL) & (metrics.epilepsy_type == "Generalized")].iloc[0]
    drow = delong[delong.model == MODEL].iloc[0]
    # CSV delta = Focal − Generalized; plot Focal then Generalized
    delta_f_minus_g = float(drow["delta"])
    p = float(drow["p"])

    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    plot_two(
        ax, foc, gen,
        ("Focal", "Generalized"),
        p=p, delta=delta_f_minus_g, y_min=0.40,
    )
    sig = stars(p) or "ns"
    lines = [
        "Aim1 Interaction — subgroup AUROC by epilepsy type (lightSpike)",
        f"  Focal       N={int(foc.n)} AUC={foc.auroc:.3f} "
        f"95%CI=[{foc.delong_ci_lo:.3f},{foc.delong_ci_hi:.3f}]",
        f"  Generalized N={int(gen.n)} AUC={gen.auroc:.3f} "
        f"95%CI=[{gen.delong_ci_lo:.3f},{gen.delong_ci_hi:.3f}]",
        f"  DeLong Δ(Focal−Gen)={delta_f_minus_g:+.3f} p={p:.4g} {sig}",
        "",
    ]
    save_fig(fig, "FigAim1_subgroup_auroc_by_epilepsy_type_interaction", lines)


def main() -> None:
    fig_prior()
    print()
    fig_epilepsy_type()


if __name__ == "__main__":
    main()
