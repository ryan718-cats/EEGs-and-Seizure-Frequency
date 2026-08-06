#!/usr/bin/env python3
"""Subgroup AUROC by Generalized epilepsy subtype — Full Model4 Youden.

Point + DeLong 95% CI (bagged OOS) for Model4_IED_slow on lightSpike Aim2.
No title. Pairwise Bonferroni-significant contrasts annotated above.
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
    / "General"
    / "aim2_generalized_poor1mo_youden_model4_cont_compare_lightSpike"
)
PAPER_OUT = (
    ML_DIR.parent
    / "output"
    / "FigAim2_subgroup_auroc_by_subtype_model4_full_lightSpike.png"
)

FONT = 15

# Display labels + colors (same family as other Aim2 subgroup figs)
DISPLAY = {
    "Unspecified Generalized": "Unspecified",
    "Juvenile Myoclonic": "JME",
    "GTCA": "GTCA",
    "Other Specified Generalized (LGS)": "LGS",
}
# Uniform blue (same family as other Aim figures)
BLUE = np.array([0.36, 0.45, 0.66])
COLORS = {name: BLUE for name in (
    "Unspecified Generalized",
    "Juvenile Myoclonic",
    "GTCA",
    "Other Specified Generalized (LGS)",
)}
ORDER = [
    "Unspecified Generalized",
    "Juvenile Myoclonic",
    "GTCA",
    "Other Specified Generalized (LGS)",
]


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


def main() -> None:
    stratum = pd.read_csv(OUT_DIR / "bagged_metrics_by_subtype.csv")
    pairwise = pd.read_csv(OUT_DIR / "subtype_pairwise_delong.csv")
    stratum = stratum[stratum["model"] == "Model4_IED_slow"].copy()
    pairwise = pairwise[pairwise["model"] == "Model4_IED_slow"].copy()

    rows = []
    for name in ORDER:
        r = stratum[stratum["stratum"] == name].iloc[0]
        rows.append(r)

    xs = np.arange(len(rows))
    aucs = np.array([float(r["bagged_auc"]) for r in rows])
    lo = np.array([float(r["delong_ci_lo"]) for r in rows])
    hi = np.array([min(1.0, float(r["delong_ci_hi"])) for r in rows])  # clip display
    ns = [int(r["n_patients"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for x, auc, lo_i, hi_i, name in zip(xs, aucs, lo, hi, ORDER):
        c = COLORS[name]
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

    labels = [
        f"{DISPLAY[name]}\nAUC = {auc:.2f}"
        for name, auc in zip(ORDER, aucs)
    ]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=FONT - 2)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Classification AUC", fontsize=FONT + 1)
    ax.tick_params(axis="y", labelsize=FONT - 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    # Annotate Bonferroni-significant pairwise contrasts
    idx = {name: i for i, name in enumerate(ORDER)}
    sig = pairwise[pairwise["p_bonferroni"] < 0.05].copy()
    sig = sig.sort_values("p_bonferroni")
    y_base = float(max(hi)) + 0.045
    tick = 0.016
    for k, (_, row) in enumerate(sig.iterrows()):
        i = idx[row["subtype_a"]]
        j = idx[row["subtype_b"]]
        if i > j:
            i, j = j, i
        y = y_base + k * 0.075
        ax.plot([i, i, j, j], [y - tick, y, y, y - tick], color="0.45", lw=1.3)
        star = stars(float(row["p_bonferroni"]))
        ann = f"{star + ' ' if star else ''}{p_fmt(float(row['p_bonferroni']))} (Bonf.)"
        ax.text(
            (i + j) / 2,
            y + 0.008,
            ann.strip(),
            ha="center",
            va="bottom",
            fontsize=FONT - 2,
            color="0.35",
            fontweight="bold",
        )

    y_top = y_base + max(0, len(sig) - 1) * 0.075 + 0.08 if len(sig) else 1.02
    ax.set_ylim(0.40, max(1.05, y_top))

    fig.tight_layout()
    out_png = OUT_DIR / "subgroup_auroc_by_subtype_model4_full.png"
    PAPER_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(PAPER_OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    lines = [
        "Subgroup AUROC by Generalized subtype — Full Model4 Youden (lightSpike)",
        f"OUT: {OUT_DIR}",
        "",
    ]
    for name, r in zip(ORDER, rows):
        lines.append(
            f"  {DISPLAY[name]:12s} N={int(r.n_patients):3d} AUC={r.bagged_auc:.3f} "
            f"SE={r.delong_se:.4f} 95%CI=[{r.delong_ci_lo:.3f},{min(1.0, r.delong_ci_hi):.3f}]"
        )
    lines.append("")
    lines.append("Pairwise DeLong (Bonferroni-adjusted p):")
    for _, row in pairwise.iterrows():
        sig_lab = stars(float(row["p_bonferroni"])) or "ns"
        lines.append(
            f"  {DISPLAY[row['subtype_a']]} vs {DISPLAY[row['subtype_b']]}: "
            f"Δ={row['delta_a_minus_b']:+.3f} p={row['p']:.4g} "
            f"p_bonf={row['p_bonferroni']:.4g} {sig_lab}"
        )
    lines.append("")
    lines.append(f"Saved {out_png}")
    lines.append(f"Saved {PAPER_OUT}")
    txt = OUT_DIR / "subgroup_auroc_by_subtype_model4_full.txt"
    txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
