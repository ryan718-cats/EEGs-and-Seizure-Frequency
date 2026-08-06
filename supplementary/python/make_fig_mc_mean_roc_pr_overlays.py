#!/usr/bin/env python3
"""MC-mean ROC / PR overlays (1000 permutations) — Aim1 + Aim2 Gen/Focal.

Curves: each MC split is interpolated onto a common grid and drawn lightly;
the solid line is the pointwise mean. Legend metrics / CIs: MC mean AUROC or
AUPRC with bootstrap 95% CI of the mean. Δ / p in caption text: bagged OOS
DeLong (AUROC) or paired bootstrap (AUPRC).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent
PAPER_OUT = ML_DIR.parent / "output"

C_CLIN = np.array([0.30, 0.47, 0.66])  # blue
C_FULL = np.array([0.80, 0.47, 0.27])  # orange
N_GRID = 401
MEAN_LW = 2.6
LIGHT_ALPHA = 0.035
LIGHT_LW = 0.55


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "p=NA"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _fmt_delta(d: float, lo: float, hi: float) -> str:
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def interp_roc_matrix(csv_path: Path, fpr_grid: np.ndarray) -> np.ndarray:
    df = pd.read_csv(csv_path)
    seeds = np.sort(df["seed"].unique())
    mat = np.empty((len(seeds), len(fpr_grid)), dtype=float)
    for i, seed in enumerate(seeds):
        sub = df.loc[df["seed"] == seed, ["fpr", "tpr"]].to_numpy(dtype=float)
        fpr, tpr = sub[:, 0], sub[:, 1]
        order = np.argsort(fpr)
        fpr, tpr = fpr[order], tpr[order]
        if fpr[0] > 0:
            fpr, tpr = np.r_[0.0, fpr], np.r_[0.0, tpr]
        if fpr[-1] < 1:
            fpr, tpr = np.r_[fpr, 1.0], np.r_[tpr, tpr[-1]]
        keep = np.r_[True, fpr[1:] != fpr[:-1]]
        # keep last of each tie (scan reverse then restore)
        fpr_r, tpr_r = fpr[::-1], tpr[::-1]
        keep_r = np.r_[True, fpr_r[1:] != fpr_r[:-1]]
        fpr_u, tpr_u = fpr_r[keep_r][::-1], tpr_r[keep_r][::-1]
        mat[i] = np.interp(fpr_grid, fpr_u, tpr_u)
    return np.clip(mat, 0.0, 1.0)


def interp_pr_matrix(csv_path: Path, recall_grid: np.ndarray) -> np.ndarray:
    """Interpolate precision vs recall for each MC seed onto a common recall grid."""
    df = pd.read_csv(csv_path)
    seeds = np.sort(df["seed"].unique())
    mat = np.empty((len(seeds), len(recall_grid)), dtype=float)
    for i, seed in enumerate(seeds):
        sub = df.loc[df["seed"] == seed, ["recall", "precision"]].to_numpy(dtype=float)
        rec, prec = sub[:, 0], sub[:, 1]
        # Drop sklearn's artificial (recall=0, precision=1) if present after real zeros;
        # keep real operating points, then ensure endpoints for interpolation.
        order = np.argsort(rec)
        rec, prec = rec[order], prec[order]
        # unique recall ascending: keep last precision at each recall
        keep = np.r_[rec[1:] != rec[:-1], True]
        rec_u, prec_u = rec[keep], prec[keep]
        if rec_u[0] > 0:
            # extrapolate leftmost with first observed precision
            rec_u = np.r_[0.0, rec_u]
            prec_u = np.r_[prec_u[0], prec_u]
        if rec_u[-1] < 1:
            rec_u = np.r_[rec_u, 1.0]
            prec_u = np.r_[prec_u, prec_u[-1]]
        mat[i] = np.interp(recall_grid, rec_u, prec_u)
    return np.clip(mat, 0.0, 1.0)


def _add_light_curves(
    ax,
    x_grid: np.ndarray,
    mat: np.ndarray,
    rgb: np.ndarray,
    *,
    alpha: float = LIGHT_ALPHA,
    lw: float = LIGHT_LW,
) -> None:
    """Draw each MC-permutation curve lightly behind the mean."""
    from matplotlib.collections import LineCollection

    segs = np.stack(
        [np.column_stack([x_grid, mat[i]]) for i in range(mat.shape[0])],
        axis=0,
    )
    color = (*rgb.tolist(), float(alpha))
    lc = LineCollection(segs, colors=[color], linewidths=lw, rasterized=True, zorder=1)
    ax.add_collection(lc)


def plot_roc_overlay(
    mat_clin: np.ndarray,
    mat_full: np.ndarray,
    fpr_grid: np.ndarray,
    label_clin: str,
    label_full: str,
    out_paths: list[Path],
) -> None:
    mean_clin = mat_clin.mean(axis=0)
    mean_full = mat_full.mean(axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 6.8), dpi=150)
    ax.set_facecolor("white")
    _add_light_curves(ax, fpr_grid, mat_clin, C_CLIN)
    _add_light_curves(ax, fpr_grid, mat_full, C_FULL)
    ax.plot([0, 1], [0, 1], "--", color="0.55", lw=1.3, zorder=2)
    ax.plot(fpr_grid, mean_clin, color=C_CLIN, lw=MEAN_LW, zorder=4, label=label_clin)
    ax.plot(fpr_grid, mean_full, color=C_FULL, lw=MEAN_LW, zorder=4, label=label_full)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False positive rate", fontsize=16)
    ax.set_ylabel("True positive rate", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved {p}")
    plt.close(fig)


def plot_pr_overlay(
    mat_clin: np.ndarray,
    mat_full: np.ndarray,
    recall_grid: np.ndarray,
    label_clin: str,
    label_full: str,
    prevalence: float,
    out_paths: list[Path],
) -> None:
    mean_clin = mat_clin.mean(axis=0)
    mean_full = mat_full.mean(axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 6.8), dpi=150)
    ax.set_facecolor("white")
    _add_light_curves(ax, recall_grid, mat_clin, C_CLIN)
    _add_light_curves(ax, recall_grid, mat_full, C_FULL)
    ax.axhline(prevalence, color="0.55", ls="--", lw=1.3, zorder=2)
    ax.plot(recall_grid, mean_clin, color=C_CLIN, lw=MEAN_LW, zorder=4, label=label_clin)
    ax.plot(recall_grid, mean_full, color=C_FULL, lw=MEAN_LW, zorder=4, label=label_full)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved {p}")
    plt.close(fig)


def row_metrics(summ: pd.DataFrame, model: str) -> dict:
    r = summ[summ["model"] == model].iloc[0]
    # Aim1 uses auc_mean; some tables use auroc_mean
    auc = float(r["auc_mean"] if "auc_mean" in r.index else r["auroc_mean"])
    auc_lo = float(r["auc_ci_lo"] if "auc_ci_lo" in r.index else r["auroc_ci_lo"])
    auc_hi = float(r["auc_ci_hi"] if "auc_ci_hi" in r.index else r["auroc_ci_hi"])
    return {
        "auc": auc,
        "auc_lo": auc_lo,
        "auc_hi": auc_hi,
        "auprc": float(r["auprc_mean"]),
        "auprc_lo": float(r["auprc_ci_lo"]),
        "auprc_hi": float(r["auprc_ci_hi"]),
    }


def write_caption(
    path: Path,
    section: str,
    metric: str,
    clin_name: str,
    full_name: str,
    m_clin: dict,
    m_full: dict,
    delta: float,
    d_lo: float,
    d_hi: float,
    p: float,
    note: str,
) -> None:
    if metric == "AUROC":
        clin_s = f"{m_clin['auc']:.3f} [{m_clin['auc_lo']:.3f}, {m_clin['auc_hi']:.3f}]"
        full_s = f"{m_full['auc']:.3f} [{m_full['auc_lo']:.3f}, {m_full['auc_hi']:.3f}]"
    else:
        clin_s = (
            f"{m_clin['auprc']:.3f} [{m_clin['auprc_lo']:.3f}, {m_clin['auprc_hi']:.3f}]"
        )
        full_s = (
            f"{m_full['auprc']:.3f} [{m_full['auprc_lo']:.3f}, {m_full['auprc_hi']:.3f}]"
        )
    sig = "statistically different" if p < 0.05 else "not statistically different"
    body = (
        f"{section}. {full_name} ({full_s}) vs {clin_name} ({clin_s}) "
        f"difference was {sig} ({_fmt_delta(delta, d_lo, d_hi)}; {_fmt_p(p)})."
    )
    text = (
        f"{body}\n\n"
        f"Notes:\n"
        f"- Point estimates and 95% CIs are MC means over 1000 patient-grouped "
        f"80/20 permutations (bootstrap CI of the mean).\n"
        f"- ROC/PR curves are the pointwise mean of the 1000 split curves "
        f"(interpolated to a common grid).\n"
        f"- Difference Δ / p: {note}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"Saved {path}")
    print(body)


def run_pair(
    *,
    name: str,
    section_auroc: str,
    section_auprc: str,
    clin_key: str,
    full_key: str,
    clin_label: str,
    full_label: str,
    roc_clin: Path,
    roc_full: Path,
    pr_clin: Path,
    pr_full: Path,
    summ_csv: Path,
    out_roc: list[Path],
    out_pr: list[Path],
    caption_auroc: Path,
    caption_auprc: Path,
    prevalence: float,
    delong_row: pd.Series,
    auprc_row: pd.Series,
) -> None:
    print("=" * 72)
    print(name)
    summ = pd.read_csv(summ_csv)
    m_clin = row_metrics(summ, clin_key)
    m_full = row_metrics(summ, full_key)

    fpr_grid = np.linspace(0.0, 1.0, N_GRID)
    print("  interpolating ROC...")
    mat_roc_c = interp_roc_matrix(roc_clin, fpr_grid)
    mat_roc_f = interp_roc_matrix(roc_full, fpr_grid)
    plot_roc_overlay(
        mat_roc_c,
        mat_roc_f,
        fpr_grid,
        f"{clin_label}\nAUROC = {m_clin['auc']:.3f} "
        f"[{m_clin['auc_lo']:.3f}, {m_clin['auc_hi']:.3f}]",
        f"{full_label}\nAUROC = {m_full['auc']:.3f} "
        f"[{m_full['auc_lo']:.3f}, {m_full['auc_hi']:.3f}]",
        out_roc,
    )

    recall_grid = np.linspace(0.0, 1.0, N_GRID)
    print("  interpolating PR...")
    # If both paths point at a multi-model CSV, filter by model key first
    pr_c_path, pr_f_path, tmp_paths = _maybe_split_pr_csv(
        pr_clin, pr_full, clin_key, full_key,
    )
    try:
        mat_pr_c = interp_pr_matrix(pr_c_path, recall_grid)
        mat_pr_f = interp_pr_matrix(pr_f_path, recall_grid)
        plot_pr_overlay(
            mat_pr_c,
            mat_pr_f,
            recall_grid,
            f"{clin_label}\nAUPRC = {m_clin['auprc']:.3f} "
            f"[{m_clin['auprc_lo']:.3f}, {m_clin['auprc_hi']:.3f}]",
            f"{full_label}\nAUPRC = {m_full['auprc']:.3f} "
            f"[{m_full['auprc_lo']:.3f}, {m_full['auprc_hi']:.3f}]",
            prevalence,
            out_pr,
        )
    finally:
        for t in tmp_paths:
            t.unlink(missing_ok=True)

    write_caption(
        caption_auroc,
        section_auroc,
        "AUROC",
        clin_label,
        full_label,
        m_clin,
        m_full,
        float(delong_row["delta_full_minus_clinical"]),
        float(delong_row["delong_ci_lo"]),
        float(delong_row["delong_ci_hi"]),
        float(delong_row["delong_p"]),
        "bagged out-of-sample paired DeLong test (full − clinical).",
    )
    write_caption(
        caption_auprc,
        section_auprc,
        "AUPRC",
        clin_label,
        full_label,
        m_clin,
        m_full,
        float(auprc_row["delta_full_minus_clinical"]),
        float(auprc_row["boot_ci_lo"]),
        float(auprc_row["boot_ci_hi"]),
        float(auprc_row["boot_p"]),
        "bagged out-of-sample paired bootstrap of AUPRC (full − clinical).",
    )


def main() -> None:
    by_type = ML_DIR / "outputs" / "by_type" / "aim2_clinical_vs_full_youden_lightSpike"
    aim1 = ML_DIR / "outputs" / "Combined" / "aim1_innerCV_pick_1000mc_lightSpike"
    delong = pd.read_csv(
        ML_DIR
        / "outputs"
        / "by_type"
        / "oob_paired_delong_clinical_vs_full_lightSpike"
        / "oob_paired_delong_clinical_vs_full.csv"
    )
    auprc_boot = pd.read_csv(
        ML_DIR
        / "outputs"
        / "by_type"
        / "oob_paired_bootstrap_auprc_clinical_vs_full_lightSpike"
        / "oob_paired_bootstrap_auprc_clinical_vs_full.csv"
    )

    # --- Aim2 Generalized ---
    gen = by_type / "General"
    d_gen = delong[delong["cohort"] == "Generalized"].iloc[0]
    a_gen = auprc_boot[auprc_boot["cohort"] == "Generalized"].iloc[0]
    # prevalence from RESULTS or bagged preds
    bag_g = pd.read_csv(gen / "bagged_oos_predictions.csv")
    prev_g = float(bag_g["y_true"].mean())
    run_pair(
        name="Aim2 Generalized",
        section_auroc="C. Generalized Epilepsy Model AUROC Metrics",
        section_auprc="C. Generalized Epilepsy Model AUPRC Metrics",
        clin_key="Model1_clinical",
        full_key="Model4_IED_slow",
        clin_label="Clinical Model",
        full_label="Full Model",
        roc_clin=gen / "mc_roc_curves_Model1_clinical.csv",
        roc_full=gen / "mc_roc_curves_Model4_IED_slow.csv",
        pr_clin=gen / "mc_pr_curves.csv",  # filtered below if multi-model
        pr_full=gen / "mc_pr_curves.csv",
        summ_csv=gen / "mc_auc_auprc_summary.csv",
        out_roc=[
            gen / "FigAim2_gen_roc_clinical_vs_full_lightSpike.png",
            PAPER_OUT / "FigAim2_gen_roc_clinical_vs_full_lightSpike.png",
        ],
        out_pr=[
            gen / "FigAim2_gen_pr_clinical_vs_full_lightSpike.png",
            PAPER_OUT / "FigAim2_gen_pr_clinical_vs_full_lightSpike.png",
        ],
        caption_auroc=PAPER_OUT / "caption_Aim2_gen_AUROC_clinical_vs_full.txt",
        caption_auprc=PAPER_OUT / "caption_Aim2_gen_AUPRC_clinical_vs_full.txt",
        prevalence=prev_g,
        delong_row=d_gen,
        auprc_row=a_gen,
    )
    # --- Aim2 Focal ---
    foc = by_type / "Focal"
    d_foc = delong[delong["cohort"] == "Focal"].iloc[0]
    a_foc = auprc_boot[auprc_boot["cohort"] == "Focal"].iloc[0]
    bag_f = pd.read_csv(foc / "bagged_oos_predictions.csv")
    prev_f = float(bag_f["y_true"].mean())
    run_pair(
        name="Aim2 Focal",
        section_auroc="D. Focal Epilepsy Model AUROC Metrics",
        section_auprc="D. Focal Epilepsy Model AUPRC Metrics",
        clin_key="Model1_clinical",
        full_key="Model4_IED_slow",
        clin_label="Clinical Model",
        full_label="Full Model",
        roc_clin=foc / "mc_roc_curves_Model1_clinical.csv",
        roc_full=foc / "mc_roc_curves_Model4_IED_slow.csv",
        pr_clin=foc / "mc_pr_curves.csv",
        pr_full=foc / "mc_pr_curves.csv",
        summ_csv=foc / "mc_auc_auprc_summary.csv",
        out_roc=[
            foc / "FigAim2_focal_roc_clinical_vs_full_lightSpike.png",
            PAPER_OUT / "FigAim2_focal_roc_clinical_vs_full_lightSpike.png",
        ],
        out_pr=[
            foc / "FigAim2_focal_pr_clinical_vs_full_lightSpike.png",
            PAPER_OUT / "FigAim2_focal_pr_clinical_vs_full_lightSpike.png",
        ],
        caption_auroc=PAPER_OUT / "caption_Aim2_focal_AUROC_clinical_vs_full.txt",
        caption_auprc=PAPER_OUT / "caption_Aim2_focal_AUPRC_clinical_vs_full.txt",
        prevalence=prev_f,
        delong_row=d_foc,
        auprc_row=a_foc,
    )
    # --- Aim1 pooled ---
    d_p = delong[delong["cohort"] == "Pooled (Interactions Aim1)"].iloc[0]
    a_p = auprc_boot[auprc_boot["cohort"] == "Pooled (Interactions Aim1)"].iloc[0]
    bag_p = pd.read_csv(aim1 / "bagged_oos_predictions.csv")
    prev_p = float(bag_p["y_true"].mean())
    # Prefer bootstrap-of-mean CI summary (narrow), not percentile-of-MC table
    summ_aim1 = aim1 / "mc_auc_auprc_summary.csv"
    run_pair(
        name="Aim1 Pooled Interactions",
        section_auroc="B. Pooled (Focal + Generalized) AUROC Metrics",
        section_auprc="B. Pooled (Focal + Generalized) AUPRC Metrics",
        clin_key="Model_clinical",
        full_key="Model_interaction",
        clin_label="Clinical Model",
        full_label="Interactions Model",
        roc_clin=aim1 / "mc_roc_curves_Model_clinical.csv",
        roc_full=aim1 / "mc_roc_curves_Model_interaction.csv",
        pr_clin=aim1 / "mc_pr_curves_Model_clinical.csv",
        pr_full=aim1 / "mc_pr_curves_Model_interaction.csv",
        summ_csv=summ_aim1,
        out_roc=[
            aim1 / "FigAim1_roc_clinical_vs_interaction_lightSpike.png",
            PAPER_OUT / "FigAim1_roc_clinical_vs_interaction_lightSpike.png",
        ],
        out_pr=[
            aim1 / "FigAim1_pr_clinical_vs_interaction_lightSpike.png",
            PAPER_OUT / "FigAim1_pr_clinical_vs_interaction_lightSpike.png",
        ],
        caption_auroc=PAPER_OUT / "caption_Aim1_pooled_AUROC_clinical_vs_interaction.txt",
        caption_auprc=PAPER_OUT / "caption_Aim1_pooled_AUPRC_clinical_vs_interaction.txt",
        prevalence=prev_p,
        delong_row=d_p,
        auprc_row=a_p,
    )

    # Update Aim2 RESULTS_SUMMARY text to clarify MC vs bagged test
    _rewrite_aim2_results(gen, foc, d_gen, d_foc, a_gen, a_foc)
    print("DONE")


def _maybe_split_pr_csv(
    pr_clin: Path,
    pr_full: Path,
    clin_key: str,
    full_key: str,
) -> tuple[Path, Path, list[Path]]:
    """If a shared multi-model PR CSV is passed twice, write filtered temps."""
    if pr_clin.resolve() != pr_full.resolve():
        return pr_clin, pr_full, []
    head = pd.read_csv(pr_clin, nrows=5)
    if "model" not in head.columns:
        return pr_clin, pr_full, []
    full = pd.read_csv(pr_clin)
    tmp_c = pr_clin.with_name(f"_tmp_pr_{clin_key}.csv")
    tmp_f = pr_clin.with_name(f"_tmp_pr_{full_key}.csv")
    full[full["model"] == clin_key].to_csv(tmp_c, index=False)
    full[full["model"] == full_key].to_csv(tmp_f, index=False)
    return tmp_c, tmp_f, [tmp_c, tmp_f]


def _rewrite_aim2_results(gen: Path, foc: Path, d_gen, d_foc, a_gen, a_foc) -> None:
    for path, drow, arow, summ_csv in [
        (gen / "RESULTS_SUMMARY.txt", d_gen, a_gen, gen / "mc_auc_auprc_summary.csv"),
        (foc / "RESULTS_SUMMARY.txt", d_foc, a_foc, foc / "mc_auc_auprc_summary.csv"),
    ]:
        old = path.read_text().splitlines()
        header = []
        for ln in old:
            if ln.startswith("MC mean") or ln.startswith("Pairwise"):
                break
            header.append(ln)
        summ = pd.read_csv(summ_csv)
        mc_lines = ["MC mean AUROC / AUPRC [bootstrap 95% CI of the mean]:"]
        for _, r in summ.iterrows():
            mc_lines.append(
                f"  {r['model']}: AUROC={r['auc_mean']:.3f} "
                f"[{r['auc_ci_lo']:.3f}, {r['auc_ci_hi']:.3f}]  "
                f"AUPRC={r['auprc_mean']:.3f} "
                f"[{r['auprc_ci_lo']:.3f}, {r['auprc_ci_hi']:.3f}]"
            )
        # keep original MC sign-test line if present
        sign = [ln for ln in old if "wins" in ln]
        out = header + [""] + mc_lines + [""]
        out.append("MC sign test (same-seed AUROC; descriptive only):")
        out.extend(sign if sign else ["  (see pairwise_model_bootstrap.csv)"])
        out.append("")
        out.append(
            "Primary difference test (bagged OOS; use for Δ / p in figure captions):"
        )
        out.append(
            f"  AUROC DeLong Δ(full−clinical)="
            f"{_fmt_delta(float(drow['delta_full_minus_clinical']), float(drow['delong_ci_lo']), float(drow['delong_ci_hi']))}  "
            f"{_fmt_p(float(drow['delong_p']))}"
        )
        out.append(
            f"  AUPRC bootstrap Δ(full−clinical)="
            f"{_fmt_delta(float(arow['delta_full_minus_clinical']), float(arow['boot_ci_lo']), float(arow['boot_ci_hi']))}  "
            f"{_fmt_p(float(arow['boot_p']))}"
        )
        out.append("")
        out.append(
            "Reporting rule: AUROC/AUPRC point estimates + CIs = MC mean over 1000 "
            "permutations; curve plots = MC-mean ROC/PR; Δ/p = bagged OOS tests above."
        )
        path.write_text("\n".join(out) + "\n")
        print(f"Updated {path}")


if __name__ == "__main__":
    main()
