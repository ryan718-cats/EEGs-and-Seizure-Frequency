#!/usr/bin/env python3
"""Average confusion matrices across 1000 MC splits @ threshold 0.5.

1) Aim1 Model_interaction (locked best clf from lightSpike run)
2) Aim2 Model4_IED_slow full Youden (Generalized only; locked best clf)

Uses same patient-grouped 80/20 seeds and GridSearch fit as the paper pipelines.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))

import aim1_pooled_ml as a1  # noqa: E402
import aim2_generalized_ml as a2  # noqa: E402
import generalized_ml as gml  # noqa: E402
import statistical_tests as st  # noqa: E402

PAPER_OUT = ML_DIR.parent / "output"
AIM1_OUT = (
    ML_DIR
    / "outputs"
    / "Combined"
    / "aim1_model5_interaction_poor1.0mo_matched_prior_youden_grid_lightSpike"
)
AIM2_OUT = (
    ML_DIR
    / "outputs"
    / "General"
    / "aim2_generalized_poor1mo_youden_model4_cont_compare_lightSpike"
)

THRESH = 0.5
N_MC = 1000


def _cm_one_split(
    seed: int,
    analysis_df: pd.DataFrame,
    cols: list[str],
    y_col: str,
    clf_key: str,
    test_size: float,
    n_cv: int,
) -> np.ndarray | None:
    y_all = analysis_df[y_col].values.astype(int)
    tr_idx, te_idx = gml._patient_grouped_split(analysis_df, y_all, test_size, seed)
    y_tr = y_all[tr_idx]
    y_te = y_all[te_idx]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None
    train_df = analysis_df.iloc[tr_idx]
    test_df = analysis_df.iloc[te_idx]
    X_tr = gml.prepare_X(train_df, cols, ref_df=train_df)
    X_te = gml.prepare_X(test_df, cols, ref_df=train_df)
    try:
        est, _ = gml.fit_clf(clf_key, X_tr, y_tr, seed, n_cv, class_weight="none")
        p = est.predict_proba(X_te)[:, 1]
    except Exception:
        return None
    if not np.isfinite(p).all():
        return None
    y_pred = (p >= THRESH).astype(int)
    # labels [0,1] → [[TN, FP], [FN, TP]]
    return confusion_matrix(y_te, y_pred, labels=[0, 1]).astype(float)


def average_cms(
    analysis_df: pd.DataFrame,
    cols: list[str],
    y_col: str,
    clf_key: str,
    n_mc: int,
    n_jobs: int,
    test_size: float,
    n_cv: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    cms = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_cm_one_split)(
            seed, analysis_df, cols, y_col, clf_key, test_size, n_cv,
        )
        for seed in range(n_mc)
    )
    valid = [c for c in cms if c is not None]
    stack = np.stack(valid, axis=0)
    return stack.mean(axis=0), stack.std(axis=0, ddof=1), len(valid)


def plot_cm(
    mean_cm: np.ndarray,
    std_cm: np.ndarray,
    title: str,
    out_png: Path,
    paper_png: Path,
    n_valid: int,
    clf_key: str,
) -> None:
    # Also row-normalize rates for annotation context
    row_sums = mean_cm.sum(axis=1, keepdims=True)
    rate = np.divide(mean_cm, row_sums, out=np.zeros_like(mean_cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(mean_cm, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Moderate control", "Poor control"], fontsize=12)
    ax.set_yticklabels(["Moderate control", "Poor control"], fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)

    # Annotate mean ± sd and row %
    for i in range(2):
        for j in range(2):
            txt = f"{mean_cm[i, j]:.1f}\n±{std_cm[i, j]:.1f}\n({100 * rate[i, j]:.1f}%)"
            color = "white" if mean_cm[i, j] > mean_cm.max() * 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean count across MC test splits", fontsize=11)
    fig.tight_layout()
    paper_png.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paper_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {paper_png}")


def save_summary(
    path: Path,
    mean_cm: np.ndarray,
    std_cm: np.ndarray,
    n_valid: int,
    model: str,
    clf_key: str,
    n_patients: int,
) -> None:
    tn, fp, fn, tp = mean_cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    acc = (tp + tn) / mean_cm.sum() if mean_cm.sum() else np.nan
    rows = [{
        "model": model,
        "clf": clf_key,
        "threshold": THRESH,
        "n_mc_valid": n_valid,
        "n_patients": n_patients,
        "mean_TN": tn, "mean_FP": fp, "mean_FN": fn, "mean_TP": tp,
        "sd_TN": std_cm[0, 0], "sd_FP": std_cm[0, 1],
        "sd_FN": std_cm[1, 0], "sd_TP": std_cm[1, 1],
        "sens_from_mean": sens, "spec_from_mean": spec,
        "ppv_from_mean": ppv, "npv_from_mean": npv,
        "acc_from_mean": acc,
    }]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {path}")
    print(
        f"  mean CM [[TN={tn:.1f}, FP={fp:.1f}], [FN={fn:.1f}, TP={tp:.1f}]]  "
        f"sens={sens:.3f} spec={spec:.3f} acc={acc:.3f}"
    )


def run_aim1(n_jobs: int, n_mc: int) -> None:
    print("=" * 72)
    print("Aim1 Interaction — average CM across MC splits")
    sel = {}
    for line in (AIM1_OUT / "selected_algorithms_per_model.txt").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            sel[k.strip()] = v.strip()
    clf_key = sel["Model_interaction"]
    print(f"Locked clf: {clf_key}")

    tables = st._load_combined_data()
    analysis_df, clinical_cols, _, _ = a1.build_analysis_df(
        tables["sz_freq_class"],
        binary_markers=True,
        poor_cutoff=1.0,
        ternary_outcome=False,
    )
    cols = a1._build_feature_sets(clinical_cols)["Model_interaction"]
    print(f"N={len(analysis_df)}  features={cols}")

    mean_cm, std_cm, n_valid = average_cms(
        analysis_df, cols, "outcome", clf_key, n_mc, n_jobs,
        a1.TEST_SIZE, a1.N_CV_FOLDS,
    )
    save_summary(
        AIM1_OUT / "mc_mean_confusion_matrix_Model_interaction.csv",
        mean_cm, std_cm, n_valid, "Model_interaction", clf_key, len(analysis_df),
    )
    # also save raw matrix
    pd.DataFrame(
        mean_cm,
        index=["true_not_poor", "true_poor"],
        columns=["pred_not_poor", "pred_poor"],
    ).to_csv(AIM1_OUT / "mc_mean_confusion_matrix_Model_interaction_counts.csv")
    plot_cm(
        mean_cm, std_cm,
        "Interactions model — mean confusion matrix",
        AIM1_OUT / "mc_mean_confusion_matrix_Model_interaction.png",
        PAPER_OUT / "FigAim1_mc_mean_confusion_matrix_interaction_lightSpike.png",
        n_valid, clf_key,
    )


def run_aim2(n_jobs: int, n_mc: int) -> None:
    print("=" * 72)
    print("Aim2 Model4 Youden (Generalized) — average CM across MC splits")
    sel = {}
    for line in (AIM2_OUT / "selected_algorithms_per_model.txt").read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            sel[parts[0]] = parts[1]
    clf_key = sel["Model4_IED_slow"]
    print(f"Locked clf: {clf_key}")

    tables = st._load_combined_data()
    analysis_df = a2.build_analysis_df(
        tables["sz_freq_class"],
        poor_cutoff=1.0,
        binary_markers=True,
    )
    cols = a2.build_feature_sets(binary_markers=True)["Model4_IED_slow"]
    print(f"N={len(analysis_df)}  features={cols}")

    mean_cm, std_cm, n_valid = average_cms(
        analysis_df, cols, "sz_freq_class", clf_key, n_mc, n_jobs,
        a2.TEST_SIZE, a2.N_CV_FOLDS,
    )
    save_summary(
        AIM2_OUT / "mc_mean_confusion_matrix_Model4_IED_slow.csv",
        mean_cm, std_cm, n_valid, "Model4_IED_slow", clf_key, len(analysis_df),
    )
    pd.DataFrame(
        mean_cm,
        index=["true_not_poor", "true_poor"],
        columns=["pred_not_poor", "pred_poor"],
    ).to_csv(AIM2_OUT / "mc_mean_confusion_matrix_Model4_IED_slow_counts.csv")
    plot_cm(
        mean_cm, std_cm,
        "Full model (thresholded) — Generalized — mean confusion matrix",
        AIM2_OUT / "mc_mean_confusion_matrix_Model4_IED_slow.png",
        PAPER_OUT / "FigAim2_mc_mean_confusion_matrix_model4_full_lightSpike.png",
        n_valid, clf_key,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-mc", type=int, default=N_MC)
    p.add_argument("--n-jobs", type=int, default=int(os.environ.get("ML_TOTAL_CORES", 6)))
    p.add_argument("--aim1-only", action="store_true")
    p.add_argument("--aim2-only", action="store_true")
    args = p.parse_args()
    do_aim1 = not args.aim2_only
    do_aim2 = not args.aim1_only
    if do_aim1:
        run_aim1(args.n_jobs, args.n_mc)
    if do_aim2:
        run_aim2(args.n_jobs, args.n_mc)
    print("DONE")


if __name__ == "__main__":
    main()
