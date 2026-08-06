#!/usr/bin/env python3
"""Interactions model: auto Youden markers vs clinician report (test-time swap).

Train every MC split on automatic Youden IED/slowing (usual Interactions model).
For patients with usable report IED AND report_SLOWING (present/absent), also
predict with report-coded spikes/slowing (+ type× interactions) — no retrain.

Compares bagged OOS AUROC on that subset + Wilcoxon signed-rank on paired
per-split AUROCs (auto vs report).

Usage:
  python aim1_report_vs_auto_markers_oob.py
  python aim1_report_vs_auto_markers_oob.py --n-mc 1000 --n-jobs 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

import aim1_pooled_ml as a1
import aim2_generalized_ml as a2
import generalized_ml as gml

HERE = Path(__file__).resolve().parent
COHORT = (
    HERE
    / "outputs"
    / "Combined"
    / "aim1_model5_interaction_poor1.0mo_matched_prior_youden_grid_lightSpike"
    / "analysis_cohort.csv"
)
PAPER = HERE.parent / "output"
OUT_DIR = (
    HERE
    / "outputs"
    / "Combined"
    / "aim1_report_vs_auto_markers_oob_interaction_lightSpike"
)

MODEL_KEY = "Model_interaction"
CLF_KEY = "lr"  # locked from Aim1 lightSpike run
FEATURE_COLS = [
    "age_at_eeg",
    "gender",
    "epilepsy_type_focal",
    "prior_sz_poor",
    "prior_sz_unavailable",
    "spikes",
    "slowing",
    "type_x_spikes",
    "type_x_slowing",
]


def _norm_pid(s: pd.Series) -> pd.Series:
    return a2._norm_id(s).astype(str)


def load_patient_slowing_annotation(analysis_pids: pd.Series) -> pd.DataFrame:
    """Patient-level report_SLOWING present/absent/unknown (same rules as IED)."""
    rep = pd.read_csv(a2.DEID_REPORT_CSV, low_memory=False)
    rep = a2._filter_outpatient_routine_report(rep)
    rep = rep.copy()
    rep["patient_id"] = _norm_pid(rep["patient_id"])
    keep = set(_norm_pid(analysis_pids).dropna().astype(str))
    rep = rep[rep["patient_id"].isin(keep)].copy()
    s = rep["report_SLOWING"].astype(str).str.strip().str.lower()
    status = np.full(len(rep), "unknown", dtype=object)
    status[s.eq("present").to_numpy()] = "present"
    status[s.eq("absent").to_numpy()] = "absent"
    sess = pd.DataFrame({"patient_id": rep["patient_id"], "report_status": status})
    agg = sess.groupby("patient_id", sort=False)["report_status"].agg(
        has_present=lambda x: bool((x == "present").any()),
        has_absent=lambda x: bool((x == "absent").any()),
    )
    out_status = pd.Series("unknown", index=agg.index, dtype=object)
    out_status[agg["has_absent"] & ~agg["has_present"]] = "absent"
    out_status[agg["has_present"]] = "present"
    out = out_status.rename("slowing_annotation").reset_index()
    all_pids = pd.DataFrame({"patient_id": sorted(keep)})
    out = all_pids.merge(out, on="patient_id", how="left")
    out["slowing_annotation"] = out["slowing_annotation"].fillna("unknown")
    return out


def prepare_analysis_df(cohort_csv: Path) -> tuple[pd.DataFrame, pd.Index]:
    df = pd.read_csv(cohort_csv)
    df["patient_id"] = _norm_pid(df["patient_id"])
    df["gender"] = df["gender"].astype(float)
    df["outcome"] = df["outcome"].astype(int)
    df["epilepsy_type_focal"] = (df["epilepsy_type"] == "Focal").astype(float)
    df["spikes"] = df["spikes"].astype(float)
    df["slowing"] = df["slowing"].astype(float)
    df["prior_sz_poor"] = df["prior_sz_poor"].astype(float)
    df["prior_sz_unavailable"] = df["prior_sz_unavailable"].astype(float)
    df["type_x_spikes"] = df["epilepsy_type_focal"] * df["spikes"]
    df["type_x_slowing"] = df["epilepsy_type_focal"] * df["slowing"]

    ied = a2.load_patient_ied_annotation(df["patient_id"])
    ied["patient_id"] = _norm_pid(ied["patient_id"])
    slow = load_patient_slowing_annotation(df["patient_id"])
    slow["patient_id"] = _norm_pid(slow["patient_id"])
    df = df.merge(ied[["patient_id", "ied_annotation"]], on="patient_id", how="left")
    df = df.merge(slow[["patient_id", "slowing_annotation"]], on="patient_id", how="left")

    has_both = (
        df["ied_annotation"].isin(["present", "absent"])
        & df["slowing_annotation"].isin(["present", "absent"])
    )
    df["spikes_report"] = np.where(
        df["ied_annotation"] == "present",
        1.0,
        np.where(df["ied_annotation"] == "absent", 0.0, np.nan),
    )
    df["slowing_report"] = np.where(
        df["slowing_annotation"] == "present",
        1.0,
        np.where(df["slowing_annotation"] == "absent", 0.0, np.nan),
    )
    df["has_both_reports"] = has_both
    df["type_x_spikes_report"] = df["epilepsy_type_focal"] * df["spikes_report"]
    df["type_x_slowing_report"] = df["epilepsy_type_focal"] * df["slowing_report"]

    n_both = int(has_both.sum())
    print(
        f"[Reports] patients with usable IED AND slowing annotations: {n_both}/{len(df)}"
    )
    print(f"  IED among them: {df.loc[has_both, 'ied_annotation'].value_counts().to_dict()}")
    print(
        f"  Slowing among them: "
        f"{df.loc[has_both, 'slowing_annotation'].value_counts().to_dict()}"
    )
    agree_ied = (
        df.loc[has_both, "spikes"].astype(int)
        == (df.loc[has_both, "ied_annotation"] == "present").astype(int)
    ).mean()
    agree_sl = (
        df.loc[has_both, "slowing"].astype(int)
        == (df.loc[has_both, "slowing_annotation"] == "present").astype(int)
    ).mean()
    print(f"  Agreement auto vs report: IED={agree_ied:.3f}, slowing={agree_sl:.3f}")
    return df.reset_index(drop=True), df.index[has_both]


def _swap_report_features(test_df: pd.DataFrame) -> pd.DataFrame:
    """Copy of test_df with report markers for patients who have both reports."""
    out = test_df.copy()
    m = out["has_both_reports"].to_numpy()
    out.loc[m, "spikes"] = out.loc[m, "spikes_report"].to_numpy()
    out.loc[m, "slowing"] = out.loc[m, "slowing_report"].to_numpy()
    out.loc[m, "type_x_spikes"] = out.loc[m, "type_x_spikes_report"].to_numpy()
    out.loc[m, "type_x_slowing"] = out.loc[m, "type_x_slowing_report"].to_numpy()
    return out


def run_one_seed(seed: int, analysis_df: pd.DataFrame) -> dict | None:
    y_all = analysis_df["outcome"].to_numpy(dtype=int)
    tr_idx, te_idx = gml._patient_grouped_split(
        analysis_df, y_all, a1.TEST_SIZE, seed
    )
    y_tr = y_all[tr_idx]
    y_te = y_all[te_idx]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None

    train_df = analysis_df.iloc[tr_idx]
    test_df = analysis_df.iloc[te_idx]
    test_report = _swap_report_features(test_df)

    X_tr = gml.prepare_X(train_df, FEATURE_COLS, ref_df=train_df)
    X_te_auto = gml.prepare_X(test_df, FEATURE_COLS, ref_df=train_df)
    X_te_rep = gml.prepare_X(test_report, FEATURE_COLS, ref_df=train_df)
    try:
        est = gml.fit_clf_default(CLF_KEY, X_tr, y_tr, class_weight="none")
        p_auto = est.predict_proba(X_te_auto)[:, 1]
        p_rep = est.predict_proba(X_te_rep)[:, 1]
    except Exception:
        return None

    both_mask = test_df["has_both_reports"].to_numpy()
    y_both = y_te[both_mask]
    p_auto_both = p_auto[both_mask]
    p_rep_both = p_rep[both_mask]

    def _auc(y, p):
        y = np.asarray(y)
        p = np.asarray(p)
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        if len(y) < 5 or len(np.unique(y)) < 2:
            return np.nan
        return float(roc_auc_score(y, p))

    return {
        "seed": seed,
        "test_pids": test_df["patient_id"].astype(str).to_numpy(),
        "y_test": y_te.astype(int),
        "p_auto": np.asarray(p_auto, dtype=float),
        "p_report": np.asarray(p_rep, dtype=float),
        "both_mask": both_mask,
        "auc_auto_subset": _auc(y_both, p_auto_both),
        "auc_report_subset": _auc(y_both, p_rep_both),
        "n_both_in_test": int(both_mask.sum()),
        "n_both_poor": int(y_both.sum()) if len(y_both) else 0,
    }


def bag_from_results(analysis_df: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    pids = analysis_df["patient_id"].astype(str).to_numpy()
    y = analysis_df["outcome"].to_numpy(dtype=int)
    both = analysis_df["has_both_reports"].to_numpy()
    sums_auto = {pid: 0.0 for pid in pids}
    sums_rep = {pid: 0.0 for pid in pids}
    counts = {pid: 0 for pid in pids}
    for res in results:
        for pid, ya, yr in zip(res["test_pids"], res["p_auto"], res["p_report"]):
            if not (np.isfinite(ya) and np.isfinite(yr)):
                continue
            sums_auto[pid] += float(ya)
            sums_rep[pid] += float(yr)
            counts[pid] += 1
    rows = []
    for i, pid in enumerate(pids):
        n = counts[pid]
        rows.append(
            {
                "patient_id": pid,
                "y_true": int(y[i]),
                "has_both_reports": bool(both[i]),
                "n_heldout": n,
                "p_auto": sums_auto[pid] / n if n else np.nan,
                "p_report": sums_rep[pid] / n if n else np.nan,
                "ied_annotation": analysis_df.iloc[i]["ied_annotation"],
                "slowing_annotation": analysis_df.iloc[i]["slowing_annotation"],
                "spikes_auto": float(analysis_df.iloc[i]["spikes"]),
                "slowing_auto": float(analysis_df.iloc[i]["slowing"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mc", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--cohort-csv", type=Path, default=COHORT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    analysis_df, _ = prepare_analysis_df(args.cohort_csv)
    n_both = int(analysis_df["has_both_reports"].sum())
    if n_both < 20:
        raise RuntimeError(f"Too few dual-report patients ({n_both})")

    print(
        f"\nRe-running {args.n_mc} MC splits: train=auto Youden, "
        f"predict=auto AND report-swap for dual-report patients"
    )
    print(f"Model={MODEL_KEY}  clf={CLF_KEY}  n_jobs={args.n_jobs}")

    seeds = range(args.seed_start, args.seed_start + args.n_mc)
    raw = Parallel(n_jobs=args.n_jobs)(
        delayed(run_one_seed)(seed, analysis_df)
        for seed in tqdm(list(seeds), desc="MC report-vs-auto", unit="split")
    )
    results = [r for r in raw if r is not None]
    print(f"Kept {len(results)}/{args.n_mc} successful splits")

    bag = bag_from_results(analysis_df, results)
    bag.to_csv(out_dir / "bagged_oos_auto_vs_report.csv", index=False)

    sub = bag[bag["has_both_reports"]].copy()
    y = sub["y_true"].to_numpy()
    p_auto = sub["p_auto"].to_numpy()
    p_rep = sub["p_report"].to_numpy()

    def _metrics(y, p, label):
        ok = np.isfinite(y) & np.isfinite(p)
        y2, p2 = y[ok].astype(int), p[ok].astype(float)
        auc = float(roc_auc_score(y2, p2))
        ap = float(average_precision_score(y2, p2))
        d = a1.bagged_auc_delong_stats(y2, p2)
        return {
            "features": label,
            "n_patients": int(ok.sum()),
            "n_poor": int(y2.sum()),
            "prevalence": float(y2.mean()),
            "bagged_auroc": auc,
            "bagged_auprc": ap,
            "delong_se": d["delong_se"],
            "delong_ci_lo": d["delong_ci_lo"],
            "delong_ci_hi": d["delong_ci_hi"],
        }

    rows = [
        _metrics(y, p_auto, "automatic_Youden"),
        _metrics(y, p_rep, "clinician_report"),
    ]
    summary = pd.DataFrame(rows)

    # Paired DeLong on bagged preds (same patients)
    cmp_bag = a2.unpaired_delong_compare(y, p_auto, y, p_rep)  # wrong — need paired
    # Use aim1 paired bootstrap / or sklearn style: difference via DeLong paired if available
    # Prefer MC signed-rank as primary; also report bagged ΔAUC
    delta_bag = float(summary.loc[0, "bagged_auroc"] - summary.loc[1, "bagged_auroc"])

    # Per-MC AUCs on dual-report subset
    mc_rows = []
    for r in results:
        mc_rows.append(
            {
                "seed": r["seed"],
                "n_both_in_test": r["n_both_in_test"],
                "n_both_poor": r["n_both_poor"],
                "auc_auto": r["auc_auto_subset"],
                "auc_report": r["auc_report_subset"],
                "delta_auto_minus_report": (
                    r["auc_auto_subset"] - r["auc_report_subset"]
                    if np.isfinite(r["auc_auto_subset"])
                    and np.isfinite(r["auc_report_subset"])
                    else np.nan
                ),
            }
        )
    mc_df = pd.DataFrame(mc_rows)
    mc_df.to_csv(out_dir / "mc_auc_auto_vs_report_dual_report_subset.csv", index=False)

    paired = mc_df.dropna(subset=["auc_auto", "auc_report"])
    diffs = paired["delta_auto_minus_report"].to_numpy()
    if len(diffs) >= 10:
        # Wilcoxon signed-rank: H0 median(auto - report) = 0
        # alternative two-sided; also report one-sided auto > report
        w_two = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        w_greater = stats.wilcoxon(diffs, zero_method="wilcox", alternative="greater")
        sign_pos = int((diffs > 0).sum())
        sign_neg = int((diffs < 0).sum())
        sign_zero = int((diffs == 0).sum())
        # simple sign test p
        sign_p = float(stats.binomtest(sign_pos, sign_pos + sign_neg, 0.5).pvalue)
    else:
        w_two = w_greater = None
        sign_pos = sign_neg = sign_zero = 0
        sign_p = np.nan

    # Patient-level paired predicted-prob difference among dual-report (for signed-rank alt)
    d_prob = p_auto - p_rep
    okp = np.isfinite(d_prob)
    if okp.sum() >= 10 and np.any(d_prob[okp] != 0):
        w_prob = stats.wilcoxon(
            d_prob[okp], zero_method="wilcox", alternative="two-sided"
        )
    else:
        w_prob = None

    cmp_row = {
        "n_dual_report_patients": n_both,
        "n_mc_kept": len(results),
        "n_mc_paired_auc": int(len(paired)),
        "bagged_auroc_auto": float(summary.loc[0, "bagged_auroc"]),
        "bagged_auroc_report": float(summary.loc[1, "bagged_auroc"]),
        "bagged_delta_auto_minus_report": delta_bag,
        "bagged_auprc_auto": float(summary.loc[0, "bagged_auprc"]),
        "bagged_auprc_report": float(summary.loc[1, "bagged_auprc"]),
        "mc_mean_auc_auto": float(paired["auc_auto"].mean()) if len(paired) else np.nan,
        "mc_mean_auc_report": float(paired["auc_report"].mean()) if len(paired) else np.nan,
        "mc_mean_delta": float(np.mean(diffs)) if len(diffs) else np.nan,
        "wilcoxon_auc_stat": float(w_two.statistic) if w_two else np.nan,
        "wilcoxon_auc_p_two_sided": float(w_two.pvalue) if w_two else np.nan,
        "wilcoxon_auc_p_auto_greater": float(w_greater.pvalue) if w_greater else np.nan,
        "sign_test_n_auto_higher": sign_pos,
        "sign_test_n_report_higher": sign_neg,
        "sign_test_n_tie": sign_zero,
        "sign_test_p": sign_p,
        "wilcoxon_prob_stat": float(w_prob.statistic) if w_prob else np.nan,
        "wilcoxon_prob_p_two_sided": float(w_prob.pvalue) if w_prob else np.nan,
    }
    cmp_df = pd.DataFrame([cmp_row])
    summary.to_csv(out_dir / "bagged_metrics_dual_report_subset.csv", index=False)
    cmp_df.to_csv(out_dir / "auto_vs_report_comparison.csv", index=False)
    summary.to_csv(PAPER / "aim1_report_vs_auto_bagged_metrics_lightSpike.csv", index=False)
    cmp_df.to_csv(PAPER / "aim1_report_vs_auto_comparison_lightSpike.csv", index=False)

    lines = [
        "Aim1 Interactions — automatic Youden vs clinician report markers (OOB)",
        "=" * 72,
        "Design: train each MC split on auto Youden IED/slowing; at test time,",
        "  dual-report patients also scored with report present/absent swapped in",
        "  for spikes/slowing (+ type× interactions). No retrain.",
        f"Cohort: N={len(analysis_df)}; dual-report subset N={n_both}",
        f"Model: {MODEL_KEY} / {CLF_KEY}; n_mc={args.n_mc} (kept {len(results)})",
        "",
        "Bagged OOS on dual-report patients:",
        f"  Auto Youden:      AUROC={summary.loc[0,'bagged_auroc']:.3f} "
        f"[{summary.loc[0,'delong_ci_lo']:.3f}, {summary.loc[0,'delong_ci_hi']:.3f}]  "
        f"AUPRC={summary.loc[0,'bagged_auprc']:.3f}",
        f"  Clinician report: AUROC={summary.loc[1,'bagged_auroc']:.3f} "
        f"[{summary.loc[1,'delong_ci_lo']:.3f}, {summary.loc[1,'delong_ci_hi']:.3f}]  "
        f"AUPRC={summary.loc[1,'bagged_auprc']:.3f}",
        f"  Δ (auto − report) bagged AUROC = {delta_bag:+.3f}",
        "",
        "Paired MC-split AUROC (dual-report patients in each test fold):",
        f"  n_paired={len(paired)}; mean AUC auto={cmp_row['mc_mean_auc_auto']:.3f}, "
        f"report={cmp_row['mc_mean_auc_report']:.3f}, "
        f"mean Δ={cmp_row['mc_mean_delta']:+.3f}",
        f"  Wilcoxon signed-rank (auto−report): "
        f"p_two={cmp_row['wilcoxon_auc_p_two_sided']:.4g}, "
        f"p_auto>report={cmp_row['wilcoxon_auc_p_auto_greater']:.4g}",
        f"  Sign test: auto higher in {sign_pos}, report higher in {sign_neg}, "
        f"ties={sign_zero}; p={sign_p:.4g}",
        f"  Wilcoxon on patient bagged p_auto−p_report: "
        f"p={cmp_row['wilcoxon_prob_p_two_sided']:.4g}",
        "",
        f"Saved: {out_dir}",
    ]
    txt = "\n".join(lines) + "\n"
    (out_dir / "RESULTS_SUMMARY.txt").write_text(txt)
    (PAPER / "aim1_report_vs_auto_markers_oob_SUMMARY.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
