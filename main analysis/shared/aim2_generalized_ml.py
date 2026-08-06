#!/usr/bin/env python3
"""Aim 2 — Generalized-epilepsy-only ML (bagged OOS AUC / AUPRC / F1).

Models (clinical = age + sex + matched prior 2yr sz control as one-hot)
  Model1_clinical : clinical alone
  Model2_IED      : clinical + IEDs (continuous log rate or Youden binary)
  Model3_slow     : clinical + slowing (continuous foc/gen or Youden binary)
  Model4_IED_slow : clinical + IEDs + slowing

No ADR / delta connectivity.

Classifiers (GridSearch hyperparameter optimization only):
  lr (logistic regression), svm (linear SVM), sgd (log-loss SGD)
  For EACH feature model separately, pick the algorithm with highest mean
  MC test AUROC for that model (not a global pick-the-winner across models).

Primary analysis
  1. 1000× stratified 80/20 patient splits (Generalized only)
  2. Per model: select lr|svm|sgd by mean MC AUROC; report with that choice
  3. MC mean AUROC/AUPRC/F1 + bootstrap 95% CI; boxplots; bagged OOS preds
  4. Pairwise model comparisons (paired bootstrap AUROC)
  5. Bagged AUROC DeLong SE + 95% CI per model

OOS strata (bagged predictions)
  - Prior 2yr sz control: documented vs not available
  - IED (Youden feature): present vs absent (Model2/3/4)
  - Generalized epilepsy subtype (Model4 only)
  - Sex: male vs female (all models)

Examples
  python aim2_generalized_ml.py --n-mc 2
  python aim2_generalized_ml.py --n-mc 1000 --poor-cutoff 1.0 --binary-markers
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm.auto import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import generalized_ml as gml
import statistical_tests as st

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_MC = 1000
TEST_SIZE = 0.20
N_CV_FOLDS = gml.N_CV_FOLDS
N_JOBS = gml.N_JOBS
N_BOOTSTRAP = gml.N_BOOTSTRAP
OUT_DIR = Path(os.environ.get(
    'ML_OUT_DIR',
    str(_SCRIPT_DIR / 'outputs' / 'General' / 'aim2_generalized_poor1mo_youden'),
))

POOR_CUTOFF_DEFAULT = 1.0

DEID_REPORT_CSV = Path(os.environ.get(
    'AIM2_DEID_REPORT_CSV',
    str(_SCRIPT_DIR.parent / 'Routineeegpec-Deidreport_DATA_LABELS_2026-06-22_1317.csv'),
))

# Clinical: age, sex, matched prior (ref=not_poor)
DEMO_COLS = ['age_at_eeg', 'gender']
PRIOR_COLS = ['prior_sz_poor', 'prior_sz_unavailable']
CLINICAL_COLS = DEMO_COLS + PRIOR_COLS
IED_COLS = ['spikes']
SLOWING_COLS_CONT = ['pct_foc_slowing', 'pct_gen_slowing']
SLOWING_COLS_BINARY = ['slowing']

MODEL4_YOUDEN = 'Model4_IED_slow'
MODEL4_CONTINUOUS = 'Model4_IED_slow_continuous'
# Per-model ROC CSVs for graphing (Aim 2 Generalized models)
ROC_EXPORT_MODELS = (MODEL4_YOUDEN, MODEL4_CONTINUOUS, 'Model1_clinical')


def build_feature_sets(binary_markers: bool) -> dict[str, list[str]]:
    slowing_cols = SLOWING_COLS_BINARY if binary_markers else SLOWING_COLS_CONT
    return {
        'Model1_clinical': CLINICAL_COLS,
        'Model2_IED': CLINICAL_COLS + IED_COLS,
        'Model3_slow': CLINICAL_COLS + slowing_cols,
        MODEL4_YOUDEN: CLINICAL_COLS + IED_COLS + slowing_cols,
    }


def model4_continuous_feature_cols() -> list[str]:
    """Full Model4 with continuous log IED + pct foc/gen slowing."""
    return CLINICAL_COLS + ['spike_rate_log', 'pct_foc_slowing', 'pct_gen_slowing']


FEATURE_SETS = build_feature_sets(binary_markers=True)
MODEL_ORDER = list(FEATURE_SETS.keys())

# Prior-availability bagged OOS — all models
PRIOR_STRATUM_MODELS = list(MODEL_ORDER)

# IED Youden-feature bagged OOS — marker models
IED_STRATUM_MODELS = ['Model2_IED', 'Model3_slow', 'Model4_IED_slow']
IED_STRATUM_COL = 'ied_youden'
IED_STRATA = ('present', 'absent')

# Sex bagged OOS — all models
SEX_STRATUM_MODELS = list(MODEL_ORDER)
SEX_STRATUM_COL = 'sex_stratum'
SEX_STRATA = ('female', 'male')

# Subtype bagged OOS DeLong — full model only
SUBTYPE_MODEL = MODEL4_YOUDEN
SUBTYPE_KEEP = [
    'Unspecified Generalized',
    'Juvenile Myoclonic',
    'GTCA',
    'Other Specified Generalized (LGS)',
]
SUBTYPE_DROP = ['Absence', 'Unclassified']

# Only these three — GridSearch tunes HPs (e.g. L1 vs L2 inside lr)
_CLF_KEYS = ('lr', 'svm', 'sgd')
CLASSIFIERS = {k: gml.CLASSIFIERS[k] for k in _CLF_KEYS}
gml.CLASSIFIERS = CLASSIFIERS


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _safe_auroc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    ok = np.isfinite(y_prob)
    y_true, y_prob = y_true[ok], y_prob[ok]
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_prob))


def _safe_auprc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    ok = np.isfinite(y_prob)
    y_true, y_prob = y_true[ok], y_prob[ok]
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float('nan')
    return float(average_precision_score(y_true, y_prob))


def _safe_f1(y_true, y_prob, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    ok = np.isfinite(y_prob)
    y_true, y_prob = y_true[ok], y_prob[ok]
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float('nan')
    y_pred = (y_prob >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def _eval_metrics(y_true, y_prob):
    """AUROC + AUPRC + F1 (threshold=0.5) for best-clf reporting."""
    return {
        'auroc': _safe_auroc(y_true, y_prob),
        'auprc': _safe_auprc(y_true, y_prob),
        'f1': _safe_f1(y_true, y_prob),
    }


def _curve_tables(y_true, y_prob, seed: int, model_key: str, best_clf: str):
    """ROC + PR curve point tables for one MC iteration (for later graphing)."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    ok = np.isfinite(y_prob)
    y_true, y_prob = y_true[ok], y_prob[ok]
    empty_roc = pd.DataFrame(columns=[
        'seed', 'model', 'best_clf', 'point', 'fpr', 'tpr', 'threshold', 'auroc',
    ])
    empty_pr = pd.DataFrame(columns=[
        'seed', 'model', 'best_clf', 'point', 'recall', 'precision', 'threshold', 'auprc',
    ])
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return empty_roc, empty_pr

    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    roc_thr_full = np.concatenate([roc_thr, [np.nan]])
    roc_df = pd.DataFrame({
        'seed': seed,
        'model': model_key,
        'best_clf': best_clf,
        'point': np.arange(len(fpr)),
        'fpr': fpr,
        'tpr': tpr,
        'threshold': roc_thr_full[:len(fpr)],
        'auroc': auroc,
    })

    precision, recall, pr_thr = precision_recall_curve(y_true, y_prob)
    pr_thr_full = np.concatenate([pr_thr, [np.nan]])
    pr_df = pd.DataFrame({
        'seed': seed,
        'model': model_key,
        'best_clf': best_clf,
        'point': np.arange(len(precision)),
        'recall': recall,
        'precision': precision,
        'threshold': pr_thr_full[:len(precision)],
        'auprc': auprc,
    })
    return roc_df, pr_df


# ---------------------------------------------------------------------------
# MATLAB-matching IED annotation (resolve_reported_spike_status)
# ---------------------------------------------------------------------------

def _norm_id(series: pd.Series) -> pd.Series:
    """Normalize IDs to string ints (handles float64 '1021.0' from CSV NaNs)."""
    out = pd.to_numeric(series, errors='coerce')
    # keep as nullable Int64 then string; dropna handled by caller when needed
    return out.astype('Int64').astype(str).replace('<NA>', np.nan)


def resolve_reported_spike_status(report: pd.DataFrame) -> pd.DataFrame:
    """Port of MATLAB resolve_reported_spike_status (Conrad et al. pipeline).

    Combines report_SPORADIC_EPILEPTIFORM_DISCHARGES with
    jay_focal_epi / jay_multifocal_epi / jay_gen_epi into session-level
    ReportStatus ∈ {present, absent, unknown}.
    """
    need = [
        'patient_id', 'session_number',
        'report_SPORADIC_EPILEPTIFORM_DISCHARGES',
        'jay_focal_epi', 'jay_multifocal_epi', 'jay_gen_epi',
    ]
    missing = [c for c in need if c not in report.columns]
    if missing:
        raise KeyError(f'DEID report missing columns for spike status: {missing}')

    report = report.copy()
    report['patient_id'] = _norm_id(report['patient_id'])
    report['session_number'] = _norm_id(report['session_number'])
    report = report.dropna(subset=['patient_id', 'session_number']).copy()

    any_spikes = report['report_SPORADIC_EPILEPTIFORM_DISCHARGES'].astype(str).str.strip()
    is_main_present = any_spikes.eq('present')
    is_main_absent = any_spikes.eq('absent')

    raw_f = report['jay_focal_epi'].astype(str).str.strip().str.lower()
    raw_m = report['jay_multifocal_epi'].astype(str).str.strip().str.lower()
    raw_g = report['jay_gen_epi'].astype(str).str.strip().str.lower()

    is_f_p, is_f_a = raw_f.eq('present'), raw_f.eq('absent')
    is_m_p, is_m_a = raw_m.eq('present'), raw_m.eq('absent')
    is_g_p, is_g_a = raw_g.eq('present'), raw_g.eq('absent')

    present_jay_any = is_f_p | is_m_p | is_g_p
    all_jay_absent = is_f_a & is_m_a & is_g_a
    blank_main = ~(is_main_present | is_main_absent)
    blank_jay_all = (
        ~(is_f_p | is_f_a) & ~(is_m_p | is_m_a) & ~(is_g_p | is_g_a)
    )

    discordant = (all_jay_absent & is_main_present) | (
        is_main_absent & is_f_p & is_m_p & is_g_p
    )
    if discordant.any():
        raise ValueError(
            f'Discordant spike presence between main and jay_* columns '
            f'({int(discordant.sum())} sessions).'
        )

    status = np.full(len(report), 'unknown', dtype=object)
    status[is_main_present.to_numpy() | present_jay_any.to_numpy()] = 'present'
    status[(all_jay_absent & blank_main).to_numpy()] = 'absent'
    status[(is_main_absent & blank_jay_all).to_numpy()] = 'absent'

    out = pd.DataFrame({
        'patient_id': report['patient_id'].astype(str),
        'session_number': report['session_number'].astype(str),
        'report_status': status,
    })
    return out


def patient_reported_spike_status(session_status: pd.DataFrame) -> pd.DataFrame:
    """Port of MATLAB Table1 / Fig S2 patient-level spike status.

    present  = any EEG with ReportStatus == present
    absent   = ≥1 EEG absent and never present
    unknown  = otherwise (no usable report)
    """
    agg = session_status.groupby('patient_id', sort=False)['report_status'].agg(
        has_present=lambda s: bool((s == 'present').any()),
        has_absent=lambda s: bool((s == 'absent').any()),
    )
    status = pd.Series('unknown', index=agg.index, dtype=object)
    status[agg['has_absent'] & ~agg['has_present']] = 'absent'
    status[agg['has_present']] = 'present'
    return status.rename('ied_annotation').reset_index()


def _filter_outpatient_routine_report(rep: pd.DataFrame) -> pd.DataFrame:
    """Match MATLAB filter_outpatient_routine: outpatient ∩ duration ≤ 4h."""
    need = ['patient_id', 'session_number', 'acquired_on',
            'report_PATIENT_CLASS', 'jay_in_or_out']
    missing = [c for c in need if c not in rep.columns]
    if missing:
        raise KeyError(f'DEID report missing outpatient-filter columns: {missing}')

    acq = rep['acquired_on'].fillna('').astype(str).str.lower().str.strip()
    cls = rep['report_PATIENT_CLASS'].fillna('').astype(str).str.lower().str.strip()
    jay = rep['jay_in_or_out'].fillna('').astype(str).str.lower().str.strip()
    is_outpt = (
        acq.str.contains('spe', regex=False)
        | acq.str.contains('radnor', regex=False)
        | (cls == 'outpatient')
        | (jay == 'out')
    )
    rep_out = rep.loc[is_outpt].copy()

    sc = pd.read_csv(gml.SPIKE_CSV)
    sc['Duration_sec'] = pd.to_numeric(sc['Duration_sec'], errors='coerce')
    sc_routine = sc[sc['Duration_sec'].notna() & (sc['Duration_sec'] <= 4 * 3600)].copy()
    keys = sc_routine.rename(columns={'Patient': 'patient_id', 'Session': 'session_number'})[
        ['patient_id', 'session_number']
    ].copy()
    keys['patient_id'] = _norm_id(keys['patient_id'])
    keys['session_number'] = _norm_id(keys['session_number'])
    keys = keys.dropna().drop_duplicates()

    rep_out['patient_id'] = _norm_id(rep_out['patient_id'])
    rep_out['session_number'] = _norm_id(rep_out['session_number'])
    kept = rep_out.merge(keys, on=['patient_id', 'session_number'], how='inner')
    print(
        f'[IED annotation] outpatient+routine report rows: '
        f'{len(kept)}/{len(rep)} (MATLAB filter_outpatient_routine)'
    )
    return kept


def load_patient_ied_annotation(analysis_pids: pd.Series) -> pd.DataFrame:
    """Load DEID report → patient-level IED present/absent/unknown.

    Uses MATLAB resolve_reported_spike_status + Table1 patient aggregation,
    restricted to outpatient routine EEGs (ReportForKeptSessions analogue).
    """
    if not DEID_REPORT_CSV.is_file():
        raise FileNotFoundError(f'DEID report not found: {DEID_REPORT_CSV}')
    print(f'[IED annotation] DEID report: {DEID_REPORT_CSV}')
    rep = pd.read_csv(DEID_REPORT_CSV, low_memory=False)
    rep = _filter_outpatient_routine_report(rep)
    sess = resolve_reported_spike_status(rep)
    # Restrict to analysis cohort patients (Generalized ML set)
    keep_pids = set(_norm_id(analysis_pids).dropna().astype(str))
    sess = sess[sess['patient_id'].isin(keep_pids)].copy()
    pat = patient_reported_spike_status(sess)
    n_pre = int((pat['ied_annotation'] == 'present').sum())
    n_abs = int((pat['ied_annotation'] == 'absent').sum())
    n_unk = int((pat['ied_annotation'] == 'unknown').sum())
    print(
        f'[IED annotation] patients in analysis with report rows: '
        f'{len(pat)}  present={n_pre}  absent={n_abs}  unknown={n_unk}'
    )
    # Patients in analysis with no report rows → unknown
    all_pids = pd.DataFrame({'patient_id': sorted(keep_pids)})
    out = all_pids.merge(pat, on='patient_id', how='left')
    out['ied_annotation'] = out['ied_annotation'].fillna('unknown')
    n_miss = int((~out['patient_id'].isin(pat['patient_id'])).sum())
    if n_miss:
        print(f'[IED annotation] {n_miss} analysis patients had no DEID rows → unknown')
    return out


def _resolve_epilepsy_specific(values) -> str:
    """MATLAB resolve_epilepsy_specific: majority vote; empty on tie."""
    vals = [
        str(v).strip() for v in values
        if pd.notna(v) and str(v).strip() and str(v).strip().lower() != 'nan'
    ]
    if not vals:
        return ''
    uniq, counts = np.unique(vals, return_counts=True)
    order = np.argsort(-counts)
    if len(uniq) > 1 and counts[order[0]] == counts[order[1]]:
        return ''
    return str(uniq[order[0]])


def map_general_subtype(raw: str) -> str:
    """MATLAB map_general_subtype (make_fig_general_slowing_by_subtype.m)."""
    s = str(raw).strip().lower()
    if (not s) or s == 'unknown or mrn not found' or s == 'unclassified or unspecified':
        return 'Unclassified'
    if s == 'unspecified generalized':
        return 'Unspecified Generalized'
    if 'absence' in s:
        return 'Absence'
    if s == 'juvenile myoclonic':
        return 'Juvenile Myoclonic'
    if s == 'gtca':
        return 'GTCA'
    if s == 'other specified generalized' or 'lgs' in s:
        return 'Other Specified Generalized (LGS)'
    if s == 'combined generalized and focal':
        return 'Combined Generalized and Focal'
    return 'Unclassified'


def load_patient_epilepsy_subtype(analysis_pids: pd.Series) -> pd.DataFrame:
    """Patient-level generalized subtype from DEID epilepsy_specific."""
    if not DEID_REPORT_CSV.is_file():
        raise FileNotFoundError(f'DEID report not found: {DEID_REPORT_CSV}')
    rep = pd.read_csv(DEID_REPORT_CSV, low_memory=False)
    if 'epilepsy_specific' not in rep.columns:
        raise KeyError('DEID report missing epilepsy_specific')
    rep = rep.copy()
    rep['patient_id'] = _norm_id(rep['patient_id'])
    rep = rep.dropna(subset=['patient_id'])

    keep_pids = set(_norm_id(analysis_pids).dropna().astype(str))
    rows = []
    for pid, sub in rep.groupby('patient_id', sort=False):
        if str(pid) not in keep_pids:
            continue
        raw = _resolve_epilepsy_specific(sub['epilepsy_specific'].tolist())
        label = map_general_subtype(raw) if raw else 'Unclassified'
        rows.append({
            'patient_id': str(pid),
            'subtype_raw': raw if raw else '',
            'epilepsy_subtype': label,
        })
    out = pd.DataFrame(rows)
    all_pids = pd.DataFrame({'patient_id': sorted(keep_pids)})
    out = all_pids.merge(out, on='patient_id', how='left')
    out['epilepsy_subtype'] = out['epilepsy_subtype'].fillna('Unclassified')
    out['subtype_raw'] = out['subtype_raw'].fillna('')

    vc = out['epilepsy_subtype'].value_counts()
    print('[Subtype] patient-level mapped labels (Aim2 cohort):')
    for lab, n in vc.items():
        flag = ' KEEP' if lab in SUBTYPE_KEEP else (' DROP' if lab in SUBTYPE_DROP else '')
        print(f'  {lab}: {n}{flag}')
    return out


# ---------------------------------------------------------------------------
# Cohort / features
# ---------------------------------------------------------------------------

def build_analysis_df(
    patient_df: pd.DataFrame,
    poor_cutoff: float = POOR_CUTOFF_DEFAULT,
    binary_markers: bool = True,
    epilepsy_type: str = 'Generalized',
) -> pd.DataFrame:
    """Type-stratified cohort: poor>cutoff outcome, matched prior, IED/slowing.

    epilepsy_type: 'Generalized' or 'Focal' (Temporal+Frontal pooled as Focal upstream).
    """
    df = patient_df.copy()
    if epilepsy_type not in ('Generalized', 'Focal'):
        raise ValueError(f'epilepsy_type must be Generalized or Focal, got {epilepsy_type!r}')
    df = df[df['epilepsy_type'] == epilepsy_type].copy()
    if 'mean_sz_freq' not in df.columns:
        raise KeyError('mean_sz_freq required for poor-cutoff outcome')
    df['sz_freq_class'] = df['mean_sz_freq'].map(
        lambda x: st._code_outcome_binary(x, poor_cutoff)
    )
    df = df.dropna(subset=['sz_freq_class', 'age_at_eeg', 'gender']).copy()
    df['sz_freq_class'] = df['sz_freq_class'].astype(int)
    df['gender'] = df['gender'].astype(float)

    df['spikes'] = df['spike_rate_log']
    df['spike_rate_log'] = df['spikes'].astype(float)
    df['slowing_pct'] = df['pct_gen_slowing'].fillna(0) + df['pct_foc_slowing'].fillna(0)
    df['pct_foc_slowing'] = df['pct_foc_slowing'].fillna(0)
    df['pct_gen_slowing'] = df['pct_gen_slowing'].fillna(0)

    if 'mean_sz_freq_pre' not in df.columns:
        raise KeyError('mean_sz_freq_pre required for matched prior')
    df['prior_sz_control'] = df['mean_sz_freq_pre'].map(
        lambda x: st._code_prior_from_cutoff(x, poor_cutoff)
    ).astype(int)
    df['prior_sz_poor'] = (df['prior_sz_control'] == 1).astype(float)
    df['prior_sz_unavailable'] = (df['prior_sz_control'] == 2).astype(float)
    df['prior_availability'] = np.where(
        df['prior_sz_control'] == 2, 'unavailable', 'available',
    )

    df['ied_youden'] = np.where(
        df['spikes'] >= st.SPIKE_YOUDEN_THRESH, 'present', 'absent',
    )
    df['sex_stratum'] = np.where(df['gender'] < 0.5, 'female', 'male')

    if binary_markers:
        n_sp = int((df['spikes'] >= st.SPIKE_YOUDEN_THRESH).sum())
        n_sl = int((df['slowing_pct'] >= st.SLOWING_YOUDEN_THRESH_PCT).sum())
        print(
            f'[Binary markers] spikes>={st.SPIKE_YOUDEN_THRESH}: {n_sp}/{len(df)}; '
            f'slowing>={st.SLOWING_YOUDEN_THRESH_PCT}%: {n_sl}/{len(df)}'
        )
        df['spikes'] = (df['spikes'] >= st.SPIKE_YOUDEN_THRESH).astype(float)
        df['slowing'] = (df['slowing_pct'] >= st.SLOWING_YOUDEN_THRESH_PCT).astype(float)

    slowing_need = SLOWING_COLS_BINARY if binary_markers else SLOWING_COLS_CONT
    need = [
        'sz_freq_class', 'age_at_eeg', 'gender', 'spikes',
        *slowing_need,
        'prior_sz_poor', 'prior_sz_unavailable',
    ]
    df = df.dropna(subset=need).copy()
    df['patient_id'] = df['patient_id'].astype(str)

    n_poor = int((df['sz_freq_class'] == 1).sum())
    print(
        f'[Outcome] poor = mean > {poor_cutoff}/mo: {n_poor}/{len(df)} '
        f'({100 * n_poor / len(df):.1f}%)'
    )
    vc = df['prior_sz_control'].value_counts().sort_index()
    print(
        '[Prior matched] '
        f"not_poor={int(vc.get(0, 0))}, poor={int(vc.get(1, 0))}, "
        f"NA={int(vc.get(2, 0))}"
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# DeLong (unpaired strata comparison)
# ---------------------------------------------------------------------------

def delong_auc_and_var(y_true, y_score) -> tuple[float, float]:
    """DeLong AUC and variance for a single classifier (Sun & Xu / DeLong 1988)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    ok = np.isfinite(y_score)
    y_true, y_score = y_true[ok], y_score[ok]
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float('nan'), float('nan')
    auc, v10, v01, n_pos, n_neg = gml._auc_placement(y_score, y_true)
    if n_pos < 2 or n_neg < 2:
        return float(auc), float('nan')
    var = float(np.var(v10, ddof=1) / n_pos + np.var(v01, ddof=1) / n_neg)
    return float(auc), var


def bagged_auc_delong_stats(y_true, y_score) -> dict:
    """Point AUROC + DeLong SE and normal-approx 95% CI for one score vector."""
    auc, var = delong_auc_and_var(y_true, y_score)
    out = {
        'bagged_auc': auc,
        'delong_var': var,
        'delong_se': np.nan,
        'delong_ci_lo': np.nan,
        'delong_ci_hi': np.nan,
    }
    if np.isfinite(var) and var >= 0:
        se = float(np.sqrt(var))
        out['delong_se'] = se
        out['delong_ci_lo'] = float(auc - 1.96 * se)
        out['delong_ci_hi'] = float(auc + 1.96 * se)
    return out


def unpaired_delong_compare(y_a, p_a, y_b, p_b) -> dict:
    """Δ = AUC_A − AUC_B with SE = sqrt(Var_A + Var_B) (independent samples)."""
    auc_a, var_a = delong_auc_and_var(y_a, p_a)
    auc_b, var_b = delong_auc_and_var(y_b, p_b)
    out = {
        'auc_a': auc_a, 'auc_b': auc_b, 'var_a': var_a, 'var_b': var_b,
        'delta': np.nan, 'se': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
        'z': np.nan, 'p': np.nan,
        'n_a': int(np.isfinite(p_a).sum()), 'n_b': int(np.isfinite(p_b).sum()),
    }
    if not (np.isfinite(auc_a) and np.isfinite(auc_b)
            and np.isfinite(var_a) and np.isfinite(var_b) and (var_a + var_b) > 0):
        return out
    delta = auc_a - auc_b
    se = float(np.sqrt(var_a + var_b))
    z = delta / se
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    out.update({
        'delta': float(delta), 'se': se,
        'ci_lo': float(delta - 1.96 * se), 'ci_hi': float(delta + 1.96 * se),
        'z': float(z), 'p': p,
    })
    return out


# ---------------------------------------------------------------------------
# One MC iteration
# ---------------------------------------------------------------------------

def run_mc_iteration(seed: int, analysis_df: pd.DataFrame, feature_sets: dict) -> dict | None:
    """Fit lr / svm / sgd with GridSearch; store test probs for each."""
    y_all = analysis_df['sz_freq_class'].values.astype(int)
    tr_idx, te_idx = gml._patient_grouped_split(analysis_df, y_all, TEST_SIZE, seed)
    y_tr = y_all[tr_idx]
    y_te = y_all[te_idx]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None

    train_df = analysis_df.iloc[tr_idx]
    test_df = analysis_df.iloc[te_idx]
    test_pids = test_df['patient_id'].astype(str).values

    proba_by_clf: dict = {}
    metrics_by_clf: dict = {}
    for model_key, cols in feature_sets.items():
        X_tr = gml.prepare_X(train_df, cols, ref_df=train_df)
        X_te = gml.prepare_X(test_df, cols, ref_df=train_df)
        proba_by_clf[model_key] = {}
        metrics_by_clf[model_key] = {}
        for clf_key in CLASSIFIERS:
            try:
                est, _ = gml.fit_clf(
                    clf_key, X_tr, y_tr, seed, N_CV_FOLDS, class_weight='none',
                )
                p = est.predict_proba(X_te)[:, 1]
            except Exception:
                p = np.full(len(y_te), np.nan)
            proba_by_clf[model_key][clf_key] = np.asarray(p, dtype=float)
            metrics_by_clf[model_key][clf_key] = _eval_metrics(y_te, p)

    return {
        'seed': seed,
        'test_pids': test_pids,
        'y_test': y_te.astype(int),
        'proba_by_clf': proba_by_clf,
        'metrics_by_clf': metrics_by_clf,
        # Filled by apply_selected_clfs_per_model
        'best_clf': {},
        'proba': {},
        'metrics': {},
        'roc_curves': {},
        'pr_curves': {},
    }


def select_clf_per_model(mc_results: list, model_keys: list[str]) -> tuple[dict[str, str], pd.DataFrame]:
    """For each feature model, pick lr|svm|sgd with highest mean MC test AUROC."""
    rows = []
    selected: dict[str, str] = {}
    for model_key in model_keys:
        means = {}
        for clf_key in _CLF_KEYS:
            aucs = []
            for res in mc_results:
                if res is None:
                    continue
                a = res['metrics_by_clf'][model_key][clf_key]['auroc']
                if np.isfinite(a):
                    aucs.append(float(a))
            mean_auc = float(np.mean(aucs)) if aucs else float('nan')
            means[clf_key] = mean_auc
            rows.append({
                'model': model_key,
                'algorithm': clf_key,
                'n_mc': len(aucs),
                'auc_mean': mean_auc,
                'auc_std': float(np.std(aucs, ddof=1)) if len(aucs) > 1 else np.nan,
            })
        valid = {k: v for k, v in means.items() if np.isfinite(v)}
        if not valid:
            raise RuntimeError(f'No finite AUCs for model {model_key}')
        selected[model_key] = max(valid, key=valid.get)
    rank = (
        pd.DataFrame(rows)
        .sort_values(['model', 'auc_mean'], ascending=[True, False])
        .reset_index(drop=True)
    )
    return selected, rank


def apply_selected_clfs_per_model(
    mc_results: list,
    feature_sets: dict,
    selected_by_model: dict[str, str],
) -> list:
    """Lock each feature model to its own chosen algorithm (lr|svm|sgd)."""
    for res in mc_results:
        if res is None:
            continue
        for model_key in feature_sets:
            clf = selected_by_model[model_key]
            p = np.asarray(res['proba_by_clf'][model_key][clf], dtype=float)
            res['proba'][model_key] = p
            res['metrics'][model_key] = res['metrics_by_clf'][model_key][clf]
            res['best_clf'][model_key] = clf
            roc_df, pr_df = _curve_tables(
                res['y_test'], p, res['seed'], model_key, clf,
            )
            res['roc_curves'][model_key] = roc_df
            res['pr_curves'][model_key] = pr_df
    return mc_results


# ---------------------------------------------------------------------------
# Bagging + summaries
# ---------------------------------------------------------------------------

def bag_oos_predictions(mc_results: list, analysis_df: pd.DataFrame, model_key: str):
    """Mean held-out prediction per patient for one model."""
    pids = analysis_df['patient_id'].astype(str).values
    y = analysis_df['sz_freq_class'].values.astype(int)
    ied = analysis_df[IED_STRATUM_COL].astype(str).values
    subtype = analysis_df['epilepsy_subtype'].astype(str).values
    prior_av = analysis_df['prior_availability'].astype(str).values
    sex = analysis_df[SEX_STRATUM_COL].astype(str).values
    sums = {pid: 0.0 for pid in pids}
    counts = {pid: 0 for pid in pids}
    clf_votes: dict[str, int] = {}

    for res in mc_results:
        if res is None:
            continue
        preds = res['proba'][model_key]
        for pid, pr in zip(res['test_pids'], preds):
            if not np.isfinite(pr):
                continue
            sums[pid] += float(pr)
            counts[pid] += 1
        clf = res['best_clf'][model_key]
        clf_votes[clf] = clf_votes.get(clf, 0) + 1

    bagged = np.full(len(pids), np.nan)
    n_held = np.zeros(len(pids), dtype=int)
    for i, pid in enumerate(pids):
        if counts[pid] > 0:
            bagged[i] = sums[pid] / counts[pid]
            n_held[i] = counts[pid]

    return {
        'patient_id': pids,
        'y_true': y,
        'ied_youden': ied,
        'epilepsy_subtype': subtype,
        'prior_availability': prior_av,
        'sex_stratum': sex,
        'bagged_pred': bagged,
        'n_heldout': n_held,
        'clf_votes': clf_votes,
    }


def summarize_mc_metrics(mc_results: list, model_key: str, rng=None) -> dict:
    """Mean AUROC / AUPRC across MC test folds + bootstrap 95% CI of the mean."""
    rng = rng or np.random.default_rng(0)
    aurocs, auprcs, f1s = [], [], []
    for res in mc_results:
        if res is None:
            continue
        m = res['metrics'][model_key]
        if np.isfinite(m['auroc']):
            aurocs.append(m['auroc'])
        if np.isfinite(m['auprc']):
            auprcs.append(m['auprc'])
        if np.isfinite(m['f1']):
            f1s.append(m['f1'])

    aurocs = np.asarray(aurocs, dtype=float)
    auprcs = np.asarray(auprcs, dtype=float)
    f1s = np.asarray(f1s, dtype=float)
    auc_lo, auc_hi = gml.bootstrap_mean_ci(aurocs, n_boot=N_BOOTSTRAP, rng=rng)
    pr_lo, pr_hi = gml.bootstrap_mean_ci(auprcs, n_boot=N_BOOTSTRAP, rng=rng)
    f1_lo, f1_hi = gml.bootstrap_mean_ci(f1s, n_boot=N_BOOTSTRAP, rng=rng)
    return {
        'model': model_key,
        'n_mc': int(len(aurocs)),
        'auc_mean': float(np.nanmean(aurocs)) if len(aurocs) else np.nan,
        'auc_ci_lo': auc_lo,
        'auc_ci_hi': auc_hi,
        'auprc_mean': float(np.nanmean(auprcs)) if len(auprcs) else np.nan,
        'auprc_ci_lo': pr_lo,
        'auprc_ci_hi': pr_hi,
        'f1_mean': float(np.nanmean(f1s)) if len(f1s) else np.nan,
        'f1_ci_lo': f1_lo,
        'f1_ci_hi': f1_hi,
    }


def pairwise_mc_auroc_sign_test(
    mc_df: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> dict:
    """Paired MC-split AUROC comparison (no bagged / patient bootstrap).

    For each MC seed, compare test-set AUROC of model_a vs model_b.
    Δ = AUROC_b − AUROC_a across splits; 95% CI = 2.5/97.5 percentiles of Δ.
    Two-sided sign-test p from win counts (neither model privileged):
        p = 2 * (min(n_a_wins, n_b_wins) + 1) / (n_mc + 1)
    """
    piv = mc_df.pivot(index='seed', columns='model', values='auroc')
    if model_a not in piv.columns or model_b not in piv.columns:
        return {
            'model_a': model_a, 'model_b': model_b, 'n_mc': 0,
            'auc_a_mean': np.nan, 'auc_b_mean': np.nan,
            'diff_b_minus_a': np.nan, 'diff_ci_lo': np.nan, 'diff_ci_hi': np.nan,
            'n_a_wins': 0, 'n_b_wins': 0, 'n_ties': 0,
            'better_model': '', 'p': np.nan,
        }
    a = piv[model_a]
    b = piv[model_b]
    both = a.notna() & b.notna()
    da = a[both].to_numpy(dtype=float)
    db = b[both].to_numpy(dtype=float)
    deltas = db - da
    n = int(len(deltas))
    n_a_wins = int(np.sum(deltas < 0))
    n_b_wins = int(np.sum(deltas > 0))
    n_ties = int(np.sum(deltas == 0))
    n_min = min(n_a_wins, n_b_wins)
    p = float(min(2.0 * (n_min + 1) / (n + 1), 1.0)) if n > 0 else np.nan
    mean_d = float(np.mean(deltas)) if n else np.nan
    better = model_b if (n and mean_d > 0) else (model_a if n else '')
    if n and mean_d == 0:
        better = 'tie'
    return {
        'model_a': model_a,
        'model_b': model_b,
        'n_mc': n,
        'auc_a_mean': float(np.mean(da)) if n else np.nan,
        'auc_b_mean': float(np.mean(db)) if n else np.nan,
        'diff_b_minus_a': mean_d,
        'diff_ci_lo': float(np.percentile(deltas, 2.5)) if n else np.nan,
        'diff_ci_hi': float(np.percentile(deltas, 97.5)) if n else np.nan,
        'n_a_wins': n_a_wins,
        'n_b_wins': n_b_wins,
        'n_ties': n_ties,
        'better_model': better,
        'p': p,
    }


def pairwise_paper_comparisons(
    mc_df: pd.DataFrame,
    bagged: dict,
    model_keys: list[str],
    feature_sets: dict,
    n_boot: int = N_BOOTSTRAP,
    rng=None,
) -> list[dict]:
    """Pairwise comparisons from MC test AUROCs (sign test across permutations).

    ``bagged`` / ``n_boot`` / ``rng`` kept for call-site compatibility; unused.
    """
    del bagged, n_boot, rng
    rows = []
    for model_a, model_b in combinations(model_keys, 2):
        r = pairwise_mc_auroc_sign_test(mc_df, model_a, model_b)
        rows.append({
            'stratum': 'all',
            'model_a': r['model_a'],
            'model_b': r['model_b'],
            'model_simple': r['model_a'],  # legacy alias (not used for p)
            'model_complex': r['model_b'],
            'n_features_a': len(feature_sets.get(model_a, [])),
            'n_features_b': len(feature_sets.get(model_b, [])),
            'n_features_simple': len(feature_sets.get(model_a, [])),
            'n_features_complex': len(feature_sets.get(model_b, [])),
            'n_patients': int(mc_df['seed'].nunique()) if len(mc_df) else 0,
            'n_splits': r['n_mc'],
            'auc_a': r['auc_a_mean'],
            'auc_b': r['auc_b_mean'],
            'auc_simple': r['auc_a_mean'],
            'auc_complex': r['auc_b_mean'],
            'diff': r['diff_b_minus_a'],
            'diff_ci_lo': r['diff_ci_lo'],
            'diff_ci_hi': r['diff_ci_hi'],
            'p': r['p'],
            'n_a_wins': r['n_a_wins'],
            'n_b_wins': r['n_b_wins'],
            'n_ties': r['n_ties'],
            'better_model': r['better_model'],
            'n_simpler': r['n_a_wins'],  # legacy alias
            'n_boot_valid': r['n_mc'],
            'auprc_simple': np.nan,
            'auprc_complex': np.nan,
            'auprc_diff': np.nan,
        })
    return rows


def pairwise_model_comparisons(
    bagged: dict,
    model_keys: list[str],
    feature_sets: dict,
    mask: np.ndarray | None = None,
    stratum: str = 'all',
    rng=None,
) -> list[dict]:
    """All pairwise bagged-OOS bootstrap AUROC comparisons (two-sided p)."""
    rng = rng or np.random.default_rng(1)
    n_feats = {k: len(feature_sets[k]) for k in model_keys}
    rows = []
    for ki, kj in combinations(model_keys, 2):
        if n_feats[ki] <= n_feats[kj]:
            k_simple, k_complex = ki, kj
        else:
            k_simple, k_complex = kj, ki

        y = bagged[k_simple]['y_true']
        p_s = bagged[k_simple]['bagged_pred']
        p_c = bagged[k_complex]['bagged_pred']
        ok = np.isfinite(p_s) & np.isfinite(p_c)
        if mask is not None:
            ok = ok & mask

        cmp_ = gml.bootstrap_compare_auroc(
            y[ok], p_s[ok], p_c[ok],
            n_boot=N_BOOTSTRAP, rng=rng,
        )
        rows.append({
            'stratum': stratum,
            'model_simple': k_simple,
            'model_complex': k_complex,
            'n_features_simple': n_feats[k_simple],
            'n_features_complex': n_feats[k_complex],
            'n_patients': int(ok.sum()),
            'auc_simple': cmp_['auc_simple'],
            'auc_complex': cmp_['auc_complex'],
            'diff': cmp_['diff'],
            'diff_ci_lo': cmp_['diff_ci_lo'],
            'diff_ci_hi': cmp_['diff_ci_hi'],
            'p': cmp_['p'],
            'auprc_simple': _safe_auprc(y[ok], p_s[ok]),
            'auprc_complex': _safe_auprc(y[ok], p_c[ok]),
            'n_boot_valid': cmp_['n_boot_valid'],
        })
    return rows


def stratum_bagged_metrics(bag: dict, stratum: str, stratum_col: str = 'ied_annotation') -> dict:
    y = bag['y_true']
    p = bag['bagged_pred']
    labels = np.asarray(bag[stratum_col], dtype=str)
    ok = np.isfinite(p) & (labels == stratum)
    delong = bagged_auc_delong_stats(y[ok], p[ok])
    return {
        'model': None,  # filled by caller
        'stratum': stratum,
        'n_patients': int(ok.sum()),
        'n_pos': int(y[ok].sum()) if ok.any() else 0,
        'prevalence': float(y[ok].mean()) if ok.any() else np.nan,
        'bagged_auc': delong['bagged_auc'],
        'delong_se': delong['delong_se'],
        'delong_var': delong['delong_var'],
        'delong_ci_lo': delong['delong_ci_lo'],
        'delong_ci_hi': delong['delong_ci_hi'],
        'bagged_auprc': _safe_auprc(y[ok], p[ok]),
        'n_heldout_mean': float(np.mean(bag['n_heldout'][ok])) if ok.any() else np.nan,
    }


def pairwise_subtype_delong(bag: dict, subtypes: list[str], model_key: str) -> list[dict]:
    """Unpaired DeLong AUC comparisons across subtypes + Bonferroni (global)."""
    y = bag['y_true']
    p = bag['bagged_pred']
    lab = np.asarray(bag['epilepsy_subtype'], dtype=str)
    ok = np.isfinite(p)

    usable = []
    for s in subtypes:
        mask = ok & (lab == s)
        n_pos = int(y[mask].sum()) if mask.any() else 0
        n_neg = int(mask.sum()) - n_pos
        if n_pos >= 2 and n_neg >= 2:
            usable.append(s)
        else:
            print(f'  [Subtype] skip {s}: N={int(mask.sum())} pos={n_pos} neg={n_neg} (<2/class)')

    pairs = list(combinations(usable, 2))
    n_tests = len(pairs)
    rows = []
    for a, b in pairs:
        ma = ok & (lab == a)
        mb = ok & (lab == b)
        cmp_ = unpaired_delong_compare(y[ma], p[ma], y[mb], p[mb])
        p_raw = cmp_['p']
        p_bonf = float(min(1.0, p_raw * n_tests)) if np.isfinite(p_raw) else np.nan
        rows.append({
            'model': model_key,
            'subtype_a': a,
            'subtype_b': b,
            'n_a': cmp_['n_a'],
            'n_b': cmp_['n_b'],
            'auc_a': cmp_['auc_a'],
            'auc_b': cmp_['auc_b'],
            'var_a': cmp_['var_a'],
            'var_b': cmp_['var_b'],
            'delta_a_minus_b': cmp_['delta'],
            'se': cmp_['se'],
            'ci_lo': cmp_['ci_lo'],
            'ci_hi': cmp_['ci_hi'],
            'p': p_raw,
            'p_bonferroni': p_bonf,
            'n_tests_bonferroni': n_tests,
        })
    return rows



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare_model4_youden_vs_continuous(
    mc_results: list,
    analysis_df: pd.DataFrame,
    selected_by_model: dict[str, str],
    out_dir: Path,
    n_boot: int = N_BOOTSTRAP,
) -> dict:
    """Paired MC-split comparison of Model4 Youden vs continuous (sign test)."""
    del n_boot  # no patient bootstrap
    rng = np.random.default_rng(42)
    summ_y = summarize_mc_metrics(mc_results, MODEL4_YOUDEN, rng=rng)
    summ_c = summarize_mc_metrics(mc_results, MODEL4_CONTINUOUS, rng=rng)

    mc_df = pd.DataFrame([
        {
            'seed': res['seed'],
            'model': mk,
            'auroc': res['metrics'][mk]['auroc'],
            'auprc': res['metrics'][mk]['auprc'],
        }
        for res in mc_results if res is not None
        for mk in (MODEL4_YOUDEN, MODEL4_CONTINUOUS)
        if mk in res['metrics']
    ])
    cmp_ = pairwise_mc_auroc_sign_test(mc_df, MODEL4_YOUDEN, MODEL4_CONTINUOUS)

    # Bagged AUCs kept for descriptive reporting only (not used for p)
    bag_y = bag_oos_predictions(mc_results, analysis_df, MODEL4_YOUDEN)
    bag_c = bag_oos_predictions(mc_results, analysis_df, MODEL4_CONTINUOUS)
    y = bag_y['y_true']
    p_y = bag_y['bagged_pred']
    p_c = bag_c['bagged_pred']
    ok = np.isfinite(p_y) & np.isfinite(p_c)
    delong_y = bagged_auc_delong_stats(y[ok], p_y[ok])
    delong_c = bagged_auc_delong_stats(y[ok], p_c[ok])

    row = {
        'model_youden': MODEL4_YOUDEN,
        'model_continuous': MODEL4_CONTINUOUS,
        'n_patients': int(ok.sum()),
        'selected_clf_youden': selected_by_model.get(MODEL4_YOUDEN, ''),
        'selected_clf_continuous': selected_by_model.get(MODEL4_CONTINUOUS, ''),
        'mc_auroc_youden': summ_y['auc_mean'],
        'mc_auroc_youden_ci_lo': summ_y['auc_ci_lo'],
        'mc_auroc_youden_ci_hi': summ_y['auc_ci_hi'],
        'mc_auroc_continuous': summ_c['auc_mean'],
        'mc_auroc_continuous_ci_lo': summ_c['auc_ci_lo'],
        'mc_auroc_continuous_ci_hi': summ_c['auc_ci_hi'],
        'mc_auroc_diff_cont_minus_youden': cmp_['diff_b_minus_a'],
        'mc_diff_ci_lo': cmp_['diff_ci_lo'],
        'mc_diff_ci_hi': cmp_['diff_ci_hi'],
        'n_youden_wins': cmp_['n_a_wins'],
        'n_continuous_wins': cmp_['n_b_wins'],
        'n_ties': cmp_['n_ties'],
        'better_model': cmp_['better_model'],
        'p_mc_sign_two_sided': cmp_['p'],
        'n_mc': cmp_['n_mc'],
        'bagged_auc_youden': delong_y['bagged_auc'],
        'bagged_auc_youden_ci_lo': delong_y['delong_ci_lo'],
        'bagged_auc_youden_ci_hi': delong_y['delong_ci_hi'],
        'bagged_auc_continuous': delong_c['bagged_auc'],
        'bagged_auc_continuous_ci_lo': delong_c['delong_ci_lo'],
        'bagged_auc_continuous_ci_hi': delong_c['delong_ci_hi'],
        # legacy aliases for old summary readers
        'bagged_auc_diff_cont_minus_youden': cmp_['diff_b_minus_a'],
        'bagged_auc_diff_ci_lo': cmp_['diff_ci_lo'],
        'bagged_auc_diff_ci_hi': cmp_['diff_ci_hi'],
        'p_bootstrap_two_sided': cmp_['p'],
        'n_boot_valid': cmp_['n_mc'],
        'auprc_youden': _safe_auprc(y[ok], p_y[ok]),
        'auprc_continuous': _safe_auprc(y[ok], p_c[ok]),
    }
    save_table(pd.DataFrame([row]), 'model4_youden_vs_continuous_bootstrap.csv', out_dir)

    lines = [
        'Model4 full model — Youden binary vs continuous markers (paired MC splits)',
        '=' * 72,
        f'Youden features:     {build_feature_sets(True)[MODEL4_YOUDEN]}',
        f'Continuous features: {model4_continuous_feature_cols()}',
        f'Selected clf Youden:      {row["selected_clf_youden"]}',
        f'Selected clf Continuous:  {row["selected_clf_continuous"]}',
        '',
        'MC mean test AUROC [bootstrap 95% CI of mean]:',
        f'  Youden:      {row["mc_auroc_youden"]:.3f} '
        f'[{row["mc_auroc_youden_ci_lo"]:.3f}, {row["mc_auroc_youden_ci_hi"]:.3f}]',
        f'  Continuous:  {row["mc_auroc_continuous"]:.3f} '
        f'[{row["mc_auroc_continuous_ci_lo"]:.3f}, {row["mc_auroc_continuous_ci_hi"]:.3f}]',
        '',
        'Paired MC sign test (per-split test AUROC; H0: equal):',
        f'  Δ(cont−youden) = {row["mc_auroc_diff_cont_minus_youden"]:+.4f} '
        f'[{row["mc_diff_ci_lo"]:+.4f}, {row["mc_diff_ci_hi"]:+.4f}]',
        f'  wins: Youden={row["n_youden_wins"]}  Continuous={row["n_continuous_wins"]}  '
        f'ties={row["n_ties"]}',
        f'  better={row["better_model"]}  p={row["p_mc_sign_two_sided"]:.4g}  '
        f'(n_mc={row["n_mc"]})',
        '',
        'Bagged OOS AUROC [DeLong 95% CI] (descriptive only):',
        f'  Youden:      {row["bagged_auc_youden"]:.3f} '
        f'[{row["bagged_auc_youden_ci_lo"]:.3f}, {row["bagged_auc_youden_ci_hi"]:.3f}]',
        f'  Continuous:  {row["bagged_auc_continuous"]:.3f} '
        f'[{row["bagged_auc_continuous_ci_lo"]:.3f}, {row["bagged_auc_continuous_ci_hi"]:.3f}]',
        '',
        f'AUPRC (bagged, descriptive): Youden={row["auprc_youden"]:.3f}, '
        f'Continuous={row["auprc_continuous"]:.3f}',
    ]
    summary_path = out_dir / 'model4_youden_vs_continuous_summary.txt'
    summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n=== Model4 Youden vs continuous (paired) ===')
    print('\n'.join(lines))
    print(f'Saved: {summary_path}')
    return row


def write_aim2_results_summary(out_dir: Path, include_model4_cont: bool = False) -> Path:
    """Organize key CSV outputs into aim2_results_summary.txt."""
    out_dir = Path(out_dir)
    mc = pd.read_csv(out_dir / 'mc_metrics_summary.csv')
    pair_path = out_dir / 'pairwise_model_mc_splits.csv'
    if not pair_path.is_file():
        pair_path = out_dir / 'pairwise_model_bootstrap.csv'
    pair = pd.read_csv(pair_path)
    prior = pd.read_csv(out_dir / 'bagged_metrics_by_prior_availability.csv')
    prior_delong = pd.read_csv(out_dir / 'prior_available_vs_unavailable_delong.csv')
    ied = pd.read_csv(out_dir / 'bagged_metrics_by_ied_youden_stratum.csv')
    sex = pd.read_csv(out_dir / 'bagged_metrics_by_sex_stratum.csv')
    sel = (out_dir / 'selected_algorithms_per_model.txt').read_text(encoding='utf-8').strip()

    lines = [
        'Aim 2 — Generalized epilepsy ML results summary',
        '=' * 72,
        f'OUT_DIR: {out_dir}/',
        '',
        '0. What this pipeline extracts',
        '-' * 24,
        'Per 1000 MC splits (80/20 patient-grouped, Generalized N=490):',
        '  • MC test metrics: AUROC, AUPRC, F1@0.5 per model×algorithm',
        '  • Algorithm pick: best of lr/svm/sgd by mean MC AUROC (per feature model)',
        '  • Pairwise models: per-split test AUROC sign test across MC permutations',
        '      p = 2*(min(n_A_wins, n_B_wins)+1)/(n_mc+1); CI = percentiles of Δ',
        '  • Bagged OOS strata (descriptive / DeLong only):',
        '      - prior 2yr sz freq documented vs unavailable (+ DeLong)',
        '      - Youden IED present vs absent, Models 2–4 (+ DeLong)',
        '      - sex female vs male (+ DeLong)',
        '      - Generalized subtype, Model4 Youden only (+ DeLong Bonferroni)',
        '  • Model4 Youden vs continuous: same MC sign test on paired splits',
        '',
        '1. Selected algorithms',
        '-' * 24,
        sel,
        '',
        '2. MC mean metrics (locked algorithm)',
        '-' * 24,
    ]
    for _, r in mc.iterrows():
        lines.append(
            f"{r['model']}: AUROC={r['auc_mean']:.3f} [{r['auc_ci_lo']:.3f},{r['auc_ci_hi']:.3f}]  "
            f"AUPRC={r['auprc_mean']:.3f}  F1={r['f1_mean']:.3f}"
        )

    lines += [
        '',
        '3. Pairwise comparisons — Youden models (MC permutation sign test)',
        '-' * 24,
    ]
    pair_all = pair[pair['stratum'] == 'all'] if 'stratum' in pair.columns else pair
    for _, r in pair_all.iterrows():
        sig = '*' if r['p'] < 0.05 else ''
        a = r['model_a'] if 'model_a' in r.index and pd.notna(r.get('model_a')) else r['model_simple']
        b = r['model_b'] if 'model_b' in r.index and pd.notna(r.get('model_b')) else r['model_complex']
        better = r['better_model'] if 'better_model' in r.index else ''
        n_a = int(r['n_a_wins']) if 'n_a_wins' in r.index and pd.notna(r.get('n_a_wins')) else -1
        n_b = int(r['n_b_wins']) if 'n_b_wins' in r.index and pd.notna(r.get('n_b_wins')) else -1
        win_txt = f'  wins {n_a}:{n_b}' if n_a >= 0 else ''
        lines.append(
            f"{a} vs {b}: Δ(B−A)={r['diff']:+.3f} "
            f"[{r['diff_ci_lo']:+.3f},{r['diff_ci_hi']:+.3f}] "
            f"better={better} p={r['p']:.4g}{sig}{win_txt}"
        )

    if include_model4_cont and (out_dir / 'model4_youden_vs_continuous_bootstrap.csv').is_file():
        m4 = pd.read_csv(out_dir / 'model4_youden_vs_continuous_bootstrap.csv').iloc[0]
        lines += [
            '', '4. Model4 Youden vs continuous (MC permutation sign test)', '-' * 24,
            f"MC mean AUROC: Youden={m4['mc_auroc_youden']:.3f}  "
            f"Continuous={m4['mc_auroc_continuous']:.3f}",
            f"Δ(cont−youden)={m4['mc_auroc_diff_cont_minus_youden']:+.4f} "
            f"[{m4.get('mc_diff_ci_lo', m4['bagged_auc_diff_ci_lo']):+.4f},"
            f"{m4.get('mc_diff_ci_hi', m4['bagged_auc_diff_ci_hi']):+.4f}] "
            f"better={m4.get('better_model', '')} "
            f"p={m4.get('p_mc_sign_two_sided', m4['p_bootstrap_two_sided']):.4g}",
        ]

    lines += ['', '5. Bagged OOS by prior 2yr sz freq info (available vs unavailable)', '-' * 24]
    for model in prior_delong['model'].unique():
        d = prior_delong[prior_delong['model'] == model].iloc[0]
        av = prior[(prior['model'] == model) & (prior['stratum'] == 'available')].iloc[0]
        un = prior[(prior['model'] == model) & (prior['stratum'] == 'unavailable')].iloc[0]
        lines.append(f"{model}:")
        lines.append(
            f"  available   N={int(av['n_patients'])} AUC={av['bagged_auc']:.3f} "
            f"SE={av.get('delong_se', float('nan')):.4f} "
            f"95%CI=[{av.get('delong_ci_lo', float('nan')):.3f},"
            f"{av.get('delong_ci_hi', float('nan')):.3f}] "
            f"AUPRC={av['bagged_auprc']:.3f}"
        )
        lines.append(
            f"  unavailable N={int(un['n_patients'])} AUC={un['bagged_auc']:.3f} "
            f"SE={un.get('delong_se', float('nan')):.4f} "
            f"95%CI=[{un.get('delong_ci_lo', float('nan')):.3f},"
            f"{un.get('delong_ci_hi', float('nan')):.3f}] "
            f"AUPRC={un['bagged_auprc']:.3f}"
        )
        lines.append(
            f"  DeLong Δ(avail−unavail)={d['delta_available_minus_unavailable']:+.3f} "
            f"SE={d['se']:.4f} 95%CI=[{d['ci_lo']:.3f},{d['ci_hi']:.3f}] p={d['p']:.4g}"
        )

    lines += ['', '6. Bagged OOS by Youden IED stratum (Models 2–4)', '-' * 24]
    for model in ied['model'].unique():
        pr_df = ied[(ied['model'] == model) & (ied['stratum'] == 'present')]
        ab_df = ied[(ied['model'] == model) & (ied['stratum'] == 'absent')]
        if pr_df.empty or ab_df.empty:
            continue
        pr = pr_df.iloc[0]
        ab = ab_df.iloc[0]
        lines.append(
            f"{model}: present AUC={pr['bagged_auc']:.3f}  absent AUC={ab['bagged_auc']:.3f}"
        )

    lines += ['', '7. Bagged OOS by sex (all Youden models)', '-' * 24]
    for model in sex['model'].unique():
        f = sex[(sex['model'] == model) & (sex['stratum'] == 'female')].iloc[0]
        m = sex[(sex['model'] == model) & (sex['stratum'] == 'male')].iloc[0]
        lines.append(
            f"{model}: female AUC={f['bagged_auc']:.3f}  male AUC={m['bagged_auc']:.3f}"
        )

    lines += ['', 'See CSV/PNG files in OUT_DIR for full tables, ROC curves, subtype DeLong.']
    path = out_dir / 'aim2_results_summary.txt'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Saved: {path}')
    return path


def save_table(df: pd.DataFrame, filename: str, out_dir: Path | None = None) -> Path:
    dest = Path(out_dir) if out_dir is not None else OUT_DIR
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / filename
    df.to_csv(path, index=False)
    print(f'  Saved {path}')
    return path


def plot_mc_metric_boxplots(mc_df: pd.DataFrame, out_dir: Path) -> None:
    """Boxplots of per-MC AUC/AUPRC/F1 for each model (best algorithm locked)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics = [
        ('auroc', 'AUROC'),
        ('auprc', 'AUPRC'),
        ('f1', 'F1 (threshold=0.5)'),
    ]
    models = [m for m in MODEL_ORDER if m in set(mc_df['model'])]
    for col, label in metrics:
        if col not in mc_df.columns:
            continue
        data = [mc_df.loc[mc_df['model'] == m, col].dropna().values for m in models]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(data, labels=models, showfliers=False)
        ax.set_ylabel(label)
        ax.set_title(f'MC {label} by model (best algorithm per model)')
        ax.tick_params(axis='x', rotation=20)
        fig.tight_layout()
        path = out_dir / f'mc_{col}_boxplot.png'
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {path}')


def plot_mc_auroc_auprc_box_swarm(mc_df: pd.DataFrame, out_dir: Path) -> Path:
    """Box + swarm via MATLAB (same style as make_fig_general_slowing_by_subtype.m)."""
    del mc_df  # MATLAB reads mc_iteration_metrics.csv from out_dir
    out_dir = Path(out_dir)
    out_png = out_dir / 'mc_auroc_auprc_bars_dots_models1to4.png'
    metrics_csv = out_dir / 'mc_iteration_metrics.csv'
    mfile_dir = Path(__file__).resolve().parent
    matlab = 'matlab'
    cmd = (
        f"cd('{mfile_dir}'); "
        f"make_fig_aim2_mc_auroc_auprc_bars("
        f"'MetricsCsv','{metrics_csv}',"
        f"'OutPng','{out_png}',"
        f"'SavePng',true);"
    )
    import subprocess
    try:
        subprocess.run(
            [matlab, '-batch', cmd],
            check=True,
            cwd=str(mfile_dir),
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f'  MATLAB figure failed ({e}); falling back to matplotlib')
        return _plot_mc_auroc_auprc_matplotlib(out_dir, out_png)
    print(f'  Saved {out_png}')
    return out_png


def _plot_mc_auroc_auprc_matplotlib(out_dir: Path, out_png: Path) -> Path:
    """Fallback if MATLAB unavailable."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mc_df = pd.read_csv(Path(out_dir) / 'mc_iteration_metrics.csv')
    order = [
        'Model1_clinical', 'Model2_IED', 'Model3_slow', 'Model4_IED_slow',
    ]
    labels = ['Clinical', 'IED', 'Slowing', 'Full model']
    sub = mc_df[mc_df['model'].isin(order)].copy()
    prev = 0.365
    prev_path = Path(out_dir) / 'bagged_metrics_by_prior_availability.csv'
    if prev_path.is_file():
        b = pd.read_csv(prev_path)
        hit = b[(b['model'] == 'Model4_IED_slow') & (b['stratum'] == 'all')]
        if len(hit):
            prev = float(hit.iloc[0]['prevalence'])

    c_auc = (0.30, 0.47, 0.66)
    c_ap = (0.80, 0.47, 0.27)
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(12, 8.6), facecolor='white')
    plt.rcParams.update({'font.size': 18, 'font.family': 'sans-serif'})

    def _one(ax_, x0, y, color):
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if len(y) < 2:
            return
        # violin-ish density via many jittered points
        parts = ax_.violinplot(
            [y], positions=[x0], widths=0.55, showmeans=False,
            showmedians=False, showextrema=False,
        )
        for b in parts['bodies']:
            b.set_facecolor(color)
            b.set_alpha(0.18)
            b.set_edgecolor(color)
        ax_.boxplot(
            [y], positions=[x0], widths=0.22, showfliers=False,
            patch_artist=True, zorder=3,
            medianprops=dict(color='k', linewidth=1.5),
            whiskerprops=dict(color=color, linewidth=1.2),
            capprops=dict(color=color, linewidth=1.2),
            boxprops=dict(facecolor=(*color, 0.25), edgecolor=color, linewidth=1.2),
        )
        # swarm overlay
        n_bins = max(20, int(np.sqrt(len(y))))
        edges = np.linspace(y.min(), y.max() + 1e-12, n_bins + 1)
        bins = np.digitize(y, edges) - 1
        x = np.zeros(len(y))
        for b in range(n_bins):
            idx = np.where(bins == b)[0]
            m = len(idx)
            if m == 0:
                continue
            half = min(0.22, 0.02 + 0.01 * m)
            x[idx] = x0 + (np.linspace(-half, half, m) if m > 1 else 0.0)
        ax_.scatter(x, y, s=10, c=[color], alpha=0.22, linewidths=0, zorder=2)

    for k, m in enumerate(order):
        x_base = 1 + k * 2
        _one(ax, x_base, sub.loc[sub['model'] == m, 'auroc'], c_auc)
        _one(ax, x_base + 1, sub.loc[sub['model'] == m, 'auprc'], c_ap)

    ax.axhline(0.5, color=c_auc, ls='--', lw=1.6)
    ax.axhline(prev, color=c_ap, ls='--', lw=1.6)
    ax.set_xlim(0.3, 8.7)
    ax.set_ylim(0.3, 1.0)
    ax.set_xticks([1.5, 3.5, 5.5, 7.5])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    ax.set_xlabel('Model')
    ax.grid(True, axis='y', color='0.85')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved {out_png}')
    return out_png


def _append_stratum_metrics(
    bagged: dict,
    model_keys: list[str],
    stratum_col: str,
    strata: tuple[str, ...],
    selection_label: str,
    overall_metric_rows: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Bagged AUC/AUPRC metrics for each stratum × model."""
    rows: list[dict] = list(overall_metric_rows or [])
    delong_rows: list[dict] = []
    if len(strata) != 2:
        raise ValueError('Delong compare currently supports exactly two strata')
    s0, s1 = strata
    for model_key in model_keys:
        bag = bagged[model_key]
        for stratum in strata:
            row = stratum_bagged_metrics(bag, stratum, stratum_col=stratum_col)
            row['model'] = model_key
            row['selection_mode'] = selection_label
            row['stratum_col'] = stratum_col
            rows.append(row)
            print(
                f"  {model_key} | {stratum_col}={stratum}: N={row['n_patients']} "
                f"(pos={row['n_pos']}, prev={row['prevalence']:.3f})  "
                f"AUC={row['bagged_auc']:.3f}  AUPRC={row['bagged_auprc']:.3f}"
            )
        y = bag['y_true']
        p_pred = bag['bagged_pred']
        labels = np.asarray(bag[stratum_col], dtype=str)
        ok = np.isfinite(p_pred)
        m0 = ok & (labels == s0)
        m1 = ok & (labels == s1)
        cmp_ = unpaired_delong_compare(y[m0], p_pred[m0], y[m1], p_pred[m1])
        delong_rows.append({
            'model': model_key,
            'selection_mode': selection_label,
            'stratum_col': stratum_col,
            f'auc_{s0}': cmp_['auc_a'],
            f'auc_{s1}': cmp_['auc_b'],
            f'var_{s0}': cmp_['var_a'],
            f'var_{s1}': cmp_['var_b'],
            f'delta_{s0}_minus_{s1}': cmp_['delta'],
            'se': cmp_['se'],
            'ci_lo': cmp_['ci_lo'],
            'ci_hi': cmp_['ci_hi'],
            'p': cmp_['p'],
            f'n_{s0}': cmp_['n_a'],
            f'n_{s1}': cmp_['n_b'],
        })
        print(
            f"  {model_key} | {s0} vs {s1} ΔAUC={cmp_['delta']:.3f}  "
            f"95% CI=[{cmp_['ci_lo']:.3f}, {cmp_['ci_hi']:.3f}]  p={cmp_['p']:.4g}"
        )
    return rows, delong_rows


def run_post_mc_analyses(
    mc_results: list,
    analysis_df: pd.DataFrame,
    out_dir: Path,
    selection_label: str,
    selected_by_model: dict[str, str] | None = None,
    bagged_model_keys: list[str] | None = None,
    prior_stratum_models: list[str] | None = None,
    pairwise_model_keys: list[str] | None = None,
) -> None:
    """MC summaries, bagged OOS, pairwise, prior/IED/subtype strata → out_dir."""
    print(f'\n######## Reporting mode: {selection_label} → {out_dir} ########')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_by_model = selected_by_model or {}
    bagged_model_keys = bagged_model_keys or list(MODEL_ORDER)
    prior_stratum_models = prior_stratum_models or list(MODEL_ORDER)
    pairwise_model_keys = pairwise_model_keys or list(MODEL_ORDER)

    mc_rows = []
    roc_frames = []
    pr_frames = []
    for res in mc_results:
        if res is None:
            continue
        for mk in FEATURE_SETS:
            mc_rows.append({
                'seed': res['seed'],
                'model': mk,
                'selection_mode': selection_label,
                'best_clf': res['best_clf'][mk],
                'auroc': res['metrics'][mk]['auroc'],
                'auprc': res['metrics'][mk]['auprc'],
                'f1': res['metrics'][mk]['f1'],
                'n_test': len(res['y_test']),
            })
            roc_df = res['roc_curves'].get(mk)
            pr_df = res['pr_curves'].get(mk)
            if roc_df is not None and len(roc_df):
                roc_frames.append(roc_df)
            if pr_df is not None and len(pr_df):
                pr_frames.append(pr_df)
    save_table(pd.DataFrame(mc_rows), 'mc_iteration_metrics.csv', out_dir)
    plot_mc_metric_boxplots(pd.DataFrame(mc_rows), out_dir)
    plot_mc_auroc_auprc_box_swarm(pd.DataFrame(mc_rows), out_dir)
    if roc_frames:
        roc_all = pd.concat(roc_frames, ignore_index=True)
        save_table(roc_all, 'mc_roc_curves.csv', out_dir)
        for mk in ROC_EXPORT_MODELS:
            sub = roc_all.loc[roc_all['model'] == mk]
            if len(sub):
                n_seeds = int(sub['seed'].nunique())
                save_table(sub, f'mc_roc_curves_{mk}.csv', out_dir)
                print(f'  ROC export {mk}: {n_seeds} MC seeds, {len(sub)} curve points')
    if pr_frames:
        save_table(pd.concat(pr_frames, ignore_index=True), 'mc_pr_curves.csv', out_dir)
    print(f'  Saved per-MC ROC/PR curve points '
          f'({len(roc_frames)} ROC tables, {len(pr_frames)} PR tables)')

    summary_rows = []
    bagged = {}
    bagged_df = None
    overall_metric_rows = []

    for model_key in bagged_model_keys:
        print(f'\n--- {model_key} ---')
        summ = summarize_mc_metrics(mc_results, model_key, rng=np.random.default_rng(0))
        summ['selection_mode'] = selection_label
        summary_rows.append(summ)
        print(f"  MC AUROC  mean={summ['auc_mean']:.3f}  "
              f"95% CI=[{summ['auc_ci_lo']:.3f}, {summ['auc_ci_hi']:.3f}]")
        print(f"  MC AUPRC  mean={summ['auprc_mean']:.3f}  "
              f"95% CI=[{summ['auprc_ci_lo']:.3f}, {summ['auprc_ci_hi']:.3f}]")
        print(f"  MC F1     mean={summ['f1_mean']:.3f}  "
              f"95% CI=[{summ['f1_ci_lo']:.3f}, {summ['f1_ci_hi']:.3f}]")

        bag = bag_oos_predictions(mc_results, analysis_df, model_key)
        bagged[model_key] = bag
        ok = np.isfinite(bag['bagged_pred'])
        delong = bagged_auc_delong_stats(bag['y_true'][ok], bag['bagged_pred'][ok])
        overall = {
            'model': model_key,
            'stratum': 'all',
            'selection_mode': selection_label,
            'selected_algorithm': selected_by_model.get(model_key, ''),
            'n_patients': int(ok.sum()),
            'n_pos': int(bag['y_true'][ok].sum()),
            'prevalence': float(bag['y_true'][ok].mean()) if ok.any() else np.nan,
            'bagged_auc': delong['bagged_auc'],
            'delong_se': delong['delong_se'],
            'delong_var': delong['delong_var'],
            'delong_ci_lo': delong['delong_ci_lo'],
            'delong_ci_hi': delong['delong_ci_hi'],
            'bagged_auprc': _safe_auprc(bag['y_true'][ok], bag['bagged_pred'][ok]),
            'n_heldout_mean': float(np.mean(bag['n_heldout'][ok])) if ok.any() else np.nan,
            'best_clf_mode': max(bag['clf_votes'], key=bag['clf_votes'].get) if bag['clf_votes'] else '',
            'clf_votes': str(bag['clf_votes']),
        }
        overall_metric_rows.append(overall)
        print(f"  Best-clf votes: {bag['clf_votes']}")
        print(
            f"  Bagged overall AUC={overall['bagged_auc']:.3f}  "
            f"DeLong SE={overall['delong_se']:.4f}  "
            f"95% CI=[{overall['delong_ci_lo']:.3f}, {overall['delong_ci_hi']:.3f}]  "
            f"AUPRC={overall['bagged_auprc']:.3f}"
        )

        if bagged_df is None:
            bagged_df = pd.DataFrame({
                'patient_id': bag['patient_id'],
                'y_true': bag['y_true'],
                'ied_youden': bag['ied_youden'],
                'sex_stratum': bag['sex_stratum'],
                'epilepsy_subtype': bag['epilepsy_subtype'],
                'prior_availability': bag['prior_availability'],
                'n_heldout': bag['n_heldout'],
            })
        bagged_df[f'bagged_pred_{model_key}'] = bag['bagged_pred']

    save_table(bagged_df, 'bagged_oos_predictions.csv', out_dir)
    save_table(pd.DataFrame(summary_rows), 'mc_metrics_summary.csv', out_dir)
    save_table(pd.DataFrame(summary_rows), 'mc_auc_auprc_summary.csv', out_dir)

    mc_df = pd.DataFrame(mc_rows)

    print('\n--- Pairwise model comparisons (MC permutation sign test on test AUROC) ---')
    mc_pair_rows = pairwise_paper_comparisons(
        mc_df, bagged, pairwise_model_keys, FEATURE_SETS, rng=np.random.default_rng(1),
    )
    for r in mc_pair_rows:
        r['selection_mode'] = selection_label
        print(
            f"  {r['model_a']} vs {r['model_b']}: "
            f"Δ(B−A)={r['diff']:+.3f}  "
            f"95% CI=[{r['diff_ci_lo']:+.3f}, {r['diff_ci_hi']:+.3f}]  "
            f"better={r['better_model']}  p={r['p']:.4g}  "
            f"(wins {r['n_a_wins']}:{r['n_b_wins']}, n_mc={r['n_splits']})"
        )
    save_table(pd.DataFrame(mc_pair_rows), 'pairwise_model_mc_splits.csv', out_dir)
    # Primary pairwise table for summary (same content)
    save_table(pd.DataFrame(mc_pair_rows), 'pairwise_model_bootstrap.csv', out_dir)

    print('\n--- Bagged OOS pairwise (supplementary descriptive only; not used for p) ---')
    pair_rows = pairwise_model_comparisons(
        bagged, pairwise_model_keys, FEATURE_SETS,
        mask=None, stratum='all', rng=np.random.default_rng(1),
    )
    for r in pair_rows:
        r['selection_mode'] = selection_label
        print(
            f"  {r['model_complex']} − {r['model_simple']}: "
            f"ΔAUC={r['diff']:.3f}  "
            f"95% CI=[{r['diff_ci_lo']:.3f}, {r['diff_ci_hi']:.3f}]  "
            f"p={r['p']:.4g}  "
            f"(bagged AUC {r['auc_simple']:.3f} vs {r['auc_complex']:.3f}; "
            f"AUPRC {r['auprc_simple']:.3f} vs {r['auprc_complex']:.3f})"
        )
    save_table(pd.DataFrame(pair_rows), 'pairwise_model_bagged_bootstrap_supplementary.csv', out_dir)

    print('\n--- Bagged OOS by prior 2yr sz-control availability '
          f'(models: {prior_stratum_models}) ---')
    prior_stratum_rows = list(overall_metric_rows)
    prior_delong_rows = []
    for model_key in prior_stratum_models:
        bag = bagged[model_key]
        for stratum in ('available', 'unavailable'):
            row = stratum_bagged_metrics(bag, stratum, stratum_col='prior_availability')
            row['model'] = model_key
            row['selection_mode'] = selection_label
            prior_stratum_rows.append(row)
            print(
                f"  {model_key} | prior {stratum}: N={row['n_patients']} "
                f"(pos={row['n_pos']}, prev={row['prevalence']:.3f})  "
                f"AUC={row['bagged_auc']:.3f}  "
                f"DeLong SE={row['delong_se']:.4f}  "
                f"95% CI=[{row['delong_ci_lo']:.3f}, {row['delong_ci_hi']:.3f}]  "
                f"AUPRC={row['bagged_auprc']:.3f}"
            )

        y = bag['y_true']
        p_pred = bag['bagged_pred']
        pav = bag['prior_availability']
        ok = np.isfinite(p_pred)
        av = ok & (pav == 'available')
        un = ok & (pav == 'unavailable')
        cmp_ = unpaired_delong_compare(y[av], p_pred[av], y[un], p_pred[un])
        prior_delong_rows.append({
            'model': model_key,
            'selection_mode': selection_label,
            'auc_prior_available': cmp_['auc_a'],
            'auc_prior_unavailable': cmp_['auc_b'],
            'var_available': cmp_['var_a'],
            'var_unavailable': cmp_['var_b'],
            'delta_available_minus_unavailable': cmp_['delta'],
            'se': cmp_['se'],
            'ci_lo': cmp_['ci_lo'],
            'ci_hi': cmp_['ci_hi'],
            'p': cmp_['p'],
            'n_available': cmp_['n_a'],
            'n_unavailable': cmp_['n_b'],
        })
        print(
            f"  {model_key} | available vs unavailable ΔAUC={cmp_['delta']:.3f}  "
            f"95% CI=[{cmp_['ci_lo']:.3f}, {cmp_['ci_hi']:.3f}]  p={cmp_['p']:.4g}"
        )

    print('\n--- Pairwise model comparisons within prior-availability strata ---')
    ref_prior = bagged[prior_stratum_models[0]]['prior_availability']
    prior_pair_rows = []
    for stratum in ('available', 'unavailable'):
        mask = ref_prior == stratum
        stratum_pairs = pairwise_model_comparisons(
            bagged, prior_stratum_models, FEATURE_SETS,
            mask=mask, stratum=f'prior_{stratum}',
            rng=np.random.default_rng(4 if stratum == 'available' else 5),
        )
        for r in stratum_pairs:
            r['selection_mode'] = selection_label
        prior_pair_rows.extend(stratum_pairs)
        for r in stratum_pairs:
            print(
                f"  [{stratum}] {r['model_complex']} − {r['model_simple']}: "
                f"ΔAUC={r['diff']:.3f}  "
                f"95% CI=[{r['diff_ci_lo']:.3f}, {r['diff_ci_hi']:.3f}]  "
                f"p={r['p']:.4g}  (N={r['n_patients']})"
            )

    save_table(pd.DataFrame(prior_stratum_rows), 'bagged_metrics_by_prior_availability.csv', out_dir)
    save_table(pd.DataFrame(prior_delong_rows), 'prior_available_vs_unavailable_delong.csv', out_dir)
    save_table(pd.DataFrame(prior_pair_rows), 'pairwise_model_bootstrap_prior_strata.csv', out_dir)

    print('\n--- Bagged OOS by Youden IED feature '
          f'(models: {IED_STRATUM_MODELS}) ---')
    ied_stratum_rows, ied_delong_rows = _append_stratum_metrics(
        bagged, IED_STRATUM_MODELS, IED_STRATUM_COL, IED_STRATA, selection_label,
        overall_metric_rows=overall_metric_rows,
    )

    print('\n--- Pairwise model comparisons within Youden IED strata ---')
    ref_ied = bagged[IED_STRATUM_MODELS[0]][IED_STRATUM_COL]
    ied_pair_rows = []
    for stratum in IED_STRATA:
        mask = ref_ied == stratum
        stratum_pairs = pairwise_model_comparisons(
            bagged, IED_STRATUM_MODELS, FEATURE_SETS,
            mask=mask, stratum=f'ied_youden_{stratum}',
            rng=np.random.default_rng(2 if stratum == 'present' else 3),
        )
        for r in stratum_pairs:
            r['selection_mode'] = selection_label
        ied_pair_rows.extend(stratum_pairs)
        for r in stratum_pairs:
            print(
                f"  [{stratum}] {r['model_complex']} − {r['model_simple']}: "
                f"ΔAUC={r['diff']:.3f}  "
                f"95% CI=[{r['diff_ci_lo']:.3f}, {r['diff_ci_hi']:.3f}]  "
                f"p={r['p']:.4g}  (N={r['n_patients']})"
            )

    save_table(pd.DataFrame(ied_pair_rows), 'pairwise_model_bagged_bootstrap_ied_strata.csv', out_dir)
    save_table(pd.DataFrame(ied_stratum_rows), 'bagged_metrics_by_ied_youden_stratum.csv', out_dir)
    save_table(pd.DataFrame(ied_delong_rows), 'ied_youden_present_vs_absent_delong.csv', out_dir)

    print('\n--- Bagged OOS by sex '
          f'(models: {SEX_STRATUM_MODELS}) ---')
    sex_stratum_rows, sex_delong_rows = _append_stratum_metrics(
        bagged, SEX_STRATUM_MODELS, SEX_STRATUM_COL, SEX_STRATA, selection_label,
    )
    print('\n--- Pairwise model comparisons within sex strata ---')
    ref_sex = bagged[SEX_STRATUM_MODELS[0]][SEX_STRATUM_COL]
    sex_pair_rows = []
    for stratum in SEX_STRATA:
        mask = ref_sex == stratum
        stratum_pairs = pairwise_model_comparisons(
            bagged, SEX_STRATUM_MODELS, FEATURE_SETS,
            mask=mask, stratum=f'sex_{stratum}',
            rng=np.random.default_rng(6 if stratum == 'female' else 7),
        )
        for r in stratum_pairs:
            r['selection_mode'] = selection_label
        sex_pair_rows.extend(stratum_pairs)
        for r in stratum_pairs:
            print(
                f"  [{stratum}] {r['model_complex']} − {r['model_simple']}: "
                f"ΔAUC={r['diff']:.3f}  "
                f"95% CI=[{r['diff_ci_lo']:.3f}, {r['diff_ci_hi']:.3f}]  "
                f"p={r['p']:.4g}  (N={r['n_patients']})"
            )
    save_table(pd.DataFrame(sex_stratum_rows), 'bagged_metrics_by_sex_stratum.csv', out_dir)
    save_table(pd.DataFrame(sex_delong_rows), 'sex_female_vs_male_delong.csv', out_dir)
    save_table(pd.DataFrame(sex_pair_rows), 'pairwise_model_bootstrap_sex_strata.csv', out_dir)

    print(f'\n--- Bagged OOS by epilepsy subtype ({SUBTYPE_MODEL} only) ---')
    print(f'  Kept subtypes: {SUBTYPE_KEEP}')
    print(f'  Dropped: {SUBTYPE_DROP}')
    bag1 = bagged[SUBTYPE_MODEL]
    subtype_metric_rows = []
    for s in SUBTYPE_KEEP:
        row = stratum_bagged_metrics(bag1, s, stratum_col='epilepsy_subtype')
        row['model'] = SUBTYPE_MODEL
        row['selection_mode'] = selection_label
        subtype_metric_rows.append(row)
        print(
            f"  {s}: N={row['n_patients']} (pos={row['n_pos']}, "
            f"prev={row['prevalence']:.3f})  "
            f"AUC={row['bagged_auc']:.3f}  "
            f"DeLong SE={row['delong_se']:.4f}  "
            f"95% CI=[{row['delong_ci_lo']:.3f}, {row['delong_ci_hi']:.3f}]  "
            f"AUPRC={row['bagged_auprc']:.3f}"
        )

    print('\n--- Pairwise subtype DeLong (unpaired) + Bonferroni ---')
    subtype_pair_rows = pairwise_subtype_delong(bag1, SUBTYPE_KEEP, SUBTYPE_MODEL)
    for r in subtype_pair_rows:
        r['selection_mode'] = selection_label
        print(
            f"  {r['subtype_a']} vs {r['subtype_b']}: "
            f"ΔAUC={r['delta_a_minus_b']:.3f}  "
            f"95% CI=[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  "
            f"p={r['p']:.4g}  p_bonf={r['p_bonferroni']:.4g}  "
            f"(AUC {r['auc_a']:.3f} vs {r['auc_b']:.3f}; "
            f"N={r['n_a']}/{r['n_b']})"
        )
    save_table(pd.DataFrame(subtype_metric_rows), 'bagged_metrics_by_subtype.csv', out_dir)
    save_table(pd.DataFrame(subtype_pair_rows), 'subtype_pairwise_delong.csv', out_dir)

    
    print('\n=== MC metrics summary (AUROC / AUPRC / F1) ===')
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print('\n=== Pairwise bagged bootstrap (stratum=all) ===')
    print(pd.DataFrame(pair_rows).query("stratum == 'all'")[[
        'model_simple', 'model_complex', 'auc_simple', 'auc_complex',
        'diff', 'diff_ci_lo', 'diff_ci_hi', 'p',
        'auprc_simple', 'auprc_complex',
    ]].to_string(index=False))
    print('\n=== Bagged metrics by prior availability ===')
    print(pd.DataFrame(prior_stratum_rows).to_string(index=False))
    print('\n=== Bagged metrics by Youden IED feature ===')
    print(pd.DataFrame(ied_stratum_rows).to_string(index=False))
    print('\n=== Bagged metrics by sex ===')
    print(pd.DataFrame(sex_stratum_rows).to_string(index=False))
    print(f'\n=== Bagged metrics by subtype ({SUBTYPE_MODEL}) ===')
    print(pd.DataFrame(subtype_metric_rows).to_string(index=False))
    if subtype_pair_rows:
        print('\n=== Subtype pairwise DeLong + Bonferroni ===')
        print(pd.DataFrame(subtype_pair_rows)[[
            'subtype_a', 'subtype_b', 'auc_a', 'auc_b',
            'delta_a_minus_b', 'ci_lo', 'ci_hi', 'p', 'p_bonferroni',
        ]].to_string(index=False))
    print(f'\nDone reporting [{selection_label}] → {out_dir}')


def main() -> None:
    global FEATURE_SETS, MODEL_ORDER, PRIOR_STRATUM_MODELS, SEX_STRATUM_MODELS

    p = argparse.ArgumentParser(description='Aim 2 Generalized-only ML')
    p.add_argument('--n-mc', type=int, default=N_MC)
    p.add_argument('--n-jobs', type=int, default=N_JOBS)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--poor-cutoff', type=float, default=POOR_CUTOFF_DEFAULT,
                   help='Poor if post-EEG mean > cutoff/mo; prior matched to same cutoff')
    p.add_argument('--continuous-markers', action='store_true',
                   help='Use continuous log IED + pct slowing instead of Youden binary')
    p.add_argument(
        '--model4-continuous-compare', action='store_true',
        help=(
            'Youden 4-model run plus Model4 continuous on same MC splits; '
            'paired bootstrap Youden vs continuous full model'
        ),
    )
    args = p.parse_args()
    if args.continuous_markers and args.model4_continuous_compare:
        p.error('Use --model4-continuous-compare OR --continuous-markers, not both')
    binary_markers = not bool(args.continuous_markers)

    youden_feature_sets = build_feature_sets(binary_markers=True)
    FEATURE_SETS = dict(youden_feature_sets)
    MODEL_ORDER = list(youden_feature_sets.keys())
    mc_model_keys = list(MODEL_ORDER)
    if args.model4_continuous_compare:
        if not binary_markers:
            p.error('--model4-continuous-compare requires Youden markers (default)')
        FEATURE_SETS[MODEL4_CONTINUOUS] = model4_continuous_feature_cols()
        mc_model_keys = list(FEATURE_SETS.keys())
    elif not binary_markers:
        FEATURE_SETS = build_feature_sets(binary_markers=False)
        MODEL_ORDER = list(FEATURE_SETS.keys())
        mc_model_keys = list(MODEL_ORDER)

    PRIOR_STRATUM_MODELS = list(MODEL_ORDER)
    SEX_STRATUM_MODELS = list(MODEL_ORDER)

    print('=' * 80)
    print('Aim 2 — Generalized epilepsy only (poor>cutoff + matched prior + IED/slowing)')
    print(f'OUT_DIR:  {OUT_DIR}')
    print(f'N_MC:     {args.n_mc}  (80/20, seed {args.seed_start}..{args.seed_start + args.n_mc - 1})')
    print(f'N_JOBS:   {args.n_jobs}  |  inner CV folds: {N_CV_FOLDS}')
    print(f'Outcome:  poor if mean > {args.poor_cutoff}/mo; else moderate/else (incl SF)')
    print(f'Prior:    matched to cutoff {args.poor_cutoff}/mo (not_poor / poor / NA)')
    print(
        f'Markers:  {"Youden binary" if binary_markers else "continuous"} '
        f'(IED>={st.SPIKE_YOUDEN_THRESH}, slowing>={st.SLOWING_YOUDEN_THRESH_PCT}% when binary)'
    )
    if args.model4_continuous_compare:
        print(
            f'Also fit {MODEL4_CONTINUOUS} on same splits: '
            f'{model4_continuous_feature_cols()}'
        )
        print(
            'ROC exports: Model1_clinical, Model4 Youden, Model4 continuous '
            '(Model5 = Aim1 pooled interaction — run separately)'
        )
    print('Classifiers: lr, svm, sgd (GridSearch; L2 only × C/alpha)')
    print('Per feature model: pick its own best of {lr,svm,sgd} by mean MC AUROC')
    print('MC metrics: AUROC, AUPRC, F1 (threshold=0.5) → CSV + boxplots')
    print('OOS strata: prior available/unavailable | Youden IED | subtype | sex')
    print(f'Subtype DeLong: {SUBTYPE_MODEL} only; keep={SUBTYPE_KEEP}; drop={SUBTYPE_DROP}')
    print('=' * 80)

    patient_tables = st._load_combined_data()
    analysis_df = build_analysis_df(
        patient_tables['sz_freq_class'],
        poor_cutoff=args.poor_cutoff,
        binary_markers=binary_markers,
    )

    ied_ann = load_patient_ied_annotation(analysis_df['patient_id'])
    analysis_df = analysis_df.merge(ied_ann, on='patient_id', how='left')
    analysis_df['ied_annotation'] = analysis_df['ied_annotation'].fillna('unknown')

    sub_ann = load_patient_epilepsy_subtype(analysis_df['patient_id'])
    analysis_df = analysis_df.merge(
        sub_ann[['patient_id', 'epilepsy_subtype', 'subtype_raw']],
        on='patient_id', how='left',
    )
    analysis_df['epilepsy_subtype'] = analysis_df['epilepsy_subtype'].fillna('Unclassified')

    n_pre = int((analysis_df['ied_youden'] == 'present').sum())
    n_abs = int((analysis_df['ied_youden'] == 'absent').sum())
    n_prior_av = int((analysis_df['prior_availability'] == 'available').sum())
    n_prior_un = int((analysis_df['prior_availability'] == 'unavailable').sum())
    n_fem = int((analysis_df['sex_stratum'] == 'female').sum())
    n_male = int((analysis_df['sex_stratum'] == 'male').sum())
    print(f'\nAnalysis cohort: N={len(analysis_df)} Generalized, '
          f'prevalence={analysis_df["sz_freq_class"].mean():.3f}')
    print(f'  Youden IED: present={n_pre}, absent={n_abs}')
    print(f'  Prior 2yr control: available={n_prior_av}, unavailable={n_prior_un}')
    print(f'  Sex: female={n_fem}, male={n_male}')
    for mk, cols in FEATURE_SETS.items():
        print(f'  {mk}: {len(cols)} features — {cols}')

    cohort_cols = [
        'patient_id', 'sz_freq_class', 'ied_youden', 'ied_annotation',
        'epilepsy_subtype', 'subtype_raw', 'sex_stratum',
        'age_at_eeg', 'gender', 'prior_sz_control',
        'prior_sz_poor', 'prior_sz_unavailable', 'prior_availability',
        'spikes', 'slowing_pct', 'pct_foc_slowing', 'pct_gen_slowing',
    ]
    if binary_markers and 'slowing' in analysis_df.columns:
        cohort_cols.append('slowing')
    save_table(analysis_df[[c for c in cohort_cols if c in analysis_df.columns]].copy(), 'analysis_cohort.csv')

    print(f'\nRunning {args.n_mc} MC iterations...')
    seeds = range(args.seed_start, args.seed_start + args.n_mc)
    gen = Parallel(n_jobs=args.n_jobs, return_as='generator')(
        delayed(run_mc_iteration)(seed, analysis_df, FEATURE_SETS) for seed in seeds
    )
    mc_results = list(tqdm(gen, total=args.n_mc, desc='MC 80/20', unit='split'))
    n_ok = sum(r is not None for r in mc_results)
    print(f'  Kept {n_ok}/{args.n_mc} successful iterations')

    selected_by_model, clf_rank = select_clf_per_model(mc_results, mc_model_keys)
    save_table(clf_rank, 'algorithm_auc_ranking_by_model.csv', OUT_DIR)
    print('\n=== Per-model algorithm selection (mean MC AUROC among lr/svm/sgd) ===')
    for mk in mc_model_keys:
        sub = clf_rank[clf_rank['model'] == mk]
        print(f'  {mk}:')
        for _, row in sub.iterrows():
            mark = ' ← SELECTED' if row['algorithm'] == selected_by_model[mk] else ''
            print(f"    {row['algorithm']:4s}  mean AUC={row['auc_mean']:.4f}{mark}")
    with open(OUT_DIR / 'selected_algorithms_per_model.txt', 'w', encoding='utf-8') as f:
        for mk in mc_model_keys:
            f.write(f'{mk}\t{selected_by_model[mk]}\n')

    apply_selected_clfs_per_model(mc_results, FEATURE_SETS, selected_by_model)

    bagged_model_keys = list(MODEL_ORDER)
    prior_stratum_models = list(MODEL_ORDER)
    if args.model4_continuous_compare:
        bagged_model_keys.append(MODEL4_CONTINUOUS)
        prior_stratum_models.append(MODEL4_CONTINUOUS)
        print(
            f'ROC curve exports (all 1000 MC splits): '
            f'{list(ROC_EXPORT_MODELS)}'
        )

    run_post_mc_analyses(
        mc_results, analysis_df,
        out_dir=OUT_DIR,
        selection_label='per_model_best_of_lr_svm_sgd',
        selected_by_model=selected_by_model,
        bagged_model_keys=bagged_model_keys,
        prior_stratum_models=prior_stratum_models,
        pairwise_model_keys=list(MODEL_ORDER),
    )

    if args.model4_continuous_compare:
        compare_model4_youden_vs_continuous(
            mc_results, analysis_df, selected_by_model, OUT_DIR,
        )

    write_aim2_results_summary(
        OUT_DIR, include_model4_cont=bool(args.model4_continuous_compare),
    )

    print(f'\nAll done. Outputs: {OUT_DIR}')
    print(f'  Selected algorithms: {selected_by_model}')


if __name__ == '__main__':
    main()
