#!/usr/bin/env python3
"""Aim 1 — pooled Focal + Generalized multinomial ML (clinical + interaction).

Models (both fitted across N_MC splits; bagged OOS per model)
  Model_clinical   : age, sex, epilepsy type, prior 2yr sz control
  Model_interaction: clinical + IEDs + slowing + type×IED + type×slowing
  (no ADR; no main-effects-only model)

Prior
  Default: any-seizure vs none (+ NA) → prior_sz_any, prior_sz_unavailable
  --poor-cutoff C: matched to outcome cutoff → prior_sz_poor, prior_sz_unavailable

Outcome
  Default with --poor-cutoff: BINARY poor vs not-poor (matches logistic Aim 1)
    poor = post-EEG mean > cutoff/mo
    not poor = mean <= cutoff/mo (seizure-free + moderate collapsed)
  --ternary-outcome: SF / moderate / poor (3-level multinomial)

Algorithms
  Fit lr / svm / sgd with default hyperparameters (no GridSearch).
  Model_interaction: clinical + markers + type×IED + type×slowing (all algorithms).

Strata on bagged OOS (each model)
  Focal vs Generalized (macro + OvR DeLong)
  Pairwise clinical vs interaction (overall + within Focal / Generalized)
  Prior 2yr sz freq documented vs not (bagged metrics + DeLong + model comparison)
  Prior strata (DeLong per OvR class)

Examples
  python aim1_pooled_ml.py --n-mc 2
  python aim1_pooled_ml.py --binary-markers --poor-cutoff 1.0 --n-mc 1000 --n-jobs 50
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
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC
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

OUTCOME_LABELS = {0: 'seizure_free', 1: 'moderate', 2: 'poor'}
OUTCOME_CLASSES = (0, 1, 2)
PRIOR_STRATA_BINARY = ('none', 'any', 'unavailable')
PRIOR_STRATA_MATCHED = ('not_poor', 'poor', 'unavailable')
PRIOR_INFO_STRATA = ('available', 'unavailable')

DEMO_COLS = ['age_at_eeg', 'gender', 'epilepsy_type_focal']
PRIOR_COLS_BINARY = ['prior_sz_any', 'prior_sz_unavailable']
PRIOR_COLS_MATCHED = ['prior_sz_poor', 'prior_sz_unavailable']
MARKER_COLS = ['spikes', 'slowing']
INTERACTION_TERMS = ['type_x_spikes', 'type_x_slowing']
MODEL_ORDER = ['Model_clinical', 'Model_interaction']

_CLF_KEYS = ('lr', 'svm', 'sgd')

# Multiclass-capable classifiers (default hyperparameters only)
_CLASSIFIERS_MULTI = {
    'sgd': {
        'pipeline': SGDClassifier(
            loss='log_loss', random_state=0, class_weight=None,
            penalty='l2', max_iter=20000, tol=1e-4, n_jobs=1,
        ),
        'param_grid': {
            'alpha': gml.ALPHA_GRID,
            'penalty': ['l2'],
        },
        'needs_scaling': True,
    },
    'lr': {
        'pipeline': LogisticRegression(
            solver='lbfgs', penalty='l2',
            max_iter=20000, tol=1e-4, random_state=0, class_weight=None,
        ),
        'param_grid': {
            'penalty': ['l2'],
            'C': gml.C_GRID,
        },
        'needs_scaling': True,
    },
    'svm': {
        'pipeline': CalibratedClassifierCV(
            LinearSVC(
                penalty='l2', dual=True,
                class_weight=None, max_iter=20000, random_state=0,
            ),
            method='sigmoid', cv=3,
        ),
        'param_grid': {
            'estimator__C': gml.C_GRID,
        },
        'needs_scaling': True,
    },
}


def _build_feature_sets(clinical_cols: list[str]) -> dict[str, list[str]]:
    return {
        'Model_clinical': list(clinical_cols),
        'Model_interaction': list(clinical_cols) + MARKER_COLS + INTERACTION_TERMS,
    }


def _cols_for_algorithm(
    model_key: str,
    clf_key: str,
    feature_sets: dict[str, list[str]],
    clinical_cols: list[str],
) -> list[str]:
    del clf_key, clinical_cols  # same feature set for every algorithm
    return feature_sets[model_key]


def _default_out_dir(
    binary_markers: bool,
    poor_cutoff: float | None,
    ternary_outcome: bool,
) -> Path:
    tag = 'youden' if binary_markers else 'cont'
    if poor_cutoff is not None and not ternary_outcome:
        return (
            _SCRIPT_DIR / 'outputs' / 'Combined'
            / f'aim1_pooled_clinical_interaction_poor{poor_cutoff:g}mo_matched_prior_{tag}'
        )
    if poor_cutoff is not None and ternary_outcome:
        return (
            _SCRIPT_DIR / 'outputs' / 'Combined'
            / f'aim1_pooled_clinical_interaction_poor{poor_cutoff:g}mo_ternary_matched_prior_{tag}'
        )
    return (
        _SCRIPT_DIR / 'outputs' / 'Combined'
        / f'aim1_pooled_clinical_interaction_ternary_prior_binary_{tag}'
    )


# ---------------------------------------------------------------------------
# Multiclass metrics / predict_proba alignment
# ---------------------------------------------------------------------------

def _align_proba(est, X, classes=OUTCOME_CLASSES) -> np.ndarray:
    """Return (n, 3) proba columns ordered as classes 0,1,2."""
    raw = np.asarray(est.predict_proba(X), dtype=float)
    out = np.full((raw.shape[0], len(classes)), np.nan, dtype=float)
    clf_classes = list(getattr(est, 'classes_', classes))
    # Pipeline / CalibratedClassifierCV: classes_ on final estimator
    if not hasattr(est, 'classes_') and hasattr(est, 'named_steps'):
        last = list(est.named_steps.values())[-1]
        clf_classes = list(getattr(last, 'classes_', classes))
    for j, c in enumerate(classes):
        if c in clf_classes:
            out[:, j] = raw[:, clf_classes.index(c)]
    return out


def _safe_macro_auroc(y_true, proba) -> float:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    ok = np.all(np.isfinite(proba), axis=1)
    y_true, proba = y_true[ok], proba[ok]
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float('nan')
    try:
        return float(roc_auc_score(
            y_true, proba, multi_class='ovr', average='macro',
            labels=list(OUTCOME_CLASSES),
        ))
    except ValueError:
        return float('nan')


def _safe_macro_auprc(y_true, proba) -> float:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    ok = np.all(np.isfinite(proba), axis=1)
    y_true, proba = y_true[ok], proba[ok]
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float('nan')
    y_bin = label_binarize(y_true, classes=list(OUTCOME_CLASSES))
    if y_bin.shape[1] == 1:
        return float('nan')
    aps = []
    for j in range(y_bin.shape[1]):
        if y_bin[:, j].sum() == 0 or y_bin[:, j].sum() == len(y_bin):
            continue
        aps.append(average_precision_score(y_bin[:, j], proba[:, j]))
    return float(np.mean(aps)) if aps else float('nan')


def _eval_metrics(y_true, proba, ternary: bool) -> dict:
    if ternary:
        return {
            'auroc': _safe_macro_auroc(y_true, proba),
            'auprc': _safe_macro_auprc(y_true, proba),
        }
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float).ravel()
    ok = np.isfinite(p)
    y_true, p = y_true[ok], p[ok]
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return {'auroc': float('nan'), 'auprc': float('nan')}
    return {
        'auroc': float(roc_auc_score(y_true, p)),
        'auprc': float(average_precision_score(y_true, p)),
    }


def _finite_proba_ok(proba, ternary: bool) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    if ternary:
        return np.all(np.isfinite(p), axis=1)
    return np.isfinite(p.ravel() if p.ndim > 1 else p)


def _safe_binary_auroc_auprc(y_true, proba) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float).ravel()
    ok = np.isfinite(p)
    y_true, p = y_true[ok], p[ok]
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float('nan'), float('nan')
    return float(roc_auc_score(y_true, p)), float(average_precision_score(y_true, p))


def _ovr_auroc_auprc(y_true, proba, class_id: int) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    score = proba[:, class_id]
    ok = np.isfinite(score)
    y_bin = (y_true[ok] == class_id).astype(int)
    s = score[ok]
    if len(y_bin) < 2 or len(np.unique(y_bin)) < 2:
        return float('nan'), float('nan')
    return float(roc_auc_score(y_bin, s)), float(average_precision_score(y_bin, s))


# ---------------------------------------------------------------------------
# Fit (default hyperparameters; no GridSearch)
# ---------------------------------------------------------------------------

def fit_clf_multiclass(clf_key: str, X_train, y_train):
    cfg = _CLASSIFIERS_MULTI[clf_key]
    estimator = clone(cfg['pipeline'])
    if cfg['needs_scaling']:
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', estimator)])
    else:
        pipe = estimator
    pipe.fit(X_train, y_train)
    return pipe


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------

def _code_outcome_3way(mean_sz_freq: float, poor_cutoff: float) -> float:
    """0=SF, 1=moderate (0 < mean <= cutoff), 2=poor (mean > cutoff)."""
    if mean_sz_freq is None or (isinstance(mean_sz_freq, float) and not np.isfinite(mean_sz_freq)):
        return np.nan
    try:
        x = float(mean_sz_freq)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(x):
        return np.nan
    if x == 0.0:
        return 0.0
    if x <= poor_cutoff:
        return 1.0
    return 2.0


def build_analysis_df(
    patient_df: pd.DataFrame,
    binary_markers: bool = False,
    poor_cutoff: float | None = None,
    ternary_outcome: bool = False,
) -> tuple[pd.DataFrame, list[str], tuple[str, ...], bool]:
    """Pooled cohort; outcome binary (poor>cutoff) or ternary SF/mod/poor."""
    cut = float(poor_cutoff if poor_cutoff is not None else st.POOR_CUTOFF_PER_MONTH)
    matched_prior = poor_cutoff is not None
    prior_strata = PRIOR_STRATA_MATCHED if matched_prior else PRIOR_STRATA_BINARY
    prior_cols = PRIOR_COLS_MATCHED if matched_prior else PRIOR_COLS_BINARY
    use_ternary = bool(ternary_outcome)

    df = patient_df.copy()
    if 'mean_sz_freq' not in df.columns:
        raise KeyError('mean_sz_freq required for outcome coding')
    if use_ternary:
        df['outcome'] = df['mean_sz_freq'].map(lambda x: _code_outcome_3way(x, cut))
    else:
        df['outcome'] = df['mean_sz_freq'].map(lambda x: st._code_outcome_binary(x, cut))
    df = df.dropna(subset=['outcome', 'age_at_eeg', 'gender', 'epilepsy_type']).copy()
    df['outcome'] = df['outcome'].astype(int)
    df['gender'] = df['gender'].astype(float)
    df['epilepsy_type_focal'] = (df['epilepsy_type'] == 'Focal').astype(float)

    df['spikes'] = df['spike_rate_log']
    df['slowing'] = df['pct_gen_slowing'].fillna(0) + df['pct_foc_slowing'].fillna(0)

    if matched_prior:
        if 'mean_sz_freq_pre' not in df.columns:
            raise KeyError('cutoff-matched prior requires mean_sz_freq_pre')
        df['prior_sz_control'] = df['mean_sz_freq_pre'].map(
            lambda x: st._code_prior_from_cutoff(x, cut)
        ).astype(int)
        df['prior_sz_poor'] = (df['prior_sz_control'] == 1).astype(float)
        df['prior_sz_unavailable'] = (df['prior_sz_control'] == 2).astype(float)
        df['prior_stratum'] = np.where(
            df['prior_sz_control'] == 2, 'unavailable',
            np.where(df['prior_sz_control'] == 1, 'poor', 'not_poor'),
        )
        df['prior_info'] = np.where(df['prior_stratum'] == 'unavailable', 'unavailable', 'available')
        vc = df['prior_sz_control'].value_counts().sort_index()
        print(
            f'[Prior matched to cutoff {cut}/mo] '
            f'0(not_poor)={int(vc.get(0, 0))}, 1(poor)={int(vc.get(1, 0))}, '
            f'2(NA)={int(vc.get(2, 0))}'
        )
        vc_pi = df['prior_info'].value_counts()
        print(
            f'[Prior 2yr sz freq info] available={int(vc_pi.get("available", 0))}, '
            f'unavailable={int(vc_pi.get("unavailable", 0))}'
        )
    else:
        if 'prior_sz_control' not in df.columns:
            raise KeyError(
                'prior_sz_control missing — rebuild cohort via statistical_tests._load_combined_data'
            )
        df['prior_sz_control'] = df['prior_sz_control'].fillna(3).astype(int)
        prior4 = df['prior_sz_control'].values
        prior_bin = np.where(prior4 == 0, 0, np.where(np.isin(prior4, [1, 2]), 1, 2))
        df['prior_sz_control_bin'] = prior_bin.astype(int)
        df['prior_sz_any'] = (df['prior_sz_control_bin'] == 1).astype(float)
        df['prior_sz_unavailable'] = (df['prior_sz_control_bin'] == 2).astype(float)
        df['prior_stratum'] = np.where(
            df['prior_sz_control_bin'] == 0, 'none',
            np.where(df['prior_sz_control_bin'] == 1, 'any', 'unavailable'),
        )
        df['prior_info'] = np.where(df['prior_stratum'] == 'unavailable', 'unavailable', 'available')
        vc_pr = df['prior_stratum'].value_counts()
        print(
            '[Prior stratum] '
            + ', '.join(f'{k}={int(vc_pr.get(k, 0))}' for k in prior_strata)
        )
        vc_pi = df['prior_info'].value_counts()
        print(
            f'[Prior 2yr sz freq info] available={int(vc_pi.get("available", 0))}, '
            f'unavailable={int(vc_pi.get("unavailable", 0))}'
        )

    if binary_markers:
        n_sp = int((df['spikes'] >= st.SPIKE_YOUDEN_THRESH).sum())
        n_sl = int((df['slowing'] >= st.SLOWING_YOUDEN_THRESH_PCT).sum())
        print(
            f'[Binary markers] spikes>={st.SPIKE_YOUDEN_THRESH}: {n_sp}/{len(df)}; '
            f'slowing>={st.SLOWING_YOUDEN_THRESH_PCT}%: {n_sl}/{len(df)}'
        )
        df['spikes'] = (df['spikes'] >= st.SPIKE_YOUDEN_THRESH).astype(float)
        df['slowing'] = (df['slowing'] >= st.SLOWING_YOUDEN_THRESH_PCT).astype(float)

    t = df['epilepsy_type_focal']
    df['type_x_spikes'] = t * df['spikes']
    df['type_x_slowing'] = t * df['slowing']

    need = [
        'outcome', 'age_at_eeg', 'gender', 'epilepsy_type_focal',
        *prior_cols, 'spikes', 'slowing',
    ]
    df = df.dropna(subset=need).copy()
    df['patient_id'] = df['patient_id'].astype(str)

    if use_ternary:
        vc_out = df['outcome'].value_counts().sort_index()
        print(
            f'[Outcome TERNARY @ poor>{cut}/mo boundary] '
            + ', '.join(f'{OUTCOME_LABELS[k]}={int(vc_out.get(k, 0))}' for k in OUTCOME_CLASSES)
        )
    else:
        n_poor = int((df['outcome'] == 1).sum())
        print(
            f'[Outcome BINARY] poor = mean > {cut}/mo: {n_poor}/{len(df)} '
            f'({100 * n_poor / len(df):.1f}%); '
            f'not_poor (<= {cut}/mo incl SF)={len(df) - n_poor}'
        )
    clinical_cols = DEMO_COLS + prior_cols
    return df.reset_index(drop=True), clinical_cols, prior_strata, use_ternary


# ---------------------------------------------------------------------------
# DeLong (binary OvR)
# ---------------------------------------------------------------------------

def delong_auc_and_var(y_true, y_score) -> tuple[float, float]:
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


def unpaired_delong_compare(y_a, p_a, y_b, p_b) -> dict:
    """Δ = AUC_A − AUC_B; SE = sqrt(Var_A + Var_B)."""
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
# MC iteration + algorithm selection (Aim-2 style)
# ---------------------------------------------------------------------------

def run_mc_iteration(
    seed: int,
    analysis_df: pd.DataFrame,
    feature_sets: dict,
    clinical_cols: list[str],
    clf_keys: tuple[str, ...] = _CLF_KEYS,
    ternary_outcome: bool = False,
    grid_search: bool = False,
) -> dict | None:
    y_all = analysis_df['outcome'].values.astype(int)
    tr_idx, te_idx = gml._patient_grouped_split(analysis_df, y_all, TEST_SIZE, seed)
    y_tr = y_all[tr_idx]
    y_te = y_all[te_idx]
    min_classes = 3 if ternary_outcome else 2
    if len(np.unique(y_tr)) < min_classes or len(np.unique(y_te)) < min_classes:
        return None

    train_df = analysis_df.iloc[tr_idx]
    test_df = analysis_df.iloc[te_idx]
    test_pids = test_df['patient_id'].astype(str).values

    proba_by_clf: dict = {}
    metrics_by_clf: dict = {}
    for model_key in feature_sets:
        proba_by_clf[model_key] = {}
        metrics_by_clf[model_key] = {}
        for clf_key in clf_keys:
            cols = _cols_for_algorithm(model_key, clf_key, feature_sets, clinical_cols)
            X_tr = gml.prepare_X(train_df, cols, ref_df=train_df)
            X_te = gml.prepare_X(test_df, cols, ref_df=train_df)
            try:
                if ternary_outcome:
                    est = fit_clf_multiclass(clf_key, X_tr, y_tr)
                    p = _align_proba(est, X_te)
                elif grid_search:
                    est, _ = gml.fit_clf(
                        clf_key, X_tr, y_tr, seed, N_CV_FOLDS, class_weight='none',
                    )
                    p = est.predict_proba(X_te)[:, 1]
                else:
                    est = gml.fit_clf_default(clf_key, X_tr, y_tr, class_weight='none')
                    p = est.predict_proba(X_te)[:, 1]
            except Exception:
                if ternary_outcome:
                    p = np.full((len(y_te), len(OUTCOME_CLASSES)), np.nan)
                else:
                    p = np.full(len(y_te), np.nan)
            proba_by_clf[model_key][clf_key] = p
            metrics_by_clf[model_key][clf_key] = _eval_metrics(y_te, p, ternary_outcome)

    return {
        'seed': seed,
        'test_pids': test_pids,
        'y_test': y_te.astype(int),
        'proba_by_clf': proba_by_clf,
        'metrics_by_clf': metrics_by_clf,
        'ternary_outcome': ternary_outcome,
        'best_clf': {},
        'proba': {},
        'metrics': {},
    }


def select_clf_per_model(
    mc_results: list,
    model_keys: list[str],
    feature_sets: dict[str, list[str]],
    clinical_cols: list[str],
    clf_keys: tuple[str, ...] = _CLF_KEYS,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Lock each feature model to the algorithm with highest mean MC macro AUROC."""
    rows = []
    selected: dict[str, str] = {}
    for model_key in model_keys:
        means = {}
        for clf_key in clf_keys:
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
                'n_features': len(feature_sets[model_key]),
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
    for res in mc_results:
        if res is None:
            continue
        ternary = bool(res.get('ternary_outcome'))
        for model_key in feature_sets:
            clf = selected_by_model[model_key]
            p = np.asarray(res['proba_by_clf'][model_key][clf], dtype=float)
            res['proba'][model_key] = p
            res['metrics'][model_key] = res['metrics_by_clf'][model_key][clf]
            res['best_clf'][model_key] = clf
            if not ternary and p.ndim == 1:
                res.setdefault('roc_curves', {})[model_key] = _binary_roc_table(
                    res['y_test'], p, res['seed'], model_key, clf,
                )
    return mc_results


def _binary_roc_table(y_true, y_prob, seed: int, model_key: str, best_clf: str) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    ok = np.isfinite(y_prob)
    y_true, y_prob = y_true[ok], y_prob[ok]
    empty = pd.DataFrame(columns=[
        'seed', 'model', 'best_clf', 'point', 'fpr', 'tpr', 'threshold', 'auroc',
    ])
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return empty
    auroc = float(roc_auc_score(y_true, y_prob))
    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    roc_thr_full = np.concatenate([roc_thr, [np.nan]])
    return pd.DataFrame({
        'seed': seed,
        'model': model_key,
        'best_clf': best_clf,
        'point': np.arange(len(fpr)),
        'fpr': fpr,
        'tpr': tpr,
        'threshold': roc_thr_full[:len(fpr)],
        'auroc': auroc,
    })


# ---------------------------------------------------------------------------
# Bagging + strata
# ---------------------------------------------------------------------------

def bag_oos_predictions(
    mc_results: list,
    analysis_df: pd.DataFrame,
    model_key: str,
    ternary_outcome: bool = False,
) -> dict:
    """Mean held-out probability per patient (3-class vector or P(poor))."""
    pids = analysis_df['patient_id'].astype(str).values
    y = analysis_df['outcome'].values.astype(int)
    epi = analysis_df['epilepsy_type'].astype(str).values
    prior = analysis_df['prior_stratum'].astype(str).values
    prior_info = analysis_df['prior_info'].astype(str).values
    n = len(pids)
    clf_votes: dict[str, int] = {}

    if ternary_outcome:
        k = len(OUTCOME_CLASSES)
        sums = {pid: np.zeros(k, dtype=float) for pid in pids}
        counts = {pid: 0 for pid in pids}
        for res in mc_results:
            if res is None:
                continue
            preds = np.asarray(res['proba'][model_key], dtype=float)
            for pid, pr in zip(res['test_pids'], preds):
                if not np.all(np.isfinite(pr)):
                    continue
                sums[pid] += pr
                counts[pid] += 1
            clf = res['best_clf'][model_key]
            clf_votes[clf] = clf_votes.get(clf, 0) + 1
        bagged = np.full((n, k), np.nan)
        n_held = np.zeros(n, dtype=int)
        for i, pid in enumerate(pids):
            if counts[pid] > 0:
                bagged[i] = sums[pid] / counts[pid]
                n_held[i] = counts[pid]
    else:
        sums = {pid: 0.0 for pid in pids}
        counts = {pid: 0 for pid in pids}
        for res in mc_results:
            if res is None:
                continue
            preds = np.asarray(res['proba'][model_key], dtype=float).ravel()
            for pid, pr in zip(res['test_pids'], preds):
                if not np.isfinite(pr):
                    continue
                sums[pid] += float(pr)
                counts[pid] += 1
            clf = res['best_clf'][model_key]
            clf_votes[clf] = clf_votes.get(clf, 0) + 1
        bagged = np.full(n, np.nan)
        n_held = np.zeros(n, dtype=int)
        for i, pid in enumerate(pids):
            if counts[pid] > 0:
                bagged[i] = sums[pid] / counts[pid]
                n_held[i] = counts[pid]

    return {
        'patient_id': pids,
        'y_true': y,
        'epilepsy_type': epi,
        'prior_stratum': prior,
        'prior_info': prior_info,
        'bagged_proba': bagged,
        'n_heldout': n_held,
        'clf_votes': clf_votes,
        'ternary_outcome': ternary_outcome,
    }


def summarize_mc_metrics(mc_results: list, model_key: str, rng=None) -> dict:
    rng = rng or np.random.default_rng(0)
    aurocs, auprcs = [], []
    for res in mc_results:
        if res is None:
            continue
        m = res['metrics'][model_key]
        if np.isfinite(m['auroc']):
            aurocs.append(m['auroc'])
        if np.isfinite(m['auprc']):
            auprcs.append(m['auprc'])
    aurocs = np.asarray(aurocs, dtype=float)
    auprcs = np.asarray(auprcs, dtype=float)
    auc_lo, auc_hi = gml.bootstrap_mean_ci(aurocs, n_boot=N_BOOTSTRAP, rng=rng)
    pr_lo, pr_hi = gml.bootstrap_mean_ci(auprcs, n_boot=N_BOOTSTRAP, rng=rng)
    return {
        'model': model_key,
        'n_mc': int(len(aurocs)),
        'auc_mean': float(np.nanmean(aurocs)) if len(aurocs) else np.nan,
        'auc_ci_lo': auc_lo,
        'auc_ci_hi': auc_hi,
        'auprc_mean': float(np.nanmean(auprcs)) if len(auprcs) else np.nan,
        'auprc_ci_lo': pr_lo,
        'auprc_ci_hi': pr_hi,
    }


def _stratum_metric_row(
    bag: dict, mask: np.ndarray, stratum_name: str, stratum_col: str,
) -> dict:
    y = bag['y_true'][mask]
    p = bag['bagged_proba'][mask]
    row = {stratum_col: stratum_name, 'n': int(mask.sum())}
    if bag.get('ternary_outcome', True):
        row['macro_auroc'] = _safe_macro_auroc(y, p)
        row['macro_auprc'] = _safe_macro_auprc(y, p)
        for c in OUTCOME_CLASSES:
            auc, ap = _ovr_auroc_auprc(y, p, c)
            row[f'ovr_{OUTCOME_LABELS[c]}_auroc'] = auc
            row[f'ovr_{OUTCOME_LABELS[c]}_auprc'] = ap
            # DeLong SE / 95% CI for this OvR score within the stratum
            d = bagged_auc_delong_stats((y == c).astype(int), p[:, c] if p.ndim == 2 else p)
            row[f'ovr_{OUTCOME_LABELS[c]}_delong_se'] = d['delong_se']
            row[f'ovr_{OUTCOME_LABELS[c]}_delong_ci_lo'] = d['delong_ci_lo']
            row[f'ovr_{OUTCOME_LABELS[c]}_delong_ci_hi'] = d['delong_ci_hi']
    else:
        auc, ap = _safe_binary_auroc_auprc(y, p)
        d = bagged_auc_delong_stats(y, p)
        row['auroc'] = auc
        row['auprc'] = ap
        row['delong_se'] = d['delong_se']
        row['delong_ci_lo'] = d['delong_ci_lo']
        row['delong_ci_hi'] = d['delong_ci_hi']
        row['prevalence_poor'] = float(np.mean(y == 1)) if len(y) else np.nan
    return row


def bagged_auc_delong_stats(y_true, y_score) -> dict:
    """Point AUROC + DeLong SE and normal-approx 95% CI."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float).ravel()
    ok = np.isfinite(y_score)
    y_true, y_score = y_true[ok], y_score[ok]
    out = {
        'bagged_auc': float('nan'),
        'delong_se': float('nan'),
        'delong_ci_lo': float('nan'),
        'delong_ci_hi': float('nan'),
    }
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return out
    auc, var = delong_auc_and_var(y_true, y_score)
    out['bagged_auc'] = float(auc)
    if np.isfinite(var) and var >= 0:
        se = float(np.sqrt(var))
        out['delong_se'] = se
        out['delong_ci_lo'] = float(auc - 1.96 * se)
        out['delong_ci_hi'] = float(auc + 1.96 * se)
    return out


def pairwise_delong_strata(
    bag: dict,
    stratum_col: str,
    levels: tuple[str, ...],
) -> list[dict]:
    """Unpaired DeLong across strata (ternary OvR or binary poor)."""
    rows = []
    y = bag['y_true']
    proba = bag['bagged_proba']
    labels = bag[stratum_col]
    ok = _finite_proba_ok(proba, bag.get('ternary_outcome', True))

    if bag.get('ternary_outcome', True):
        for a, b in combinations(levels, 2):
            ma = ok & (labels == a)
            mb = ok & (labels == b)
            for c in OUTCOME_CLASSES:
                ya = (y[ma] == c).astype(int)
                yb = (y[mb] == c).astype(int)
                pa = proba[ma, c]
                pb = proba[mb, c]
                cmp_ = unpaired_delong_compare(ya, pa, yb, pb)
                rows.append({
                    'stratum_col': stratum_col,
                    'group_a': a,
                    'group_b': b,
                    'ovr_class': OUTCOME_LABELS[c],
                    **{k: cmp_[k] for k in (
                        'auc_a', 'auc_b', 'delta', 'se', 'ci_lo', 'ci_hi', 'p', 'n_a', 'n_b',
                    )},
                })
        return rows

    for a, b in combinations(levels, 2):
        ma = ok & (labels == a)
        mb = ok & (labels == b)
        cmp_ = unpaired_delong_compare(y[ma], proba[ma], y[mb], proba[mb])
        rows.append({
            'stratum_col': stratum_col,
            'group_a': a,
            'group_b': b,
            'ovr_class': 'poor',
            **{k: cmp_[k] for k in (
                'auc_a', 'auc_b', 'delta', 'se', 'ci_lo', 'ci_hi', 'p', 'n_a', 'n_b',
            )},
        })
    return rows


def bootstrap_compare_macro_auroc(
    y_true,
    proba_simple,
    proba_complex,
    n_boot: int = N_BOOTSTRAP,
    rng=None,
) -> dict:
    """Paired patient bootstrap on macro OvR AUROC (complex − simple)."""
    y_true = np.asarray(y_true, dtype=int)
    ps = np.asarray(proba_simple, dtype=float)
    pc = np.asarray(proba_complex, dtype=float)
    ok = np.all(np.isfinite(ps), axis=1) & np.all(np.isfinite(pc), axis=1)
    y_true, ps, pc = y_true[ok], ps[ok], pc[ok]
    n = len(y_true)
    nan_out = {
        'auc_simple': np.nan, 'auc_complex': np.nan, 'diff': np.nan,
        'diff_ci_lo': np.nan, 'diff_ci_hi': np.nan,
        'n_boot_valid': 0, 'n_simpler': 0, 'p': np.nan,
    }
    if n < 2 or len(np.unique(y_true)) < 2:
        return nan_out

    auc_simple_obs = _safe_macro_auroc(y_true, ps)
    auc_complex_obs = _safe_macro_auroc(y_true, pc)
    rng = rng or np.random.default_rng(0)
    diffs = []
    n_simpler = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        auc_s = _safe_macro_auroc(yt, ps[idx])
        auc_c = _safe_macro_auroc(yt, pc[idx])
        if not (np.isfinite(auc_s) and np.isfinite(auc_c)):
            continue
        diffs.append(auc_c - auc_s)
        if auc_s > auc_c:
            n_simpler += 1
    if len(diffs) < 10:
        return nan_out

    diffs = np.asarray(diffs, dtype=float)
    n_valid = len(diffs)
    p = float(min(2 * (min(n_simpler, n_valid - n_simpler) + 1) / (n_boot + 1), 1.0))
    return {
        'auc_simple': float(auc_simple_obs),
        'auc_complex': float(auc_complex_obs),
        'diff': float(auc_complex_obs - auc_simple_obs),
        'diff_ci_lo': float(np.percentile(diffs, 2.5)),
        'diff_ci_hi': float(np.percentile(diffs, 97.5)),
        'n_boot_valid': n_valid,
        'n_simpler': int(n_simpler),
        'p': p,
    }


def pairwise_mc_auroc_sign_test(mc_df: pd.DataFrame, model_a: str, model_b: str) -> dict:
    """Paired MC-split AUROC sign test (no bagged patient bootstrap)."""
    metric = 'auroc' if 'auroc' in mc_df.columns else 'macro_auroc'
    piv = mc_df.pivot(index='seed', columns='model', values=metric)
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
    p = float(min(2.0 * (n_min + 1) / (n + 1), 1.0)) if n else np.nan
    mean_d = float(np.mean(deltas)) if n else np.nan
    better = model_b if (n and mean_d > 0) else (model_a if n else '')
    if n and mean_d == 0:
        better = 'tie'
    return {
        'model_a': model_a, 'model_b': model_b, 'n_mc': n,
        'auc_a_mean': float(np.mean(da)) if n else np.nan,
        'auc_b_mean': float(np.mean(db)) if n else np.nan,
        'diff_b_minus_a': mean_d,
        'diff_ci_lo': float(np.percentile(deltas, 2.5)) if n else np.nan,
        'diff_ci_hi': float(np.percentile(deltas, 97.5)) if n else np.nan,
        'n_a_wins': n_a_wins, 'n_b_wins': n_b_wins, 'n_ties': n_ties,
        'better_model': better, 'p': p,
    }


def pairwise_model_comparisons(
    bags: dict[str, dict],
    model_keys: list[str],
    feature_sets: dict[str, list[str]],
    mask: np.ndarray | None = None,
    stratum: str = 'all',
    rng=None,
    ternary_outcome: bool = False,
) -> list[dict]:
    """Pairwise bagged-OOS AUROC (supplementary; not primary inference)."""
    rng = rng or np.random.default_rng(1)
    n_feats = {k: len(feature_sets[k]) for k in model_keys}
    rows = []
    for ki, kj in combinations(model_keys, 2):
        if n_feats[ki] <= n_feats[kj]:
            k_simple, k_complex = ki, kj
        else:
            k_simple, k_complex = kj, ki

        y = bags[k_simple]['y_true']
        p_s = bags[k_simple]['bagged_proba']
        p_c = bags[k_complex]['bagged_proba']
        ok = _finite_proba_ok(p_s, ternary_outcome) & _finite_proba_ok(p_c, ternary_outcome)
        if mask is not None:
            ok = ok & mask
        if ternary_outcome:
            cmp_ = bootstrap_compare_macro_auroc(
                y[ok], p_s[ok], p_c[ok], n_boot=N_BOOTSTRAP, rng=rng,
            )
        else:
            cmp_ = gml.bootstrap_compare_auroc(
                y[ok], p_s[ok], p_c[ok], n_boot=N_BOOTSTRAP, rng=rng,
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
            'n_boot_valid': cmp_['n_boot_valid'],
            'n_simpler': cmp_['n_simpler'],
        })
    return rows


def focal_vs_generalized_delong_for_model(
    bag: dict, model_key: str,
) -> list[dict]:
    rows = pairwise_delong_strata(bag, 'epilepsy_type', ('Focal', 'Generalized'))
    for row in rows:
        row['model'] = model_key
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_table(df: pd.DataFrame, filename: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, index=False)
    print(f'  Saved {path}')
    return path


def main() -> None:
    p = argparse.ArgumentParser(
        description='Aim 1 pooled multinomial ML — clinical + interaction models')
    p.add_argument('--n-mc', type=int, default=N_MC)
    p.add_argument('--n-jobs', type=int, default=N_JOBS)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument(
        '--classifiers', nargs='+', default=list(_CLF_KEYS),
        choices=list(_CLF_KEYS),
        help='Algorithms to choose among (default: lr svm sgd)',
    )
    p.add_argument(
        '--binary-markers', action='store_true',
        help='Binarize IEDs/slowing at clinician Youden thresholds',
    )
    p.add_argument(
        '--poor-cutoff', type=float, default=None,
        help='Poor if post-EEG mean > cutoff/mo; prior matched to same cutoff',
    )
    p.add_argument(
        '--ternary-outcome', action='store_true',
        help='Use SF/moderate/poor (3-level). Default with --poor-cutoff is binary poor vs not-poor.',
    )
    p.add_argument(
        '--grid-search', action='store_true',
        help='Tune lr/svm/sgd with GridSearchCV each MC split (binary outcome only)',
    )
    p.add_argument(
        '--out-dir', type=Path, default=None,
        help='Override output directory',
    )
    p.add_argument(
        '--cohort-csv', type=Path, default=None,
        help='Pre-exported patient cohort CSV (local runs; skips raw data rebuild)',
    )
    args = p.parse_args()

    ternary_outcome = bool(args.ternary_outcome)
    cohort_csv = args.cohort_csv
    if cohort_csv is None and os.environ.get('AIM1_COHORT_CSV'):
        cohort_csv = Path(os.environ['AIM1_COHORT_CSV'])
    out_dir = args.out_dir or Path(os.environ.get(
        'ML_OUT_DIR', str(_default_out_dir(
            args.binary_markers, args.poor_cutoff, ternary_outcome,
        )),
    ))

    if cohort_csv is not None:
        patient_df = pd.read_csv(cohort_csv)
        print(f'Loaded cohort CSV: {cohort_csv}  (N={len(patient_df)} rows)')
    else:
        patient_tables = st._load_combined_data()
        patient_df = patient_tables['sz_freq_class']
    analysis_df, clinical_cols, prior_strata, ternary_outcome = build_analysis_df(
        patient_df,
        binary_markers=args.binary_markers,
        poor_cutoff=args.poor_cutoff,
        ternary_outcome=ternary_outcome,
    )
    feature_sets = _build_feature_sets(clinical_cols)
    clf_keys = tuple(args.classifiers)
    cut = float(args.poor_cutoff if args.poor_cutoff is not None else st.POOR_CUTOFF_PER_MONTH)
    outcome_label = 'ternary SF/mod/poor' if ternary_outcome else 'binary poor vs not-poor (<=cutoff incl SF)'

    print('=' * 80)
    print('Aim 1 — pooled Focal + Generalized (clinical + interaction models)')
    print(f'OUT_DIR:  {out_dir}')
    print(f'N_MC:     {args.n_mc}  (80/20, seed {args.seed_start}..'
          f'{args.seed_start + args.n_mc - 1})')
    print(f'N_JOBS:   {args.n_jobs}')
    print(f'Classifiers: {list(clf_keys)}  |  default hyperparameters (no tuning)')
    print(f'Outcome:  {outcome_label}; poor boundary > {cut}/mo')
    if args.poor_cutoff is not None:
        print(f'Prior: matched to {cut}/mo (not_poor / poor / NA)')
    else:
        print('Prior: any-seizure vs none (+ NA)')
    print(
        'Markers: '
        + (
            f'BINARY Youden (spikes>={st.SPIKE_YOUDEN_THRESH}, '
            f'slowing>={st.SLOWING_YOUDEN_THRESH_PCT}%)'
            if args.binary_markers else
            'continuous (log spike rate; combined pct slowing)'
        )
    )
    print('Algorithm pick: highest mean MC AUROC per model')
    print('=' * 80)

    n_foc = int((analysis_df['epilepsy_type'] == 'Focal').sum())
    n_gen = int((analysis_df['epilepsy_type'] == 'Generalized').sum())
    print(
        f'\nAnalysis cohort: N={len(analysis_df)} '
        f'(Focal={n_foc}, Generalized={n_gen})'
    )
    for mk, cols in feature_sets.items():
        print(f'  {mk}: {len(cols)} features — {cols}')

    cohort_cols = [
        'patient_id', 'epilepsy_type', 'outcome', 'mean_sz_freq',
        'age_at_eeg', 'gender', 'prior_sz_control', 'prior_stratum',
        'spikes', 'slowing',
    ]
    if args.poor_cutoff is not None:
        cohort_cols += ['prior_sz_poor', 'prior_sz_unavailable']
    else:
        cohort_cols += ['prior_sz_control_bin', 'prior_sz_any', 'prior_sz_unavailable']
    save_table(analysis_df[cohort_cols].copy(), 'analysis_cohort.csv', out_dir)

    print(f'\nRunning {args.n_mc} MC iterations (all algorithms × both models)...')
    if args.grid_search and not ternary_outcome:
        print('Hyperparameter tuning: GridSearchCV (L2 only × C/alpha)')
    seeds = range(args.seed_start, args.seed_start + args.n_mc)
    gen = Parallel(n_jobs=args.n_jobs, return_as='generator')(
        delayed(run_mc_iteration)(
            seed, analysis_df, feature_sets, clinical_cols, clf_keys, ternary_outcome,
            bool(args.grid_search) and not ternary_outcome,
        )
        for seed in seeds
    )
    mc_results = list(tqdm(gen, total=args.n_mc, desc='MC 80/20', unit='split'))
    n_ok = sum(r is not None for r in mc_results)
    print(f'  Kept {n_ok}/{args.n_mc} successful iterations')

    selected, rank = select_clf_per_model(
        mc_results, MODEL_ORDER, feature_sets, clinical_cols, clf_keys,
    )
    save_table(rank, 'algorithm_auc_ranking_by_model.csv', out_dir)
    sel_path = out_dir / 'selected_algorithms_per_model.txt'
    sel_path.write_text(
        '\n'.join(f'{mk}: {clf}' for mk, clf in selected.items()) + '\n',
        encoding='utf-8',
    )
    print(f'  Selected algorithms: {selected}')
    print(f'  Saved {sel_path}')

    apply_selected_clfs_per_model(mc_results, feature_sets, selected)

    mc_rows = []
    for res in mc_results:
        if res is None:
            continue
        for mk in feature_sets:
            mc_rows.append({
                'seed': res['seed'],
                'model': mk,
                'best_clf': res['best_clf'][mk],
                'auroc': res['metrics'][mk]['auroc'],
                'auprc': res['metrics'][mk]['auprc'],
                'n_test': len(res['y_test']),
            })
    save_table(pd.DataFrame(mc_rows), 'mc_iteration_metrics.csv', out_dir)

    # Full MC ROC curves (binary) for graphing Model_clinical / Model_interaction
    if not ternary_outcome:
        roc_frames = []
        for res in mc_results:
            if res is None:
                continue
            for mk in feature_sets:
                roc_df = res.get('roc_curves', {}).get(mk)
                if roc_df is not None and len(roc_df):
                    roc_frames.append(roc_df)
        if roc_frames:
            roc_all = pd.concat(roc_frames, ignore_index=True)
            save_table(roc_all, 'mc_roc_curves.csv', out_dir)
            for mk in ('Model_clinical', 'Model_interaction'):
                sub = roc_all.loc[roc_all['model'] == mk]
                if len(sub):
                    save_table(sub, f'mc_roc_curves_{mk}.csv', out_dir)
                    print(
                        f'  ROC export {mk}: {sub["seed"].nunique()} MC seeds, '
                        f'{len(sub)} curve points'
                    )

    mc_summaries = []
    for mk in MODEL_ORDER:
        summ = summarize_mc_metrics(mc_results, mk, rng=np.random.default_rng(0))
        mc_summaries.append(summ)
        print(
            f"\n{mk} MC AUROC mean={summ['auc_mean']:.3f} "
            f"[{summ['auc_ci_lo']:.3f}, {summ['auc_ci_hi']:.3f}]"
        )
    save_table(pd.DataFrame(mc_summaries), 'mc_auc_auprc_summary.csv', out_dir)

    bags = {}
    bag_df = pd.DataFrame({
        'patient_id': analysis_df['patient_id'].astype(str).values,
        'epilepsy_type': analysis_df['epilepsy_type'].astype(str).values,
        'prior_stratum': analysis_df['prior_stratum'].astype(str).values,
        'prior_info': analysis_df['prior_info'].astype(str).values,
        'y_true': analysis_df['outcome'].values.astype(int),
    })
    overall_rows = []
    type_metric_rows = []
    type_delong_rows = []
    prior_metric_rows = []
    prior_delong_rows = []
    prior_info_metric_rows = []
    prior_info_delong_rows = []

    for mk in MODEL_ORDER:
        bag = bag_oos_predictions(
            mc_results, analysis_df, mk, ternary_outcome=ternary_outcome,
        )
        bags[mk] = bag
        print(f"{mk} best-clf votes: {bag['clf_votes']}")
        bag_df[f'{mk}_n_heldout'] = bag['n_heldout']
        if ternary_outcome:
            for j, lab in OUTCOME_LABELS.items():
                bag_df[f'{mk}_bagged_p_{lab}'] = bag['bagged_proba'][:, j]
        else:
            bag_df[f'{mk}_bagged_p_poor'] = bag['bagged_proba']

        if ternary_outcome:
            overall = {
                'model': mk,
                'macro_auroc': _safe_macro_auroc(bag['y_true'], bag['bagged_proba']),
                'macro_auprc': _safe_macro_auprc(bag['y_true'], bag['bagged_proba']),
                'selected_clf': selected[mk],
                'clf_votes': str(bag['clf_votes']),
            }
            for c in OUTCOME_CLASSES:
                auc, ap = _ovr_auroc_auprc(bag['y_true'], bag['bagged_proba'], c)
                overall[f'ovr_{OUTCOME_LABELS[c]}_auroc'] = auc
                overall[f'ovr_{OUTCOME_LABELS[c]}_auprc'] = ap
            print(
                f"  Bagged overall macro AUROC={overall['macro_auroc']:.3f}  "
                f"AUPRC={overall['macro_auprc']:.3f}"
            )
        else:
            auc, ap = _safe_binary_auroc_auprc(bag['y_true'], bag['bagged_proba'])
            d = bagged_auc_delong_stats(bag['y_true'], bag['bagged_proba'])
            overall = {
                'model': mk,
                'bagged_auroc': auc,
                'bagged_auprc': ap,
                'delong_se': d['delong_se'],
                'delong_ci_lo': d['delong_ci_lo'],
                'delong_ci_hi': d['delong_ci_hi'],
                'n_patients': int(_finite_proba_ok(bag['bagged_proba'], False).sum()),
                'prevalence_poor': float(np.mean(bag['y_true'] == 1)),
                'selected_clf': selected[mk],
                'clf_votes': str(bag['clf_votes']),
            }
            print(
                f"  Bagged AUROC={auc:.3f}  "
                f"DeLong SE={d['delong_se']:.4f}  "
                f"95% CI=[{d['delong_ci_lo']:.3f}, {d['delong_ci_hi']:.3f}]  "
                f"AUPRC={ap:.3f}"
            )
        overall_rows.append(overall)

        for et in ('Focal', 'Generalized'):
            mask = (
                _finite_proba_ok(bag['bagged_proba'], ternary_outcome)
                & (bag['epilepsy_type'] == et)
            )
            row = _stratum_metric_row(bag, mask, et, 'epilepsy_type')
            row['model'] = mk
            type_metric_rows.append(row)

        type_delong_rows.extend(focal_vs_generalized_delong_for_model(bag, mk))

        for ps in prior_strata:
            mask = (
                _finite_proba_ok(bag['bagged_proba'], ternary_outcome)
                & (bag['prior_stratum'] == ps)
            )
            row = _stratum_metric_row(bag, mask, ps, 'prior_stratum')
            row['model'] = mk
            prior_metric_rows.append(row)

        prior_delong_rows.extend([
            {**row, 'model': mk}
            for row in pairwise_delong_strata(bag, 'prior_stratum', prior_strata)
        ])

        for pi in PRIOR_INFO_STRATA:
            mask = (
                _finite_proba_ok(bag['bagged_proba'], ternary_outcome)
                & (bag['prior_info'] == pi)
            )
            row = _stratum_metric_row(bag, mask, pi, 'prior_info')
            row['model'] = mk
            prior_info_metric_rows.append(row)

        prior_info_delong_rows.extend([
            {**row, 'model': mk}
            for row in pairwise_delong_strata(bag, 'prior_info', PRIOR_INFO_STRATA)
        ])

    save_table(bag_df, 'bagged_oos_predictions.csv', out_dir)
    save_table(pd.DataFrame(overall_rows), 'bagged_overall_metrics.csv', out_dir)
    save_table(pd.DataFrame(type_metric_rows), 'bagged_metrics_by_epilepsy_type.csv', out_dir)
    save_table(pd.DataFrame(type_delong_rows), 'focal_vs_generalized_delong.csv', out_dir)
    save_table(pd.DataFrame(prior_metric_rows), 'bagged_metrics_by_prior_stratum.csv', out_dir)
    save_table(pd.DataFrame(prior_delong_rows), 'prior_stratum_pairwise_delong.csv', out_dir)
    save_table(pd.DataFrame(prior_info_metric_rows), 'bagged_metrics_by_prior_info.csv', out_dir)
    save_table(
        pd.DataFrame(prior_info_delong_rows),
        'prior_info_available_vs_unavailable_delong.csv',
        out_dir,
    )

    pair_rows = []
    mc_df_pair = pd.DataFrame(mc_rows)
    for ki, kj in combinations(MODEL_ORDER, 2):
        r = pairwise_mc_auroc_sign_test(mc_df_pair, ki, kj)
        pair_rows.append({
            'stratum': 'all',
            'model_a': r['model_a'],
            'model_b': r['model_b'],
            'model_simple': r['model_a'],
            'model_complex': r['model_b'],
            'n_features_simple': len(feature_sets[ki]),
            'n_features_complex': len(feature_sets[kj]),
            'n_splits': r['n_mc'],
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
        })
    bagged_pair_rows = pairwise_model_comparisons(
        bags, MODEL_ORDER, feature_sets,
        rng=np.random.default_rng(1),
        ternary_outcome=ternary_outcome,
    )
    type_pair_rows = []
    prior_info_pair_rows = []
    ref_epi = bags[MODEL_ORDER[0]]['epilepsy_type']
    ref_prior_info = bags[MODEL_ORDER[0]]['prior_info']
    for et in ('Focal', 'Generalized'):
        mask = ref_epi == et
        type_pair_rows.extend(pairwise_model_comparisons(
            bags, MODEL_ORDER, feature_sets,
            mask=mask, stratum=f'epilepsy_type_{et}',
            rng=np.random.default_rng(1),
            ternary_outcome=ternary_outcome,
        ))
    for pi in PRIOR_INFO_STRATA:
        mask = ref_prior_info == pi
        prior_info_pair_rows.extend(pairwise_model_comparisons(
            bags, MODEL_ORDER, feature_sets,
            mask=mask, stratum=f'prior_info_{pi}',
            rng=np.random.default_rng(1),
            ternary_outcome=ternary_outcome,
        ))
    save_table(pd.DataFrame(pair_rows), 'pairwise_model_mc_splits.csv', out_dir)
    save_table(pd.DataFrame(pair_rows), 'pairwise_model_bootstrap.csv', out_dir)
    save_table(
        pd.DataFrame(bagged_pair_rows),
        'pairwise_model_bagged_bootstrap_supplementary.csv',
        out_dir,
    )
    save_table(pd.DataFrame(type_pair_rows), 'pairwise_model_bootstrap_epilepsy_type_strata.csv', out_dir)
    save_table(
        pd.DataFrame(prior_info_pair_rows),
        'pairwise_model_bootstrap_prior_info_strata.csv',
        out_dir,
    )

    print('\n=== Pairwise clinical vs interaction (MC permutation sign test) ===')
    print(pd.DataFrame(pair_rows).to_string(index=False))
    delong_cols = ['model', 'group_a', 'group_b', 'ovr_class', 'auc_a', 'auc_b', 'delta', 'p']
    print('\n=== Focal vs Generalized (bagged DeLong, both models) ===')
    print(pd.DataFrame(type_delong_rows)[delong_cols].to_string(index=False))
    print('\n=== MC summary ===')
    print(pd.DataFrame(mc_summaries).to_string(index=False))
    print('\n=== Bagged metrics by epilepsy type ===')
    print(pd.DataFrame(type_metric_rows).to_string(index=False))
    print('\n=== Bagged metrics by prior 2yr sz freq info (available vs unavailable) ===')
    print(pd.DataFrame(prior_info_metric_rows).to_string(index=False))
    print('\n=== Prior info available vs unavailable (bagged DeLong) ===')
    print(pd.DataFrame(prior_info_delong_rows)[delong_cols].to_string(index=False))
    print('\n=== Clinical vs interaction by prior info (bagged bootstrap) ===')
    print(pd.DataFrame(prior_info_pair_rows).to_string(index=False))
    print(f'\nDone. Outputs in {out_dir}')


if __name__ == '__main__':
    main()
