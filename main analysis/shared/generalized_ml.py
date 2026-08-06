#!/usr/bin/env python3
"""
Generalized epilepsy ML — single entry point.

Outcome: sz_freq_class (seizure control) only — recurrence has been removed.

Modes:
  primary  — single 80/20 split, bootstrap AUROC comparisons, no-IED holdout (single 50/50),
             Model6_all (Table 2 Model 5) permutation importance, IED-rate quartile
             leave-one-group-out analysis (DeLong, IED vs Slowing / IED vs EEG), SHAP
  spike    — leave-one-spike-group-out only (tertile or quartile)
  forest   — DeLong forest plots from spike CSV outputs

Examples:
  python generalized_ml.py primary
  python generalized_ml.py spike --target sz_freq_class --spike-split tertile
  python generalized_ml.py forest --prefix tertile --target sz_freq_class
"""


# ==================== from seizure_frequency_metrics.py ====================
from __future__ import annotations

import json

import numpy as np
import pandas as pd

FOLLOWUP_DAYS = 730
POOR_CUTOFF_PER_MONTH = 4.33

SZ_FREQ_CLASS_LABEL = (
    'Poor vs moderate control from post-EEG MeanSzFreq (NOT MATLAB all-visit MeanSzFreq): '
    'for each routine EEG, mean Freq_R1 over unique clinic days in (0, 2y], then averaged '
    'across valid routine sessions per patient; EEG features = mean across valid sessions'
)


def parse_json(s):
    s = str(s).strip() if pd.notna(s) else ''
    if s in ('', '[]', 'null', '[null]'):
        return []
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return []


def hasz(v):
    if v is None:
        return np.nan
    try:
        iv = int(v)
        return np.nan if iv == 2 else float(iv)
    except (TypeError, ValueError):
        return np.nan


def _mean_sz_freq_in_eeg_window(
    dates,
    freqs,
    haszs,
    eeg_dt,
    followup_days: int = FOLLOWUP_DAYS,
    *,
    direction: str = 'post',
) -> float:
    """Mean Freq_R1 over unique visit days in a pre- or post-EEG window.

    Rule 1 (HasSz==0, missing freq → 0), then mean across qualifying visit days.
      post:  0 < (visit - EEG) <= followup_days
      pre:   -followup_days <= (visit - EEG) < 0
    """
    if direction not in ('post', 'pre'):
        raise ValueError(f'direction must be post|pre, got {direction!r}')

    n = len(dates)
    if not dates or len(freqs) != n:
        return np.nan
    if len(haszs) != n:
        haszs = [None] * n

    try:
        eeg_dt = pd.to_datetime(eeg_dt)
    except (TypeError, ValueError):
        return np.nan

    day_buckets: dict = {}
    for d, f_raw, h_raw in zip(dates, freqs, haszs):
        try:
            vdt = pd.to_datetime(str(d).strip())
            gap = (vdt - eeg_dt).days
            if direction == 'post':
                if gap <= 0 or gap > followup_days:
                    continue
            else:
                if gap >= 0 or gap < -followup_days:
                    continue
            try:
                f = float(f_raw) if f_raw is not None else np.nan
            except (TypeError, ValueError):
                f = np.nan
            if not np.isfinite(f) or f < 0:
                f = np.nan
            key = vdt.date()
            if key not in day_buckets:
                day_buckets[key] = {'freqs': [], 'haszs': []}
            day_buckets[key]['freqs'].append(f)
            day_buckets[key]['haszs'].append(hasz(h_raw))
        except (TypeError, ValueError):
            pass

    freq_r1 = []
    for vals in day_buckets.values():
        valid_f = [f for f in vals['freqs'] if np.isfinite(f)]
        freq_agg = float(np.mean(valid_f)) if valid_f else np.nan
        valid_h = [h for h in vals['haszs'] if np.isfinite(h)]
        has_agg = float(max(valid_h)) if valid_h else np.nan
        if not np.isfinite(freq_agg) and np.isfinite(has_agg) and has_agg == 0.0:
            freq_agg = 0.0
        if np.isfinite(freq_agg):
            freq_r1.append(freq_agg)

    return float(np.mean(freq_r1)) if freq_r1 else np.nan


def mean_sz_freq_in_post_eeg_window(
    dates,
    freqs,
    haszs,
    eeg_dt,
    followup_days: int = FOLLOWUP_DAYS,
) -> float:
    """Mean Freq_R1 over unique visit days with 0 < (visit - EEG) <= followup_days.

    ML sz_freq_class outcome (not MATLAB all-visit MeanSzFreq). Uses
    build_visit_level_table_R1 + Rule 1 (HasSz==0, missing freq → 0), restricted
    to the post-EEG window, then mean across qualifying visit days.
    """
    return _mean_sz_freq_in_eeg_window(
        dates, freqs, haszs, eeg_dt, followup_days=followup_days, direction='post',
    )


def mean_sz_freq_in_pre_eeg_window(
    dates,
    freqs,
    haszs,
    eeg_dt,
    followup_days: int = FOLLOWUP_DAYS,
) -> float:
    """Mean Freq_R1 over unique visit days with -followup_days <= (visit - EEG) < 0."""
    return _mean_sz_freq_in_eeg_window(
        dates, freqs, haszs, eeg_dt, followup_days=followup_days, direction='pre',
    )


def add_session_mean_sz_freq_2yr(
    session_df: pd.DataFrame,
    followup_days: int = FOLLOWUP_DAYS,
) -> pd.DataFrame:
    """Add mean_sz_freq_2yr per EEG session row (requires visit arrays + start_time_deid)."""
    out = session_df.copy()

    def _row(row):
        try:
            eeg_dt = pd.to_datetime(str(row['start_time_deid']).strip())
        except (TypeError, ValueError):
            return np.nan
        return mean_sz_freq_in_post_eeg_window(
            parse_json(row.get('visit_dates_deid', '[]')),
            parse_json(row.get('sz_freqs', '[]')),
            parse_json(row.get('visit_hasSz', '[]')),
            eeg_dt,
            followup_days=followup_days,
        )

    out['mean_sz_freq_2yr'] = out.apply(_row, axis=1)
    return out


def add_session_mean_sz_freq_2yr_pre(
    session_df: pd.DataFrame,
    followup_days: int = FOLLOWUP_DAYS,
) -> pd.DataFrame:
    """Add mean_sz_freq_2yr_pre per EEG session (2y clinic window before EEG)."""
    out = session_df.copy()

    def _row(row):
        try:
            eeg_dt = pd.to_datetime(str(row['start_time_deid']).strip())
        except (TypeError, ValueError):
            return np.nan
        return mean_sz_freq_in_pre_eeg_window(
            parse_json(row.get('visit_dates_deid', '[]')),
            parse_json(row.get('sz_freqs', '[]')),
            parse_json(row.get('visit_hasSz', '[]')),
            eeg_dt,
            followup_days=followup_days,
        )

    out['mean_sz_freq_2yr_pre'] = out.apply(_row, axis=1)
    return out


def build_patient_sz_from_session_means(
    session_df: pd.DataFrame,
    poor_cutoff: float = POOR_CUTOFF_PER_MONTH,
) -> pd.DataFrame:
    """Patient MeanSzFreq = mean of session mean_sz_freq_2yr (finite values only)."""
    if 'mean_sz_freq_2yr' not in session_df.columns:
        raise ValueError('session_df must include mean_sz_freq_2yr')
    valid = session_df.dropna(subset=['mean_sz_freq_2yr'])
    if valid.empty:
        return pd.DataFrame(columns=['patient_id', 'mean_sz_freq', 'sz_freq_class'])
    out = (
        valid.groupby('patient_id', as_index=False)['mean_sz_freq_2yr']
        .mean()
        .rename(columns={'mean_sz_freq_2yr': 'mean_sz_freq'})
    )
    out['sz_freq_class'] = np.where(
        out['mean_sz_freq'].notna(),
        (out['mean_sz_freq'] > poor_cutoff).astype(float),
        np.nan,
    )
    return out


def build_patient_prior_sz_control(
    session_df: pd.DataFrame,
    poor_cutoff: float = POOR_CUTOFF_PER_MONTH,
) -> pd.DataFrame:
    """Patient pre-EEG 2y seizure control coded as 0 / 1 / 2 / 3.

    0 = seizure-free (mean pre-EEG MeanSzFreq == 0)
    1 = moderate    (0 < mean pre-EEG MeanSzFreq <= cutoff)
    2 = poor        (mean pre-EEG MeanSzFreq > cutoff)
    3 = not available (no documented pre-EEG visits in the 2y window)

    Patient mean is the mean of session-level mean_sz_freq_2yr_pre over sessions
    with a finite pre-window value (same aggregation style as the post-EEG label).
    """
    if 'mean_sz_freq_2yr_pre' not in session_df.columns:
        raise ValueError('session_df must include mean_sz_freq_2yr_pre')

    pid = session_df['patient_id'].astype(str)
    all_pids = pd.DataFrame({'patient_id': pid.unique()})
    valid = session_df.loc[session_df['mean_sz_freq_2yr_pre'].notna()].copy()
    valid['patient_id'] = valid['patient_id'].astype(str)
    if valid.empty:
        all_pids['mean_sz_freq_pre'] = np.nan
        all_pids['prior_sz_control'] = 3
        return all_pids

    means = (
        valid.groupby('patient_id', as_index=False)['mean_sz_freq_2yr_pre']
        .mean()
        .rename(columns={'mean_sz_freq_2yr_pre': 'mean_sz_freq_pre'})
    )
    out = all_pids.merge(means, on='patient_id', how='left')
    freq = out['mean_sz_freq_pre']
    prior = np.full(len(out), 3, dtype=int)  # not available
    has = freq.notna()
    prior[has & (freq == 0)] = 0  # seizure-free
    prior[has & (freq > 0) & (freq <= poor_cutoff)] = 1  # moderate
    prior[has & (freq > poor_cutoff)] = 2  # poor
    out['prior_sz_control'] = prior
    return out

# ==================== from ml_classifier_selection.py ====================

import numpy as np


def select_best_clf_by_test_auroc(
    proba: dict,
    y_test,
    model_keys,
    classifiers: dict,
    eval_metrics,
) -> dict[str, str]:
    """Pick the classifier with highest AUROC on the evaluation (test) fold."""
    default_clf = list(classifiers.keys())[0]
    y_test = np.asarray(y_test, dtype=int)
    best_map: dict[str, str] = {}
    for model_key in model_keys:
        best_clf = default_clf
        best_auroc = -np.inf
        for clf_key in classifiers:
            p = proba.get(model_key, {}).get(clf_key)
            if p is None:
                continue
            m = eval_metrics(y_test, p)
            auroc = m.get('auroc', np.nan) if isinstance(m, dict) else np.nan
            if np.isfinite(auroc) and auroc > best_auroc:
                best_auroc = auroc
                best_clf = clf_key
        best_map[model_key] = best_clf
    return best_map

# ==================== from slowing_session_features.py ====================

import glob
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd

_RE_PID_SES = re.compile(r'sub-Penn(\d+)_ses-(\d+)')

# Required for a session to serve as index EEG in Python ML pipelines.
INDEX_EEG_REQUIRE_COLS = ['age_at_eeg', 'gender', 'spike_rate_per_hour']

RECURRENCE_2YR_LABEL = (
    'Any clinic visit with HasSz==1 within 2y after the index (earliest valid) EEG'
)


def _clip_stats(path):
    fname = os.path.splitext(os.path.basename(path))[0]
    try:
        cls = pd.read_csv(path, usecols=['pred_class'])['pred_class'].to_numpy()
        n = len(cls)
        if n == 0:
            return fname, np.nan, np.nan, np.nan, 0
        return (
            fname,
            100 * np.mean(cls == 0),
            100 * np.mean(cls == 1),
            100 * np.mean(cls == 2),
            n,
        )
    except Exception:
        return fname, np.nan, np.nan, np.nan, 0


def build_session_slowing_table(step1_roots: str | list[str], n_jobs: int = 1) -> pd.DataFrame | None:
    """Build session-level slowing table from Morgoth pred_SLOWING_1sStep CSVs."""
    if isinstance(step1_roots, str):
        step1_roots = [step1_roots]

    paths = []
    for root in step1_roots:
        step1_dir = os.path.join(root, 'pred_SLOWING_1sStep')
        if os.path.isdir(step1_dir):
            paths.extend(sorted(glob.glob(os.path.join(step1_dir, '*.csv'))))

    if not paths:
        return None

    with Pool(n_jobs) as pool:
        results = pool.map(_clip_stats, paths)

    pct_df = pd.DataFrame(
        results,
        columns=['file_name', 'pct_no_slowing', 'pct_foc_slowing', 'pct_gen_slowing', 'n_win_slow'],
    )
    parsed = pct_df['file_name'].map(lambda s: _RE_PID_SES.search(str(s)))
    pct_df['patient_id'] = parsed.map(lambda m: m.group(1) if m else None)
    pct_df['session'] = parsed.map(lambda m: m.group(2) if m else None)
    pct_df = pct_df.dropna(subset=['patient_id'])

    pct_cols = ['pct_no_slowing', 'pct_foc_slowing', 'pct_gen_slowing']
    rows = []
    for (pid, ses), grp in pct_df.groupby(['patient_id', 'session']):
        w = grp['n_win_slow'].to_numpy(float)
        row = {'patient_id': pid, 'session': ses, 'n_win_slow': float(w.sum())}
        for col in pct_cols:
            v = grp[col].to_numpy(float)
            ok = np.isfinite(v)
            ws = w[ok].sum()
            row[col] = float(np.dot(w[ok], v[ok]) / ws) if ws > 0 else np.nan
        rows.append(row)

    slow_df = pd.DataFrame(rows)
    slow_df[['patient_id', 'session']] = slow_df[['patient_id', 'session']].astype(str)
    return slow_df


def select_first_index_eeg_per_patient(
    session_df: pd.DataFrame,
    patient_col: str = 'patient_id',
    time_col: str = 'start_time_deid',
    session_col: str = 'session',
    require_cols: list[str] | None = None,
    require_slowing: bool = False,
) -> pd.DataFrame:
    """Keep the earliest *valid* routine EEG session per patient (index event).

    Sessions are sorted chronologically; the first row passing validity checks
    is chosen. Invalid sessions are skipped in favour of the next earliest
    session for that patient. Patients with no valid session are dropped.

    When require_slowing is True (MATLAB OR/PPV parity), the session must have
    finite pct_foc_slowing and/or pct_gen_slowing.
    """
    if require_cols is None:
        require_cols = list(INDEX_EEG_REQUIRE_COLS)

    df = _prepare_index_sort_cols(session_df, patient_col, time_col, session_col)
    valid_mask = df.apply(
        lambda row: _row_valid_index(row, require_cols, require_slowing), axis=1)

    picked = []
    for _, grp in df.groupby(patient_col, sort=False):
        valid_rows = grp[valid_mask.loc[grp.index]]
        if len(valid_rows):
            picked.append(valid_rows.iloc[0])
    if not picked:
        return session_df.iloc[0:0].copy()
    out = pd.DataFrame(picked).drop(columns=['_index_time', '_session_ord'])
    return out.reset_index(drop=True)


def _prepare_index_sort_cols(session_df, patient_col, time_col, session_col):
    df = session_df.copy()
    df['_index_time'] = pd.to_datetime(df[time_col], errors='coerce')
    df['_session_ord'] = pd.to_numeric(df[session_col], errors='coerce')
    return df.sort_values([patient_col, '_index_time', '_session_ord'], na_position='last')


def _row_valid_index(row, require_cols, require_slowing=False) -> bool:
    if pd.isna(row['_index_time']):
        return False
    for col in require_cols:
        if col not in row.index:
            return False
        if pd.isna(row[col]):
            return False
    if require_slowing:
        foc = row.get('pct_foc_slowing', np.nan)
        gen = row.get('pct_gen_slowing', np.nan)
        if pd.isna(foc) and pd.isna(gen):
            return False
    return True


def filter_valid_eeg_sessions(
    session_df: pd.DataFrame,
    patient_col: str = 'patient_id',
    time_col: str = 'start_time_deid',
    session_col: str = 'session',
    require_cols: list[str] | None = None,
    require_slowing: bool = False,
) -> pd.DataFrame:
    """All routine EEG sessions passing index EEG validity checks."""
    if require_cols is None:
        require_cols = list(INDEX_EEG_REQUIRE_COLS)
    df = _prepare_index_sort_cols(session_df, patient_col, time_col, session_col)
    valid_mask = df.apply(
        lambda row: _row_valid_index(row, require_cols, require_slowing), axis=1)
    out = df.loc[valid_mask].drop(columns=['_index_time', '_session_ord'])
    return out.reset_index(drop=True)


def log_index_eeg_grouping(before_df: pd.DataFrame, after_df: pd.DataFrame,
                           patient_col: str = 'patient_id') -> None:
    """Print index EEG selection summary (shared by General and Focal ML)."""
    n_sessions = len(before_df)
    n_dropped = before_df[patient_col].nunique() - after_df[patient_col].nunique()
    print(f"[Index EEG] Earliest valid session per patient: {len(after_df)} patients "
          f"(from {n_sessions} routine sessions; {n_dropped} patients with no valid session)")


def aggregate_mean_eeg_per_patient(
    session_df: pd.DataFrame,
    patient_sz_df: pd.DataFrame | None = None,
    n_eeg_sessions_df: pd.DataFrame | None = None,
    require_cols: list[str] | None = None,
    require_slowing: bool = False,
    poor_cutoff_per_month: float = 4.33,
) -> pd.DataFrame:
    """Mean EEG features across all valid routine sessions per patient (sz_freq_class).

    When mean_sz_freq_2yr is present on session rows, patient mean_sz_freq and
    sz_freq_class are derived from the mean of session-level 2-year-window values.
    """
    if require_cols is None:
        require_cols = list(INDEX_EEG_REQUIRE_COLS)

    valid = filter_valid_eeg_sessions(
        session_df, require_cols=require_cols, require_slowing=require_slowing)
    if valid.empty:
        return session_df.iloc[0:0].copy()

    bg_prefixes = ('frontal_', 'central_', 'temporal_', 'parietal_', 'occipital_')
    feature_cols = [
        'spike_rate_per_hour',
        'pct_no_slowing', 'pct_foc_slowing', 'pct_gen_slowing',
    ]
    feature_cols += [c for c in valid.columns if c.startswith(bg_prefixes)]
    feature_cols = [c for c in feature_cols if c in valid.columns]
    mean_cols = feature_cols + (['age_at_eeg'] if 'age_at_eeg' in valid.columns else [])

    agg_dict = {c: 'mean' for c in mean_cols}
    if 'gender' in valid.columns:
        agg_dict['gender'] = 'first'

    out = valid.groupby('patient_id', as_index=False).agg(agg_dict)
    sess_counts = (
        valid.groupby('patient_id').size().rename('n_eeg_sessions_used').reset_index())
    out = out.merge(sess_counts, on='patient_id', how='left')
    out['patient_id'] = out['patient_id'].astype(int)

    if 'mean_sz_freq_2yr' in valid.columns:
        sz = build_patient_sz_from_session_means(valid, poor_cutoff=poor_cutoff_per_month)
        sz['patient_id'] = pd.to_numeric(sz['patient_id'], errors='coerce').astype(int)
        out = out.merge(sz, on='patient_id', how='left')
    elif patient_sz_df is not None and not patient_sz_df.empty:
        patient_sz_df = patient_sz_df.copy()
        out = out.merge(patient_sz_df, on='patient_id', how='left')
    if n_eeg_sessions_df is not None:
        out['patient_id'] = out['patient_id'].astype(str)
        out = out.merge(n_eeg_sessions_df, on='patient_id', how='left')
        out['patient_id'] = out['patient_id'].astype(int)
    elif 'n_eeg_sessions' not in out.columns:
        out['n_eeg_sessions'] = out['n_eeg_sessions_used']

    out['patient_id'] = out['patient_id'].astype(str)
    out['spike_rate_log'] = np.log10(out['spike_rate_per_hour'] + 0.1)
    out['mean_sz_freq_log'] = np.log(out['mean_sz_freq'] + 0.1)
    return out


def aggregate_sessions_to_patients(
    session_df: pd.DataFrame,
    patient_sz_df: pd.DataFrame,
    n_eeg_sessions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collapse index EEG sessions (one row/patient) to patient table + MeanSzFreq."""
    bg_prefixes = ('frontal_', 'central_', 'temporal_', 'parietal_', 'occipital_')
    feature_cols = [
        'spike_rate_per_hour',
        'pct_no_slowing', 'pct_foc_slowing', 'pct_gen_slowing',
    ]
    feature_cols += [c for c in session_df.columns if c.startswith(bg_prefixes)]
    feature_cols = [c for c in feature_cols if c in session_df.columns]

    agg = {c: 'first' for c in feature_cols}
    agg['age_at_eeg'] = 'first'
    agg['gender'] = 'first'
    agg['seizure_recurrence_2yr'] = 'first'
    agg['n_followup_visits'] = 'first'

    out = session_df.groupby('patient_id', as_index=False).agg(agg)
    out['patient_id'] = out['patient_id'].astype(int)
    patient_sz_df = patient_sz_df.copy()
    patient_sz_df['patient_id'] = pd.to_numeric(patient_sz_df['patient_id'], errors='coerce').astype('Int64')
    patient_sz_df = patient_sz_df.dropna(subset=['patient_id'])
    patient_sz_df['patient_id'] = patient_sz_df['patient_id'].astype(int)
    out = out.merge(patient_sz_df, on='patient_id', how='left')
    if n_eeg_sessions_df is not None:
        out['patient_id'] = out['patient_id'].astype(str)
        out = out.merge(n_eeg_sessions_df, on='patient_id', how='left')
        out['patient_id'] = out['patient_id'].astype(int)
    else:
        out['n_eeg_sessions'] = 1
    out['patient_id'] = out['patient_id'].astype(str)
    out['spike_rate_log'] = np.log10(out['spike_rate_per_hour'] + 0.1)
    out['mean_sz_freq_log'] = np.log(out['mean_sz_freq'] + 0.1)
    return out

# ==================== from shap_beeswarm_io.py ====================

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def _bundle_dir(out_dir: str | Path, target: str) -> Path:
    return Path(out_dir) / f'shap_beeswarm_{target}'


def _model_prefix(model_key: str) -> str:
    return model_key.replace('/', '_')


def save_shap_beeswarm_bundle(
    out_dir: str | Path,
    target: str,
    model_key: str,
    clf_key: str,
    feature_names: Sequence[str],
    shap_iter_results: Iterable[Mapping[str, Any] | None],
    explainer_label: str,
    cohort: str = '',
) -> Path | None:
    """Write combined arrays, manifest, per-split NPZ, and metadata JSON."""
    valid = [r for r in shap_iter_results if r is not None]
    if not valid:
        return None

    bundle = _bundle_dir(out_dir, target)
    splits_dir = bundle / 'splits'
    bundle.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    prefix = _model_prefix(model_key)
    shap_chunks: list[np.ndarray] = []
    feat_chunks: list[np.ndarray] = []
    manifest_rows: list[dict[str, Any]] = []
    row_idx = 0

    for r in valid:
        shap_vals = np.asarray(r['shap_values'], dtype=np.float32)
        feat_vals = np.asarray(r['feature_values'], dtype=np.float32)
        pids = np.asarray(r['patient_ids'])
        y_true = np.asarray(r['y_true'])
        seed = int(r['mc_seed'])
        n = shap_vals.shape[0]

        shap_chunks.append(shap_vals)
        feat_chunks.append(feat_vals)

        split_path = splits_dir / f'{prefix}_seed_{seed:03d}.npz'
        np.savez_compressed(
            split_path,
            shap_values=shap_vals,
            feature_values=feat_vals,
            patient_ids=pids,
            y_true=y_true,
            mc_seed=np.array([seed], dtype=np.int32),
            n_background=np.array([int(r['n_background'])], dtype=np.int32),
        )

        for i in range(n):
            manifest_rows.append({
                'row_idx': row_idx + i,
                'mc_seed': seed,
                'patient_id': pids[i],
                'y_true': int(y_true[i]),
                'split_file': split_path.name,
            })
        row_idx += n

    shap_all = np.vstack(shap_chunks)
    feat_all = np.vstack(feat_chunks)
    manifest = pd.DataFrame(manifest_rows)

    np.save(bundle / f'{prefix}_shap_values.npy', shap_all)
    np.save(bundle / f'{prefix}_feature_values.npy', feat_all)
    manifest.to_csv(bundle / f'{prefix}_manifest.csv', index=False)

    meta = {
        'target': target,
        'model': model_key,
        'classifier': clf_key,
        'shap_explainer': explainer_label,
        'cohort': cohort,
        'feature_names': list(feature_names),
        'n_features': len(feature_names),
        'n_rows': int(shap_all.shape[0]),
        'n_mc_splits': len(valid),
        'mc_seeds': sorted(int(r['mc_seed']) for r in valid),
        'files': {
            'shap_values': f'{prefix}_shap_values.npy',
            'feature_values': f'{prefix}_feature_values.npy',
            'manifest': f'{prefix}_manifest.csv',
            'splits_dir': 'splits',
        },
    }
    meta_path = bundle / f'{prefix}_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2)

    return bundle


def load_shap_beeswarm_bundle(
    out_dir: str | Path,
    target: str,
    model_key: str,
) -> dict[str, Any]:
    """Load metadata + arrays for one model/target bundle."""
    bundle = _bundle_dir(out_dir, target)
    prefix = _model_prefix(model_key)
    meta_path = bundle / f'{prefix}_meta.json'
    if not meta_path.is_file():
        raise FileNotFoundError(f'Missing SHAP beeswarm bundle: {meta_path}')

    with open(meta_path, encoding='utf-8') as fh:
        meta = json.load(fh)

    shap_values = np.load(bundle / meta['files']['shap_values'])
    feature_values = np.load(bundle / meta['files']['feature_values'])
    manifest = pd.read_csv(bundle / meta['files']['manifest'])
    return {
        'meta': meta,
        'shap_values': shap_values,
        'feature_values': feature_values,
        'manifest': manifest,
        'bundle_dir': bundle,
    }


def select_beeswarm_rows(
    bundle: Mapping[str, Any],
    mc_seed: int | None = 0,
    pool_splits: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Select rows for plotting.

    mc_seed: use only one MC split (default 0). None = all rows.
    pool_splits='mean': average SHAP / feature values per patient across splits.
    """
    shap_values = bundle['shap_values']
    feature_values = bundle['feature_values']
    manifest: pd.DataFrame = bundle['manifest']
    feature_names: list[str] = bundle['meta']['feature_names']

    if pool_splits == 'mean':
        rows = []
        for pid, grp in manifest.groupby('patient_id'):
            idx = grp['row_idx'].values.astype(int)
            rows.append({
                'shap': shap_values[idx].mean(axis=0),
                'feat': feature_values[idx].mean(axis=0),
            })
        if not rows:
            return shap_values[:0], feature_values[:0], feature_names
        shap_sel = np.vstack([r['shap'] for r in rows])
        feat_sel = np.vstack([r['feat'] for r in rows])
        return shap_sel, feat_sel, feature_names

    if mc_seed is not None:
        idx = manifest.loc[manifest['mc_seed'] == mc_seed, 'row_idx'].values.astype(int)
        return shap_values[idx], feature_values[idx], feature_names

    return shap_values, feature_values, feature_names


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: Sequence[str],
    out_path: str | Path,
    *,
    title: str = '',
    top_n: int = 20,
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 150,
) -> Path:
    """Classic SHAP summary (beeswarm) plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import shap

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if shap_values.size == 0:
        raise ValueError('No SHAP rows selected for beeswarm plot')

    plt.figure(figsize=figsize)
    shap.summary_plot(
        shap_values,
        feature_values,
        feature_names=list(feature_names),
        max_display=top_n,
        show=False,
    )
    if title:
        plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return out_path


def plot_shap_beeswarm_default(
    out_dir: str | Path,
    target: str,
    model_key: str,
    *,
    mc_seed: int = 0,
    top_n: int = 20,
) -> Path | None:
    """Load bundle and write default beeswarm PNG next to other SHAP outputs."""
    bundle = load_shap_beeswarm_bundle(out_dir, target, model_key)
    shap_sel, feat_sel, names = select_beeswarm_rows(bundle, mc_seed=mc_seed)
    meta = bundle['meta']
    title = (
        f"SHAP summary — {target}\n"
        f"{meta['model']} / {meta['classifier']} ({meta['shap_explainer']}); "
        f"MC seed {mc_seed}, n={len(shap_sel)}"
    )
    out_path = Path(out_dir) / f"shap_beeswarm_{target}_{_model_prefix(model_key)}.png"
    plot_shap_beeswarm(
        shap_sel, feat_sel, names, out_path, title=title, top_n=top_n,
    )
    return out_path

# ==================== primary MC analysis ====================
import json, glob, os, re, sys, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_approximation import RBFSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

# ---------------------------------------------------------------------------
# Config — pioneer (pioneer.seas.upenn.edu) or local (Conrad_Lab / SEIZURE_FREQ_ROOT)
# ---------------------------------------------------------------------------

from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PIONEER_ROOT = Path('/mnt/sauce/littlab/users/cho6/seizure_frequency_preprocessing')


def _resolve_project_root() -> Path:
    """Folder containing spike_counts.csv (repo root or Conrad_Lab root)."""
    if os.environ.get('SEIZURE_FREQ_ROOT'):
        return Path(os.environ['SEIZURE_FREQ_ROOT']).expanduser().resolve()
    for candidate in [
        _SCRIPT_DIR,
        _SCRIPT_DIR.parent,
        _SCRIPT_DIR.parent.parent,
        _SCRIPT_DIR.parent.parent.parent,
        _PIONEER_ROOT,
    ]:
        if (candidate / 'spike_counts.csv').is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot find project root (spike_counts.csv). "
        "On pioneer: run from paper_approach/ml_analysis/. "
        "Locally: open Conrad_Lab in Cursor or set SEIZURE_FREQ_ROOT."
    )


ROOT        = str(_resolve_project_root())
SPIKE_CSV   = os.path.join(ROOT, 'spike_counts.csv')
REPORT_CSV  = os.path.join(ROOT, 'Routineeegpec-Deidreport_DATA_LABELS_2025-12-12_1339 (2).csv')
BG_FEAT_CSV = os.path.join(ROOT, 'paper_approach', 'background_features.csv')
SLOWING_DIR = os.path.join(ROOT, 'morgoth', 'results', 'light_SLOWING_General')
LOG_DIR     = os.path.join(str(_SCRIPT_DIR), 'logs')
OUT_DIR     = os.environ.get(
    'ML_OUT_DIR',
    os.path.join(ROOT, 'paper_approach', 'ml_analysis', 'outputs', 'General'),
)

PRIMARY_SPLIT_SEED  = int(os.environ.get('ML_PRIMARY_SPLIT_SEED', '0'))
HOLDOUT_SPLIT_SEED  = int(os.environ.get('ML_HOLDOUT_SPLIT_SEED', str(PRIMARY_SPLIT_SEED)))
TEST_SIZE           = 0.20
N_CV_FOLDS          = int(os.environ.get('ML_N_CV_FOLDS', '5'))
N_BOOTSTRAP         = int(os.environ.get('ML_N_BOOTSTRAP', '1000'))
PRIMARY_SPIKE_SPLITS = [
    s.strip() for s in os.environ.get('ML_SPIKE_SPLITS', 'tertile').split(',') if s.strip()
]
PERM_MODEL_KEY      = 'Model6_all'
PERM_N_REPEATS      = int(os.environ.get('ML_PERM_N_REPEATS', '10'))
PERM_TOP_N_PLOT     = int(os.environ.get('ML_PERM_TOP_N', '10'))
PERM_BAR_COLOR      = '#2EB8AD'


def _check_required_files():
    checks = [
        ('spike_counts', SPIKE_CSV),
        ('report CSV', REPORT_CSV),
        ('background_features', BG_FEAT_CSV),
        ('slowing step1 dir', os.path.join(SLOWING_DIR, 'pred_SLOWING_1sStep')),
    ]
    missing = [(label, path) for label, path in checks if not os.path.exists(path)]
    if missing:
        print(f"PROJECT ROOT: {ROOT}")
        print("MISSING FILES:")
        for label, path in missing:
            print(f"  [{label}] {path}")
        raise SystemExit(1)


TARGETS                 = ['sz_freq_class']

TARGET_META = {
    'sz_freq_class': {
        'kind': 'outcome',
        'label': SZ_FREQ_CLASS_LABEL,
        'short': 'seizure_control',
        'perm_title': 'Seizure control',
    },
}
FOLLOWUP_DAYS         = 730
POOR_CUTOFF_PER_MONTH = 4.33

SHAP_MODELS       = ['Model4_eeg', 'Model6_all']
N_SHAP_MC_SPLITS  = int(os.environ.get('ML_SHAP_MC_SPLITS', '50'))
SHAP_BG_SIZE      = int(os.environ.get('ML_SHAP_BG_SIZE', '100'))
SHAP_TEST_MAX     = int(os.environ.get('ML_SHAP_TEST_MAX', '200'))
SHAP_TOP_N_PLOT   = 20
SHAP_TOP_N_LOG    = 10
SHAP_BEESWARM_SEED = int(os.environ.get('ML_SHAP_BEESWARM_SEED', '0'))
SHAP_BEESWARM_TOP_N = int(os.environ.get('ML_SHAP_BEESWARM_TOP_N', '20'))
PCA_VARIANCE      = 0.95  # retained by fit_clf_fixed default arg (PCA unused in analyses)
TOTAL_CORES       = int(os.environ.get('ML_TOTAL_CORES', '20'))  # pioneer default 20; set lower if sharing node
PARALLEL_COHORTS  = 1
N_JOBS            = max(1, TOTAL_CORES // PARALLEL_COHORTS)
EPI_TYPE          = 'General'
METRICS           = ['auroc']


def save_table(df, filename):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved {path}")


TARGET_PATIENT_DATA = {}
FEATURE_SETS_CLEAN = {}
dataset_patient = None

_PRIMARY_LOADED = False


def _load_primary_data():
    global _PRIMARY_LOADED, TARGET_PATIENT_DATA, FEATURE_SETS_CLEAN, dataset_patient
    if _PRIMARY_LOADED:
        return
    _check_required_files()
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    print('=' * 80)
    print(f'ML Analysis — {EPI_TYPE}')
    print(f'ROOT:      {ROOT}')
    print(f'OUT_DIR:   {OUT_DIR}')
    print(f'LOG_DIR:   {LOG_DIR}')
    print(f'Workers:   {N_JOBS} parallel MC CV splits (ML_TOTAL_CORES={TOTAL_CORES})')
    print(f'Splits: primary=single 80/20 (seed={PRIMARY_SPLIT_SEED}), '
          f'holdout=single 50/50 no-IED (seed={HOLDOUT_SPLIT_SEED}), '
          f'shap={N_SHAP_MC_SPLITS}, cv_folds={N_CV_FOLDS}, '
          f'spike_strat={PRIMARY_SPIKE_SPLITS}')
    print(f'Targets:   {TARGETS}')
    print('=' * 80)

    # ===========================================================================
    # 1. Load and prepare data  (runs once, outside target loop)
    # ===========================================================================

    sc  = pd.read_csv(SPIKE_CSV)
    rep = pd.read_csv(REPORT_CSV)

    sc['count_0_46']   = pd.to_numeric(sc['count_0_46'],   errors='coerce')
    sc['Duration_sec'] = pd.to_numeric(sc['Duration_sec'], errors='coerce')
    # Spike rate uses light-preprocessed segment spans (same track as slowing);
    # Duration_sec still gates routine EEGs (<=4h).
    from pathlib import Path as _Path
    light_path = _Path(ROOT) / 'light_preprocessed_duration.csv'
    light = pd.read_csv(light_path)
    sc = sc.merge(
        light.rename(columns={'Patient': 'Patient', 'Session': 'Session'})[
            ['Patient', 'Session', 'light_duration_sec']
        ],
        on=['Patient', 'Session'], how='left',
    )
    sc['spike_rate_per_hour'] = (
        sc['count_0_46'] / pd.to_numeric(sc['light_duration_sec'], errors='coerce') * 3600
    )
    sc.loc[
        ~np.isfinite(sc['light_duration_sec']) | (sc['light_duration_sec'] <= 0),
        'spike_rate_per_hour',
    ] = np.nan
    sc_routine = sc[sc['Duration_sec'].notna() & (sc['Duration_sec'] <= 4 * 3600)].copy()
    sc_routine  = sc_routine.rename(columns={'Patient': 'patient_id', 'Session': 'session_number'})
    sc_routine[['patient_id','session_number']] = sc_routine[['patient_id','session_number']].astype(int)

    acq = rep['acquired_on'].fillna('').str.lower().str.strip()
    cls = rep['report_PATIENT_CLASS'].fillna('').str.lower().str.strip()
    jay = rep['jay_in_or_out'].fillna('').str.lower().str.strip()
    is_outpt = (acq.str.contains('spe') | acq.str.contains('radnor') |
                (cls == 'outpatient') | (jay == 'out'))
    outpt_keys = rep[is_outpt][['patient_id','session_number']].drop_duplicates().dropna().astype(int)

    rep['patient_id']     = pd.to_numeric(rep['patient_id'],     errors='coerce').astype('Int64')
    rep['session_number'] = pd.to_numeric(rep['session_number'], errors='coerce').astype('Int64')
    rep_filt = rep.merge(outpt_keys, on=['patient_id','session_number'])
    merged   = sc_routine.merge(rep_filt, on=['patient_id','session_number'])

    # ── patient-level epilepsy typing ───────────────────────────────────────────
    NESD_LABEL = "non-epileptic seizure disorder"
    BAD_TYPES  = {"uncertain if epilepsy", "unknown or mrn not found", ""}

    pat_typing = rep[['patient_id','epilepsy_type','epilepsy_specific']].copy()
    pat_typing['patient_id'] = pd.to_numeric(pat_typing['patient_id'], errors='coerce')
    pat_typing = pat_typing.dropna(subset=['patient_id'])
    pat_typing['patient_id'] = pat_typing['patient_id'].astype(int)
    pat_typing['etype_norm'] = pat_typing['epilepsy_type'].fillna('').str.lower().str.strip()
    pat_typing['spec_norm']  = pat_typing['epilepsy_specific'].fillna('').str.lower().str.strip()

    def _dedupe_patient_type(grp):
        etypes = [e for e in grp['etype_norm'].unique() if e != '']
        specs  = [s for s in grp['spec_norm'].unique()  if s != '']
        return pd.Series({'etype_norm': etypes[0] if etypes else '',
                          'spec_norm':  specs[0]  if specs  else ''})

    pat_type_dedup = pat_typing.groupby('patient_id', group_keys=False).apply(_dedupe_patient_type).reset_index()
    pat_type_dedup['EpiType3'] = ''
    pat_type_dedup.loc[pat_type_dedup['spec_norm'].str.contains('temporal'), 'EpiType3'] = 'Temporal'
    pat_type_dedup.loc[pat_type_dedup['spec_norm'].str.contains('frontal'),  'EpiType3'] = 'Frontal'
    pat_type_dedup.loc[(pat_type_dedup['EpiType3'] == '') & (pat_type_dedup['etype_norm'] == 'general'), 'EpiType3'] = 'General'

    is_nesd     = pat_type_dedup['etype_norm'] == NESD_LABEL
    is_bad_type = pat_type_dedup['etype_norm'].isin(BAD_TYPES)
    valid_pats  = pat_type_dedup.loc[~is_nesd & ~is_bad_type, 'patient_id']
    print(f"[Typing] Excluded: {is_nesd.sum()} NESD, {is_bad_type.sum()} bad/unknown type")

    merged = merged[merged['patient_id'].isin(valid_pats)].copy()
    merged = merged.drop(columns=['EpiType3'], errors='ignore')
    merged = merged.merge(pat_type_dedup[['patient_id','EpiType3']], on='patient_id', how='left')

    cohort = merged[merged['EpiType3'] == 'General'].copy()
    print(f"General epilepsy: {len(cohort)} patient-sessions, {cohort['patient_id'].nunique()} unique patients")

    # ── visit type filter ────────────────────────────────────────────────────────
    ALLOWABLE_VISITS = {
        "CONSULT VISIT", "ESTABLISHED PATIENT VISIT",
        "FOLLOW-UP PATIENT CLINIC", "NEW PATIENT CLINIC", "NEW PATIENT VISIT",
        "NPV MANAGEMENT DURING COVID-19", "NPV NEUROLOGY",
        "RETURN ANNUAL VISIT", "RETURN PATIENT EXTENDED", "RETURN PATIENT VISIT",
        "RPV MANAGEMENT DURING COVID-19", "TELEHEALTH VIDEO VISIT RETURN",
    }

    def _filter_visit_arrays(row):
        def _parse(s, default):
            s = str(s).strip() if pd.notna(s) else ''
            if s in ('', '[]', 'null', '[null]', '<missing>'): return default
            try:    return json.loads(s)
            except: return default

        vt    = _parse(row.get('visit_type'),       [])
        dates = _parse(row.get('visit_dates_deid'), [])
        szf   = _parse(row.get('sz_freqs'),         [])
        hs    = _parse(row.get('visit_hasSz'),       [])

        if not vt or not (len(vt) == len(dates) == len(szf) == len(hs)):
            return pd.Series({'visit_type': '[]', 'visit_dates_deid': '[]',
                              'sz_freqs':   '[]', 'visit_hasSz':      '[]'})

        keep = [i for i, v in enumerate(vt) if str(v).strip() in ALLOWABLE_VISITS]
        if not keep:
            return pd.Series({'visit_type': '[]', 'visit_dates_deid': '[]',
                              'sz_freqs':   '[]', 'visit_hasSz':      '[]'})

        return pd.Series({
            'visit_type':       json.dumps([vt[i]    for i in keep]),
            'visit_dates_deid': json.dumps([dates[i] for i in keep]),
            'sz_freqs':         json.dumps([szf[i]   for i in keep]),
            'visit_hasSz':      json.dumps([hs[i]    for i in keep]),
        })

    filtered_arrays = cohort.apply(_filter_visit_arrays, axis=1)
    cohort[['visit_type','visit_dates_deid','sz_freqs','visit_hasSz']] = filtered_arrays

    n_with_visits = (cohort['visit_dates_deid'] != '[]').sum()
    print(f"[Visit-type filter] {n_with_visits}/{len(cohort)} patient-sessions retain >=1 qualifying clinic visit")

    # ── clinical variables ───────────────────────────────────────────────────────

    def _parse_json(s):
        s = str(s).strip() if pd.notna(s) else ''
        if s in ('', '[]', 'null', '[null]'): return []
        try: return json.loads(s)
        except: return []

    def compute_age(row):
        try:
            return (pd.to_datetime(str(row['start_time_deid']).strip()) -
                    pd.to_datetime(str(row['deid_birth_date']).strip())).days / 365.25
        except: return np.nan

    cohort['age_at_eeg'] = cohort.apply(compute_age, axis=1)
    cohort['gender']     = cohort['nlp_gender'].map({'F': 0.0, 'M': 1.0})

    # Per-session MeanSzFreq in (0, 2y] post-EEG → patient mean across valid sessions
    session_sz = add_session_mean_sz_freq_2yr(
        cohort.drop_duplicates(subset=['patient_id', 'session_number'])
    )
    n_sess_label = int(session_sz['mean_sz_freq_2yr'].notna().sum())
    print(f"[MeanSzFreq 2yr window] {n_sess_label}/{len(session_sz)} sessions with "
          f"documented seizure frequency in (0, {FOLLOWUP_DAYS}d] post-EEG")

    base_cols = ['patient_id', 'session_number', 'start_time_deid', 'spike_rate_per_hour',
                 'age_at_eeg', 'gender']
    dataset = (cohort[base_cols]
               .drop_duplicates(subset=['patient_id', 'session_number'])
               .rename(columns={'session_number': 'session'})
               .copy())
    dataset[['patient_id', 'session']] = dataset[['patient_id', 'session']].astype(str)
    _sz = session_sz[['patient_id', 'session_number', 'mean_sz_freq_2yr']].copy()
    _sz['patient_id'] = _sz['patient_id'].astype(str)
    _sz = _sz.rename(columns={'session_number': 'session'})
    _sz['session'] = _sz['session'].astype(str)
    dataset = dataset.merge(_sz, on=['patient_id', 'session'], how='left')
    print(f"Base session table: {len(dataset)} patient-sessions, {dataset['patient_id'].nunique()} patients")

    # ── background EEG features ──────────────────────────────────────────────────
    bg = pd.read_csv(BG_FEAT_CSV)
    bg = bg[bg['subtype'] == 'General'].copy()
    bg[['patient_id','session']] = bg[['patient_id','session']].astype(str)
    bg = bg.drop(columns=['subtype'])
    dataset = dataset.merge(bg, on=['patient_id','session'], how='left')
    print(f"After EEG features: {len(dataset)} rows, {len(dataset.columns)} columns")

    # ── MORGOTH slowing ──────────────────────────────────────────────────────────

    slow_df = build_session_slowing_table(SLOWING_DIR, n_jobs=N_JOBS)
    if slow_df is not None:
        dataset = dataset.merge(slow_df, on=['patient_id', 'session'], how='left')
        n_match = dataset['pct_foc_slowing'].notna().sum()
        print(f"Slowing matched: {n_match}/{len(dataset)} sessions with pct slowing")
    else:
        print("Slowing not yet available — run run_light_preprocessed.sh first")
        for col in ['pct_no_slowing', 'pct_foc_slowing', 'pct_gen_slowing', 'n_win_slow']:
            dataset[col] = np.nan

    n_eeg_sessions_total = (
        cohort.groupby('patient_id')['session_number']
        .nunique()
        .rename('n_eeg_sessions')
        .reset_index()
    )
    n_eeg_sessions_total['patient_id'] = n_eeg_sessions_total['patient_id'].astype(str)

    # ── Patient-level dataset ────────────────────────────────────────────────────
    # Sz control: mean EEG features + mean 2yr-window sz freq, across all valid sessions.

    valid_sz_sessions = filter_valid_eeg_sessions(dataset)
    patient_sz = build_patient_sz_from_session_means(
        valid_sz_sessions, poor_cutoff=POOR_CUTOFF_PER_MONTH)
    print(f"[MeanSzFreq 2yr window] {patient_sz['mean_sz_freq'].notna().sum()} patients with label")

    dataset_patient_mean = aggregate_mean_eeg_per_patient(
        dataset,
        patient_sz_df=None,
        n_eeg_sessions_df=n_eeg_sessions_total,
        poor_cutoff_per_month=POOR_CUTOFF_PER_MONTH,
    )

    n_multi_eeg = int((n_eeg_sessions_total['n_eeg_sessions'] > 1).sum())
    print(f"\n  Patient-level (mean EEG — sz_freq_class): {len(dataset_patient_mean)} patients")
    print(f"  Patients with >1 routine EEG: {n_multi_eeg}")
    if 'n_eeg_sessions_used' in dataset_patient_mean.columns:
        used = dataset_patient_mean['n_eeg_sessions_used']
        print(f"  Mean valid sessions averaged (sz_freq_class): "
              f"mean={used.mean():.2f}, max={used.max()}")

    TARGET_PATIENT_DATA = {
        'sz_freq_class': dataset_patient_mean,
    }
    dataset_patient = dataset_patient_mean  # column discovery for feature sets

    for tgt in TARGETS:
        kind = TARGET_META[tgt]['kind']
        dfp = TARGET_PATIENT_DATA[tgt]
        n  = dfp[tgt].notna().sum()
        vc = dfp[tgt].value_counts().to_dict()
        print(f"  [{kind}] {tgt} (mean EEG): {n} patients with label; class distribution: {vc}")


    def build_data_overview(cohort, dataset, dataset_patient):
        """Summary counts for the full pipeline (analysis units are patients)."""
        rows = []

        def add(category, metric, value, target=''):
            rows.append({'category': category, 'metric': metric, 'value': value, 'target': target})

        add('cohort_filtering', 'n_patient_sessions_general_epilepsy', len(cohort))
        add('cohort_filtering', 'n_unique_patients_general_epilepsy', cohort['patient_id'].nunique())
        add('cohort_filtering', 'n_patient_sessions_before_aggregation', len(dataset))
        add('cohort_filtering', 'n_patients_patient_level_dataset', len(dataset_patient))

        demo = dataset_patient.dropna(subset=['age_at_eeg', 'gender'])
        add('demographics', 'n_patients_with_demographics', len(demo))
        add('demographics', 'age_mean_years', round(demo['age_at_eeg'].mean(), 2))
        add('demographics', 'age_std_years', round(demo['age_at_eeg'].std(), 2))
        add('demographics', 'n_female', int((demo['gender'] == 0).sum()))
        add('demographics', 'n_male', int((demo['gender'] == 1).sum()))

        for tgt in TARGETS:
            sub = TARGET_PATIENT_DATA[tgt].dropna(subset=[tgt, 'age_at_eeg', 'gender'])
            add('analysis_cohort', 'outcome_kind', TARGET_META[tgt]['kind'], target=tgt)
            add('analysis_cohort', 'outcome_description', TARGET_META[tgt]['label'], target=tgt)
            add('analysis_cohort', 'n_patients', len(sub), target=tgt)
            add('analysis_cohort', 'n_patients_ied',
                int((sub['spike_rate_per_hour'] != 0).sum()), target=tgt)
            add('analysis_cohort', 'n_patients_no_ied',
                int((sub['spike_rate_per_hour'] == 0).sum()), target=tgt)
            add('analysis_cohort', 'n_patients_class_0', int((sub[tgt] == 0).sum()), target=tgt)
            add('analysis_cohort', 'n_patients_class_1', int((sub[tgt] == 1).sum()), target=tgt)
            add('analysis_cohort', 'mean_spike_rate_per_hour',
                round(sub['spike_rate_per_hour'].mean(), 4), target=tgt)
            add('analysis_cohort', 'median_spike_rate_per_hour',
                round(sub['spike_rate_per_hour'].median(), 4), target=tgt)

        return pd.DataFrame(rows)


    data_overview = build_data_overview(cohort, dataset, dataset_patient)
    save_table(data_overview, 'data_overview.csv')
    print("\n=== Data overview (analysis unit = patient) ===")
    print(data_overview.to_string(index=False))


    # ===========================================================================
    # 2. Feature sets  (defined once — columns don't change between targets)
    # ===========================================================================

    DEMO_COLS    = ['age_at_eeg', 'gender']
    IED_COLS     = ['spike_rate_log']
    SLOWING_COLS = ['pct_foc_slowing', 'pct_gen_slowing']  # index EEG Morgoth window %
    EEG_BG_COLS  = [c for c in dataset_patient.columns
                    if any(c.startswith(r) for r in
                           ['frontal_','central_','temporal_','parietal_','occipital_'])]

    FEATURE_SETS = {
        'Model1_null':     DEMO_COLS,
        'Model2_IED':      IED_COLS + DEMO_COLS,
        'Model3_slowing':  SLOWING_COLS + DEMO_COLS,
        'Model4_eeg':      EEG_BG_COLS + DEMO_COLS,
        # Models 5 & 6 get direct classifiers on the combined feature sets…
        'Model5_slow_eeg': SLOWING_COLS + EEG_BG_COLS + DEMO_COLS,
        'Model6_all':      IED_COLS + SLOWING_COLS + EEG_BG_COLS + DEMO_COLS,
    }

    FEATURE_SETS_CLEAN = {k: [c for c in cols if c in dataset_patient.columns]
                          for k, cols in FEATURE_SETS.items()}

    for k, cols in FEATURE_SETS_CLEAN.items():
        missing = [c for c in FEATURE_SETS[k] if c not in dataset_patient.columns]
        print(f"{k:<22}: {len(cols):>3} features"
              + (f"  MISSING: {missing}" if missing else ''))


    _PRIMARY_LOADED = True



# ===========================================================================
# 3. Classifier definitions  (SGD + linear models; GridSearchCV tuning)
# ===========================================================================

# Lower C / higher alpha = stronger regularization.
# L2-only grids (no L1 / elasticnet).
C_GRID = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
ALPHA_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 0.1]
GAMMA_GRID = ['scale', 'auto', 0.01, 0.1, 1.0]


class NearestCentroidWithProba(NearestCentroid):
    """NearestCentroid + softmin-of-distance predict_proba for AUROC scoring."""

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        dists = pairwise_distances(X, self.centroids_, metric=self.metric)
        neg = -dists
        neg -= neg.max(axis=1, keepdims=True)
        exp = np.exp(neg)
        return exp / exp.sum(axis=1, keepdims=True)


CLASSIFIERS = {
    'sgd': {
        'pipeline': SGDClassifier(
            loss='log_loss',
            random_state=0,
            class_weight=None,
            max_iter=20000,
            tol=1e-4,
            n_jobs=1,
        ),
        'param_grid': {
            'alpha': ALPHA_GRID,
            'penalty': ['l2'],
        },
        'needs_scaling': True,
    },
    'lr': {
        'pipeline': LogisticRegression(
            solver='lbfgs',
            penalty='l2',
            max_iter=20000,
            tol=1e-4,
            random_state=0,
            class_weight=None,
        ),
        # L2 only × C
        'param_grid': {
            'penalty': ['l2'],
            'C': C_GRID,
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
            'estimator__C': C_GRID,
        },
        'needs_scaling': True,
    },
    'svm_rbf': {
        'pipeline': SVC(
            kernel='rbf',
            class_weight=None,
            probability=True,
            max_iter=200000,
            random_state=0,
            cache_size=1000,
        ),
        'param_grid': {
            'C': C_GRID,
            'gamma': GAMMA_GRID,
        },
        'needs_scaling': True,
    },
    'svm_poly': {
        'pipeline': SVC(
            kernel='poly',
            class_weight=None,
            probability=True,
            max_iter=200000,
            random_state=0,
            cache_size=1000,
        ),
        'param_grid': {
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3],
        },
        'needs_scaling': True,
    },
    'sgd_rbf': {
        'pipeline': Pipeline([
            ('approx', RBFSampler(random_state=0)),
            ('clf', SGDClassifier(
                loss='log_loss',
                random_state=0,
                class_weight=None,
                max_iter=20000,
                tol=1e-4,
                n_jobs=1,
            )),
        ]),
        'param_grid': {
            'approx__gamma': [0.1, 1.0, 10.0],
            'approx__n_components': [50, 100],
            'clf__alpha': ALPHA_GRID,
            'clf__penalty': ['l1', 'l2'],
        },
        'needs_scaling': True,
    },
    'dt': {
        'pipeline': DecisionTreeClassifier(
            class_weight=None,
            random_state=0,
        ),
        'param_grid': {
            'max_depth': [2, 3, 4, 5, 8, None],
            'min_samples_leaf': [5, 10, 20, 40],
            'criterion': ['gini', 'entropy'],
        },
        'needs_scaling': False,
    },
    'knn': {
        'pipeline': KNeighborsClassifier(
            n_neighbors=15,
            weights='distance',
            metric='minkowski',
            p=2,
            n_jobs=1,
        ),
        'param_grid': [
            {
                'n_neighbors': [5, 9, 15, 21, 31],
                'weights': ['uniform', 'distance'],
                'metric': ['minkowski'],
                'p': [1, 2],
            },
            {
                'n_neighbors': [5, 9, 15, 21, 31],
                'weights': ['uniform', 'distance'],
                'metric': ['cosine'],
            },
        ],
        'needs_scaling': True,
    },
    'ridge': {
        'pipeline': CalibratedClassifierCV(
            RidgeClassifier(random_state=0, class_weight=None),
            method='sigmoid', cv=3,
        ),
        'param_grid': {
            'estimator__alpha': [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0],
        },
        'needs_scaling': True,
    },
    'ncentroid': {
        'pipeline': NearestCentroidWithProba(metric='euclidean'),
        'param_grid': {
            'metric': ['euclidean', 'manhattan'],
            'shrink_threshold': [None, 0.1, 0.5, 1.0],
        },
        'needs_scaling': True,
    },
    'gnb': {
        'pipeline': GaussianNB(),
        'param_grid': {
            'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
        },
        'needs_scaling': True,
    },
    'lda': {
        'pipeline': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        'param_grid': [
            {'solver': ['svd'], 'shrinkage': [None]},
            {'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]},
        ],
        'needs_scaling': True,
    },
    'rf': {
        'pipeline': RandomForestClassifier(
            n_estimators=300,
            random_state=0,
            class_weight=None,
            n_jobs=1,
        ),
        'param_grid': {
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 8, None],
            'min_samples_leaf': [1, 5, 10],
            'max_features': ['sqrt', None],
        },
        'needs_scaling': False,
    },
    'xgb': {
        'pipeline': XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=0,
            n_jobs=1,
            tree_method='hist',
            verbosity=0,
        ),
        'param_grid': {
            'n_estimators': [100, 200, 300],
            'max_depth': [2, 3, 5],
            'learning_rate': [0.03, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'min_child_weight': [1, 5],
        },
        'needs_scaling': False,
    },
}

# Manuscript Table 2 feature sets. Internal FEATURE_SETS keys are unchanged
# (Model1_null..Model6_all) so nothing downstream breaks; this just maps them
# to the paper's Table 2 numbering (1-5) for printed/saved output labels.
# Model1_null (age+gender only, no EEG marker) is not part of Table 2 — kept
# as an internal baseline and compared against every Table 2 model.
IED_MODEL_KEY     = 'Model2_IED'
SLOWING_MODEL_KEY = 'Model3_slowing'
EEG_MODEL_KEY     = 'Model5_slow_eeg'
TABLE2_LABELS = {
    'Model1_null':     'Null (age + gender only; not part of Table 2)',
    'Model2_IED':      'Table 2 Model 1: IEDs + age + gender',
    'Model3_slowing':  'Table 2 Model 2: Slowing + age + gender',
    'Model4_eeg':       'Table 2 Model 3: Continuous EEG features + age + gender',
    'Model5_slow_eeg': 'Table 2 Model 4: Slowing + Continuous EEG features + age + gender',
    'Model6_all':      'Table 2 Model 5: IEDs + Slowing + Continuous EEG features + age + gender',
}


def _use_feature_scaler() -> bool:
    """Return False when ML_USE_SCALER=0 (disable StandardScaler on distance/linear clfs)."""
    v = os.environ.get('ML_USE_SCALER', '1').strip().lower()
    return v not in ('0', 'false', 'no', 'off')


def _effective_needs_scaling(cfg) -> bool:
    return bool(cfg['needs_scaling']) and _use_feature_scaler()


def _pipeline_param_grid(cfg):
    """Build GridSearchCV param_grid; prefix keys with clf__ when scaled Pipeline."""
    param_grid = cfg['param_grid']
    if not _effective_needs_scaling(cfg):
        return param_grid

    def _prefix(d):
        # Always add outer Pipeline's clf__ step. Nested keys like clf__alpha
        # (inner Pipeline) become clf__clf__alpha.
        return {f'clf__{k}': v for k, v in d.items()}

    if isinstance(param_grid, dict):
        return _prefix(param_grid)
    if isinstance(param_grid, list):
        return [_prefix(d) for d in param_grid]
    return param_grid


# ===========================================================================
# 4. Helper functions
# ===========================================================================


def ratio_class_weight(y, minority_scale: float = 1.0) -> dict:
    """Class weights from train counts: minority gets scale * (n_maj / n_min).

    Example with prev≈0.25 and scale=1: {0: 1.0, 1: 3.0}.
    scale>1 further upweights the minority class.
    """
    y = np.asarray(y).astype(int)
    n0 = max(int((y == 0).sum()), 1)
    n1 = max(int((y == 1).sum()), 1)
    # Assume label 1 is the minority (poor control); fall back if not
    if n1 <= n0:
        return {0: 1.0, 1: float(minority_scale) * (n0 / n1)}
    return {1: 1.0, 0: float(minority_scale) * (n1 / n0)}


def _apply_class_weight(estimator, clf_key: str, class_weight):
    """Set class_weight on the underlying estimator (handles nested Pipelines)."""
    if clf_key in ('knn', 'ncentroid', 'gnb', 'lda', 'xgb'):
        return  # no class_weight support (xgb uses scale_pos_weight in fit_clf)
    if clf_key in ('sgd', 'lr', 'svm_rbf', 'svm_poly', 'dt', 'rf'):
        estimator.set_params(class_weight=class_weight)
    elif clf_key in ('svm', 'ridge'):
        estimator.set_params(estimator__class_weight=class_weight)
    elif clf_key == 'sgd_rbf':
        estimator.set_params(clf__class_weight=class_weight)
    else:
        try:
            estimator.set_params(class_weight=class_weight)
        except ValueError:
            pass


def fit_clf(clf_key, X_train, y_train, seed, n_cv_folds=N_CV_FOLDS,
            class_weight='none', minority_scale: float = 1.0):
    """Tune via stratified CV; return (best_estimator, best_cv_auroc).

    class_weight:
      'none' / None — equal weights (default)
      'balanced'    — sklearn balanced
      'ratio'       — {maj: 1, min: minority_scale * n_maj/n_min}
      dict          — passed through
    """
    import warnings; warnings.filterwarnings('ignore')
    cfg = CLASSIFIERS[clf_key]
    cv  = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)

    estimator = clone(cfg['pipeline'])
    if clf_key == 'xgb':
        pos = max(int((y_train == 1).sum()), 1)
        neg = max(int((y_train == 0).sum()), 1)
        if class_weight in (None, 'none'):
            spw = 1.0
        elif class_weight == 'ratio':
            spw = (neg / pos) * minority_scale
        else:
            spw = neg / pos
        estimator.set_params(scale_pos_weight=spw)
    else:
        if class_weight in (None, 'none'):
            cw = None
        elif class_weight == 'balanced':
            cw = 'balanced'
        elif class_weight == 'ratio':
            cw = ratio_class_weight(y_train, minority_scale=minority_scale)
        elif isinstance(class_weight, dict):
            cw = class_weight
        else:
            cw = class_weight
        _apply_class_weight(estimator, clf_key, cw)

    if _effective_needs_scaling(cfg):
        pipe       = Pipeline([('scaler', StandardScaler()), ('clf', estimator)])
        param_grid = _pipeline_param_grid(cfg)
    else:
        pipe       = estimator
        param_grid = cfg['param_grid']

    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=1, refit=True)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_score_


def fit_clf_default(clf_key, X_train, y_train, class_weight='none', minority_scale: float = 1.0):
    """Fit sklearn defaults only (no GridSearch); return fitted estimator."""
    import warnings; warnings.filterwarnings('ignore')
    cfg = CLASSIFIERS[clf_key]
    estimator = clone(cfg['pipeline'])
    if clf_key == 'xgb':
        if class_weight in (None, 'none'):
            estimator.set_params(scale_pos_weight=1.0)
        else:
            pos = max(int((y_train == 1).sum()), 1)
            neg = max(int((y_train == 0).sum()), 1)
            scale = minority_scale if class_weight == 'ratio' else 1.0
            estimator.set_params(scale_pos_weight=(neg / pos) * scale)
    else:
        if class_weight in (None, 'none'):
            cw = None
        elif class_weight == 'balanced':
            cw = 'balanced'
        elif class_weight == 'ratio':
            cw = ratio_class_weight(y_train, minority_scale=minority_scale)
        elif isinstance(class_weight, dict):
            cw = class_weight
        else:
            cw = class_weight
        _apply_class_weight(estimator, clf_key, cw)

    if _effective_needs_scaling(cfg):
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', estimator)])
    else:
        pipe = estimator
    pipe.fit(X_train, y_train)
    return pipe


def get_best_params_from_train(train_df, y_train, feature_sets):
    """
    Run GridSearchCV on the training split for every (model, clf).
    Returns {model_key: {clf_key: best_params_dict}}.
    """
    result = {}
    for model_key in feature_sets:
        result[model_key] = {}
        cols = feature_sets[model_key]
        X = prepare_X(train_df, cols, ref_df=train_df)
        for clf_key in CLASSIFIERS:
            try:
                cfg = CLASSIFIERS[clf_key]
                cv  = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
                estimator = clone(cfg['pipeline'])
                if clf_key == 'xgb':
                    pos = max(int((y_train == 1).sum()), 1)
                    neg = max(int((y_train == 0).sum()), 1)
                    estimator.set_params(scale_pos_weight=neg / pos)
                if _effective_needs_scaling(cfg):
                    pipe       = Pipeline([('scaler', StandardScaler()), ('clf', estimator)])
                    param_grid = _pipeline_param_grid(cfg)
                else:
                    pipe       = estimator
                    param_grid = cfg['param_grid']
                gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc',
                                  n_jobs=1, refit=False)
                gs.fit(X, y_train)
                result[model_key][clf_key] = gs.best_params_
            except Exception as e:
                print(f"  HP extraction failed ({model_key}/{clf_key}): {e}")
                result[model_key][clf_key] = {}
    return result


def fit_clf_fixed(clf_key, X_train, y_train, best_params, use_pca=False,
                  pca_variance=PCA_VARIANCE):
    """
    Fit with pre-determined hyperparameters (no CV search).
    If use_pca=True, prepends StandardScaler → PCA(pca_variance) to the pipeline.
    best_params: dict from gs.best_params_ — may or may not have 'clf__' prefix.
    """
    import warnings; warnings.filterwarnings('ignore')
    cfg       = CLASSIFIERS[clf_key]
    estimator = clone(cfg['pipeline'])
    if clf_key == 'xgb':
        pos = max(int((y_train == 1).sum()), 1)
        neg = max(int((y_train == 0).sum()), 1)
        estimator.set_params(scale_pos_weight=neg / pos)

    if use_pca:
        pca_steps = []
        if _use_feature_scaler():
            pca_steps.append(('scaler', StandardScaler()))
        pca_steps.extend([
            ('pca', PCA(n_components=pca_variance, random_state=0)),
            ('clf', estimator),
        ])
        pipe = Pipeline(pca_steps)
        # Normalise params to clf__ prefix
        clf_params = {(k if k.startswith('clf__') else f'clf__{k}'): v
                      for k, v in best_params.items()}
        if clf_params:
            pipe.set_params(**clf_params)
    elif _effective_needs_scaling(cfg):
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', estimator)])
        if best_params:
            pipe.set_params(**best_params)   # already carries clf__ prefix
    else:
        pipe = estimator
        if best_params:
            pipe.set_params(**best_params)   # no prefix for tree models

    pipe.fit(X_train, y_train)
    return pipe


def best_clf_for_model(model_key, agg_dict):
    """Best classifier for model_key by mean test AUROC across MC splits."""
    clf_dict = agg_dict.get(model_key, {})
    pool = {k: v for k, v in clf_dict.items() if k in CLASSIFIERS}
    if not pool:
        return list(CLASSIFIERS.keys())[0]
    return max(pool, key=lambda k: np.nanmean(pool[k]['auroc']))


def _eval_metrics(y_true, y_prob):
    if not np.isfinite(y_prob).all() or len(np.unique(y_true)) < 2:
        return {'auroc': np.nan}
    return {'auroc': roc_auc_score(y_true, y_prob)}


def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAP, alpha=0.05, rng=None):
    """
    Bootstrap percentile CI for the mean of a 1-D sample.
    Unlike a raw 2.5–97.5 percentile interval on the sample, this resamples
    with replacement and takes the central interval of bootstrap replicate means.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = rng or np.random.default_rng(0)
    n = len(values)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        boot_means[b] = np.mean(values[rng.integers(0, n, size=n)])
    lo_pct = 100 * alpha / 2
    hi_pct = 100 * (1 - alpha / 2)
    return float(np.percentile(boot_means, lo_pct)), float(np.percentile(boot_means, hi_pct))


def bootstrap_auroc_ci(y_true, y_prob, n_boot=N_BOOTSTRAP, alpha=0.05, rng=None):
    """
    Bootstrap percentile CI for AUROC on a fixed test set (patient resampling).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    ok = np.isfinite(y_prob)
    y_true, y_prob = y_true[ok], y_prob[ok]
    n = len(y_true)
    if n < 2 or len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    rng = rng or np.random.default_rng(0)
    boot_aurocs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aurocs.append(roc_auc_score(yt, yp))
    if len(boot_aurocs) < 10:
        return np.nan, np.nan
    boot_aurocs = np.asarray(boot_aurocs)
    lo_pct = 100 * alpha / 2
    hi_pct = 100 * (1 - alpha / 2)
    return float(np.percentile(boot_aurocs, lo_pct)), float(np.percentile(boot_aurocs, hi_pct))


def bootstrap_diff_ci(diffs, n_boot=N_BOOTSTRAP, alpha=0.05, rng=None):
    """Bootstrap percentile CI for the mean of paired AUROC differences."""
    return bootstrap_mean_ci(diffs, n_boot=n_boot, alpha=alpha, rng=rng)


# ===========================================================================
# Bootstrap AUROC comparison (primary + no-IED holdout model comparisons)
# ===========================================================================

def bootstrap_compare_auroc(y_true, pred_simple, pred_complex, n_boot=N_BOOTSTRAP, rng=None):
    """
    Paired bootstrap AUROC comparison on a single held-out test set.

    The test set is resampled with replacement n_boot times; for each resample,
    diff = AUC(complex) - AUC(simple). The 95% CI is the 2.5th/97.5th percentile
    of that diff distribution (observed diff is the point estimate). Two-sided
    p = min(2 * (min(n_simpler, n_valid - n_simpler) + 1) / (n_boot + 1), 1.0),
    where n_simpler counts resamples in which the simpler model had the higher
    AUC — taking the min of the two tail counts makes this a proper two-sided
    test regardless of which model happens to win more often.
    """
    y_true = np.asarray(y_true, dtype=int)
    pred_simple = np.asarray(pred_simple, dtype=float)
    pred_complex = np.asarray(pred_complex, dtype=float)
    ok = np.isfinite(pred_simple) & np.isfinite(pred_complex)
    y_true, pred_simple, pred_complex = y_true[ok], pred_simple[ok], pred_complex[ok]
    n = len(y_true)
    nan_out = {
        'auc_simple': np.nan, 'auc_complex': np.nan, 'diff': np.nan,
        'diff_ci_lo': np.nan, 'diff_ci_hi': np.nan,
        'n_boot_valid': 0, 'n_simpler': 0, 'p': np.nan,
    }
    if n < 2 or len(np.unique(y_true)) < 2:
        return nan_out

    auc_simple_obs = roc_auc_score(y_true, pred_simple)
    auc_complex_obs = roc_auc_score(y_true, pred_complex)

    rng = rng or np.random.default_rng(0)
    diffs = []
    n_simpler = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        auc_s = roc_auc_score(yt, pred_simple[idx])
        auc_c = roc_auc_score(yt, pred_complex[idx])
        diffs.append(auc_c - auc_s)
        if auc_s > auc_c:
            n_simpler += 1
    if len(diffs) < 10:
        return nan_out

    diffs = np.asarray(diffs)
    n_valid = len(diffs)
    p = min(2 * (min(n_simpler, n_valid - n_simpler) + 1) / (n_boot + 1), 1.0)
    return {
        'auc_simple': float(auc_simple_obs),
        'auc_complex': float(auc_complex_obs),
        'diff': float(auc_complex_obs - auc_simple_obs),
        'diff_ci_lo': float(np.percentile(diffs, 2.5)),
        'diff_ci_hi': float(np.percentile(diffs, 97.5)),
        'n_boot_valid': len(diffs),
        'n_simpler': n_simpler,
        'p': float(p),
    }


def bootstrap_pairwise_comparisons(y_true, proba_dict, model_keys, best_clf_map,
                                   feature_sets_clean, n_boot=N_BOOTSTRAP, rng=None):
    """
    All pairwise bootstrap AUROC comparisons (bootstrap_compare_auroc) among
    model_keys, using each model's best_clf_map-selected classifier.

    The "simpler" model in each pair is the one with fewer feature columns
    (ties broken by earlier position in model_keys) — this only decides which
    model anchors the diff sign / n_simpler count, not which model is "better".
    """
    rng = rng or np.random.default_rng(0)
    default_clf = list(CLASSIFIERS.keys())[0]
    keys = [k for k in model_keys if k in proba_dict]
    n_feats = {k: len(feature_sets_clean.get(k, [])) for k in keys}
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            if n_feats[ki] <= n_feats[kj]:
                k_simple, k_complex = ki, kj
            else:
                k_simple, k_complex = kj, ki
            c_simple = best_clf_map.get(k_simple, default_clf)
            c_complex = best_clf_map.get(k_complex, default_clf)
            p_simple = proba_dict[k_simple].get(c_simple)
            p_complex = proba_dict[k_complex].get(c_complex)
            if p_simple is None or p_complex is None:
                continue
            if not (np.isfinite(p_simple).all() and np.isfinite(p_complex).all()):
                continue
            res = bootstrap_compare_auroc(y_true, p_simple, p_complex, n_boot=n_boot, rng=rng)
            rows.append({
                'model_simple': k_simple, 'model_complex': k_complex,
                'label_simple': TABLE2_LABELS.get(k_simple, k_simple),
                'label_complex': TABLE2_LABELS.get(k_complex, k_complex),
                'classifier_simple': c_simple, 'classifier_complex': c_complex,
                'n_features_simple': n_feats[k_simple], 'n_features_complex': n_feats[k_complex],
                'auc_simple': res['auc_simple'], 'auc_complex': res['auc_complex'],
                'diff': res['diff'], 'diff_ci_lo': res['diff_ci_lo'], 'diff_ci_hi': res['diff_ci_hi'],
                'n_boot': n_boot, 'n_simpler': res['n_simpler'], 'p': res['p'],
            })
    out = pd.DataFrame(rows)
    return out


def print_bootstrap_pairwise(df, title):
    hdr = (f"{'Simpler':<18} {'Clf':<6} {'Complex':<18} {'Clf':<6}  "
           f"{'AUC_s':>6} {'AUC_c':>6} {'diff':>7} {'95% Boot CI':>18} {'p':>8}")
    print(f"\n{title}")
    print(hdr)
    print('-' * len(hdr))
    for _, r in df.iterrows():
        ci = f"[{r['diff_ci_lo']:.3f}, {r['diff_ci_hi']:.3f}]"
        print(
            f"{r['model_simple']:<18} {r['classifier_simple']:<6} "
            f"{r['model_complex']:<18} {r['classifier_complex']:<6}  "
            f"{r['auc_simple']:6.3f} {r['auc_complex']:6.3f} {r['diff']:7.3f} {ci:>18} "
            f"{r['p']:8.4f}"
        )


def build_auroc_summary(agg_dict, n_boot=N_BOOTSTRAP, rng=None):
    """Mean AUROC with bootstrap 95% CI (resampling MC CV iteration AUROCs)."""
    rng = rng or np.random.default_rng(0)
    rows = []
    for model_key, clf_dict in agg_dict.items():
        for clf_key, vals in clf_dict.items():
            auroc = np.array(vals['auroc'], dtype=float)
            ci_lo, ci_hi = bootstrap_mean_ci(auroc, n_boot=n_boot, rng=rng)
            rows.append({
                'model':      model_key,
                'classifier': clf_key,
                'auroc_mean': np.nanmean(auroc),
                'auroc_lo':   ci_lo,
                'auroc_hi':   ci_hi,
            })
    return pd.DataFrame(rows)


def build_single_split_auroc_summary(
    split_result,
    feature_sets_clean,
    best_clf_map,
    split_seed=PRIMARY_SPLIT_SEED,
    n_boot=N_BOOTSTRAP,
    rng=None,
):
    """Test-set AUROC per model×classifier with patient bootstrap 95% CI."""
    rng = rng or np.random.default_rng(0)
    y_te = np.asarray(split_result['_meta']['y_test'], dtype=int)
    proba = split_result['_meta']['proba']
    rows = []
    for model_key in feature_sets_clean:
        for clf_key in CLASSIFIERS:
            p = proba.get(model_key, {}).get(clf_key)
            if p is None:
                continue
            p = np.asarray(p, dtype=float)
            if len(np.unique(y_te)) < 2 or not np.isfinite(p).all():
                auroc = np.nan
                ci_lo, ci_hi = np.nan, np.nan
            else:
                auroc = roc_auc_score(y_te, p)
                ci_lo, ci_hi = bootstrap_auroc_ci(y_te, p, n_boot=n_boot, rng=rng)
            rows.append({
                'model': model_key,
                'classifier': clf_key,
                'auroc_mean': auroc,
                'auroc_lo': ci_lo,
                'auroc_hi': ci_hi,
                'test_selected': clf_key == best_clf_map.get(model_key),
                'split_seed': split_seed,
                'n_test': len(y_te),
            })
    return pd.DataFrame(rows)


def print_auroc_summary(summary, title):
    hdr = f"{'Model':<22} {'Clf':<16}  {'AUROC (95% bootstrap CI)':>26}"
    print(f"\n{title}")
    print(hdr)
    print('-' * len(hdr))
    for _, r in summary.iterrows():
        print(f"{r['model']:<22} {r['classifier']:<16}  "
              f"{r['auroc_mean']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}]")


def print_delong_pairwise(delong_df, title):
    hdr = (f"{'Model A':<22} {'Clf A':<6} {'Model B':<22} {'Clf B':<6}  "
           f"{'AUC_A':>6} {'AUC_B':>6} {'diff':>7} {'95% DeLong CI':>18} {'p':>8}")
    print(f"\n{title}")
    print(hdr)
    print('-' * len(hdr))
    for _, r in delong_df.iterrows():
        ci = f"[{r['diff_ci_lo']:.3f}, {r['diff_ci_hi']:.3f}]"
        print(
            f"{r['model_A']:<22} {r['classifier_A']:<6} "
            f"{r['model_B']:<22} {r['classifier_B']:<6}  "
            f"{r['auc_A']:6.3f} {r['auc_B']:6.3f} {r['diff']:7.3f} {ci:>18} {r['p']:8.4f}"
        )


def prepare_X(df, cols, ref_df=None):
    """Median imputation using ref_df (training set) to avoid test-set leakage."""
    ref = ref_df if ref_df is not None else df
    medians = ref[cols].median()
    X = df[cols].copy()
    for c in cols:
        X[c] = X[c].fillna(medians[c])
    return X.values.astype(np.float64)


def _patient_grouped_split(analysis_df, y_all, test_size, seed):
    """Stratified train/test split at patient level (one row per patient)."""
    patients    = analysis_df['patient_id'].values
    unique_pats = np.unique(patients)
    pat_label = np.array([
        int(np.round(np.mean(y_all[patients == p]))) for p in unique_pats
    ])
    pat_tr, pat_te = train_test_split(
        unique_pats, test_size=test_size, random_state=seed, stratify=pat_label
    )
    tr_idx = np.where(np.isin(patients, pat_tr))[0]
    te_idx = np.where(np.isin(patients, pat_te))[0]
    return tr_idx, te_idx


def fit_all_classifiers_select_best(train_df, test_df, y_train, y_test, feature_sets_clean,
                                    seed, n_cv_folds=N_CV_FOLDS):
    """Tune all model×classifier on train; pick best by test AUROC; return test proba."""
    proba = {}
    cv_scores = {}
    for model_key, cols in feature_sets_clean.items():
        X_tr = prepare_X(train_df, cols, ref_df=train_df)
        X_te = prepare_X(test_df, cols, ref_df=train_df)
        proba[model_key] = {}
        cv_scores[model_key] = {}
        for clf_key in CLASSIFIERS:
            try:
                est, cv_auroc = fit_clf(clf_key, X_tr, y_train, seed, n_cv_folds)
                proba[model_key][clf_key] = est.predict_proba(X_te)[:, 1]
                cv_scores[model_key][clf_key] = cv_auroc
            except Exception:
                proba[model_key][clf_key] = np.full(len(test_df), np.nan)
                cv_scores[model_key][clf_key] = np.nan
    best_clf = select_best_clf_by_test_auroc(
        proba, y_test, feature_sets_clean.keys(), CLASSIFIERS, _eval_metrics)
    return proba, cv_scores, best_clf


def run_mc_iteration(seed, analysis_df, target_col, feature_sets_clean,
                     n_cv_folds=N_CV_FOLDS):
    """
    One Monte Carlo CV iteration: 80/20 patient-grouped stratified split with
    train-only median imputation and inner CV hyperparameter tuning.
    """
    y_all = analysis_df[target_col].values.astype(int)
    tr_idx, te_idx = _patient_grouped_split(analysis_df, y_all, TEST_SIZE, seed)
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]
    train_df = analysis_df.iloc[tr_idx]
    test_df  = analysis_df.iloc[te_idx]

    clf_proba = {}

    for model_key, cols in feature_sets_clean.items():
        X_tr = prepare_X(train_df, cols, ref_df=train_df)
        X_te = prepare_X(test_df, cols, ref_df=train_df)
        clf_proba[model_key] = {}
        for clf_key in CLASSIFIERS:
            try:
                est, _ = fit_clf(clf_key, X_tr, y_tr, seed, n_cv_folds)
                clf_proba[model_key][clf_key] = est.predict_proba(X_te)[:, 1]
            except Exception:
                clf_proba[model_key][clf_key] = np.full(len(y_te), np.nan)

    out = {}

    for model_key in feature_sets_clean:
        out[model_key] = {clf_key: _eval_metrics(y_te, p)
                          for clf_key, p in clf_proba[model_key].items()}

    out['_meta'] = {'y_test': y_te, 'proba': clf_proba}
    return out





def aggregate_mc_iterations(mc_results):
    """Collapse list of per-iteration dicts into {model: {clf: {metric: [values]}}}."""
    agg = {}
    for mc_iter in mc_results:
        if mc_iter is None:
            continue
        for model_key, model_res in mc_iter.items():
            if str(model_key).startswith('_'):
                continue
            if model_key not in agg:
                agg[model_key] = {}
            for clf_key, metrics in model_res.items():
                if clf_key not in agg[model_key]:
                    agg[model_key][clf_key] = {m: [] for m in METRICS}
                for m in METRICS:
                    agg[model_key][clf_key][m].append(metrics.get(m, np.nan))
    return agg


def run_holdout_mc_iteration(seed, analysis_df, target_col, feature_sets_clean,
                             n_cv_folds=N_CV_FOLDS):
    """
    One Monte Carlo holdout iteration: patient-grouped 50/50 split of no-IED patients.
    Train on all IED + 50% no-IED patients; test on the other 50% no-IED patients.
    Best classifier per model is re-selected on the holdout test fold (test AUROC).
    """
    import warnings; warnings.filterwarnings('ignore')
    patients    = analysis_df['patient_id'].values
    y_all       = analysis_df[target_col].values.astype(int)
    no_ied_mask = analysis_df['spike_rate_per_hour'].values == 0
    ied_idx     = np.where(~no_ied_mask)[0]

    # Patient-grouped split of no-IED patients
    no_ied_pats = np.unique(patients[no_ied_mask])
    if len(no_ied_pats) < 4:
        return None
    # Per-patient label for stratification (redundant at patient-level but kept for consistency)
    no_ied_sessions_mask = no_ied_mask  # boolean mask into analysis_df
    pat_no_ied_label = np.array([
        int(np.round(np.mean(y_all[no_ied_sessions_mask & (patients == p)])))
        for p in no_ied_pats
    ])
    if len(np.unique(pat_no_ied_label)) < 2:
        return None

    holdout_pats, train_no_ied_pats = train_test_split(
        no_ied_pats, test_size=0.5, random_state=seed, stratify=pat_no_ied_label
    )
    holdout_idx      = np.where(no_ied_mask & np.isin(patients, holdout_pats))[0]
    train_no_ied_idx = np.where(no_ied_mask & np.isin(patients, train_no_ied_pats))[0]
    train_idx = np.concatenate([ied_idx, train_no_ied_idx])
    test_idx  = holdout_idx

    y_train = analysis_df[target_col].values.astype(int)[train_idx]
    y_test  = analysis_df[target_col].values.astype(int)[test_idx]

    if len(np.unique(y_test)) < 2:
        return None

    train_df = analysis_df.iloc[train_idx]
    test_df  = analysis_df.iloc[test_idx]

    proba_all, _, selected_clf = fit_all_classifiers_select_best(
        train_df, test_df, y_train, y_test, feature_sets_clean, seed, n_cv_folds)

    out = {}
    for model_key in feature_sets_clean:
        out[model_key] = {
            clf_key: _eval_metrics(y_test, p)
            for clf_key, p in proba_all[model_key].items()
        }

    out['_meta'] = {
        'y_test': y_test,
        'proba': proba_all,
        'selected_clf': selected_clf,
    }
    return out


def build_best_clf_map(agg_dict, feature_sets_clean):
    """Best classifier per model."""
    return {mk: best_clf_for_model(mk, agg_dict) for mk in feature_sets_clean}


def _auc_placement(scores, y_true):
    pos = y_true == 1
    neg = y_true == 0
    p = scores[pos]
    n = scores[neg]
    v10 = np.array([(pi > n).mean() + 0.5 * (pi == n).mean() for pi in p])
    v01 = np.array([(nj < p).mean() + 0.5 * (nj == p).mean() for nj in n])
    return float(v10.mean()), v10, v01, int(pos.sum()), int(neg.sum())


def delong_compare(y_true, pred1, pred2):
    """
    DeLong (1988) paired AUROC comparison.
    Returns auc_a, auc_b, diff, diff_ci_lo, diff_ci_hi, z, p_two_tailed.
    """
    y_true = np.asarray(y_true, dtype=int)
    pred1  = np.asarray(pred1, dtype=float)
    pred2  = np.asarray(pred2, dtype=float)
    ok = np.isfinite(pred1) & np.isfinite(pred2)
    y_true, pred1, pred2 = y_true[ok], pred1[ok], pred2[ok]

    if len(np.unique(y_true)) < 2:
        return {k: np.nan for k in ('auc_a', 'auc_b', 'diff', 'diff_ci_lo',
                                    'diff_ci_hi', 'z', 'p')}

    auc_a, v10_a, v01_a, n_pos, n_neg = _auc_placement(pred1, y_true)
    auc_b, v10_b, v01_b, _, _        = _auc_placement(pred2, y_true)

    s10 = np.cov(np.vstack([v10_a, v10_b])) / n_pos
    s01 = np.cov(np.vstack([v01_a, v01_b])) / n_neg
    var_diff = s10[0, 0] + s10[1, 1] - 2 * s10[0, 1] + s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]

    diff = auc_a - auc_b
    if var_diff <= 0:
        return {'auc_a': auc_a, 'auc_b': auc_b, 'diff': diff,
                'diff_ci_lo': np.nan, 'diff_ci_hi': np.nan,
                'z': np.nan, 'p': np.nan}

    se = np.sqrt(var_diff)
    z  = diff / se
    p  = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        'auc_a':       auc_a,
        'auc_b':       auc_b,
        'diff':        diff,
        'diff_ci_lo':  diff - 1.96 * se,
        'diff_ci_hi':  diff + 1.96 * se,
        'z':           float(z),
        'p':           p,
    }


# ---------------------------------------------------------------------------
# SHAP feature importance (TreeExplainer / LinearExplainer by classifier type)
# ---------------------------------------------------------------------------

def _pipeline_final_estimator(fitted):
    if isinstance(fitted, Pipeline):
        return fitted.steps[-1][1]
    return fitted


def _apply_pre_classifier_steps(fitted, X):
    if not isinstance(fitted, Pipeline):
        return X
    Xt = X
    for _, step in fitted.steps[:-1]:
        Xt = step.transform(Xt)
    return Xt


def _avg_calibrated_linear_svc(calibrated):
    """Average LinearSVC coef/intercept across CalibratedClassifierCV inner fits."""
    ests = [cc.estimator for cc in calibrated.calibrated_classifiers_]

    class _LinearProxy:
        pass

    proxy = _LinearProxy()
    proxy.coef_ = np.mean([e.coef_ for e in ests], axis=0)
    proxy.intercept_ = np.mean([e.intercept_ for e in ests], axis=0)
    return proxy


def _shap_explainer_label(clf_key):
    if clf_key == 'sgd':
        return 'LinearExplainer'
    if clf_key == 'lr':
        return 'LinearExplainer'
    if clf_key == 'svm':
        return 'LinearExplainer (avg LinearSVC, pre-calibration)'
    return 'unknown'


def compute_shap_values(fitted, X_background, X_explain, clf_key):
    """
    SHAP values (n_explain, n_features).
    sgd/lr: LinearExplainer on linear model with scaled background.
    svm: LinearExplainer on averaged inner LinearSVC (CalibratedClassifierCV
         applies sigmoid post-hoc; attributions are for the linear score).
    """
    import shap

    X_bg_t = _apply_pre_classifier_steps(fitted, X_background)
    X_ex_t = _apply_pre_classifier_steps(fitted, X_explain)
    clf = _pipeline_final_estimator(fitted)
    if isinstance(clf, CalibratedClassifierCV):
        linear_model = _avg_calibrated_linear_svc(clf)
    elif isinstance(clf, (LogisticRegression, SGDClassifier)):
        linear_model = clf
    else:
        raise TypeError(f'No SHAP explainer for classifier type {type(clf)}')

    explainer = shap.LinearExplainer(linear_model, X_bg_t)
    shap_vals = explainer.shap_values(X_ex_t)
    return np.asarray(shap_vals, dtype=float)


def _subsample_rows(X, max_rows, seed):
    n = len(X)
    if n <= max_rows:
        return X, np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_rows, replace=False)
    return X[idx], idx


def run_shap_mc_iteration(seed, analysis_df, target_col, feature_cols, clf_key):
    """One MC split: fit best classifier type, compute mean |SHAP| on test patients."""
    y_all = analysis_df[target_col].values.astype(int)
    tr_idx, te_idx = _patient_grouped_split(analysis_df, y_all, TEST_SIZE, seed)
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None

    train_df = analysis_df.iloc[tr_idx]
    test_df  = analysis_df.iloc[te_idx]
    X_tr = prepare_X(train_df, feature_cols, ref_df=train_df)
    X_te = prepare_X(test_df, feature_cols, ref_df=train_df)

    est, _ = fit_clf(clf_key, X_tr, y_tr, seed)
    X_bg, _ = _subsample_rows(X_tr, SHAP_BG_SIZE, seed)
    X_ex, ex_idx = _subsample_rows(X_te, SHAP_TEST_MAX, seed + 1)
    shap_vals = compute_shap_values(est, X_bg, X_ex, clf_key)
    return {
        'mean_abs_shap': np.mean(np.abs(shap_vals), axis=0),
        'shap_values':   shap_vals,
        'feature_values': X_ex,
        'patient_ids':   test_df['patient_id'].values[ex_idx],
        'y_true':        y_te[ex_idx],
        'mc_seed':       seed,
        'n_background':  len(X_bg),
        'n_explained':   len(X_ex),
    }


def aggregate_shap_iterations(shap_iter_results, model_key, clf_key, feature_names):
    valid = [r for r in shap_iter_results if r is not None]
    if not valid:
        return pd.DataFrame()
    stack = np.vstack([r['mean_abs_shap'] for r in valid])
    mean_abs = stack.mean(axis=0)
    explainer = _shap_explainer_label(clf_key)
    rows = [{
        'model':          model_key,
        'classifier':     clf_key,
        'feature':        feat,
        'mean_abs_shap':  float(val),
        'shap_explainer': explainer,
        'n_mc_splits':    len(valid),
    } for feat, val in zip(feature_names, mean_abs)]
    return pd.DataFrame(rows).sort_values('mean_abs_shap', ascending=False)


def plot_shap_barplots(shap_df, target, top_n=SHAP_TOP_N_PLOT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = list(shap_df['model'].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 8), squeeze=False)
    for ax, model in zip(axes[0], models):
        sub = shap_df[shap_df['model'] == model].nlargest(top_n, 'mean_abs_shap')
        ax.barh(sub['feature'].iloc[::-1], sub['mean_abs_shap'].iloc[::-1], color='steelblue')
        expl = sub['shap_explainer'].iloc[0]
        clf  = sub['classifier'].iloc[0]
        ax.set_title(f"{model}\n{clf} ({expl})", fontsize=10)
        ax.set_xlabel('Mean |SHAP|')
    fig.suptitle(f'SHAP feature importance — {target}', fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f'shap_{target}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Model 6 permutation feature importance (primary split, test set)
# ---------------------------------------------------------------------------

def display_perm_feature_name(feature: str) -> str:
    mapping = {
        'spike_rate_log': 'Log10 spike rate',
        'pct_gen_slowing': 'Generalized slowing (%)',
        'pct_foc_slowing': 'Focal slowing (%)',
        'age_at_eeg': 'Age (years)',
        'gender': 'Sex (male vs female)',
    }
    if feature in mapping:
        return mapping[feature]
    name = feature
    for region in ('frontal', 'central', 'temporal', 'parietal', 'occipital'):
        if name.startswith(region + '_'):
            suffix = name[len(region) + 1:]
            suffix = suffix.replace('_relpower', ' power').replace('_', ' ')
            return f'{region.capitalize()} {suffix}'
    return name.replace('_', ' ')


def run_perm_importance_single_split(
    analysis_df,
    target_col,
    feature_cols,
    clf_key,
    split_seed=PRIMARY_SPLIT_SEED,
):
    y_all = analysis_df[target_col].values.astype(int)
    tr_idx, te_idx = _patient_grouped_split(analysis_df, y_all, TEST_SIZE, split_seed)
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None

    train_df = analysis_df.iloc[tr_idx]
    test_df = analysis_df.iloc[te_idx]
    X_tr = prepare_X(train_df, feature_cols, ref_df=train_df)
    X_te = prepare_X(test_df, feature_cols, ref_df=train_df)

    est, _ = fit_clf(clf_key, X_tr, y_tr, split_seed)
    pi = permutation_importance(
        est, X_te, y_te,
        scoring='roc_auc',
        n_repeats=PERM_N_REPEATS,
        random_state=split_seed,
        n_jobs=1,
    )
    return {
        'importances_mean': pi.importances_mean.astype(float),
        'importances_std': pi.importances_std.astype(float),
        'n_test': len(y_te),
        'split_seed': split_seed,
    }


def build_perm_importance_tables(result, feature_cols, target, clf_key):
    imp_mean = result['importances_mean']
    imp_std = result['importances_std']
    n_repeats = PERM_N_REPEATS
    summary = pd.DataFrame({
        'target': target,
        'model': PERM_MODEL_KEY,
        'classifier': clf_key,
        'feature': feature_cols,
        'display_name': [display_perm_feature_name(f) for f in feature_cols],
        'importance_mean': imp_mean,
        'importance_std_perm': imp_std,
        'importance_se_perm': imp_std / np.sqrt(n_repeats),
        'n_repeats': n_repeats,
        'split_seed': result['split_seed'],
        'n_test': result['n_test'],
        'scoring': 'roc_auc',
        'eval_set': 'test',
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return summary


def plot_perm_importance_bar(summary_df, out_path, title, top_n=PERM_TOP_N_PLOT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sub = summary_df.nlargest(top_n, 'importance_mean').iloc[::-1]
    y_pos = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(
        y_pos,
        sub['importance_mean'].values,
        xerr=sub['importance_se_perm'].values,
        color=PERM_BAR_COLOR,
        ecolor='black',
        capsize=4,
        height=0.72,
        error_kw={'linewidth': 1.2},
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub['display_name'], fontsize=14)
    ax.set_xlabel('Feature Importance', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
    ax.grid(axis='x', alpha=0.25, linestyle='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def run_perm_importance_for_target(analysis_df, target, clf_key):
    feature_cols = FEATURE_SETS_CLEAN[PERM_MODEL_KEY]
    short = TARGET_META[target]['short']
    title = TARGET_META[target]['perm_title']
    print(f"\n--- Model 6 permutation importance ({title}) ---")
    print(f"  Classifier: {clf_key}  |  Split seed: {PRIMARY_SPLIT_SEED}  |  "
          f"Repeats: {PERM_N_REPEATS}  |  Eval: primary test set")

    result = run_perm_importance_single_split(
        analysis_df, target, feature_cols, clf_key, split_seed=PRIMARY_SPLIT_SEED,
    )
    if result is None:
        print('  Skipped: invalid split for permutation importance.')
        return

    summary = build_perm_importance_tables(result, feature_cols, target, clf_key)
    save_table(summary, f'perm_importance_{PERM_MODEL_KEY}_{short}.csv')
    png_path = os.path.join(OUT_DIR, f'perm_importance_{PERM_MODEL_KEY}_{short}.png')
    plot_perm_importance_bar(summary, png_path, title=title)
    print(f'  Saved {png_path}')
    print(f"\n  Top {PERM_TOP_N_PLOT} features:")
    for _, row in summary.head(PERM_TOP_N_PLOT).iterrows():
        print(
            f"    {row['display_name']}: "
            f"{row['importance_mean']:.4f} ± {row['importance_se_perm']:.4f} (SE)"
        )


# ===========================================================================
# 5. Per-target loop
# ===========================================================================

def run_primary():
    for TARGET in TARGETS:
        t_kind = TARGET_META[TARGET]['kind']
        t_label = TARGET_META[TARGET]['label']
        print(f"\n{'='*80}")
        print(f"TARGET: {TARGET}  [{t_kind.upper()}]")
        print(f"  {t_label}")
        print(f"{'='*80}\n")

        # ── Analysis set (patient-level; no-IED holdout varies per MC split) ──────
        analysis_df = TARGET_PATIENT_DATA[TARGET].dropna(subset=[TARGET, 'age_at_eeg', 'gender']).copy()
        analysis_df[TARGET] = analysis_df[TARGET].astype(int)

        n_ied    = (analysis_df['spike_rate_per_hour'] != 0).sum()
        n_no_ied = (analysis_df['spike_rate_per_hour'] == 0).sum()
        print(f"Analysis cohort: {len(analysis_df)} patients ({n_ied} IED + {n_no_ied} no-IED)")
        print(f"Class dist: {analysis_df[TARGET].value_counts().to_dict()}\n")

        y_all = analysis_df[TARGET].values.astype(int)
        all_model_keys = list(FEATURE_SETS_CLEAN.keys())

        # ── 5a. Primary analysis — single 80/20 patient split ─────────────────────
        print(f"Running primary 80/20 split (seed={PRIMARY_SPLIT_SEED})...")
        split_result = run_mc_iteration(
            PRIMARY_SPLIT_SEED, analysis_df, TARGET, FEATURE_SETS_CLEAN,
        )
        if split_result is None:
            print(f"  Skipped {TARGET}: invalid primary split.")
            continue

        meta = split_result['_meta']
        y_te = np.asarray(meta['y_test'], dtype=int)
        proba = meta['proba']
        best_clf_map = select_best_clf_by_test_auroc(
            proba, y_te, all_model_keys, CLASSIFIERS, _eval_metrics,
        )
        print(f"  n_test={len(y_te)}  class dist={pd.Series(y_te).value_counts().sort_index().to_dict()}")
        print("  Test-AUROC-selected classifiers:")
        for mk in all_model_keys:
            print(f"    {mk}: {best_clf_map.get(mk)}")

        # ── 5b. Primary AUROC summary + bootstrap pairwise comparisons ──────────
        rng_summary = np.random.default_rng(0)
        summary = build_single_split_auroc_summary(
            split_result, FEATURE_SETS_CLEAN, best_clf_map,
            split_seed=PRIMARY_SPLIT_SEED, rng=rng_summary,
        )
        save_table(summary, f"summary_{TARGET}.csv")
        print_auroc_summary(
            summary,
            f"Primary AUROC summary — {TARGET} "
            f"(single 80/20 split, seed={PRIMARY_SPLIT_SEED})",
        )

        boot_df = bootstrap_pairwise_comparisons(
            y_te, proba, all_model_keys, best_clf_map, FEATURE_SETS_CLEAN,
            rng=np.random.default_rng(1),
        )
        if not boot_df.empty:
            save_table(boot_df, f"bootstrap_compare_{TARGET}.csv")
            print_bootstrap_pairwise(
                boot_df,
                f"Bootstrap pairwise model comparisons ({N_BOOTSTRAP}x) — {TARGET} (primary test set)",
            )

        # ── 5c. SHAP feature importance (Model4_eeg, Model6_all, best clf) ───────
        print(f"\n--- SHAP feature importance ({N_SHAP_MC_SPLITS} MC splits, best clf) ---")
        print("  sgd/lr/svm: LinearExplainer "
              "(SVM uses averaged pre-calibration LinearSVC weights).")
        shap_frames = []
        for model_key in SHAP_MODELS:
            if model_key not in FEATURE_SETS_CLEAN:
                continue
            clf_key = best_clf_map.get(model_key, list(CLASSIFIERS.keys())[0])
            cols = FEATURE_SETS_CLEAN[model_key]
            print(f"  {model_key} / {clf_key} ({_shap_explainer_label(clf_key)}) ...")
            _shap_gen = Parallel(n_jobs=N_JOBS, return_as='generator')(
                delayed(run_shap_mc_iteration)(
                    seed, analysis_df, TARGET, cols, clf_key
                )
                for seed in range(N_SHAP_MC_SPLITS)
            )
            shap_iters = list(tqdm(_shap_gen, total=N_SHAP_MC_SPLITS,
                                   desc=f'SHAP {model_key}', unit='split'))
            df_shap = aggregate_shap_iterations(shap_iters, model_key, clf_key, cols)
            if not df_shap.empty:
                shap_frames.append(df_shap)
                print(f"    top {min(SHAP_TOP_N_LOG, len(df_shap))} features:")
                for _, row in df_shap.head(SHAP_TOP_N_LOG).iterrows():
                    print(f"      {row['feature']}: {row['mean_abs_shap']:.4f}")
                bundle_dir = save_shap_beeswarm_bundle(
                    OUT_DIR, TARGET, model_key, clf_key, cols, shap_iters,
                    _shap_explainer_label(clf_key), cohort=EPI_TYPE,
                )
                if bundle_dir is not None:
                    print(f"  Saved SHAP beeswarm bundle: {bundle_dir}")
                    beeswarm_path = plot_shap_beeswarm_default(
                        OUT_DIR, TARGET, model_key,
                        mc_seed=SHAP_BEESWARM_SEED, top_n=SHAP_BEESWARM_TOP_N,
                    )
                    if beeswarm_path is not None:
                        print(f"  Saved {beeswarm_path}")
        if shap_frames:
            shap_df = pd.concat(shap_frames, ignore_index=True)
            save_table(shap_df, f'shap_{TARGET}.csv')
            plot_shap_barplots(shap_df, TARGET)

        # ── 5d. Model 6 permutation importance (primary split, test set) ──────────
        perm_clf = best_clf_map.get(PERM_MODEL_KEY, list(CLASSIFIERS.keys())[0])
        run_perm_importance_for_target(analysis_df, TARGET, perm_clf)

        # ── 5e. No-IED holdout — single 50/50 no-IED split ──────────────────────
        print(f"\n--- No-IED holdout ({n_no_ied} no-IED patients, "
              f"single 50/50 split, seed={HOLDOUT_SPLIT_SEED}) ---")
        holdout_result = run_holdout_mc_iteration(
            HOLDOUT_SPLIT_SEED, analysis_df, TARGET, FEATURE_SETS_CLEAN,
        )
        if holdout_result is None:
            print('  Skipped: insufficient no-IED patients or single class in holdout test.')
        else:
            meta_h = holdout_result['_meta']
            y_hold = np.asarray(meta_h['y_test'], dtype=int)
            proba_h = meta_h['proba']
            holdout_best = select_best_clf_by_test_auroc(
                proba_h, y_hold, all_model_keys, CLASSIFIERS, _eval_metrics,
            )
            print(f"  n_test={len(y_hold)} (no-IED only)  "
                  f"class dist={pd.Series(y_hold).value_counts().sort_index().to_dict()}")
            print("  Test-AUROC-selected classifiers:")
            for mk in all_model_keys:
                print(f"    {mk}: {holdout_best.get(mk)}")

            holdout_summary = build_single_split_auroc_summary(
                holdout_result, FEATURE_SETS_CLEAN, holdout_best,
                split_seed=HOLDOUT_SPLIT_SEED, rng=np.random.default_rng(2),
            )
            save_table(holdout_summary, f"holdout_summary_{TARGET}.csv")
            print_auroc_summary(
                holdout_summary,
                f"No-IED holdout AUROC summary — {TARGET} "
                f"(single 50/50 split, seed={HOLDOUT_SPLIT_SEED})",
            )

            holdout_boot = bootstrap_pairwise_comparisons(
                y_hold, proba_h, all_model_keys, holdout_best, FEATURE_SETS_CLEAN,
                rng=np.random.default_rng(3),
            )
            if not holdout_boot.empty:
                save_table(holdout_boot, f"holdout_bootstrap_compare_{TARGET}.csv")
                print_bootstrap_pairwise(
                    holdout_boot,
                    f"Bootstrap pairwise model comparisons ({N_BOOTSTRAP}x) — "
                    f"{TARGET} (no-IED holdout test set)",
                )

        # ── 5f. Tertile (and optional quartile) spike-stratified analysis ─────────
        for spike_split in PRIMARY_SPIKE_SPLITS:
            run_spike_stratified_analysis(
                analysis_df, FEATURE_SETS_CLEAN, N_JOBS, TARGET, spike_split=spike_split,
            )

        print(f"\n{'='*80}")
        print(f"TARGET {TARGET} complete.")
        print(f"{'='*80}\n")
# ==================== spike-stratified analysis ====================

def save_table(df: pd.DataFrame, filename: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f'  Saved {path}')


def _prediction_diagnostics(y_prob: np.ndarray) -> dict:
    p = np.asarray(y_prob, dtype=float)
    finite = p[np.isfinite(p)]
    if finite.size == 0:
        return {'n_unique_probs': 0, 'pred_min': np.nan, 'pred_max': np.nan, 'is_degenerate': True}
    rounded = np.round(finite, 8)
    n_unique = len(np.unique(rounded))
    return {
        'n_unique_probs': int(n_unique),
        'pred_min': float(finite.min()),
        'pred_max': float(finite.max()),
        'is_degenerate': n_unique <= 1,
    }


def fit_clf_spike(clf_key, X_train, y_train, seed, n_cv_folds=N_CV_FOLDS):
    cfg = CLASSIFIERS[clf_key]
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)
    estimator = clone(cfg['pipeline'])
    if _effective_needs_scaling(cfg):
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', estimator)])
        param_grid = cfg['param_grid']
        if isinstance(param_grid, dict):
            param_grid = {f'clf__{k}': v for k, v in param_grid.items()}
    else:
        pipe = estimator
        param_grid = cfg['param_grid']
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=1, refit=True)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, float(gs.best_score_), gs.best_params_


def _eval_metrics(y_true, y_prob):
    if not np.isfinite(y_prob).all() or len(np.unique(y_true)) < 2:
        return {'auroc': np.nan}
    return {'auroc': roc_auc_score(y_true, y_prob)}


def _auc_placement(scores, y_true):
    pos = y_true == 1
    neg = y_true == 0
    p = scores[pos]
    n = scores[neg]
    v10 = np.array([(pi > n).mean() + 0.5 * (pi == n).mean() for pi in p])
    v01 = np.array([(nj < p).mean() + 0.5 * (nj == p).mean() for nj in n])
    return float(v10.mean()), v10, v01, int(pos.sum()), int(neg.sum())


def delong_compare(y_true, pred1, pred2):
    y_true = np.asarray(y_true, dtype=int)
    pred1 = np.asarray(pred1, dtype=float)
    pred2 = np.asarray(pred2, dtype=float)
    ok = np.isfinite(pred1) & np.isfinite(pred2)
    y_true, pred1, pred2 = y_true[ok], pred1[ok], pred2[ok]
    if len(np.unique(y_true)) < 2:
        return {k: np.nan for k in ('auc_a', 'auc_b', 'diff', 'diff_ci_lo', 'diff_ci_hi', 'z', 'p')}
    auc_a, v10_a, v01_a, n_pos, n_neg = _auc_placement(pred1, y_true)
    auc_b, v10_b, v01_b, _, _ = _auc_placement(pred2, y_true)
    s10 = np.cov(np.vstack([v10_a, v10_b])) / n_pos
    s01 = np.cov(np.vstack([v01_a, v01_b])) / n_neg
    var_diff = s10[0, 0] + s10[1, 1] - 2 * s10[0, 1] + s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]
    diff = auc_a - auc_b
    if var_diff <= 0:
        return {
            'auc_a': auc_a, 'auc_b': auc_b, 'diff': diff,
            'diff_ci_lo': np.nan, 'diff_ci_hi': np.nan, 'z': np.nan, 'p': np.nan,
        }
    se = np.sqrt(var_diff)
    z = diff / se
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        'auc_a': auc_a, 'auc_b': auc_b, 'diff': diff,
        'diff_ci_lo': diff - 1.96 * se, 'diff_ci_hi': diff + 1.96 * se,
        'z': float(z), 'p': p,
    }


def _fit_tertile_cell(train_df, test_df, y_tr, y_te, model_key, clf_key, cols, q_seed):
    X_tr = prepare_X(train_df, cols, ref_df=train_df)
    X_te = prepare_X(test_df, cols, ref_df=train_df)
    try:
        est, cv_auroc, best_params = fit_clf_spike(clf_key, X_tr, y_tr, q_seed)
        proba = est.predict_proba(X_te)[:, 1]
    except Exception as exc:
        print(f'    WARN fit failed {model_key}/{clf_key}: {exc}')
        proba = np.full(len(y_te), np.nan)
        cv_auroc = np.nan
        best_params = {}
    return model_key, clf_key, proba, cv_auroc, best_params


def select_best_clf_by_test_auroc_local(proba, y_te, feature_sets_clean):
    return select_best_clf_by_test_auroc(
        proba, y_te, feature_sets_clean.keys(), CLASSIFIERS, _eval_metrics)


def run_tertile_fold_parallel(q_idx, train_df, test_df, y_tr, y_te, feature_sets_clean, n_jobs):
    q_seed = 1000 + q_idx
    tasks = [
        (model_key, clf_key, cols)
        for model_key, cols in feature_sets_clean.items()
        for clf_key in CLASSIFIERS
    ]
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_fit_tertile_cell)(
            train_df, test_df, y_tr, y_te, model_key, clf_key, cols, q_seed)
        for model_key, clf_key, cols in tasks
    )
    proba = {mk: {} for mk in feature_sets_clean}
    cv_scores = {mk: {} for mk in feature_sets_clean}
    best_params = {mk: {} for mk in feature_sets_clean}
    for model_key, clf_key, p, cv_auroc, params in results:
        proba[model_key][clf_key] = p
        cv_scores[model_key][clf_key] = cv_auroc
        best_params[model_key][clf_key] = params
    return proba, cv_scores, best_params


def ied_vs_marker_delong_pairwise(y_true, proba_dict, best_clf_map):
    """
    DeLong: IED model (Table 2 Model 1) vs Slowing model (Model 2) and vs
    Slowing+Continuous-EEG combined model (Model 4) — used for the IED-rate
    quartile analysis.

    Only these 2 comparisons are run (not full pairwise).
    """
    default_clf = list(CLASSIFIERS.keys())[0]
    rows = []
    for other_key in (SLOWING_MODEL_KEY, EEG_MODEL_KEY):
        if IED_MODEL_KEY not in proba_dict or other_key not in proba_dict:
            continue
        c_ied = best_clf_map.get(IED_MODEL_KEY, default_clf)
        c_other = best_clf_map.get(other_key, default_clf)
        p_ied = proba_dict[IED_MODEL_KEY].get(c_ied)
        p_other = proba_dict[other_key].get(c_other)
        if p_ied is None or p_other is None:
            continue
        if not (np.isfinite(p_ied).all() and np.isfinite(p_other).all()):
            continue
        d = delong_compare(y_true, p_ied, p_other)
        rows.append({
            'model_A': IED_MODEL_KEY, 'model_B': other_key,
            'classifier_A': c_ied, 'classifier_B': c_other,
            'auc_A': d['auc_a'], 'auc_B': d['auc_b'],
            'diff': d['diff'], 'diff_ci_lo': d['diff_ci_lo'],
            'diff_ci_hi': d['diff_ci_hi'], 'z': d['z'], 'p': d['p'],
        })
    out = pd.DataFrame(rows)
    return out


SPIKE_SPLIT_CONFIG = {
    'tertile': {
        'n_groups': 3,
        'labels': ['T1 (lowest)', 'T2', 'T3 (highest)'],
        'prefix': 'tertile',
        'column': 'tertile',
    },
    'quartile': {
        'n_groups': 4,
        'labels': ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'],
        'prefix': 'quartile',
        'column': 'quartile',
    },
}


def _spike_group_labels(spike_split: str):
    if spike_split not in SPIKE_SPLIT_CONFIG:
        raise ValueError(f'spike_split must be one of {tuple(SPIKE_SPLIT_CONFIG)}')
    return SPIKE_SPLIT_CONFIG[spike_split]


def run_spike_stratified_analysis(analysis_df, feature_sets_clean, n_jobs, target, spike_split):
    cfg = _spike_group_labels(spike_split)
    n_groups = cfg['n_groups']
    group_col = cfg['column']
    out_prefix = cfg['prefix']

    print(f'\n--- IED {spike_split} analysis (leave-one-group-out, patients) ---')
    print('  Classifiers: SGD (L1/L2), LR saga (L1/L2), LinearSVC (L1/L2)')
    print(f'  CV scoring: roc_auc ({N_CV_FOLDS}-fold stratified)')
    print(f'  C grid: {C_GRID}')
    print(f'  {n_groups} groups by index-EEG spike rate; outcome = {target}')

    strat_cohort = analysis_df.dropna(subset=['spike_rate_per_hour']).copy()
    n_spike_missing = len(analysis_df) - len(strat_cohort)
    if n_spike_missing:
        print(f'  Excluded {n_spike_missing} patients with NaN spike_rate_per_hour')
    y_all = strat_cohort[target].values.astype(int)
    try:
        group_labels = pd.qcut(
            strat_cohort['spike_rate_per_hour'], q=n_groups, labels=cfg['labels'],
        )
    except ValueError:
        # Duplicate bin edges (e.g. many patients at spike_rate_per_hour == 0)
        # collapse some quantile boundaries — fall back to fewer groups rather
        # than crash, and warn loudly since this changes the analysis shape.
        codes = pd.qcut(
            strat_cohort['spike_rate_per_hour'], q=n_groups,
            duplicates='drop', labels=False,
        )
        n_actual = int(codes.nunique())
        fallback_labels = [f'G{i}' + (' (lowest)' if i == 0 else
                           ' (highest)' if i == n_actual - 1 else '')
                           for i in range(n_actual)]
        group_labels = codes.map(dict(enumerate(fallback_labels))).astype('category')
        group_labels = group_labels.cat.set_categories(fallback_labels, ordered=True)
        print(f'  WARNING: duplicate {spike_split} bin edges (ties in spike_rate_per_hour) '
              f'collapsed {n_groups} requested groups into {n_actual} actual groups.')
    print(f'{spike_split.capitalize()} patient counts (by index-EEG spike rate):')
    print(group_labels.value_counts().sort_index())

    q_hdr = f"{'Model':<22} {'Clf':<14}  {'AUROC':>10}  {'degen':>5}"
    detail_rows = []
    delong_rows = []
    best_clf_rows = []

    for q_idx, q_label in enumerate(group_labels.cat.categories):
        q_mask = (group_labels == q_label).values
        test_idx = np.where(q_mask)[0]
        train_idx = np.where(~q_mask)[0]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        if len(np.unique(y_te)) < 2 or len(y_te) < 10:
            print(f'\n{q_label}: skipped (n_test={len(y_te)}, classes={np.unique(y_te)})')
            continue

        print(f"\n{'=' * 70}")
        print(f'{q_label}  [n_train={len(train_idx)}, n_test={len(test_idx)}]')
        print(f'  train class dist: {dict(zip(*np.unique(y_tr, return_counts=True)))}')
        print(f'  test class dist:  {dict(zip(*np.unique(y_te, return_counts=True)))}')
        print(f'  {q_hdr}')
        print(f"  {'-' * len(q_hdr)}")

        train_df = strat_cohort.iloc[train_idx]
        test_df = strat_cohort.iloc[test_idx]

        proba, cv_scores, hp_map = run_tertile_fold_parallel(
            q_idx, train_df, test_df, y_tr, y_te, feature_sets_clean, n_jobs)
        best_clf_map = select_best_clf_by_test_auroc_local(
            proba, y_te, feature_sets_clean)

        for model_key, clf_key in best_clf_map.items():
            p = proba[model_key][clf_key]
            diag = _prediction_diagnostics(p)
            test_auroc = _eval_metrics(y_te, p)['auroc']
            best_clf_rows.append({
                group_col: q_label,
                'model': model_key,
                'best_classifier': clf_key,
                'test_auroc': test_auroc,
                'train_cv_auroc': cv_scores.get(model_key, {}).get(clf_key, np.nan),
                'is_degenerate': diag['is_degenerate'],
                'n_unique_probs': diag['n_unique_probs'],
                'best_hp': str(hp_map.get(model_key, {}).get(clf_key, {})),
            })

        for model_key in feature_sets_clean:
            for clf_key, p in proba[model_key].items():
                m = _eval_metrics(y_te, p)
                diag = _prediction_diagnostics(p)
                if not np.isfinite(m['auroc']):
                    print(f"  {model_key:<22} {clf_key:<14}  {'—':>10}  {'—':>5}")
                    continue
                degen_mark = 'Y' if diag['is_degenerate'] else ''
                print(f"  {model_key:<22} {clf_key:<14}  {m['auroc']:>10.3f}  {degen_mark:>5}")
                detail_rows.append({
                    group_col: q_label, 'model': model_key, 'classifier': clf_key,
                    'auroc': m['auroc'],
                    'train_cv_auroc': cv_scores.get(model_key, {}).get(clf_key, np.nan),
                    'is_best_classifier': clf_key == best_clf_map.get(model_key),
                    'is_degenerate': diag['is_degenerate'],
                    'n_unique_probs': diag['n_unique_probs'],
                    'pred_min': diag['pred_min'],
                    'pred_max': diag['pred_max'],
                    'best_hp': str(hp_map.get(model_key, {}).get(clf_key, {})),
                    'n_train_patients': len(train_idx),
                    'n_test_patients': len(test_idx),
                })

        delong_q = ied_vs_marker_delong_pairwise(y_te, proba, best_clf_map)
        if not delong_q.empty:
            delong_q = delong_q.copy()
            delong_q.insert(0, group_col, q_label)
            delong_rows.append(delong_q)
            print(f'\n  DeLong: IED vs Slowing, IED vs Slowing+Continuous-EEG ({q_label}, '
                  f'test-AUROC best clf):')
            print(delong_q.round(4).to_string(index=False))

    if best_clf_rows:
        save_table(pd.DataFrame(best_clf_rows), f'{out_prefix}_best_clf_{target}.csv')
    if detail_rows:
        save_table(pd.DataFrame(detail_rows), f'{out_prefix}_{target}.csv')
    if delong_rows:
        save_table(pd.concat(delong_rows, ignore_index=True), f'{out_prefix}_delong_{target}.csv')


# ==================== forest plots ====================
def make_forest_plot(name, rows, out_path):
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows, columns=["comparison", "diff", "ci_lo", "ci_hi", "p"])
    if df.empty:
        print(f"No rows for {name}")
        return
    df = df.iloc[::-1].reset_index(drop=True)
    err_lo = df["diff"] - df["ci_lo"]
    err_hi = df["ci_hi"] - df["diff"]
    sig = (df["ci_lo"] > 0) | (df["ci_hi"] < 0)
    colors = sig.map({True: "#c0392b", False: "#7f8c8d"}).tolist()
    fig_height = max(3, 0.4 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    for i, (d, lo, hi, c) in enumerate(zip(df["diff"], err_lo, err_hi, colors)):
        ax.errorbar(d, i, xerr=[[lo], [hi]], fmt="o", markersize=6, capsize=3,
                    elinewidth=1.5, ecolor=c, mfc=c, mec=c)
    ax.axvline(0, linestyle="--", color="black", linewidth=1)
    ax.set_yticks(list(range(len(df))))
    ax.set_yticklabels(df["comparison"], fontsize=9)
    ax.set_xlabel("AUROC difference (Model A − Model B), 95% DeLong CI")
    ax.set_title(name)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def run_forest(prefix, target, out_dir):
    path = os.path.join(out_dir, f"{prefix}_delong_{target}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    group_col = "tertile" if prefix == "tertile" else "quartile"
    if group_col not in df.columns:
        group_col = df.columns[0]
    for grp, sub in df.groupby(group_col):
        rows = []
        for _, r in sub.iterrows():
            lab = f"{r['model_A']} vs {r['model_B']}"
            rows.append((lab, r['diff'], r['diff_ci_lo'], r['diff_ci_hi'], r['p']))
        slug = str(grp).split()[0].replace('(', '')
        make_forest_plot(str(grp), rows, os.path.join(out_dir, f"forest_szfreq_{slug}.png"))

def main():
    import argparse
    p = argparse.ArgumentParser(description='Generalized epilepsy ML')
    p.add_argument('mode', choices=['primary', 'spike', 'forest'])
    p.add_argument('--target', choices=['sz_freq_class'], default='sz_freq_class')
    p.add_argument('--spike-split', choices=['tertile', 'quartile'], default='tertile')
    p.add_argument('--prefix', choices=['tertile', 'quartile'], default='tertile',
                   help='For forest mode: which DeLong CSV prefix')
    args = p.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.mode == 'forest':
        run_forest(args.prefix, args.target, OUT_DIR)
        return

    _load_primary_data()

    if args.mode == 'primary':
        run_primary()
    elif args.mode == 'spike':
        print('=' * 80)
        print(f'Spike-{args.spike_split} — General — {args.target}')
        print(f'OUT_DIR: {OUT_DIR}')
        print('=' * 80)
        analysis_df = (
            TARGET_PATIENT_DATA[args.target]
            .dropna(subset=[args.target, 'age_at_eeg', 'gender'])
            .copy()
        )
        analysis_df[args.target] = analysis_df[args.target].astype(int)
        print(f'Analysis cohort: {len(analysis_df)} patients')
        print(f'Class dist: {analysis_df[args.target].value_counts().to_dict()}')
        run_spike_stratified_analysis(
            analysis_df, FEATURE_SETS_CLEAN, n_jobs=N_JOBS,
            target=args.target, spike_split=args.spike_split)
    else:
        raise RuntimeError(f'unhandled mode: {args.mode}')


if __name__ == '__main__':
    main()