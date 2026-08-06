#!/usr/bin/env python3
"""Nested logistic regression: null → main → interaction (Focal vs Generalized).

Pooled cohort models:
  Null:        outcome ~ age + sex + epilepsy_type + prior_sz_control
  Main:        Null + IEDs + slowing (+ optional ADR)
  Interaction: Main + epilepsy_type × (IEDs, slowing [, ADR])
Report LRTs for all nested pairs; ORs for null + interaction (bootstrap).

Spike rate uses light-preprocessed duration (SPIKE_DURATION_CSV).

Outcome modes
  binary (default): poor vs not-poor (Binomial GLM)
  ternary (--ternary-outcome): seizure-free / moderate / poor (Multinomial logit)
  --binary-prior: collapse prior to any-seizure vs none (NA retained as level 2)

  python statistical_tests.py --binary-markers --no-adr --poor-cutoff 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import generalized_ml as gml

FOCAL_EPI_TYPES = ['Temporal', 'Frontal']
ALL_EPI_TYPES = FOCAL_EPI_TYPES + ['General']
EPI_TYPE_REF = 'Generalized'
LOG_SPIKE_EPS = 0.1

ROOT = gml.ROOT
SPIKE_CSV = gml.SPIKE_CSV
REPORT_CSV = gml.REPORT_CSV
BG_FEAT_CSV = gml.BG_FEAT_CSV
FOLLOWUP_DAYS = gml.FOLLOWUP_DAYS
POOR_CUTOFF_PER_MONTH = gml.POOR_CUTOFF_PER_MONTH
N_JOBS = gml.N_JOBS
SLOWING_DIRS = [
    os.path.join(ROOT, 'morgoth', 'results', 'light_SLOWING_General'),
    os.path.join(ROOT, 'morgoth', 'results', 'light_SLOWING_Temporal'),
    os.path.join(ROOT, 'morgoth', 'results', 'light_SLOWING_Frontal'),
]

OUTCOMES = {
    'sz_freq_class': {
        'label': gml.SZ_FREQ_CLASS_LABEL,
        'short': 'seizure_control',
        'table_key': 'mean',
    },
}

MODEL_COVARIATES = [
    'outcome', 'age', 'sex', 'epilepsy_type', 'prior_sz_control',
    'spikes', 'slowing', 'adr',
]
BG_REGIONS = ['frontal', 'central', 'temporal', 'parietal', 'occipital']
ET_TERM = f"C(epilepsy_type, Treatment(reference='{EPI_TYPE_REF}'))"
# 0=seizure-free, 1=moderate, 2=poor, 3=not available (reference = seizure-free)
# With --binary-prior: 0=no seizures, 1=any seizures, 2=not available
PRIOR_TERM = 'C(prior_sz_control, Treatment(reference=0))'
N_BOOTSTRAP = 1000
BOOT_SEED = 42
BOOT_ALPHA = 0.05

# Clinician-matched Youden thresholds (vs report annotations):
#   SpikeNet2 IEDs — ied_log_spike_threshold_sweep.py (log10[rate/hr + 0.1])
#   MORGOTH slowing — slowing_verification.ipynb (fraction_slowing → % windows)
SPIKE_YOUDEN_THRESH = 0.9273          # log10(light-preproc spike_rate/hr + 0.1); Youden vs report IED
SLOWING_YOUDEN_THRESH_PCT = 13.87     # 100 * 0.1387 fraction_slowing Youden
# Spike rate denominator: light-preprocessed segment spans (same track as slowing)
SPIKE_DURATION_CSV = 'light_preprocessed_duration.csv'
SPIKE_DURATION_COL = 'light_duration_sec'
USE_BINARY_MARKERS = False           # set True via --binary-markers
USE_ADR = True                       # set False via --no-adr
USE_TERNARY_OUTCOME = False          # set True via --ternary-outcome
USE_BINARY_PRIOR = False             # set True via --binary-prior (any vs none)
# When set, binary outcome AND prior use this MeanSzFreq cutoff (/mo)
OUTCOME_POOR_CUTOFF = None           # set via --poor-cutoff
# Post-EEG 3-level outcome labels (MeanSzFreq); reference = seizure-free (0)
OUTCOME_3_LABELS = {
    0: 'seizure-free',
    1: 'moderate',
    2: 'poor',
}
OUTCOME_3_REF = 'seizure-free'
PRIOR_4_LABELS = {
    0: 'seizure_free', 1: 'moderate', 2: 'poor', 3: 'not_available',
}
PRIOR_BINARY_LABELS = {
    0: 'no_seizures', 1: 'any_seizures', 2: 'not_available',
}
PRIOR_CUTOFF_LABELS = {
    0: 'not_poor', 1: 'poor', 2: 'not_available',
}


def _whole_brain_from_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Whole-brain ADR and delta connectivity from regional background summaries.

    Regional features in background_features.csv use median-over-channels
    (within region) then median-over-epochs. Here whole-brain is approximated
    by the unweighted mean across the five cortical regions:

      ADR = mean(alpha_relpower) / mean(delta_relpower)   # alpha/delta
      delta_connectivity = mean(delta_wpli_strength)
    """
    out = df.copy()
    alpha_cols = [f'{r}_alpha_relpower' for r in BG_REGIONS]
    delta_cols = [f'{r}_delta_relpower' for r in BG_REGIONS]
    wpli_cols = [f'{r}_delta_wpli_strength' for r in BG_REGIONS]
    missing = [c for c in alpha_cols + delta_cols + wpli_cols if c not in out.columns]
    if missing:
        raise KeyError(f'Missing background feature columns: {missing}')

    alpha_mean = out[alpha_cols].mean(axis=1, skipna=True)
    delta_mean = out[delta_cols].mean(axis=1, skipna=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        out['adr'] = np.where(delta_mean > 0, alpha_mean / delta_mean, np.nan)
    out['delta_connectivity'] = out[wpli_cols].mean(axis=1, skipna=True)
    return out


def _check_required_files() -> None:
    checks = [
        ('spike_counts', SPIKE_CSV),
        ('report CSV', REPORT_CSV),
        ('background_features', BG_FEAT_CSV),
    ]
    for epi_dir in SLOWING_DIRS:
        checks.append((
            f'{os.path.basename(epi_dir)} step1 dir',
            os.path.join(epi_dir, 'pred_SLOWING_1sStep'),
        ))
    missing = [(label, path) for label, path in checks if not os.path.exists(path)]
    if missing:
        print(f'PROJECT ROOT: {ROOT}')
        print('MISSING FILES:')
        for label, path in missing:
            print(f'  [{label}] {path}')
        raise SystemExit(1)


def _apply_spike_duration_rate(sc: pd.DataFrame) -> pd.DataFrame:
    """Replace spike_rate_per_hour using light-preprocessed segment spans.

    Rate = count_0_46 / light_duration_sec * 3600.
    Routine filter still uses raw Duration_sec (<=4h). Sessions without
    light segments → NaN rate.
    """
    duration_path = Path(ROOT) / SPIKE_DURATION_CSV
    if not duration_path.is_file():
        raise FileNotFoundError(f'Missing spike duration table: {duration_path}')
    duration = pd.read_csv(duration_path)
    pid_col = 'Patient' if 'Patient' in sc.columns else 'patient_id'
    ses_col = 'Session' if 'Session' in sc.columns else 'session_number'
    t = duration.rename(columns={'Patient': pid_col, 'Session': ses_col})
    if SPIKE_DURATION_COL not in t.columns:
        raise KeyError(f'{SPIKE_DURATION_CSV} missing column {SPIKE_DURATION_COL}')
    keep = [pid_col, ses_col, SPIKE_DURATION_COL]
    sc = sc.merge(t[keep], on=[pid_col, ses_col], how='left')
    sc['spike_rate_per_hour'] = (
        pd.to_numeric(sc['count_0_46'], errors='coerce')
        / pd.to_numeric(sc[SPIKE_DURATION_COL], errors='coerce')
        * 3600
    )
    sc.loc[
        ~np.isfinite(sc[SPIKE_DURATION_COL]) | (sc[SPIKE_DURATION_COL] <= 0),
        'spike_rate_per_hour',
    ] = np.nan
    return sc


def _apply_treated_clip_spike_rate(sc: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias → light-preprocessed duration spike rate."""
    return _apply_spike_duration_rate(sc)


def _load_combined_data() -> dict:
    _check_required_files()

    sc = pd.read_csv(SPIKE_CSV)
    rep = pd.read_csv(REPORT_CSV)

    sc['count_0_46'] = pd.to_numeric(sc['count_0_46'], errors='coerce')
    sc['Duration_sec'] = pd.to_numeric(sc['Duration_sec'], errors='coerce')
    sc = _apply_treated_clip_spike_rate(sc)
    sc_routine = sc[sc['Duration_sec'].notna() & (sc['Duration_sec'] <= 4 * 3600)].copy()
    sc_routine = sc_routine.rename(columns={'Patient': 'patient_id', 'Session': 'session_number'})
    sc_routine[['patient_id', 'session_number']] = sc_routine[['patient_id', 'session_number']].astype(int)

    acq = rep['acquired_on'].fillna('').str.lower().str.strip()
    cls = rep['report_PATIENT_CLASS'].fillna('').str.lower().str.strip()
    jay = rep['jay_in_or_out'].fillna('').str.lower().str.strip()
    is_outpt = (
        acq.str.contains('spe') | acq.str.contains('radnor') |
        (cls == 'outpatient') | (jay == 'out')
    )
    outpt_keys = rep[is_outpt][['patient_id', 'session_number']].drop_duplicates().dropna().astype(int)

    rep['patient_id'] = pd.to_numeric(rep['patient_id'], errors='coerce').astype('Int64')
    rep['session_number'] = pd.to_numeric(rep['session_number'], errors='coerce').astype('Int64')
    rep_filt = rep.merge(outpt_keys, on=['patient_id', 'session_number'])
    merged = sc_routine.merge(rep_filt, on=['patient_id', 'session_number'])

    NESD_LABEL = 'non-epileptic seizure disorder'
    BAD_TYPES = {'uncertain if epilepsy', 'unknown or mrn not found', ''}

    pat_typing = rep[['patient_id', 'epilepsy_type', 'epilepsy_specific']].copy()
    pat_typing['patient_id'] = pd.to_numeric(pat_typing['patient_id'], errors='coerce')
    pat_typing = pat_typing.dropna(subset=['patient_id'])
    pat_typing['patient_id'] = pat_typing['patient_id'].astype(int)
    pat_typing['etype_norm'] = pat_typing['epilepsy_type'].fillna('').str.lower().str.strip()
    pat_typing['spec_norm'] = pat_typing['epilepsy_specific'].fillna('').str.lower().str.strip()

    def _dedupe_patient_type(grp):
        etypes = [e for e in grp['etype_norm'].unique() if e != '']
        specs = [s for s in grp['spec_norm'].unique() if s != '']
        return pd.Series({
            'etype_norm': etypes[0] if etypes else '',
            'spec_norm': specs[0] if specs else '',
        })

    pat_type_dedup = (
        pat_typing.groupby('patient_id', group_keys=False)
        .apply(_dedupe_patient_type)
        .reset_index()
    )
    pat_type_dedup['EpiType3'] = ''
    pat_type_dedup.loc[pat_type_dedup['spec_norm'].str.contains('temporal'), 'EpiType3'] = 'Temporal'
    pat_type_dedup.loc[pat_type_dedup['spec_norm'].str.contains('frontal'), 'EpiType3'] = 'Frontal'
    pat_type_dedup.loc[
        (pat_type_dedup['EpiType3'] == '') & (pat_type_dedup['etype_norm'] == 'general'),
        'EpiType3',
    ] = 'General'

    is_nesd = pat_type_dedup['etype_norm'] == NESD_LABEL
    is_bad_type = pat_type_dedup['etype_norm'].isin(BAD_TYPES)
    valid_pats = pat_type_dedup.loc[~is_nesd & ~is_bad_type, 'patient_id']
    print(f'[Typing] Excluded: {is_nesd.sum()} NESD, {is_bad_type.sum()} bad/unknown type')

    merged = merged[merged['patient_id'].isin(valid_pats)].copy()
    merged = merged.drop(columns=['EpiType3'], errors='ignore')
    merged = merged.merge(pat_type_dedup[['patient_id', 'EpiType3']], on='patient_id', how='left')

    cohort = merged[merged['EpiType3'].isin(ALL_EPI_TYPES)].copy()
    print(
        f'Pooled cohort (General + Temporal + Frontal): {len(cohort)} patient-sessions, '
        f"{cohort['patient_id'].nunique()} unique patients"
    )
    for label in ('General', 'Temporal', 'Frontal'):
        sub = cohort.loc[cohort['EpiType3'] == label, 'patient_id']
        print(f'  {label:<9}: {(cohort["EpiType3"] == label).sum()} sessions, {sub.nunique()} patients')

    ALLOWABLE_VISITS = {
        'CONSULT VISIT', 'ESTABLISHED PATIENT VISIT',
        'FOLLOW-UP PATIENT CLINIC', 'NEW PATIENT CLINIC', 'NEW PATIENT VISIT',
        'NPV MANAGEMENT DURING COVID-19', 'NPV NEUROLOGY',
        'RETURN ANNUAL VISIT', 'RETURN PATIENT EXTENDED', 'RETURN PATIENT VISIT',
        'RPV MANAGEMENT DURING COVID-19', 'TELEHEALTH VIDEO VISIT RETURN',
    }

    def _filter_visit_arrays(row):
        def _parse(s, default):
            s = str(s).strip() if pd.notna(s) else ''
            if s in ('', '[]', 'null', '[null]', '<missing>'):
                return default
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return default

        vt = _parse(row.get('visit_type'), [])
        dates = _parse(row.get('visit_dates_deid'), [])
        szf = _parse(row.get('sz_freqs'), [])
        hs = _parse(row.get('visit_hasSz'), [])

        empty = pd.Series({
            'visit_type': '[]', 'visit_dates_deid': '[]',
            'sz_freqs': '[]', 'visit_hasSz': '[]',
        })
        if not vt or not (len(vt) == len(dates) == len(szf) == len(hs)):
            return empty

        keep = [i for i, v in enumerate(vt) if str(v).strip() in ALLOWABLE_VISITS]
        if not keep:
            return empty

        return pd.Series({
            'visit_type': json.dumps([vt[i] for i in keep]),
            'visit_dates_deid': json.dumps([dates[i] for i in keep]),
            'sz_freqs': json.dumps([szf[i] for i in keep]),
            'visit_hasSz': json.dumps([hs[i] for i in keep]),
        })

    filtered_arrays = cohort.apply(_filter_visit_arrays, axis=1)
    cohort[['visit_type', 'visit_dates_deid', 'sz_freqs', 'visit_hasSz']] = filtered_arrays
    n_with_visits = (cohort['visit_dates_deid'] != '[]').sum()
    print(f'[Visit-type filter] {n_with_visits}/{len(cohort)} patient-sessions retain >=1 qualifying clinic visit')

    def compute_age(row):
        try:
            return (
                pd.to_datetime(str(row['start_time_deid']).strip()) -
                pd.to_datetime(str(row['deid_birth_date']).strip())
            ).days / 365.25
        except (TypeError, ValueError):
            return np.nan

    cohort['age_at_eeg'] = cohort.apply(compute_age, axis=1)
    cohort['gender'] = cohort['nlp_gender'].map({'F': 0.0, 'M': 1.0})

    session_sz = gml.add_session_mean_sz_freq_2yr(
        cohort.drop_duplicates(subset=['patient_id', 'session_number'])
    )
    session_sz = gml.add_session_mean_sz_freq_2yr_pre(session_sz)
    n_sess_label = int(session_sz['mean_sz_freq_2yr'].notna().sum())
    n_sess_pre = int(session_sz['mean_sz_freq_2yr_pre'].notna().sum())
    print(
        f'[MeanSzFreq 2yr window] {n_sess_label}/{len(session_sz)} sessions with '
        f'documented seizure frequency in (0, {FOLLOWUP_DAYS}d] post-EEG'
    )
    print(
        f'[MeanSzFreq 2yr PRE window] {n_sess_pre}/{len(session_sz)} sessions with '
        f'documented seizure frequency in [-{FOLLOWUP_DAYS}d, 0) pre-EEG'
    )

    base_cols = [
        'patient_id', 'session_number', 'start_time_deid', 'spike_rate_per_hour',
        'age_at_eeg', 'gender', 'EpiType3',
    ]
    dataset = (
        cohort[base_cols]
        .drop_duplicates(subset=['patient_id', 'session_number'])
        .rename(columns={'session_number': 'session'})
        .copy()
    )
    dataset[['patient_id', 'session']] = dataset[['patient_id', 'session']].astype(str)

    _sz = session_sz[[
        'patient_id', 'session_number', 'mean_sz_freq_2yr', 'mean_sz_freq_2yr_pre',
    ]].copy()
    _sz['patient_id'] = _sz['patient_id'].astype(str)
    _sz = _sz.rename(columns={'session_number': 'session'})
    _sz['session'] = _sz['session'].astype(str)
    dataset = dataset.merge(_sz, on=['patient_id', 'session'], how='left')
    print(f'Base session table: {len(dataset)} patient-sessions, {dataset["patient_id"].nunique()} patients')

    bg = pd.read_csv(BG_FEAT_CSV)
    bg = bg[bg['subtype'].isin(ALL_EPI_TYPES)].copy()
    bg[['patient_id', 'session']] = bg[['patient_id', 'session']].astype(str)
    bg = bg.rename(columns={'subtype': 'EpiType3'})
    dataset = dataset.merge(bg, on=['patient_id', 'session', 'EpiType3'], how='left')
    dataset = dataset.drop(columns=['EpiType3'])
    print(f'After EEG features: {len(dataset)} rows, {len(dataset.columns)} columns')

    slow_df = gml.build_session_slowing_table(SLOWING_DIRS, n_jobs=N_JOBS)
    if slow_df is not None:
        dataset = dataset.merge(slow_df, on=['patient_id', 'session'], how='left')
        print(f"Slowing matched: {dataset['pct_foc_slowing'].notna().sum()}/{len(dataset)} sessions with pct slowing")
    else:
        print('Slowing not yet available — run run_light_preprocessed.sh first')
        for col in ['pct_no_slowing', 'pct_foc_slowing', 'pct_gen_slowing', 'n_win_slow']:
            dataset[col] = np.nan

    n_eeg_sessions_total = (
        cohort.groupby('patient_id')['session_number']
        .nunique()
        .rename('n_eeg_sessions')
        .reset_index()
    )
    n_eeg_sessions_total['patient_id'] = n_eeg_sessions_total['patient_id'].astype(str)

    valid_sz_sessions = gml.filter_valid_eeg_sessions(dataset)
    patient_sz = gml.build_patient_sz_from_session_means(
        valid_sz_sessions, poor_cutoff=POOR_CUTOFF_PER_MONTH)
    print(f"[MeanSzFreq 2yr window] {patient_sz['mean_sz_freq'].notna().sum()} patients with label")

    dataset_patient_mean = gml.aggregate_mean_eeg_per_patient(
        dataset,
        patient_sz_df=None,
        n_eeg_sessions_df=n_eeg_sessions_total,
        poor_cutoff_per_month=POOR_CUTOFF_PER_MONTH,
    )

    valid_for_prior = gml.filter_valid_eeg_sessions(dataset)
    prior = gml.build_patient_prior_sz_control(
        valid_for_prior, poor_cutoff=POOR_CUTOFF_PER_MONTH)
    prior['patient_id'] = prior['patient_id'].astype(str)
    dataset_patient_mean['patient_id'] = dataset_patient_mean['patient_id'].astype(str)
    dataset_patient_mean = dataset_patient_mean.merge(
        prior[['patient_id', 'mean_sz_freq_pre', 'prior_sz_control']],
        on='patient_id', how='left',
    )
    dataset_patient_mean['prior_sz_control'] = (
        dataset_patient_mean['prior_sz_control'].fillna(3).astype(int)
    )
    prior_vc = dataset_patient_mean['prior_sz_control'].value_counts().sort_index()
    print(
        '[Prior 2yr sz control] '
        f"0=seizure-free:{int(prior_vc.get(0, 0))}, "
        f"1=moderate:{int(prior_vc.get(1, 0))}, "
        f"2=poor:{int(prior_vc.get(2, 0))}, "
        f"3=not available:{int(prior_vc.get(3, 0))}"
    )

    epi_map = pat_type_dedup[['patient_id', 'EpiType3']].copy()
    epi_map['patient_id'] = epi_map['patient_id'].astype(str)
    epi_map['epilepsy_type'] = epi_map['EpiType3'].map(
        lambda t: 'Focal' if t in FOCAL_EPI_TYPES else ('Generalized' if t == 'General' else np.nan)
    )

    dataset_patient_mean = dataset_patient_mean.merge(
        epi_map[['patient_id', 'EpiType3', 'epilepsy_type']], on='patient_id', how='left')

    print(f'\n  Patient-level (mean EEG — sz_freq_class): {len(dataset_patient_mean)} patients')
    return {'sz_freq_class': dataset_patient_mean}


def _code_outcome_3way(mean_sz_freq: float) -> float:
    """0=seizure-free (ref), 1=moderate, 2=poor from post-EEG MeanSzFreq."""
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
    if x <= POOR_CUTOFF_PER_MONTH:
        return 1.0
    return 2.0


def _collapse_prior_binary(prior: int) -> int:
    """Collapse 4-level prior → none / any seizures / not available.

    0 = no seizures (was SF)
    1 = any seizures (was moderate or poor)
    2 = not available (was 3)
    """
    p = int(prior)
    if p == 0:
        return 0
    if p in (1, 2):
        return 1
    return 2  # not available (and any unexpected)


def _code_prior_from_cutoff(mean_sz_freq_pre, poor_cutoff: float) -> int:
    """Match outcome grouping: 0=not poor, 1=poor (mean > cutoff), 2=NA."""
    if mean_sz_freq_pre is None or (isinstance(mean_sz_freq_pre, float) and not np.isfinite(mean_sz_freq_pre)):
        return 2
    try:
        x = float(mean_sz_freq_pre)
    except (TypeError, ValueError):
        return 2
    if not np.isfinite(x):
        return 2
    return 1 if x > poor_cutoff else 0


def _code_outcome_binary(mean_sz_freq: float, poor_cutoff: float) -> float:
    """0 = not poor (moderate/else), 1 = poor (mean > poor_cutoff)."""
    if mean_sz_freq is None or (isinstance(mean_sz_freq, float) and not np.isfinite(mean_sz_freq)):
        return np.nan
    try:
        x = float(mean_sz_freq)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(x):
        return np.nan
    return 1.0 if x > poor_cutoff else 0.0


def build_cohort(
    patient_df: pd.DataFrame,
    outcome_col: str,
    binary_markers: bool = False,
    ternary_outcome: bool = False,
    binary_prior: bool = False,
    poor_cutoff: float | None = None,
) -> pd.DataFrame:
    df = patient_df.rename(columns={
        'spike_rate_log': 'spikes',
        'age_at_eeg': 'age',
        'gender': 'sex',
    }).copy()
    cut = float(POOR_CUTOFF_PER_MONTH if poor_cutoff is None else poor_cutoff)
    if ternary_outcome:
        if 'mean_sz_freq' not in df.columns:
            raise KeyError('ternary_outcome requires mean_sz_freq on patient table')
        df['outcome'] = df['mean_sz_freq'].map(_code_outcome_3way)
    elif poor_cutoff is not None:
        if 'mean_sz_freq' not in df.columns:
            raise KeyError('custom poor_cutoff requires mean_sz_freq on patient table')
        df['outcome'] = df['mean_sz_freq'].map(lambda x: _code_outcome_binary(x, cut))
        print(
            f'[Binary outcome] poor = mean > {cut}/mo; '
            f'moderate/else = mean <= {cut}/mo'
        )
    else:
        df['outcome'] = df[outcome_col]
    df['slowing'] = df['pct_gen_slowing'].fillna(0) + df['pct_foc_slowing'].fillna(0)
    df = _whole_brain_from_regions(df)

    # Keep the same clinical complete-case sample as before (N=1358).
    # A few patients lack background features entirely; median-impute ADR
    # (analogous to ML prepare_X median imputation).
    # prior_sz_control is always defined (0/1/2/3; missing pre-visits → 3).
    base_covars = ['outcome', 'age', 'sex', 'epilepsy_type', 'spikes', 'slowing']
    df = df.dropna(subset=base_covars).copy()
    df['prior_sz_control'] = df['prior_sz_control'].fillna(3).astype(int)
    # Prior: cutoff-matched (when --poor-cutoff) > any-vs-none > 4-level
    if poor_cutoff is not None:
        if 'mean_sz_freq_pre' not in df.columns:
            raise KeyError('cutoff-matched prior requires mean_sz_freq_pre')
        df['prior_sz_control'] = df['mean_sz_freq_pre'].map(
            lambda x: _code_prior_from_cutoff(x, cut)
        ).astype(int)
        vc = df['prior_sz_control'].value_counts().sort_index()
        print(
            f'[Prior matched to cutoff {cut}/mo] '
            f'0(not_poor)={int(vc.get(0, 0))}, 1(poor)={int(vc.get(1, 0))}, '
            f'2(NA)={int(vc.get(2, 0))}'
        )
    elif binary_prior:
        before = df['prior_sz_control'].value_counts().sort_index().to_dict()
        df['prior_sz_control'] = df['prior_sz_control'].map(_collapse_prior_binary).astype(int)
        after = df['prior_sz_control'].value_counts().sort_index().to_dict()
        print(
            '[Binary prior] collapsed 4-level → none/any/NA: '
            f'before={before} → after={{'
            f'0(no)={after.get(0, 0)}, 1(any)={after.get(1, 0)}, '
            f'2(NA)={after.get(2, 0)}}}'
        )
    n_miss = int(df['adr'].isna().sum())
    if n_miss:
        fill = float(df['adr'].median(skipna=True))
        df['adr'] = df['adr'].fillna(fill)
        print(f'[Impute] adr: filled {n_miss} patients with median={fill:.4f}')

    if binary_markers:
        # Binarize SpikeNet2 / MORGOTH scores at clinician-matched Youden cutoffs.
        n_sp = int((df['spikes'] >= SPIKE_YOUDEN_THRESH).sum())
        n_sl = int((df['slowing'] >= SLOWING_YOUDEN_THRESH_PCT).sum())
        print(
            f'[Binary markers] spikes >= {SPIKE_YOUDEN_THRESH:.4f}: '
            f'{n_sp}/{len(df)} present; '
            f'slowing >= {SLOWING_YOUDEN_THRESH_PCT:.2f}%: '
            f'{n_sl}/{len(df)} present'
        )
        df['spikes'] = (df['spikes'] >= SPIKE_YOUDEN_THRESH).astype(int)
        df['slowing'] = (df['slowing'] >= SLOWING_YOUDEN_THRESH_PCT).astype(int)

    df = df.dropna(subset=MODEL_COVARIATES).copy()
    df['outcome'] = df['outcome'].astype(int)
    df['sex'] = df['sex'].astype(int)
    df['prior_sz_control'] = df['prior_sz_control'].astype(int)
    if ternary_outcome:
        vc = df['outcome'].value_counts().sort_index()
        print(
            '[Ternary outcome] '
            + ', '.join(
                f"{OUTCOME_3_LABELS.get(int(k), k)}={int(v)}" for k, v in vc.items()
            )
        )
    else:
        n1 = int((df['outcome'] == 1).sum())
        print(
            f'[Binary outcome] poor={n1}/{len(df)} ({100 * n1 / len(df):.1f}%), '
            f'cutoff>{cut}/mo'
        )
    return df


def _lrt(alt_result, null_result) -> dict:
    stat = 2 * (alt_result.llf - null_result.llf)
    lr_df = int(round(float(alt_result.df_model) - float(null_result.df_model)))
    if lr_df <= 0:
        lr_df = max(1, int(round(stat)))  # fallback; should not happen for nested models
    return {'stat': float(stat), 'df': lr_df, 'p': float(stats.chi2.sf(stat, lr_df))}


def _bootstrap_p(boot_betas: np.ndarray) -> float:
    boot = np.asarray(boot_betas, dtype=float)
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return float('nan')
    p = 2.0 * min(float(np.mean(boot >= 0)), float(np.mean(boot <= 0)))
    return float(min(max(p, 0.0), 1.0))


def bootstrap_glm_coefs(
    df: pd.DataFrame,
    formula: str,
    terms: list[str],
    n_boot: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    # --- bootstrap setup ---
    # draw patients with replacement, refit the same logistic model, store betas
    rng = rng if rng is not None else np.random.default_rng(BOOT_SEED)
    n = len(df)
    stores: dict[str, list[float]] = {t: [] for t in terms}

    for _ in range(n_boot):
        # one bootstrap sample = n patients drawn with replacement
        sample = df.iloc[rng.integers(0, n, size=n)]
        try:
            res = smf.glm(formula, data=sample, family=sm.families.Binomial()).fit()
        except Exception:
            continue
        # skip if a level disappeared and coeffs don't match the full-sample model
        if list(res.params.index) != list(terms):
            continue
        for t in terms:
            stores[t].append(float(res.params[t]))

    return {t: np.asarray(v, dtype=float) for t, v in stores.items()}


def bootstrap_mnlogit_coefs(
    df: pd.DataFrame,
    formula: str,
    param_template: pd.DataFrame,
    n_boot: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Bootstrap multinomial logit betas; keys are '{eq_col}::{term}'."""
    rng = rng if rng is not None else np.random.default_rng(BOOT_SEED)
    n = len(df)
    term_index = list(param_template.index)
    eq_cols = list(param_template.columns)
    stores: dict[str, list[float]] = {
        f'{c}::{t}': [] for c in eq_cols for t in term_index
    }

    for _ in range(n_boot):
        sample = df.iloc[rng.integers(0, n, size=n)]
        # need all outcome levels present
        if sample['outcome'].nunique() < df['outcome'].nunique():
            continue
        try:
            res = smf.mnlogit(formula, data=sample).fit(
                method='newton', maxiter=200, disp=False)
        except Exception:
            continue
        if list(res.params.index) != term_index or list(res.params.columns) != eq_cols:
            continue
        for c in eq_cols:
            for t in term_index:
                stores[f'{c}::{t}'].append(float(res.params.loc[t, c]))

    return {k: np.asarray(v, dtype=float) for k, v in stores.items()}


def _main_formula(include_adr: bool) -> str:
    if include_adr:
        return (
            f'outcome ~ age + sex + {ET_TERM} + {PRIOR_TERM} + spikes + slowing + adr'
        )
    return (
        f'outcome ~ age + sex + {ET_TERM} + {PRIOR_TERM} + spikes + slowing'
    )


def _interaction_formula(include_adr: bool) -> str:
    if include_adr:
        return (
            f'outcome ~ age + sex + {ET_TERM} + {PRIOR_TERM} + spikes + slowing + adr + '
            f'{ET_TERM}:spikes + {ET_TERM}:slowing + {ET_TERM}:adr'
        )
    return (
        f'outcome ~ age + sex + {ET_TERM} + {PRIOR_TERM} + spikes + slowing + '
        f'{ET_TERM}:spikes + {ET_TERM}:slowing'
    )


def _fit_one(formula: str, df: pd.DataFrame, ternary_outcome: bool):
    """Plain logistic / multinomial fit — no hyperparameter search."""
    if ternary_outcome:
        return smf.mnlogit(formula, data=df).fit(
            method='newton', maxiter=200, disp=False)
    return smf.glm(formula, data=df, family=sm.families.Binomial()).fit()


def fit_nested_models(
    df: pd.DataFrame,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOT_SEED,
    include_adr: bool = True,
    ternary_outcome: bool = False,
) -> dict:
    """Null → main → interaction. LRT all pairs. Bootstrap ORs for null + interaction."""
    f_null = f'outcome ~ age + sex + {ET_TERM} + {PRIOR_TERM}'
    f_main = _main_formula(include_adr)
    f_inter = _interaction_formula(include_adr)

    print('  Fitting null / main / interaction (no HP tuning)...')
    null = _fit_one(f_null, df, ternary_outcome)
    main = _fit_one(f_main, df, ternary_outcome)
    inter = _fit_one(f_inter, df, ternary_outcome)

    if ternary_outcome:
        eq_labels = {
            null.params.columns[0]: 'moderate_vs_seizure_free',
            null.params.columns[1]: 'poor_vs_seizure_free',
        }
    else:
        eq_labels = None

    formulas = {'null': f_null, 'main': f_main, 'interaction': f_inter}
    models = {'null': null, 'main': main, 'interaction': inter}

    # Bootstrap ORs for null and interaction models
    print(f'  Bootstrapping null + interaction model ORs ({n_boot} resamples)...')
    if ternary_outcome:
        boot_null = bootstrap_mnlogit_coefs(
            df, f_null, null.params, n_boot=n_boot,
            rng=np.random.default_rng(seed))
        boot_inter = bootstrap_mnlogit_coefs(
            df, f_inter, inter.params, n_boot=n_boot,
            rng=np.random.default_rng(seed))
    else:
        boot_null = bootstrap_glm_coefs(
            df, f_null, list(null.params.index), n_boot=n_boot,
            rng=np.random.default_rng(seed))
        boot_inter = bootstrap_glm_coefs(
            df, f_inter, list(inter.params.index), n_boot=n_boot,
            rng=np.random.default_rng(seed))
    n_ok_null = int(next(iter(boot_null.values())).size) if boot_null else 0
    n_ok_inter = int(next(iter(boot_inter.values())).size) if boot_inter else 0
    print(f'    null: kept {n_ok_null}/{n_boot} successful replicates')
    print(f'    interaction: kept {n_ok_inter}/{n_boot} successful replicates')

    lrt_main_vs_null = _lrt(main, null)
    lrt_inter_vs_main = _lrt(inter, main)
    lrt_inter_vs_null = _lrt(inter, null)
    print(
        f"  LRT main vs null: chi2({lrt_main_vs_null['df']})="
        f"{lrt_main_vs_null['stat']:.2f}, p={lrt_main_vs_null['p']:.4g}"
    )
    print(
        f"  LRT interaction vs main: chi2({lrt_inter_vs_main['df']})="
        f"{lrt_inter_vs_main['stat']:.2f}, p={lrt_inter_vs_main['p']:.4g}"
    )
    print(
        f"  LRT interaction vs null: chi2({lrt_inter_vs_null['df']})="
        f"{lrt_inter_vs_null['stat']:.2f}, p={lrt_inter_vs_null['p']:.4g}"
    )

    return {
        'formulas': formulas,
        'models': models,
        'bootstrap': {'null': boot_null, 'interaction': boot_inter},
        'n_boot': n_boot,
        'lrt_main_vs_null': lrt_main_vs_null,
        'lrt_interaction_vs_main': lrt_inter_vs_main,
        'lrt_interaction_vs_null': lrt_inter_vs_null,
        'include_adr': include_adr,
        'ternary_outcome': ternary_outcome,
        'eq_labels': eq_labels,
    }


def clean_term_name(term: str) -> str:
    t = str(term)
    if t == 'Intercept':
        return 'Intercept'
    if t == 'age':
        return 'Age at EEG (years)'
    if t == 'sex':
        return 'Sex (male vs female)'
    if t == 'spikes':
        if USE_BINARY_MARKERS:
            return (
                f'IEDs present (log10[rate/hr + 0.1] >= {SPIKE_YOUDEN_THRESH:.4f}; '
                "Youden vs report)"
            )
        return 'IEDs: log10 spike rate (log10[rate/hr + 0.1])'
    if t == 'slowing':
        if USE_BINARY_MARKERS:
            return (
                f'Slowing present (pct foc+gen >= {SLOWING_YOUDEN_THRESH_PCT:.2f}%; '
                "Youden vs report)"
            )
        return 'Slowing (% windows: generalized + focal)'
    if t == 'adr':
        return 'ADR (whole-brain alpha/delta)'
    if t == 'delta_connectivity':
        return 'Delta connectivity (whole-brain wPLI strength)'
    if 'prior_sz_control' in t and 'T.1' in t:
        if OUTCOME_POOR_CUTOFF is not None:
            return (
                f'Prior 2yr: poor (mean > {OUTCOME_POOR_CUTOFF}/mo) vs moderate/else'
            )
        if USE_BINARY_PRIOR:
            return 'Prior 2yr: any seizures vs none'
        return 'Prior 2yr seizure control: moderate vs seizure-free'
    if 'prior_sz_control' in t and 'T.2' in t:
        if OUTCOME_POOR_CUTOFF is not None:
            return 'Prior 2yr: not available vs moderate/else'
        if USE_BINARY_PRIOR:
            return 'Prior 2yr: not available vs none'
        return 'Prior 2yr seizure control: poor vs seizure-free'
    if 'prior_sz_control' in t and 'T.3' in t:
        return 'Prior 2yr seizure control: not available vs seizure-free'
    if 'epilepsy_type' in t and ':spikes' in t:
        return 'Epilepsy type (Focal) × IEDs'
    if 'epilepsy_type' in t and ':slowing' in t:
        return 'Epilepsy type (Focal) × slowing'
    if 'epilepsy_type' in t and ':adr' in t:
        return 'Epilepsy type (Focal) × ADR'
    if 'epilepsy_type' in t and ':delta_connectivity' in t:
        return 'Epilepsy type (Focal) × delta connectivity'
    m = re.search(r'\[T\.([^\]]+)\]', t)
    if m and 'epilepsy_type' in t:
        return f'Epilepsy type: {m.group(1)} vs {EPI_TYPE_REF}'
    return t


def _or_row_from_boot(
    term: str,
    beta: float,
    boot: np.ndarray | None,
    se_fallback: float,
    p_fallback: float,
    alpha: float,
) -> dict:
    lo_pct = 100.0 * alpha / 2.0
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    if boot is not None and np.isfinite(boot).sum() > 0:
        boot = np.asarray(boot, dtype=float)
        boot = boot[np.isfinite(boot)]
        se = float(np.std(boot, ddof=1)) if boot.size > 1 else float('nan')
        or_boot = np.exp(boot)
        or_lo = float(np.percentile(or_boot, lo_pct))
        or_hi = float(np.percentile(or_boot, hi_pct))
        p = _bootstrap_p(boot)
        n_boot_ok = int(boot.size)
    else:
        se = float(se_fallback)
        or_lo = float(np.exp(beta - 1.96 * se))
        or_hi = float(np.exp(beta + 1.96 * se))
        p = float(p_fallback)
        n_boot_ok = 0
    return {
        'Term': term,
        'Display': clean_term_name(term),
        'Beta': beta,
        'SE': se,
        'p': p,
        'OR': float(np.exp(beta)),
        'OR_lo': or_lo,
        'OR_hi': or_hi,
        'n_boot': n_boot_ok,
    }


def make_fe_table(
    result,
    boot_coefs: dict[str, np.ndarray] | None = None,
    alpha: float = BOOT_ALPHA,
    eq_labels: dict | None = None,
) -> pd.DataFrame:
    rows = []
    params = result.params

    # Multinomial: params is DataFrame (terms × equations)
    if isinstance(params, pd.DataFrame):
        bse = result.bse
        pvalues = result.pvalues
        for eq_col in params.columns:
            eq_name = (eq_labels or {}).get(eq_col, f'eq_{eq_col}')
            for term in params.index:
                beta = float(params.loc[term, eq_col])
                key = f'{eq_col}::{term}'
                boot = None if boot_coefs is None else boot_coefs.get(key)
                se_fb = float(bse.loc[term, eq_col]) if hasattr(bse, 'loc') else float('nan')
                p_fb = float(pvalues.loc[term, eq_col]) if hasattr(pvalues, 'loc') else float('nan')
                row = _or_row_from_boot(term, beta, boot, se_fb, p_fb, alpha)
                row['Equation'] = eq_name
                rows.append(row)
    else:
        for term in params.index:
            beta = float(params[term])
            boot = None if boot_coefs is None else boot_coefs.get(term)
            row = _or_row_from_boot(
                term, beta, boot,
                float(result.bse[term]), float(result.pvalues[term]), alpha,
            )
            row['Equation'] = 'poor_vs_not_poor'
            rows.append(row)

    out = pd.DataFrame(rows)
    out['CI_fmt'] = out.apply(lambda r: f"[{r['OR_lo']:.3f}, {r['OR_hi']:.3f}]", axis=1)
    return out


def _term_line_compact(r: pd.Series) -> str:
    sig = '*' if r['p'] < 0.05 else ''
    return (
        f"  {r['Display']:<38} {r['OR']:.3f} [{r['OR_lo']:.3f}, {r['OR_hi']:.3f}]  "
        f"p={r['p']:.4g}{sig}"
    )


def _fe_lines(fe: pd.DataFrame, eq_labels: dict | None = None) -> list[str]:
    lines: list[str] = []
    if 'Equation' in fe.columns:
        for eq in fe['Equation'].unique():
            if len(fe['Equation'].unique()) > 1:
                lines.append(f'  [{eq}]')
            sub = fe[(fe['Equation'] == eq) & (fe['Term'] != 'Intercept')]
            for _, r in sub.iterrows():
                lines.append(_term_line_compact(r))
    else:
        for _, r in fe[fe['Term'] != 'Intercept'].iterrows():
            lines.append(_term_line_compact(r))
    return lines


def _term_line(r: pd.Series) -> str:
    sig = '*' if r['p'] < 0.05 else ''
    return f"  {r['Display']}: OR={r['OR']:.3f}, 95% CI={r['CI_fmt']}, p={r['p']:.4g}{sig}"


def _print_fe_by_equation(fe: pd.DataFrame, model_label: str) -> None:
    print(f'\n{model_label} — OR (bootstrap 95% CI), bootstrap p:')
    if 'Equation' not in fe.columns:
        for _, r in fe[fe['Term'] != 'Intercept'].iterrows():
            print(_term_line(r))
        return
    for eq in fe['Equation'].unique():
        print(f'  [{eq}]')
        sub = fe[(fe['Equation'] == eq) & (fe['Term'] != 'Intercept')]
        for _, r in sub.iterrows():
            print(_term_line(r))


def print_model_results(oc: str, cohort_df: pd.DataFrame, fit: dict) -> None:
    ternary = bool(fit.get('ternary_outcome'))
    if ternary:
        label = (
            'Post-EEG seizure control (3-level): seizure-free / moderate / poor '
            '(multinomial logit; reference = seizure-free)'
        )
        vc = cohort_df['outcome'].value_counts().sort_index()
        prev = ', '.join(
            f"{OUTCOME_3_LABELS.get(int(k), k)}={int(v)} ({100 * v / len(cohort_df):.1f}%)"
            for k, v in vc.items()
        )
    else:
        label = OUTCOMES[oc]['label']
        prev = f"prevalence(poor)={cohort_df['outcome'].mean():.3f}"

    lrt1 = fit['lrt_main_vs_null']
    lrt2 = fit['lrt_interaction_vs_main']
    lrt3 = fit['lrt_interaction_vs_null']

    print('=' * 72)
    print(label)
    print(
        f"N={len(cohort_df)} (Focal={int((cohort_df['epilepsy_type'] == 'Focal').sum())}, "
        f"Generalized={int((cohort_df['epilepsy_type'] == 'Generalized').sum())}), "
        f"{prev}"
    )
    print(
        f"LRT (Main vs Null): chi2({lrt1['df']})={lrt1['stat']:.2f}, p={lrt1['p']:.4g} "
        f"→ EEG main effects "
        f"{'DO' if lrt1['p'] < 0.05 else 'do NOT'} add beyond clinical covariates."
    )
    print(
        f"LRT (Interaction vs Main): chi2({lrt2['df']})={lrt2['stat']:.2f}, p={lrt2['p']:.4g} "
        f"→ type×marker interactions "
        f"{'DO' if lrt2['p'] < 0.05 else 'do NOT'} add beyond main effects."
    )
    print(
        f"LRT (Interaction vs Null): chi2({lrt3['df']})={lrt3['stat']:.2f}, p={lrt3['p']:.4g}"
    )
    print(
        f"(ORs from null + interaction models; bootstrap 95% CI / p, n_boot={fit['n_boot']}; "
        'no multiplicity correction)'
    )

    eq_labels = fit.get('eq_labels')
    fe_null = make_fe_table(
        fit['models']['null'],
        boot_coefs=fit['bootstrap']['null'],
        eq_labels=eq_labels,
    )
    fe_inter = make_fe_table(
        fit['models']['interaction'],
        boot_coefs=fit['bootstrap']['interaction'],
        eq_labels=eq_labels,
    )
    _print_fe_by_equation(fe_null, 'Null model')
    _print_fe_by_equation(fe_inter, 'Interaction model')


def save_outputs(oc: str, cohort_df: pd.DataFrame, fit: dict, out_dir: Path) -> None:
    short = OUTCOMES[oc]['short']
    if fit.get('ternary_outcome'):
        short = 'seizure_control_3way'
    subdir = out_dir / short
    subdir.mkdir(parents=True, exist_ok=True)

    eq_labels = fit.get('eq_labels')
    fe_null = make_fe_table(
        fit['models']['null'],
        boot_coefs=fit['bootstrap']['null'],
        eq_labels=eq_labels,
    )
    fe_inter = make_fe_table(
        fit['models']['interaction'],
        boot_coefs=fit['bootstrap']['interaction'],
        eq_labels=eq_labels,
    )
    fe_null.to_csv(subdir / 'or_table_null.csv', index=False)
    fe_inter.to_csv(subdir / 'or_table_interaction.csv', index=False)

    pd.DataFrame([
        {
            'comparison': 'main_vs_null',
            'chi2': fit['lrt_main_vs_null']['stat'],
            'df': fit['lrt_main_vs_null']['df'],
            'p': fit['lrt_main_vs_null']['p'],
        },
        {
            'comparison': 'interaction_vs_main',
            'chi2': fit['lrt_interaction_vs_main']['stat'],
            'df': fit['lrt_interaction_vs_main']['df'],
            'p': fit['lrt_interaction_vs_main']['p'],
        },
        {
            'comparison': 'interaction_vs_null',
            'chi2': fit['lrt_interaction_vs_null']['stat'],
            'df': fit['lrt_interaction_vs_null']['df'],
            'p': fit['lrt_interaction_vs_null']['p'],
        },
    ]).to_csv(subdir / 'lrt_results.csv', index=False)

    out_labels = (
        {0: 'seizure_free', 1: 'moderate', 2: 'poor'}
        if fit.get('ternary_outcome') else {0: 'not_poor', 1: 'poor'}
    )
    prior_labels = (
        PRIOR_CUTOFF_LABELS if OUTCOME_POOR_CUTOFF is not None else
        (PRIOR_BINARY_LABELS if USE_BINARY_PRIOR else PRIOR_4_LABELS)
    )
    pd.crosstab(
        cohort_df['epilepsy_type'],
        cohort_df['outcome'].map(out_labels),
        margins=True,
    ).to_csv(subdir / 'epilepsy_type_crosstab.csv')
    pd.crosstab(
        cohort_df['prior_sz_control'].map(prior_labels),
        cohort_df['outcome'].map(out_labels),
        margins=True,
    ).to_csv(subdir / 'prior_sz_control_crosstab.csv')
    print(f'Saved: {subdir}')


def write_results_summary(cohorts: dict, fits: dict, out_dir: Path) -> Path:
    n_boot = next(iter(fits.values()))['n_boot'] if fits else N_BOOTSTRAP
    include_adr = bool(next(iter(fits.values())).get('include_adr', True)) if fits else True
    ternary = bool(next(iter(fits.values())).get('ternary_outcome', False)) if fits else False

    if include_adr:
        model_desc = [
            '  Null model:        outcome ~ age + sex + epilepsy_type + prior_sz_control',
            '  Main-effects model: Null + IEDs + slowing + ADR',
            '  Interaction model: Main + epilepsy_type:(IEDs, slowing, ADR)',
        ]
        adr_line = '  ADR: mean(regional alpha_relpower) / mean(regional delta_relpower) [continuous]'
    else:
        model_desc = [
            '  Null model:        outcome ~ age + sex + epilepsy_type + prior_sz_control',
            '  Main-effects model: Null + IEDs + slowing',
            '  Interaction model: Main + epilepsy_type:(IEDs, slowing)',
            '                     [ADR omitted; no type × prior; no delta]',
        ]
        adr_line = '  ADR: OMITTED from this run'

    outcome_line = (
        '  Outcome: 3-level post-EEG seizure control (seizure-free / moderate / poor); '
        'multinomial logit; reference = seizure-free. No HP tuning.'
        if ternary else
        '  Outcome: binary poor vs not-poor; binomial logit. No HP tuning.'
    )
    if OUTCOME_POOR_CUTOFF is not None:
        prior_lines = [
            f'  Prior 2yr (pre-EEG): matched to outcome cutoff {OUTCOME_POOR_CUTOFF}/mo:',
            f'  0=not poor (mean <= {OUTCOME_POOR_CUTOFF}/mo), '
            f'1=poor (mean > {OUTCOME_POOR_CUTOFF}/mo),',
            '  2=not available; reference = not poor (0). No type × prior interaction.',
        ]
    elif USE_BINARY_PRIOR:
        prior_lines = [
            '  Prior 2yr (pre-EEG): collapsed to any-seizure vs none (+ not available):',
            '  0=no seizures (mean == 0), 1=any seizures (mean > 0),',
            '  2=not available; reference = no seizures (0). No type × prior interaction.',
        ]
    else:
        prior_lines = [
            '  Prior 2yr seizure control (pre-EEG): same Rule-1 MeanSzFreq aggregation as',
            f'  the post-EEG outcome, over [-{FOLLOWUP_DAYS}d, 0); coded 0=seizure-free',
            f'  (mean == 0), 1=moderate (0 < mean <= {POOR_CUTOFF_PER_MONTH}/mo),',
            f'  2=poor (mean > {POOR_CUTOFF_PER_MONTH}/mo), 3=not available;',
            '  reference = seizure-free (0). No type × prior interaction.',
        ]

    lines = [
        'Statistical tests — EEG markers vs. epilepsy type (Focal vs Generalized)',
        '=' * 72,
        'METHODS',
        '  Cohort: General + Focal (Temporal + Frontal combined) epilepsy, pooled.',
        *model_desc,
        '  LRT: Main vs Null; Interaction vs Main; Interaction vs Null.',
        '  Odds ratios: reported for Null and Interaction models (bootstrap CIs/p).',
        outcome_line,
        (
            f'  IEDs: BINARY present if log10(rate/hr + {LOG_SPIKE_EPS}) '
            f'>= {SPIKE_YOUDEN_THRESH:.4f} (Youden vs report IED); '
            f'Slowing: BINARY present if pct_foc+pct_gen >= {SLOWING_YOUDEN_THRESH_PCT:.2f}% '
            f'(Youden vs report slowing = 0.1387 fraction)'
            if USE_BINARY_MARKERS else
            f'  IEDs: log10(spike_rate_per_hour + {LOG_SPIKE_EPS}); '
            'Slowing: pct_gen_slowing + pct_foc_slowing (% windows)'
        ),
        adr_line,
        *prior_lines,
        f'  Epilepsy type reference: {EPI_TYPE_REF}',
        f'  Term-level 95% CIs: patient bootstrap percentile CI (n_boot={n_boot}).',
        '  Term-level p-values: two-sided percentile bootstrap p for H0: beta = 0.',
        '  No multiple-comparison correction.',
        '  All three models fit on the same complete-case sample.',
        '',
    ]

    for oc in OUTCOMES:
        df = cohorts[oc]
        fit = fits[oc]
        lrt1 = fit['lrt_main_vs_null']
        lrt2 = fit['lrt_interaction_vs_main']
        lrt3 = fit['lrt_interaction_vs_null']
        eq_labels = fit.get('eq_labels')

        if fit.get('ternary_outcome'):
            vc = df['outcome'].value_counts().sort_index()
            prev = ', '.join(
                f"{OUTCOME_3_LABELS.get(int(k), k)}={int(v)}" for k, v in vc.items()
            )
            header = (
                f"3-level post-EEG control (multinomial; ref=seizure-free) "
                f"(N={len(df)}, Focal={int((df['epilepsy_type'] == 'Focal').sum())}, "
                f"Generalized={int((df['epilepsy_type'] == 'Generalized').sum())}, {prev})"
            )
        else:
            header = (
                f"{OUTCOMES[oc]['label']} (N={len(df)}, "
                f"Focal={int((df['epilepsy_type'] == 'Focal').sum())}, "
                f"Generalized={int((df['epilepsy_type'] == 'Generalized').sum())}, "
                f"prevalence={df['outcome'].mean():.3f})"
            )

        lines += [
            header,
            '-' * 72,
            f"LRT (Main vs Null): chi2({lrt1['df']})={lrt1['stat']:.2f}, p={lrt1['p']:.4g}",
            f"LRT (Interaction vs Main): chi2({lrt2['df']})={lrt2['stat']:.2f}, p={lrt2['p']:.4g}",
            f"LRT (Interaction vs Null): chi2({lrt3['df']})={lrt3['stat']:.2f}, p={lrt3['p']:.4g}",
            '\nNull model ORs:',
        ]
        fe_null = make_fe_table(
            fit['models']['null'],
            boot_coefs=fit['bootstrap']['null'],
            eq_labels=eq_labels,
        )
        lines.extend(_fe_lines(fe_null, eq_labels))
        lines += ['\nInteraction model ORs:']
        fe_inter = make_fe_table(
            fit['models']['interaction'],
            boot_coefs=fit['bootstrap']['interaction'],
            eq_labels=eq_labels,
        )
        lines.extend(_fe_lines(fe_inter, eq_labels))
        lines.append('')

    path = out_dir / 'results_summary.txt'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(path.read_text())
    print(f'Saved: {path}')
    return path


def write_aim1_compact_summary(cohorts: dict, fits: dict, out_dir: Path) -> Path | None:
    """One-page Aim 1 summary with null + interaction ORs (bootstrap)."""
    if len(fits) != 1:
        return None
    oc = next(iter(fits))
    fit = fits[oc]
    df = cohorts[oc]
    eq_labels = fit.get('eq_labels')

    if USE_BINARY_MARKERS:
        marker_line = f'IED >={SPIKE_YOUDEN_THRESH:.4f}; slowing >={SLOWING_YOUDEN_THRESH_PCT:.2f}%'
    else:
        marker_line = 'continuous IED/slowing'

    if OUTCOME_POOR_CUTOFF is not None:
        n_poor = int((df['outcome'] == 1).sum())
        title = (
            f'Aim 1 — poor>{OUTCOME_POOR_CUTOFF}/mo outcome + MATCHED prior '
            f'(>{OUTCOME_POOR_CUTOFF}/mo) + '
            f'{"Youden BINARY" if USE_BINARY_MARKERS else "CONTINUOUS"} markers'
        )
        outcome_line = (
            f'Outcome:  poor = post mean > {OUTCOME_POOR_CUTOFF}/mo '
            f'(N={n_poor}, {100 * n_poor / len(df):.1f}%); '
            f'else <={OUTCOME_POOR_CUTOFF}/mo (N={len(df) - n_poor})'
        )
        prior_vc = df['prior_sz_control'].value_counts().sort_index()
        prior_line = (
            f'Prior:    same rule on pre-EEG mean — '
            f'not_poor={int(prior_vc.get(0, 0))}, poor={int(prior_vc.get(1, 0))}, '
            f'NA={int(prior_vc.get(2, 0))} (ref=not_poor)'
        )
        fname = (
            f'aim1_logistic_or_poor{OUTCOME_POOR_CUTOFF:g}mo_matched_prior_'
            f'{"youden" if USE_BINARY_MARKERS else "cont"}'
            f'{"_noadr" if not USE_ADR else ""}.txt'
        )
    elif fit.get('ternary_outcome'):
        return None
    else:
        return None

    lrt1 = fit['lrt_main_vs_null']
    lrt2 = fit['lrt_interaction_vs_main']
    lrt3 = fit['lrt_interaction_vs_null']
    lines = [
        title,
        '=' * len(title),
        outcome_line,
        prior_line,
        f'ADR {"omitted" if not USE_ADR else "included"}; n_boot={fit["n_boot"]}',
        f'Markers:  {marker_line}',
        '',
        'LRTs: '
        f'Main vs Null χ²({lrt1["df"]})={lrt1["stat"]:.2f} p={lrt1["p"]:.4g} · '
        f'Inter vs Main χ²({lrt2["df"]})={lrt2["stat"]:.2f} p={lrt2["p"]:.4g} · '
        f'Inter vs Null χ²({lrt3["df"]})={lrt3["stat"]:.2f} p={lrt3["p"]:.4g}',
        '',
        'Null model ORs [95% CI], bootstrap p:',
    ]
    fe_null = make_fe_table(
        fit['models']['null'],
        boot_coefs=fit['bootstrap']['null'],
        eq_labels=eq_labels,
    )
    lines.extend(_fe_lines(fe_null, eq_labels))
    lines += ['', 'Interaction model ORs [95% CI], bootstrap p:']
    fe_inter = make_fe_table(
        fit['models']['interaction'],
        boot_coefs=fit['bootstrap']['interaction'],
        eq_labels=eq_labels,
    )
    lines.extend(_fe_lines(fe_inter, eq_labels))
    lines += ['', f'OUT_DIR: {out_dir}/']
    path = out_dir / fname
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Saved: {path}')
    return path


def main() -> None:
    global USE_BINARY_MARKERS, USE_ADR, USE_TERNARY_OUTCOME, USE_BINARY_PRIOR
    global OUTCOME_POOR_CUTOFF
    p = argparse.ArgumentParser(
        description='Nested logistic regression — Focal vs Generalized EEG markers')
    p.add_argument(
        '--out-dir', type=Path,
        default=_SCRIPT_DIR / 'outputs' / 'Combined' / 'statistical_tests_with_prior_no_deltaconn',
    )
    p.add_argument('--n-boot', type=int, default=N_BOOTSTRAP)
    p.add_argument('--seed', type=int, default=BOOT_SEED)
    p.add_argument(
        '--binary-markers', action='store_true',
        help=(
            'Binarize IEDs / slowing at clinician-matched Youden thresholds '
            f'(spikes>={SPIKE_YOUDEN_THRESH}, slowing>={SLOWING_YOUDEN_THRESH_PCT}%%)'
        ),
    )
    p.add_argument(
        '--no-adr', action='store_true',
        help='Omit ADR and type×ADR from the interaction model',
    )
    p.add_argument(
        '--ternary-outcome', action='store_true',
        help=(
            '3-level post-EEG outcome (seizure-free/moderate/poor) via multinomial '
            'logit; reference = seizure-free'
        ),
    )
    p.add_argument(
        '--binary-prior', action='store_true',
        help=(
            'Collapse prior to any-seizure vs none (0=no, 1=any); '
            'not-available kept as level 2 so N unchanged. '
            'Ignored when --poor-cutoff is set (prior then matches that cutoff).'
        ),
    )
    p.add_argument(
        '--poor-cutoff', type=float, default=None,
        help=(
            'Binary outcome AND prior from MeanSzFreq: poor if mean > CUTOFF /mo '
            '(else = moderate/not-poor); prior NA retained. '
            'Default uses precomputed sz_freq_class (4.33). '
            'Ignored with --ternary-outcome.'
        ),
    )
    args = p.parse_args()
    USE_BINARY_MARKERS = bool(args.binary_markers)
    USE_ADR = not bool(args.no_adr)
    USE_TERNARY_OUTCOME = bool(args.ternary_outcome)
    USE_BINARY_PRIOR = bool(args.binary_prior)
    poor_cutoff = None if USE_TERNARY_OUTCOME else args.poor_cutoff
    OUTCOME_POOR_CUTOFF = poor_cutoff

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out_dir}')
    print(f'Bootstrap: n_boot={args.n_boot}, seed={args.seed}')
    if USE_TERNARY_OUTCOME:
        print('Outcome: TERNARY (SF/moderate/poor; multinomial; ref=seizure-free)')
    elif poor_cutoff is not None:
        print(
            f'Outcome: BINARY poor vs moderate/else '
            f'(poor = mean > {poor_cutoff}/mo)'
        )
    else:
        print('Outcome: BINARY (poor vs not-poor; binomial; sz_freq_class)')
    if poor_cutoff is not None:
        print(f'Prior: MATCHED to cutoff {poor_cutoff}/mo (not_poor / poor / NA)')
    else:
        print(
            'Prior: '
            + (
                'BINARY (no seizures vs any; NA retained)'
                if USE_BINARY_PRIOR else
                '4-level (SF / moderate / poor / NA)'
            )
        )
    if USE_BINARY_MARKERS:
        print(
            f'Markers: BINARY Youden thresholds '
            f'(spikes>={SPIKE_YOUDEN_THRESH}, slowing>={SLOWING_YOUDEN_THRESH_PCT}%); '
            f'ADR={"included" if USE_ADR else "OMITTED"}'
        )
    else:
        print(
            f'Markers: continuous (log spike rate; pct slowing); '
            f'ADR={"included" if USE_ADR else "OMITTED"}'
        )

    patient_tables = _load_combined_data()
    cohorts: dict = {}
    fits: dict = {}
    for oc in OUTCOMES:
        cohorts[oc] = build_cohort(
            patient_tables[oc], oc,
            binary_markers=USE_BINARY_MARKERS,
            ternary_outcome=USE_TERNARY_OUTCOME,
            binary_prior=USE_BINARY_PRIOR,
            poor_cutoff=poor_cutoff,
        )
        fits[oc] = fit_nested_models(
            cohorts[oc], n_boot=args.n_boot, seed=args.seed,
            include_adr=USE_ADR, ternary_outcome=USE_TERNARY_OUTCOME,
        )
        print_model_results(oc, cohorts[oc], fits[oc])
        save_outputs(oc, cohorts[oc], fits[oc], out_dir)

    write_results_summary(cohorts, fits, out_dir)
    write_aim1_compact_summary(cohorts, fits, out_dir)


if __name__ == '__main__':
    main()
