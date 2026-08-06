#!/usr/bin/env python3
"""
Marker PPV + OR tables (General and Focal) — matches make_fig_*_slowing_by_outcome.m.

Markers (patient-level):
  slowing — (pct_foc_slowing + pct_gen_slowing) >= 14 (% windows)
  spikes  — log10(spike_rate_per_hour + 0.1) > log10(exp(-1))
            (same rate cutoff as former ln(rate+0.1) > -1)

Outcomes:
  seizure_recurrence_2yr — index EEG, any HasSz==1 in (0, 730d] post-EEG
  sz_freq_class          — mean 2yr-window MeanSzFreq across valid sessions > 4.33/mo

OR: univariate logistic regression (marker present vs absent), Wald 95% CI.
PPV: P(outcome+ | marker+).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import generalized_ml as gml

LOG_EPS = 0.1
SLOWING_THRESH = 14.0
# Preserve former ln(rate+0.1) > -1 rate cutoff under log10:
#   rate > exp(-1) - 0.1  ↔  log10(rate+0.1) > log10(exp(-1))
SPIKE_LOG_THRESH = float(np.log10(np.exp(-1.0)))  # ≈ -0.4343
N_BOOT = 5000
ALPHA = 0.05

OUTCOME_SPECS = [
    ('2 year seizure recurrence', 'seizure_recurrence_2yr', 'index'),
    ('poor seizure control', 'sz_freq_class', 'mean'),
]


def _p_fmt(p: float) -> str:
    if not np.isfinite(p):
        return 'NaN'
    if p < 0.001:
        return '<0.001'
    if p >= 0.995:
        return '1.00'
    return f'{p:.4g}'


def add_marker_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['pct_slowing'] = out['pct_foc_slowing'].fillna(0) + out['pct_gen_slowing'].fillna(0)
    out['spike_rate_log'] = np.log10(out['spike_rate_per_hour'].astype(float) + LOG_EPS)
    out['slowing_present'] = (out['pct_slowing'] >= SLOWING_THRESH).astype(int)
    out['spikes_present'] = (out['spike_rate_log'] > SPIKE_LOG_THRESH).astype(int)
    return out


def _fit_or(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    """Logistic OR for marker (1 vs 0) predicting outcome y."""
    X = sm.add_constant(x.astype(float), has_constant='add')
    res = sm.Logit(y.astype(float), X).fit(disp=0, maxiter=200)
    beta = float(res.params[1])
    or_val = float(np.exp(beta))
    ci = res.conf_int()[1]
    or_lo, or_hi = float(np.exp(ci[0])), float(np.exp(ci[1]))
    p = float(res.pvalues[1])
    return or_val, or_lo, or_hi, p


def _bootstrap_or_ci(y: np.ndarray, x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(y)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        try:
            or_v, _, _, _ = _fit_or(y[idx], x[idx])
            if np.isfinite(or_v):
                boots.append(or_v)
        except Exception:
            pass
    if len(boots) < 50:
        return np.nan, np.nan
    lo = float(np.percentile(boots, 100 * ALPHA / 2))
    hi = float(np.percentile(boots, 100 * (1 - ALPHA / 2)))
    return lo, hi


def marker_row(
    df: pd.DataFrame,
    marker_col: str,
    marker_label: str,
    outcome_col: str,
    outcome_label: str,
    rng: np.random.Generator,
) -> dict:
    sub = df.dropna(subset=[marker_col, outcome_col]).copy()
    sub[outcome_col] = sub[outcome_col].astype(int)
    mpos = sub[marker_col].astype(int) == 1
    y = sub[outcome_col].to_numpy()
    x = sub[marker_col].to_numpy()

    n_pos = int(mpos.sum())
    ppv = float(y[mpos].mean() * 100) if n_pos else np.nan
    or_val, wald_lo, wald_hi, p = _fit_or(y, x)
    boot_lo, boot_hi = _bootstrap_or_ci(y, x, rng)
    or_lo = boot_lo if np.isfinite(boot_lo) else wald_lo
    or_hi = boot_hi if np.isfinite(boot_hi) else wald_hi

    return {
        'Marker': marker_label,
        'Outcome': outcome_label,
        'N_marker_pos': n_pos,
        'N_outcome_labeled': len(sub),
        'PPV': ppv,
        'OR': or_val,
        'OR_lo': or_lo,
        'OR_hi': or_hi,
        'p': p,
        'OR_fmt': f'{or_val:.2f} [{or_lo:.2f}, {or_hi:.2f}]',
        'PPV_fmt': f'{ppv:.1f}%' if np.isfinite(ppv) else 'NaN',
        'p_fmt': _p_fmt(p),
    }


def build_table(patient_tables: dict[str, pd.DataFrame], cohort_label: str) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(1)
    for outcome_label, outcome_col, table_key in OUTCOME_SPECS:
        df = add_marker_columns(patient_tables[table_key])
        for marker_col, marker_label in (
            ('slowing_present', 'slowing'),
            ('spikes_present', 'spikes'),
        ):
            row = marker_row(df, marker_col, marker_label, outcome_col, outcome_label, rng)
            row['Cohort'] = cohort_label
            rows.append(row)
    return pd.DataFrame(rows)


def format_display_table(tbl: pd.DataFrame) -> pd.DataFrame:
    return tbl[['Marker', 'Outcome', 'PPV_fmt', 'OR_fmt', 'p_fmt']].rename(
        columns={'PPV_fmt': 'PPV', 'OR_fmt': 'OR [95% CI]', 'p_fmt': 'p value'})


def load_general_tables() -> dict[str, pd.DataFrame]:
    gml._load_primary_data()
    return {
        'index': gml.TARGET_PATIENT_DATA['seizure_recurrence_2yr'].copy(),
        'mean': gml.TARGET_PATIENT_DATA['sz_freq_class'].copy(),
    }


def load_focal_tables() -> dict[str, pd.DataFrame]:
    """Temporal + Frontal — mirrors ml_analysis_focal.py patient tables."""
    root = gml.ROOT
    spike_csv = gml.SPIKE_CSV
    report_csv = gml.REPORT_CSV
    slowing_dirs = [
        os.path.join(root, 'morgoth', 'results', 'light_SLOWING_Temporal'),
        os.path.join(root, 'morgoth', 'results', 'light_SLOWING_Frontal'),
    ]
    focal_types = ['Temporal', 'Frontal']
    poor_cutoff = gml.POOR_CUTOFF_PER_MONTH

    sc = pd.read_csv(spike_csv)
    rep = pd.read_csv(report_csv)
    sc['count_0_46'] = pd.to_numeric(sc['count_0_46'], errors='coerce')
    sc['Duration_sec'] = pd.to_numeric(sc['Duration_sec'], errors='coerce')
    sc['spike_rate_per_hour'] = sc['count_0_46'] / sc['Duration_sec'] * 3600
    sc_routine = sc[sc['Duration_sec'].notna() & (sc['Duration_sec'] <= 4 * 3600)].copy()
    sc_routine = sc_routine.rename(columns={'Patient': 'patient_id', 'Session': 'session_number'})
    sc_routine[['patient_id', 'session_number']] = sc_routine[['patient_id', 'session_number']].astype(int)

    acq = rep['acquired_on'].fillna('').str.lower().str.strip()
    cls = rep['report_PATIENT_CLASS'].fillna('').str.lower().str.strip()
    jay = rep['jay_in_or_out'].fillna('').str.lower().str.strip()
    is_outpt = acq.str.contains('spe') | acq.str.contains('radnor') | (cls == 'outpatient') | (jay == 'out')
    outpt_keys = rep[is_outpt][['patient_id', 'session_number']].drop_duplicates().dropna().astype(int)
    rep['patient_id'] = pd.to_numeric(rep['patient_id'], errors='coerce').astype('Int64')
    rep['session_number'] = pd.to_numeric(rep['session_number'], errors='coerce').astype('Int64')
    merged = sc_routine.merge(rep.merge(outpt_keys, on=['patient_id', 'session_number']), on=['patient_id', 'session_number'])

    # typing
    NESD = 'non-epileptic seizure disorder'
    bad = {'uncertain if epilepsy', 'unknown or mrn not found', ''}
    pt = rep[['patient_id', 'epilepsy_type', 'epilepsy_specific']].copy()
    pt['patient_id'] = pd.to_numeric(pt['patient_id'], errors='coerce').dropna().astype(int)
    pt['etype'] = pt['epilepsy_type'].fillna('').str.lower().str.strip()
    pt['spec'] = pt['epilepsy_specific'].fillna('').str.lower().str.strip()

    def dedupe(g):
        et = [e for e in g['etype'].unique() if e]
        sp = [s for s in g['spec'].unique() if s]
        return pd.Series({'etype': et[0] if et else '', 'spec': sp[0] if sp else ''})

    ptd = pt.groupby('patient_id', group_keys=False).apply(dedupe).reset_index()
    ptd['EpiType3'] = ''
    ptd.loc[ptd['spec'].str.contains('temporal'), 'EpiType3'] = 'Temporal'
    ptd.loc[ptd['spec'].str.contains('frontal'), 'EpiType3'] = 'Frontal'
    ptd.loc[(ptd['EpiType3'] == '') & (ptd['etype'] == 'general'), 'EpiType3'] = 'General'
    valid = ptd.loc[(ptd['etype'] != NESD) & (~ptd['etype'].isin(bad)), 'patient_id']
    merged = merged[merged['patient_id'].isin(valid)].merge(ptd[['patient_id', 'EpiType3']], on='patient_id')
    cohort = merged[merged['EpiType3'].isin(focal_types)].copy()

    allowable = {
        'CONSULT VISIT', 'ESTABLISHED PATIENT VISIT', 'FOLLOW-UP PATIENT CLINIC',
        'NEW PATIENT CLINIC', 'NEW PATIENT VISIT', 'NPV MANAGEMENT DURING COVID-19',
        'NPV NEUROLOGY', 'RETURN ANNUAL VISIT', 'RETURN PATIENT EXTENDED',
        'RETURN PATIENT VISIT', 'RPV MANAGEMENT DURING COVID-19', 'TELEHEALTH VIDEO VISIT RETURN',
    }

    def filt_row(row):
        def parse(s):
            s = str(s).strip() if pd.notna(s) else ''
            if s in ('', '[]', 'null', '[null]'):
                return []
            return json.loads(s)

        vt = parse(row.get('visit_type'))
        dates = parse(row.get('visit_dates_deid'))
        szf = parse(row.get('sz_freqs'))
        hs = parse(row.get('visit_hasSz'))
        if not vt or not (len(vt) == len(dates) == len(szf) == len(hs)):
            return '[]', '[]', '[]', '[]'
        keep = [i for i, v in enumerate(vt) if str(v).strip() in allowable]
        if not keep:
            return '[]', '[]', '[]', '[]'
        return (
            json.dumps([vt[i] for i in keep]),
            json.dumps([dates[i] for i in keep]),
            json.dumps([szf[i] for i in keep]),
            json.dumps([hs[i] for i in keep]),
        )

    fa = cohort.apply(lambda r: pd.Series(filt_row(r), index=[
        'visit_type', 'visit_dates_deid', 'sz_freqs', 'visit_hasSz']), axis=1)
    cohort[['visit_type', 'visit_dates_deid', 'sz_freqs', 'visit_hasSz']] = fa

    def compute_age(row):
        try:
            return (pd.to_datetime(str(row['start_time_deid']).strip())
                    - pd.to_datetime(str(row['deid_birth_date']).strip())).days / 365.25
        except Exception:
            return np.nan

    def compute_rec(row):
        try:
            eeg_dt = pd.to_datetime(str(row['start_time_deid']).strip())
        except Exception:
            return np.nan, np.nan
        dates = gml.parse_json(row['visit_dates_deid'])
        haszs = gml.parse_json(row['visit_hasSz'])
        if not dates or len(dates) != len(haszs):
            return np.nan, np.nan
        post = []
        for d, h in zip(dates, haszs):
            try:
                gap = (pd.to_datetime(str(d).strip()) - eeg_dt).days
                if 0 < gap <= gml.FOLLOWUP_DAYS:
                    hz = gml.hasz(h)
                    if np.isfinite(hz):
                        post.append(hz)
            except Exception:
                pass
        if not post:
            return np.nan, np.nan
        return (1.0 if any(h == 1.0 for h in post) else 0.0), float(len(post))

    cohort['age_at_eeg'] = cohort.apply(compute_age, axis=1)
    cohort['gender'] = cohort['nlp_gender'].map({'F': 0.0, 'M': 1.0})
    rec = cohort.apply(compute_rec, axis=1, result_type='expand')
    rec.columns = ['seizure_recurrence_2yr', 'n_followup_visits']
    cohort = cohort.join(rec)

    session_sz = gml.add_session_mean_sz_freq_2yr(
        cohort.drop_duplicates(subset=['patient_id', 'session_number']))
    base = cohort.drop_duplicates(subset=['patient_id', 'session_number']).rename(
        columns={'session_number': 'session'})
    base[['patient_id', 'session']] = base[['patient_id', 'session']].astype(str)
    _sz = session_sz[['patient_id', 'session_number', 'mean_sz_freq_2yr']].copy()
    _sz['patient_id'] = _sz['patient_id'].astype(str)
    _sz = _sz.rename(columns={'session_number': 'session'})
    _sz['session'] = _sz['session'].astype(str)
    dataset = base.merge(_sz, on=['patient_id', 'session'], how='left')

    bg = pd.read_csv(gml.BG_FEAT_CSV)
    bg = bg[bg['subtype'].isin(focal_types)].copy()
    bg[['patient_id', 'session']] = bg[['patient_id', 'session']].astype(str)
    bg = bg.rename(columns={'subtype': 'EpiType3'})
    dataset = dataset.merge(bg, on=['patient_id', 'session', 'EpiType3'], how='left').drop(columns=['EpiType3'])

    slow_df = gml.build_session_slowing_table(slowing_dirs, n_jobs=4)
    if slow_df is not None:
        dataset = dataset.merge(slow_df, on=['patient_id', 'session'], how='left')

    n_eeg = cohort.groupby('patient_id')['session_number'].nunique().rename('n_eeg_sessions').reset_index()
    n_eeg['patient_id'] = n_eeg['patient_id'].astype(str)
    dataset_before = dataset.copy()
    index_df = gml.select_first_index_eeg_per_patient(dataset_before)
    valid_sz = gml.filter_valid_eeg_sessions(dataset_before)
    patient_sz = gml.build_patient_sz_from_session_means(valid_sz, poor_cutoff=poor_cutoff)
    index_pat = gml.aggregate_sessions_to_patients(index_df, patient_sz, n_eeg)
    mean_pat = gml.aggregate_mean_eeg_per_patient(
        dataset_before, patient_sz_df=None, n_eeg_sessions_df=n_eeg, poor_cutoff_per_month=poor_cutoff)
    return {'index': index_pat, 'mean': mean_pat}


def main() -> None:
    out_dir = _SCRIPT_DIR / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading General cohort...')
    gen_tbl = build_table(load_general_tables(), 'General')
    print('Loading Focal cohort...')
    foc_tbl = build_table(load_focal_tables(), 'Focal')
    combined = pd.concat([gen_tbl, foc_tbl], ignore_index=True)

    combined.to_csv(out_dir / 'marker_or_ppv_table.csv', index=False)
    txt_lines = [
        'Marker PPV + OR tables (2yr-window sz_freq_class for poor seizure control)',
        f'Slowing+: pct_foc + pct_gen >= {SLOWING_THRESH}%; '
        f'Spikes+: log10(rate+{LOG_EPS}) > {SPIKE_LOG_THRESH:.4f}',
        f'OR 95% CI: bootstrap ({N_BOOT} resamples); p: Wald logistic',
        '',
        '=== General ===',
        format_display_table(gen_tbl).to_string(index=False),
        '',
        '=== Focal (Temporal + Frontal) ===',
        format_display_table(foc_tbl).to_string(index=False),
    ]
    txt_path = out_dir / 'marker_or_ppv_table.txt'
    txt_path.write_text('\n'.join(txt_lines) + '\n', encoding='utf-8')

    print('\n=== General ===')
    print(format_display_table(gen_tbl).to_string(index=False))
    print('\n=== Focal (Temporal + Frontal) ===')
    print(format_display_table(foc_tbl).to_string(index=False))
    print(f'\nSaved: {out_dir / "marker_or_ppv_table.csv"}')
    print(f'Saved: {txt_path}')


if __name__ == '__main__':
    main()
