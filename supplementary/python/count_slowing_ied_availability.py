#!/usr/bin/env python3
"""Data-management inventory: patients usable from slowing + IEDs alone.

Does NOT require ADR / delta_connectivity / background_features.
Reports how many Generalized patients have:
  - light-preprocessed EDF in light_preprocessed/General/ (Morgoth input list)
  - Morgoth slowing output in results/light_SLOWING_General/pred_SLOWING_1sStep/
  - available continuous IED rate (spike_counts)
  - available binary IED annotation (DEID present/absent)
  - optional overlay: sz_freq_class label (pure = visit arrays only; constrained = ML spine)

Usage:
  /mnt/sauce/littlab/users/cho6/.venv/bin/python3 count_slowing_ied_availability.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import generalized_ml as gml
import aim2_generalized_ml as a2

warnings.filterwarnings('ignore')

_RE_PID_SES = re.compile(r'sub-Penn(\d+)_ses-(\d+)')

OUT_DIR = Path(os.environ.get(
    'SLOWING_IED_COUNT_OUT',
    str(_SCRIPT_DIR / 'outputs' / 'General' / 'slowing_ied_availability'),
))

LIGHT_PREPROCESSED_GENERAL = os.path.join(gml.ROOT, 'light_preprocessed', 'General')
MORGOTH_SLOWING_GENERAL = os.path.join(
    gml.ROOT, 'morgoth', 'results', 'light_SLOWING_General', 'pred_SLOWING_1sStep',
)


def _nuniq(s: pd.Series) -> int:
    return int(pd.Series(s).nunique())


def print_count(label: str, n: int, denom: int | None = None):
    if denom is None:
        print(f'  {label:<58} {n:>6d}')
    else:
        pct = 100.0 * n / denom if denom else 0.0
        print(f'  {label:<58} {n:>6d}  ({pct:5.1f}%)')


def load_all_typed_generalized_patient_ids() -> set[str]:
    """All DEID-report patients typed General (before routine/validity filters)."""
    rep = pd.read_csv(gml.REPORT_CSV, low_memory=False)
    pat_typing = rep[['patient_id', 'epilepsy_type', 'epilepsy_specific']].copy()
    pat_typing['patient_id'] = pd.to_numeric(pat_typing['patient_id'], errors='coerce')
    pat_typing = pat_typing.dropna(subset=['patient_id'])
    pat_typing['patient_id'] = pat_typing['patient_id'].astype(int).astype(str)
    pat_typing['etype_norm'] = pat_typing['epilepsy_type'].fillna('').str.lower().str.strip()
    pat_typing['spec_norm'] = pat_typing['epilepsy_specific'].fillna('').str.lower().str.strip()

    def _dedupe(grp):
        etypes = [e for e in grp['etype_norm'].unique() if e != '']
        specs = [s for s in grp['spec_norm'].unique() if s != '']
        return pd.Series({
            'etype_norm': etypes[0] if etypes else '',
            'spec_norm': specs[0] if specs else '',
        })

    dedup = (
        pat_typing.groupby('patient_id', group_keys=False)
        .apply(_dedupe)
        .reset_index()
    )
    dedup['EpiType3'] = ''
    dedup.loc[dedup['spec_norm'].str.contains('temporal'), 'EpiType3'] = 'Temporal'
    dedup.loc[dedup['spec_norm'].str.contains('frontal'), 'EpiType3'] = 'Frontal'
    dedup.loc[
        (dedup['EpiType3'] == '') & (dedup['etype_norm'] == 'general'),
        'EpiType3',
    ] = 'General'
    return set(dedup.loc[dedup['EpiType3'] == 'General', 'patient_id'])


def print_generalized_section(
    out: pd.DataFrame,
    all_gen_ids: set[str],
    slow_inv: pd.DataFrame,
    pat_ied_routine: pd.DataFrame,
    pat_ied_ann: pd.DataFrame,
    clin: pd.DataFrame,
    pure_sz: pd.DataFrame,
) -> dict:
    """Print availability counts restricted to Generalized patients."""
    gen_report_ids = all_gen_ids
    N_report = len(gen_report_ids)

    edf_ids = set(slow_inv.loc[slow_inv['has_light_preprocessed_edf'], 'patient_id'])
    morgoth_ids = set(slow_inv.loc[slow_inv['has_morgoth_slowing_output'], 'patient_id'])
    ied_ids = set(pat_ied_routine['patient_id'])
    ann_ids = set(
        pat_ied_ann.loc[pat_ied_ann['has_usable_ied_annotation'], 'patient_id']
    )
    spine_ids = set(clin.loc[clin['is_generalized'], 'patient_id'])
    constrained_label_ids = set(
        clin.loc[clin['has_sz_freq_label'] & clin['is_generalized'], 'patient_id']
    )
    pure_label_ids = set(
        pure_sz.loc[pure_sz['has_pure_sz_freq_label'], 'patient_id']
    )
    pure_gen_label_ids = pure_label_ids & gen_report_ids

    gen_spine = out[out['is_generalized']].copy()
    N_spine = len(gen_spine)
    N_morgoth_pool = len(morgoth_ids)

    print('\n' + '=' * 72)
    print('GENERALIZED PATIENTS ONLY')
    print('=' * 72)
    print(f'  Morgoth input : {LIGHT_PREPROCESSED_GENERAL}')
    print(f'  Morgoth output: {MORGOTH_SLOWING_GENERAL}')
    print()
    print_count('Typed General in DEID report (any session)', N_report)
    print_count('  → in light_preprocessed/General (Morgoth input EDFs)', len(gen_report_ids & edf_ids), N_report)
    print_count('  → have Morgoth pred_SLOWING_1sStep output', len(gen_report_ids & morgoth_ids), N_report)
    print_count('  → have routine IED rate (≤4h)', len(gen_report_ids & ied_ids), N_report)
    print_count('  → have usable IED annotation', len(gen_report_ids & ann_ids), N_report)
    print_count('  → in analysis spine (valid routine age/sex/spike)', len(gen_report_ids & spine_ids), N_report)
    print()
    print('2yr sz_freq label — PURE (visit arrays only; no visit-type / routine / age / spike gates):')
    print_count('  typed General with label', len(pure_gen_label_ids), N_report)
    print_count('  in Morgoth General pool', len(morgoth_ids & pure_gen_label_ids))
    print_count(
        '  Morgoth ∩ routine IED ∩ pure label',
        len(morgoth_ids & pure_gen_label_ids & ied_ids),
    )
    print()
    print('2yr sz_freq label — CONSTRAINED (current ML spine: allowable visits + valid routine EEG):')
    print_count('  typed General with label', len(constrained_label_ids), N_report)
    print_count('  in Morgoth General pool', len(morgoth_ids & constrained_label_ids))
    print_count(
        '  Morgoth ∩ routine IED ∩ constrained label  [Aim2 N]',
        len(morgoth_ids & constrained_label_ids & ied_ids),
    )
    print_count(
        '  extra in Morgoth pool from pure-only (not in Aim2 spine)',
        len((morgoth_ids & pure_gen_label_ids) - constrained_label_ids),
    )
    print()
    print('Morgoth General slowing pool (light_preprocessed/General):')
    print_count('  patients with input EDFs', len(edf_ids))
    print_count('  patients with Morgoth output CSVs', len(morgoth_ids))
    print_count('  input EDF but missing Morgoth output', len(edf_ids - morgoth_ids))
    print()
    print('Analysis spine (typed General + valid routine age/sex/spike):')
    print_count('  N', N_spine)
    print_count('  in Morgoth General slowing pool', int(gen_spine['has_morgoth_slowing_output'].sum()), N_spine)
    print_count('  have routine IED rate', int(gen_spine['has_usable_ied_rate_routine'].sum()), N_spine)
    print_count('  have pure sz_freq label', int(gen_spine['has_pure_sz_freq_label'].sum()), N_spine)
    print_count('  have constrained sz_freq label', int(gen_spine['has_sz_freq_label'].sum()), N_spine)
    print()
    print('Slowing + IED + PURE label (within Morgoth pool):')
    morgoth_gen = out[out['patient_id'].isin(morgoth_ids & gen_report_ids)]
    print_count('  Morgoth ∩ IED ∩ pure label', int(morgoth_gen['slow_ied_pure_label'].sum()), len(morgoth_gen))
    print_count(
        '  above + usable IED annotation',
        int((morgoth_gen['slow_ied_pure_label'] & morgoth_gen['has_usable_ied_annotation']).sum()),
        len(morgoth_gen),
    )
    print()
    print(f'Loss from Morgoth General pool (N={N_morgoth_pool}) — pure label:')
    print_count(
        '  missing pure 2yr label',
        int(N_morgoth_pool - len(morgoth_ids & pure_gen_label_ids)),
        N_morgoth_pool,
    )
    print_count(
        '  have Morgoth+IED but no pure label',
        int(len((morgoth_ids & ied_ids) - pure_gen_label_ids)),
    )
    print()
    print(f'Loss from Morgoth General pool — constrained label:')
    print_count(
        '  missing constrained 2yr label',
        int(N_morgoth_pool - len(morgoth_ids & constrained_label_ids)),
        N_morgoth_pool,
    )
    print_count('  typed General but no light_preprocessed/General EDF', len(gen_report_ids - edf_ids), N_report)

    return {
        'n_gen_report': N_report,
        'n_gen_report_light_preprocessed': len(gen_report_ids & edf_ids),
        'n_gen_report_morgoth_output': len(gen_report_ids & morgoth_ids),
        'n_gen_report_ied_rate': len(gen_report_ids & ied_ids),
        'n_gen_report_ied_annotation': len(gen_report_ids & ann_ids),
        'n_gen_report_spine': len(gen_report_ids & spine_ids),
        'n_gen_pure_label': len(pure_gen_label_ids),
        'n_gen_constrained_label': len(constrained_label_ids),
        'n_morgoth_pure_label': len(morgoth_ids & pure_gen_label_ids),
        'n_morgoth_constrained_label': len(morgoth_ids & constrained_label_ids),
        'n_morgoth_slow_ied_pure_label': len(morgoth_ids & pure_gen_label_ids & ied_ids),
        'n_morgoth_slow_ied_constrained_label': len(morgoth_ids & constrained_label_ids & ied_ids),
        'n_morgoth_pure_minus_constrained': len((morgoth_ids & pure_gen_label_ids) - constrained_label_ids),
        'n_morgoth_pool_patients': len(morgoth_ids),
        'n_gen_spine': N_spine,
        'n_gen_report_no_light_preprocessed': len(gen_report_ids - edf_ids),
    }


# ---------------------------------------------------------------------------
# Slowing — Generalized uses light_preprocessed/General + light_SLOWING_General
# (see morgoth/run_light_preprocessed.sh; not the Temporal/Frontal result dirs)
# ---------------------------------------------------------------------------

def _session_rows_from_paths(paths: list[str], source: str) -> pd.DataFrame:
    rows = []
    for p in paths:
        m = _RE_PID_SES.search(os.path.basename(p))
        if not m:
            continue
        rows.append({'patient_id': m.group(1), 'session': m.group(2), 'source': source})
    if not rows:
        return pd.DataFrame(columns=['patient_id', 'session', 'source'])
    return pd.DataFrame(rows)


def load_generalized_slowing_inventory() -> pd.DataFrame:
    """Patient-level Morgoth slowing availability for Generalized epilepsy.

    Master list = EDFs in light_preprocessed/General/ (Morgoth pipeline input).
    Processed  = segment CSVs in results/light_SLOWING_General/pred_SLOWING_1sStep/.
    """
    if not os.path.isdir(LIGHT_PREPROCESSED_GENERAL):
        raise FileNotFoundError(f'Missing Morgoth input dir: {LIGHT_PREPROCESSED_GENERAL}')

    edf_paths = sorted(glob.glob(os.path.join(LIGHT_PREPROCESSED_GENERAL, '*.edf')))
    edf_sess = _session_rows_from_paths(edf_paths, 'light_preprocessed')

    morgoth_paths: list[str] = []
    if os.path.isdir(MORGOTH_SLOWING_GENERAL):
        morgoth_paths = sorted(glob.glob(os.path.join(MORGOTH_SLOWING_GENERAL, '*.csv')))
    morgoth_sess = _session_rows_from_paths(morgoth_paths, 'morgoth_output')

    edf_pats = set(edf_sess['patient_id']) if len(edf_sess) else set()
    morgoth_pats = set(morgoth_sess['patient_id']) if len(morgoth_sess) else set()
    all_pats = sorted(edf_pats | morgoth_pats)

    edf_sess_ct = (
        edf_sess.groupby('patient_id')['session'].nunique().rename('n_light_preprocessed_sessions')
        if len(edf_sess) else pd.Series(dtype=int)
    )
    morgoth_sess_ct = (
        morgoth_sess.groupby('patient_id')['session'].nunique().rename('n_morgoth_sessions')
        if len(morgoth_sess) else pd.Series(dtype=int)
    )
    morgoth_file_ct = (
        morgoth_sess.groupby('patient_id').size().rename('n_morgoth_segment_csvs')
        if len(morgoth_sess) else pd.Series(dtype=int)
    )

    out = pd.DataFrame({'patient_id': all_pats})
    out = out.merge(edf_sess_ct.reset_index(), on='patient_id', how='left')
    out = out.merge(morgoth_sess_ct.reset_index(), on='patient_id', how='left')
    out = out.merge(morgoth_file_ct.reset_index(), on='patient_id', how='left')
    for col in ('n_light_preprocessed_sessions', 'n_morgoth_sessions', 'n_morgoth_segment_csvs'):
        out[col] = out[col].fillna(0).astype(int)

    out['has_light_preprocessed_edf'] = out['patient_id'].isin(edf_pats)
    out['has_morgoth_slowing_output'] = out['patient_id'].isin(morgoth_pats)
    # Back-compat alias used in downstream flags
    out['has_usable_slowing'] = out['has_morgoth_slowing_output']
    return out


# ---------------------------------------------------------------------------
# Continuous IEDs
# ---------------------------------------------------------------------------

def load_continuous_ied_sessions() -> pd.DataFrame:
    sc = pd.read_csv(gml.SPIKE_CSV)
    sc['count_0_46'] = pd.to_numeric(sc['count_0_46'], errors='coerce')
    sc['Duration_sec'] = pd.to_numeric(sc['Duration_sec'], errors='coerce')
    sc = sc.rename(columns={'Patient': 'patient_id', 'Session': 'session_number'})
    sc['patient_id'] = pd.to_numeric(sc['patient_id'], errors='coerce')
    sc['session_number'] = pd.to_numeric(sc['session_number'], errors='coerce')
    sc = sc.dropna(subset=['patient_id', 'session_number']).copy()
    sc['patient_id'] = sc['patient_id'].astype(int).astype(str)
    sc['session_number'] = sc['session_number'].astype(int).astype(str)
    sc['spike_rate_per_hour'] = sc['count_0_46'] / sc['Duration_sec'] * 3600
    sc['is_routine_leq_4h'] = sc['Duration_sec'].notna() & (sc['Duration_sec'] <= 4 * 3600)
    sc['has_usable_ied_rate'] = np.isfinite(sc['spike_rate_per_hour'].to_numpy(float))
    return sc


def patient_agg_ied_rate(sc: pd.DataFrame, routine_only: bool) -> pd.DataFrame:
    sub = sc.loc[sc['has_usable_ied_rate']]
    if routine_only:
        sub = sub.loc[sub['is_routine_leq_4h']]
    g = sub.groupby('patient_id', as_index=False).agg(
        n_ied_rate_sessions=('session_number', 'nunique'),
        spike_rate_per_hour=('spike_rate_per_hour', 'mean'),
    )
    g['has_usable_ied_rate'] = True
    return g


# ---------------------------------------------------------------------------
# Binary IED annotation (all outpatient+routine report patients)
# ---------------------------------------------------------------------------

def load_ied_annotation_all_patients() -> pd.DataFrame:
    """Patient-level present/absent/unknown for all outpatient routine DEID rows."""
    if not a2.DEID_REPORT_CSV.is_file():
        raise FileNotFoundError(f'DEID report not found: {a2.DEID_REPORT_CSV}')
    print(f'  report: {a2.DEID_REPORT_CSV}')
    rep = pd.read_csv(a2.DEID_REPORT_CSV, low_memory=False)
    rep = a2._filter_outpatient_routine_report(rep)
    sess = a2.resolve_reported_spike_status(rep)
    pat = a2.patient_reported_spike_status(sess)
    pat['patient_id'] = pat['patient_id'].astype(str)
    pat['has_usable_ied_annotation'] = pat['ied_annotation'].isin(['present', 'absent'])
    n_pre = int((pat['ied_annotation'] == 'present').sum())
    n_abs = int((pat['ied_annotation'] == 'absent').sum())
    n_unk = int((pat['ied_annotation'] == 'unknown').sum())
    print(f'  present={n_pre}  absent={n_abs}  unknown={n_unk}')
    return pat


# ---------------------------------------------------------------------------
# Pure sz_freq labels (visit arrays only — no extra gates)
# ---------------------------------------------------------------------------

def load_pure_sz_freq_patient_labels() -> pd.DataFrame:
    """Patient MeanSzFreq / sz_freq_class from all DEID report EEG sessions.

    Uses the same 2yr post-EEG window algorithm as generalized_ml, but:
      - no ALLOWABLE_VISITS filter
      - no outpatient / routine EEG filter
      - no age / sex / spike_rate validity filter
      - patient label = mean of session mean_sz_freq_2yr over ANY session with data
    """
    print('  (all report sessions; ~30s)')
    rep = pd.read_csv(gml.REPORT_CSV, low_memory=False)
    rep['patient_id'] = pd.to_numeric(rep['patient_id'], errors='coerce')
    rep['session_number'] = pd.to_numeric(rep['session_number'], errors='coerce')
    rep = rep.dropna(subset=['patient_id', 'session_number']).copy()
    rep['patient_id'] = rep['patient_id'].astype(int).astype(str)

    need = [
        'patient_id', 'session_number', 'start_time_deid',
        'visit_dates_deid', 'sz_freqs', 'visit_hasSz',
    ]
    sess = rep[need].drop_duplicates(subset=['patient_id', 'session_number'])
    session_sz = gml.add_session_mean_sz_freq_2yr(sess)
    n_sess = int(session_sz['mean_sz_freq_2yr'].notna().sum())
    print(f'  sessions with finite mean_sz_freq_2yr: {n_sess}/{len(sess)}')

    patient_sz = gml.build_patient_sz_from_session_means(
        session_sz, poor_cutoff=gml.POOR_CUTOFF_PER_MONTH,
    )
    patient_sz['patient_id'] = patient_sz['patient_id'].astype(str)
    patient_sz = patient_sz.rename(columns={
        'mean_sz_freq': 'pure_mean_sz_freq',
        'sz_freq_class': 'pure_sz_freq_class',
    })
    patient_sz['has_pure_sz_freq_label'] = patient_sz['pure_mean_sz_freq'].notna()
    print(f'  patients with pure label: {int(patient_sz["has_pure_sz_freq_label"].sum())}')
    return patient_sz


# ---------------------------------------------------------------------------
# Clinical labels/types WITHOUT ADR / background_features
# ---------------------------------------------------------------------------

def load_clinical_without_adr() -> pd.DataFrame:
    """Routine outpatient typed cohort + 2yr sz_freq labels; no BG merge.

    Validity for labels uses age/sex/spike_rate only (INDEX_EEG_REQUIRE_COLS),
    not ADR/connectivity and not require_slowing.
    """
    sc = pd.read_csv(gml.SPIKE_CSV)
    rep = pd.read_csv(gml.REPORT_CSV, low_memory=False)

    sc['count_0_46'] = pd.to_numeric(sc['count_0_46'], errors='coerce')
    sc['Duration_sec'] = pd.to_numeric(sc['Duration_sec'], errors='coerce')
    sc['spike_rate_per_hour'] = sc['count_0_46'] / sc['Duration_sec'] * 3600
    sc_routine = sc[sc['Duration_sec'].notna() & (sc['Duration_sec'] <= 4 * 3600)].copy()
    sc_routine = sc_routine.rename(columns={'Patient': 'patient_id', 'Session': 'session_number'})
    sc_routine[['patient_id', 'session_number']] = sc_routine[['patient_id', 'session_number']].astype(int)

    acq = rep['acquired_on'].fillna('').str.lower().str.strip()
    cls = rep['report_PATIENT_CLASS'].fillna('').str.lower().str.strip()
    jay = rep['jay_in_or_out'].fillna('').str.lower().str.strip()
    is_outpt = (
        acq.str.contains('spe') | acq.str.contains('radnor')
        | (cls == 'outpatient') | (jay == 'out')
    )
    outpt_keys = rep.loc[is_outpt, ['patient_id', 'session_number']].drop_duplicates().dropna().astype(int)

    rep = rep.copy()
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
    print(f'  [Typing] Excluded: {int(is_nesd.sum())} NESD, {int(is_bad_type.sum())} bad/unknown')

    ALL_EPI = ['General', 'Temporal', 'Frontal']
    merged = merged[merged['patient_id'].isin(valid_pats)].copy()
    merged = merged.drop(columns=['EpiType3'], errors='ignore')
    merged = merged.merge(pat_type_dedup[['patient_id', 'EpiType3']], on='patient_id', how='left')
    cohort = merged[merged['EpiType3'].isin(ALL_EPI)].copy()
    print(
        f'  Typed outpatient+routine sessions: {len(cohort):,}  '
        f'patients={cohort["patient_id"].nunique():,}'
    )

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

    def compute_age(row):
        try:
            return (
                pd.to_datetime(str(row['start_time_deid']).strip())
                - pd.to_datetime(str(row['deid_birth_date']).strip())
            ).days / 365.25
        except (TypeError, ValueError):
            return np.nan

    cohort['age_at_eeg'] = cohort.apply(compute_age, axis=1)
    cohort['gender'] = cohort['nlp_gender'].map({'F': 0.0, 'M': 1.0})

    session_sz = gml.add_session_mean_sz_freq_2yr(
        cohort.drop_duplicates(subset=['patient_id', 'session_number'])
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

    _sz = session_sz[['patient_id', 'session_number', 'mean_sz_freq_2yr']].copy()
    _sz['patient_id'] = _sz['patient_id'].astype(str)
    _sz = _sz.rename(columns={'session_number': 'session'})
    _sz['session'] = _sz['session'].astype(str)
    dataset = dataset.merge(_sz, on=['patient_id', 'session'], how='left')

    # Valid for label: age/sex/spike only — NOT ADR, NOT slowing
    valid = gml.filter_valid_eeg_sessions(dataset, require_slowing=False)
    patient_sz = gml.build_patient_sz_from_session_means(
        valid, poor_cutoff=gml.POOR_CUTOFF_PER_MONTH
    )
    patient_sz['patient_id'] = patient_sz['patient_id'].astype(str)

    epi_map = pat_type_dedup[['patient_id', 'EpiType3']].copy()
    epi_map['patient_id'] = epi_map['patient_id'].astype(str)
    epi_map['epilepsy_type'] = epi_map['EpiType3'].map(
        lambda t: 'Focal' if t in ('Temporal', 'Frontal')
        else ('Generalized' if t == 'General' else np.nan)
    )

    # All typed patients with ≥1 valid routine session (even without label)
    typed = (
        valid.groupby('patient_id', as_index=False)
        .agg(n_valid_sessions=('session', 'nunique'), EpiType3=('EpiType3', 'first'))
    )
    typed = typed.merge(epi_map[['patient_id', 'epilepsy_type']], on='patient_id', how='left')
    typed = typed.merge(patient_sz, on='patient_id', how='left')
    typed['has_sz_freq_label'] = typed['sz_freq_class'].notna()
    typed['is_generalized'] = typed['epilepsy_type'].astype(str).eq('Generalized')
    typed['is_focal'] = typed['epilepsy_type'].astype(str).eq('Focal')
    print(
        f'  Valid routine sessions (age/sex/spike): {len(valid):,}  '
        f'patients={typed["patient_id"].nunique():,}  '
        f'with label={int(typed["has_sz_freq_label"].sum()):,}  '
        f'Gen={int(typed["is_generalized"].sum()):,}'
    )
    return typed


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 72)
    print('Slowing + IED availability inventory (no ADR/connectivity required)')
    print('=' * 72)

    print('\n[1/4] Morgoth General slowing (light_preprocessed/General → light_SLOWING_General)...')
    slow_inv = load_generalized_slowing_inventory()
    n_edf = int(slow_inv['has_light_preprocessed_edf'].sum())
    n_out = int(slow_inv['has_morgoth_slowing_output'].sum())
    n_seg = int(slow_inv['n_morgoth_segment_csvs'].sum())
    print(f'  input EDFs     : {len(glob.glob(os.path.join(LIGHT_PREPROCESSED_GENERAL, "*.edf"))):,} segments, {n_edf:,} patients')
    print(f'  Morgoth output : {n_seg:,} segment CSVs, {n_out:,} patients')
    print(f'  input dir      : {LIGHT_PREPROCESSED_GENERAL}')
    print(f'  output dir     : {MORGOTH_SLOWING_GENERAL}')
    if n_edf != n_out:
        print(f'  [warn] {n_edf - n_out} patients have input EDF but no Morgoth CSV')

    print('\n[2/4] Continuous IEDs (spike_counts)...')
    sc = load_continuous_ied_sessions()
    print(f'  spike_counts sessions: {len(sc):,}  patients={_nuniq(sc["patient_id"]):,}')
    print(
        f'  routine ≤4h sessions: {int(sc["is_routine_leq_4h"].sum()):,}  '
        f'patients={_nuniq(sc.loc[sc["is_routine_leq_4h"], "patient_id"]):,}'
    )
    pat_ied_all = patient_agg_ied_rate(sc, routine_only=False)
    pat_ied_routine = patient_agg_ied_rate(sc, routine_only=True)
    print(f'  patients with usable IED rate (any duration): {len(pat_ied_all):,}')
    print(f'  patients with usable IED rate (routine ≤4h):  {len(pat_ied_routine):,}')

    print('\n[3/4] Binary IED annotation (DEID outpatient+routine)...')
    pat_ied_ann = load_ied_annotation_all_patients()
    print(f'  patients with usable annotation (present|absent): '
          f'{int(pat_ied_ann["has_usable_ied_annotation"].sum()):,}')

    print('\n[4/5] Pure 2yr sz_freq labels (visit arrays only)...')
    pure_sz = load_pure_sz_freq_patient_labels()

    print('\n[5/5] Constrained labels + types (ML spine, no ADR merge)...')
    clin = load_clinical_without_adr()

    # Outer join inventory
    base = pd.DataFrame({
        'patient_id': sorted(
            set(slow_inv['patient_id'])
            | set(pat_ied_routine['patient_id'])
            | set(pat_ied_ann['patient_id'])
            | set(clin['patient_id'])
        )
    })
    out = (
        base
        .merge(slow_inv, on='patient_id', how='left')
        .merge(pat_ied_routine.rename(columns={
            'n_ied_rate_sessions': 'n_ied_rate_sessions_routine',
            'spike_rate_per_hour': 'spike_rate_per_hour_routine',
            'has_usable_ied_rate': 'has_usable_ied_rate_routine',
        }), on='patient_id', how='left')
        .merge(pat_ied_all.rename(columns={
            'n_ied_rate_sessions': 'n_ied_rate_sessions_any',
            'spike_rate_per_hour': 'spike_rate_per_hour_any',
            'has_usable_ied_rate': 'has_usable_ied_rate_any',
        }), on='patient_id', how='left')
        .merge(pat_ied_ann, on='patient_id', how='left')
        .merge(
            pure_sz[['patient_id', 'pure_mean_sz_freq', 'pure_sz_freq_class', 'has_pure_sz_freq_label']],
            on='patient_id', how='left',
        )
        .merge(
            clin[['patient_id', 'epilepsy_type', 'EpiType3', 'sz_freq_class',
                  'mean_sz_freq', 'has_sz_freq_label', 'is_generalized', 'is_focal',
                  'n_valid_sessions']],
            on='patient_id', how='left',
        )
    )
    for col in (
        'has_light_preprocessed_edf', 'has_morgoth_slowing_output', 'has_usable_slowing',
        'has_usable_ied_rate_routine', 'has_usable_ied_rate_any',
        'has_usable_ied_annotation', 'has_pure_sz_freq_label',
        'has_sz_freq_label', 'is_generalized', 'is_focal',
    ):
        out[col] = out[col].fillna(False).astype(bool)
    for col in ('n_light_preprocessed_sessions', 'n_morgoth_sessions', 'n_morgoth_segment_csvs'):
        out[col] = out[col].fillna(0).astype(int)

    out['has_slowing_and_ied_rate'] = out['has_morgoth_slowing_output'] & out['has_usable_ied_rate_routine']
    out['has_slowing_and_ied_annot'] = out['has_morgoth_slowing_output'] & out['has_usable_ied_annotation']
    out['has_slowing_and_both_ied'] = (
        out['has_morgoth_slowing_output']
        & out['has_usable_ied_rate_routine']
        & out['has_usable_ied_annotation']
    )
    out['slow_ied_label'] = (
        out['has_morgoth_slowing_output']
        & out['has_usable_ied_rate_routine']
        & out['has_sz_freq_label']
    )
    out['slow_ied_pure_label'] = (
        out['has_morgoth_slowing_output']
        & out['has_usable_ied_rate_routine']
        & out['has_pure_sz_freq_label']
    )
    out['slow_ied_pure_label_gen'] = out['slow_ied_pure_label'] & out['is_generalized']
    out['slow_ied_label_gen'] = out['slow_ied_label'] & out['is_generalized']
    out['slow_ied_label_focal'] = out['slow_ied_label'] & out['is_focal']

    all_gen_ids = load_all_typed_generalized_patient_ids()
    gen_summary = print_generalized_section(
        out, all_gen_ids, slow_inv, pat_ied_routine, pat_ied_ann, clin, pure_sz,
    )

    N = len(out)
    print('\n' + '=' * 72)
    print('ALL PATIENTS (reference — no ADR / connectivity gate)')
    print('=' * 72)
    print_count('Universe (slowing ∪ routine-IED ∪ annot ∪ clinical)', N)
    print()
    print('Markers alone:')
    print_count('Morgoth General slowing output', int(out['has_morgoth_slowing_output'].sum()), N)
    print_count('usable IED rate (routine ≤4h)', int(out['has_usable_ied_rate_routine'].sum()), N)
    print_count('usable IED rate (any duration)', int(out['has_usable_ied_rate_any'].sum()), N)
    print_count('usable IED annotation (present|absent)', int(out['has_usable_ied_annotation'].sum()), N)
    print()
    print('Slowing ∩ IED:')
    print_count('slowing ∩ IED rate (routine)', int(out['has_slowing_and_ied_rate'].sum()), N)
    print_count('slowing ∩ IED annotation', int(out['has_slowing_and_ied_annot'].sum()), N)
    print_count('slowing ∩ IED rate ∩ IED annotation', int(out['has_slowing_and_both_ied'].sum()), N)
    print()
    print('With clinical label / type (age/sex/spike valid; no ADR):')
    print_count('has sz_freq_class label', int(out['has_sz_freq_label'].sum()), N)
    print_count('Generalized (typed valid routine)', int(out['is_generalized'].sum()), N)
    print_count('Focal (typed valid routine)', int(out['is_focal'].sum()), N)
    print()
    print('Analysis-ready for slowing+IED models:')
    print_count('slowing ∩ IED-rate ∩ label (any type)', int(out['slow_ied_label'].sum()), N)
    print_count('  → Generalized', int(out['slow_ied_label_gen'].sum()), N)
    print_count('  → Focal', int(out['slow_ied_label_focal'].sum()), N)
    print_count(
        'Gen + slowing ∩ IED-rate ∩ label ∩ usable annotation',
        int((out['slow_ied_label_gen'] & out['has_usable_ied_annotation']).sum()),
        N,
    )

    aim2_cohort = _SCRIPT_DIR / 'outputs' / 'General' / 'aim2_generalized_final' / 'analysis_cohort.csv'
    if aim2_cohort.is_file():
        a2df = pd.read_csv(aim2_cohort)
        a2_pids = set(a2df['patient_id'].astype(str))
        print()
        print('vs current Aim2 analysis_cohort.csv:')
        print_count('Aim2 N', len(a2_pids))
        in_both = out.loc[out['slow_ied_label_gen'], 'patient_id'].isin(a2_pids).sum()
        print_count('Aim2 also in Morgoth∩IED∩constrained label', int(in_both))
        pure_aim2 = int(out.loc[out['slow_ied_pure_label_gen'], 'patient_id'].nunique())
        print_count('Morgoth∩IED∩pure label (Generalized)', pure_aim2)
        extra_pure = int((out['slow_ied_pure_label_gen'] & ~out['patient_id'].isin(a2_pids)).sum())
        print_count('EXTRA from pure label beyond Aim2 N=490', extra_pure)

    out_path = OUT_DIR / 'patient_slowing_ied_flags.csv'
    out.to_csv(out_path, index=False)
    gen_out = out[out['is_generalized']].copy()
    gen_path = OUT_DIR / 'generalized_slowing_ied_flags.csv'
    gen_out.to_csv(gen_path, index=False)

    summary = {
        'all_patients': {
            'n_universe': N,
            'n_morgoth_slowing': int(out['has_morgoth_slowing_output'].sum()),
            'n_ied_rate_routine': int(out['has_usable_ied_rate_routine'].sum()),
            'n_ied_annotation': int(out['has_usable_ied_annotation'].sum()),
            'n_slowing_and_ied_rate': int(out['has_slowing_and_ied_rate'].sum()),
            'n_slow_ied_label': int(out['slow_ied_label'].sum()),
            'n_slow_ied_label_gen': int(out['slow_ied_label_gen'].sum()),
            'n_slow_ied_label_focal': int(out['slow_ied_label_focal'].sum()),
        },
        'generalized_only': gen_summary,
    }
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved all patients → {out_path}')
    print(f'Saved Generalized only → {gen_path}')
    print(f'Summary → {OUT_DIR / "summary.json"}')
    print('Done.')


if __name__ == '__main__':
    main()
