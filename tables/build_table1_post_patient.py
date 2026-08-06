#!/usr/bin/env python3
"""Table 1. Post-Patient Dataset Information (Generalized + Focal).

Cohort counts and EEG duration are restricted to patient-sessions with clips
in filtered_data/ (sum of clip spans per session from filename timestamps).
Bootstrap 95% CI: mean clip duration (minutes) across sessions (n_boot=5000).
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/mnt/sauce/littlab/users/cho6/seizure_frequency_preprocessing')
FILTERED_ROOT = ROOT / 'filtered_data'
REPORT_CSV = ROOT / 'Routineeegpec-Deidreport_DATA_LABELS_2025-12-12_1339 (2).csv'
OUT_PATH = Path(__file__).resolve().parent / 'outputs' / 'General' / 'table1_post_patient.txt'

N_BOOT = 5000
RNG = np.random.default_rng(42)
_CLIP_PAT = re.compile(r'sub-Penn(\d+)_ses-(\d+)_clip\d+_t(\d+)s-(\d+)s\.edf$')

NESD = 'non-epileptic seizure disorder'
BAD_TYPES = {'uncertain if epilepsy', 'unknown or mrn not found', ''}


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = N_BOOT, alpha: float = 0.05):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    boots = [float(np.mean(RNG.choice(x, size=len(x), replace=True))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(x)), float(lo), float(hi)


def load_patient_typing() -> pd.DataFrame:
    rep = pd.read_csv(REPORT_CSV)
    pt = rep[['patient_id', 'epilepsy_type', 'epilepsy_specific']].copy()
    pt['patient_id'] = pd.to_numeric(pt['patient_id'], errors='coerce').dropna().astype(int)
    pt['etype_norm'] = pt['epilepsy_type'].fillna('').str.lower().str.strip()
    pt['spec_norm'] = pt['epilepsy_specific'].fillna('').str.lower().str.strip()

    def _dedupe(g):
        et = [e for e in g['etype_norm'].unique() if e]
        sp = [s for s in g['spec_norm'].unique() if s]
        return pd.Series({
            'etype_norm': et[0] if et else '',
            'spec_norm': sp[0] if sp else '',
        })

    pat = pt.groupby('patient_id', group_keys=False).apply(_dedupe, include_groups=False).reset_index()
    pat['EpiType3'] = ''
    pat.loc[pat['spec_norm'].str.contains('temporal'), 'EpiType3'] = 'Temporal'
    pat.loc[pat['spec_norm'].str.contains('frontal'), 'EpiType3'] = 'Frontal'
    pat.loc[(pat['EpiType3'] == '') & (pat['etype_norm'] == 'general'), 'EpiType3'] = 'General'
    keep = ~(pat['etype_norm'].eq(NESD) | pat['etype_norm'].isin(BAD_TYPES))
    return pat.loc[keep, ['patient_id', 'EpiType3']]


def load_filtered_session_table() -> pd.DataFrame:
    """One row per patient-session in filtered_data with total clip duration (min)."""
    clip_sec: dict[tuple[int, int], float] = {}
    for subtype in ('General', 'Temporal', 'Frontal'):
        folder = FILTERED_ROOT / subtype
        if not folder.is_dir():
            continue
        for f in glob.glob(str(folder / '*.edf')):
            m = _CLIP_PAT.search(os.path.basename(f))
            if not m:
                continue
            key = (int(m.group(1)), int(m.group(2)))
            clip_sec[key] = clip_sec.get(key, 0.0) + float(int(m.group(4)) - int(m.group(3)))

    rows = [
        {'patient_id': pid, 'session_number': ses, 'duration_min': sec / 60.0}
        for (pid, ses), sec in clip_sec.items()
    ]
    sessions = pd.DataFrame(rows)
    typing = load_patient_typing()
    sessions = sessions.merge(typing, on='patient_id', how='inner')

    sessions['cohort'] = pd.Series(pd.NA, index=sessions.index, dtype='string')
    sessions.loc[sessions['EpiType3'] == 'General', 'cohort'] = 'Generalized'
    sessions.loc[sessions['EpiType3'].isin(['Temporal', 'Frontal']), 'cohort'] = 'Focal'
    return sessions.dropna(subset=['cohort']).reset_index(drop=True)


def build_table() -> pd.DataFrame:
    sessions = load_filtered_session_table()
    rows = []
    for cohort in ('Generalized', 'Focal'):
        sub = sessions[sessions['cohort'] == cohort]
        _, lo, hi = bootstrap_mean_ci(sub['duration_min'].values)
        rows.append({
            'Cohort': cohort,
            'Patients': int(sub['patient_id'].nunique()),
            'Patient-Sessions': len(sub),
            '95% CI (minutes)': f'[{lo:.2f}, {hi:.2f}]',
        })
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> str:
    cols = ['Cohort', 'Patients', 'Patient-Sessions', '95% CI (minutes)']
    widths = [max(len(c), df[c].astype(str).str.len().max()) for c in cols]

    def fmt_row(values):
        return '  '.join(str(v).ljust(w) for v, w in zip(values, widths))

    lines = [
        'Table 1. Post-Patient Dataset Information.',
        '(Patient-sessions and durations from filtered_data/ clip EDFs only.)',
        '',
        fmt_row(cols),
        fmt_row(['-' * w for w in widths]),
    ]
    for _, r in df.iterrows():
        lines.append(fmt_row([r[c] for c in cols]))
    return '\n'.join(lines) + '\n'


def main() -> None:
    table = build_table()
    text = format_table(table)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding='utf-8')
    print(text, end='')
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
