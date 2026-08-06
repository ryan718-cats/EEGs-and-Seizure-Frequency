#!/usr/bin/env python3
"""IED Youden sweep using spike rate with light-preprocessed segment duration.

Rate = count_0_46 / light_duration_sec * 3600
Score for Youden = log10(rate + 0.1)

Labels: DEID report IED present/absent (same as treated-duration sweep).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

HERE = Path(__file__).resolve().parent
ML_DIR = HERE / 'ml_analysis'
ROOT = HERE.parent
SPIKE_CSV = ROOT / 'spike_counts.csv'
LIGHT_DURATION_CSV = ROOT / 'light_preprocessed_duration.csv'
PAPER_OUT = HERE / 'output'

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import aim2_generalized_ml as a2  # noqa: E402

OUT_DIR = ML_DIR / 'outputs' / 'ied_log_spike_threshold_light_duration'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_OUT.mkdir(parents=True, exist_ok=True)

C_ACC = '#006EFF'
C_SENS = '#FF0000'
C_SPEC = '#00A218'


def load_session_log_spike_rate_light() -> pd.DataFrame:
    """count_0_46 / light_duration_sec * 3600, then log10(rate + 0.1).

    Cohort gate still uses raw spike_counts Duration_sec ≤ 4h (routine filter).
    """
    sc = pd.read_csv(SPIKE_CSV)
    sc['count_0_46'] = pd.to_numeric(sc['count_0_46'], errors='coerce')
    sc['Duration_sec'] = pd.to_numeric(sc['Duration_sec'], errors='coerce')
    sc = sc[sc['Duration_sec'].notna() & (sc['Duration_sec'] <= 4 * 3600)].copy()
    sc = sc.rename(columns={'Patient': 'patient_id', 'Session': 'session_number'})
    sc['patient_id'] = a2._norm_id(sc['patient_id'])
    sc['session_number'] = a2._norm_id(sc['session_number'])

    light = pd.read_csv(LIGHT_DURATION_CSV)
    light = light.rename(columns={'Patient': 'patient_id', 'Session': 'session_number'})
    light['patient_id'] = a2._norm_id(light['patient_id'])
    light['session_number'] = a2._norm_id(light['session_number'])

    m = sc.merge(light, on=['patient_id', 'session_number'], how='inner')
    m = m.dropna(subset=['count_0_46', 'light_duration_sec'])
    m = m[m['light_duration_sec'] > 0].copy()

    m['spike_rate_per_hour_raw'] = m['count_0_46'] / m['Duration_sec'] * 3600
    m['spike_rate_per_hour'] = m['count_0_46'] / m['light_duration_sec'] * 3600
    m['spike_rate_log'] = np.log10(m['spike_rate_per_hour'].astype(float) + 0.1)
    m['spike_rate_log_raw'] = np.log10(m['spike_rate_per_hour_raw'].astype(float) + 0.1)

    return m[[
        'patient_id', 'session_number',
        'count_0_46', 'Duration_sec', 'light_duration_sec',
        'spike_rate_per_hour', 'spike_rate_log',
        'spike_rate_per_hour_raw', 'spike_rate_log_raw',
    ]].drop_duplicates(subset=['patient_id', 'session_number'], keep='first')


def load_session_ied_labels() -> pd.DataFrame:
    if not a2.DEID_REPORT_CSV.is_file():
        raise FileNotFoundError(f'DEID report not found: {a2.DEID_REPORT_CSV}')
    rep = pd.read_csv(a2.DEID_REPORT_CSV, low_memory=False)
    rep = a2._filter_outpatient_routine_report(rep)
    sess = a2.resolve_reported_spike_status(rep)
    sess = sess[sess['report_status'].isin(['present', 'absent'])].copy()
    sess['label'] = (sess['report_status'] == 'present').astype(int)
    return sess[['patient_id', 'session_number', 'report_status', 'label']]


def main() -> None:
    print('Loading light-duration spike rates...')
    spikes = load_session_log_spike_rate_light()
    print(f'  sessions with light segments + routine spike_counts: {len(spikes)}')
    print(
        f'  light_duration_sec: median={spikes["light_duration_sec"].median():.1f}, '
        f'mean={spikes["light_duration_sec"].mean():.1f}'
    )
    print(
        f'  raw Duration_sec:   median={spikes["Duration_sec"].median():.1f}, '
        f'mean={spikes["Duration_sec"].mean():.1f}'
    )
    ratio = spikes['light_duration_sec'] / spikes['Duration_sec']
    print(f'  light/raw duration: median={ratio.median():.3f}, mean={ratio.mean():.3f}')

    print('Loading IED annotations...')
    labels = load_session_ied_labels()
    print(f'  labelled sessions (present|absent): {len(labels)}')
    print(labels['report_status'].value_counts().to_string())

    merged = labels.merge(spikes, on=['patient_id', 'session_number'], how='inner')
    print(f'\nMatched patient-sessions: {len(merged)}')
    print(merged['report_status'].value_counts().to_string())

    y_true = merged['label'].to_numpy(dtype=int)
    y_score = merged['spike_rate_log'].to_numpy(dtype=float)

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    best_thr = float(thresholds[best_idx])
    best_j = float(j[best_idx])
    best_rate = float(10 ** best_thr - 0.1)

    print('\n=== Light-preprocessed spike rate Youden ===')
    print(f'AUROC          : {auc:.3f}')
    print(f"Best threshold : {best_thr:.4f}  (Youden's J = {best_j:.3f})")
    print(f'  (= log10(spike_rate_per_hour + 0.1))')
    print(f'  equivalent rate/hour ≈ {best_rate:.4f}')

    yp_best = (y_score >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yp_best).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    acc = accuracy_score(y_true, yp_best)
    print('At Youden best:')
    print(f'  Accuracy    : {acc:.3f}')
    print(f'  Sensitivity : {sens:.3f}')
    print(f'  Specificity : {spec:.3f}')
    print(f'  TP={tp} FP={fp} TN={tn} FN={fn}')

    y_raw = merged['spike_rate_log_raw'].to_numpy(dtype=float)
    auc_raw = roc_auc_score(y_true, y_raw)
    fpr_r, tpr_r, thr_r = roc_curve(y_true, y_raw)
    j_r = tpr_r - fpr_r
    bi_r = int(np.argmax(j_r))
    print('\n=== Same matched set, RAW Duration_sec rate (reference) ===')
    print(f'AUROC          : {auc_raw:.3f}')
    print(f"Best threshold : {float(thr_r[bi_r]):.4f}  (Youden's J = {float(j_r[bi_r]):.3f})")

    lo = float(np.nanpercentile(y_score, 1))
    hi = float(np.nanpercentile(y_score, 99))
    thresholds_sweep = np.round(np.linspace(lo, hi, 20), 2)

    rows_out = []
    for thr in thresholds_sweep:
        yp = (y_score >= thr).astype(int)
        _tn, _fp, _fn, _tp = confusion_matrix(y_true, yp).ravel()
        rows_out.append({
            'threshold': float(thr),
            'accuracy': accuracy_score(y_true, yp),
            'sensitivity': _tp / (_tp + _fn) if (_tp + _fn) else np.nan,
            'specificity': _tn / (_tn + _fp) if (_tn + _fp) else np.nan,
        })
    df_sweep = pd.DataFrame(rows_out).round(3)

    # --- ROC ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#1f4e79', linewidth=2.0, label=f'AUC = {auc:.3f}')
    ax.scatter(
        fpr[best_idx], tpr[best_idx], color='red', zorder=5, s=55,
        label=f"Youden thr = {best_thr:.3f}",
    )
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.0)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate (sensitivity)')
    ax.set_title('Light-preprocessed duration:\nlog10(spike rate + 0.1) vs report IED')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    roc_name = 'ied_log_spike_roc_light_duration.png'
    roc_path = OUT_DIR / roc_name
    fig.savefig(roc_path, dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(PAPER_OUT / roc_name, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # --- Metrics vs threshold ---
    span = max(0.01, (hi - lo) * 0.02)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvspan(best_thr - span, best_thr + span, color='#FAF6E0', zorder=0)
    ax.plot(df_sweep['threshold'], df_sweep['accuracy'], marker='o', color=C_ACC)
    ax.plot(df_sweep['threshold'], df_sweep['sensitivity'], marker='s', color=C_SENS)
    ax.plot(
        df_sweep['threshold'], df_sweep['specificity'],
        marker='^', color=C_SPEC, linestyle='--', alpha=0.85,
    )
    ax.axvline(best_thr, color='red', linestyle=':', linewidth=1.5)
    ax.set_xlabel('Threshold (log10[spike rate + 0.1], light-preprocessed duration)', fontsize=15)
    ax.set_ylabel('Score', fontsize=15)
    ax.set_xticks(thresholds_sweep)
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds_sweep], rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    sweep_name = 'ied_log_spike_threshold_sweep_light_duration.png'
    sweep_path = OUT_DIR / sweep_name
    fig.savefig(sweep_path, dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(PAPER_OUT / sweep_name, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    summary = pd.DataFrame([{
        'duration_basis': 'light_preprocessed_segment_spans',
        'n_matched_sessions': len(merged),
        'auroc': auc,
        'youden_threshold_log10': best_thr,
        'youden_J': best_j,
        'equiv_spike_rate_per_hour': best_rate,
        'accuracy': acc,
        'sensitivity': sens,
        'specificity': spec,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'raw_duration_youden_threshold_log10': float(thr_r[bi_r]),
        'raw_duration_youden_J': float(j_r[bi_r]),
        'raw_duration_auroc': auc_raw,
    }])
    summary_path = OUT_DIR / 'ied_youden_light_duration_summary.csv'
    summary.to_csv(summary_path, index=False)
    df_sweep.to_csv(OUT_DIR / 'ied_log_spike_threshold_sweep_light_duration.csv', index=False)
    merged.to_csv(OUT_DIR / 'ied_log_spike_matched_sessions_light_duration.csv', index=False)

    print(f'\nSaved {roc_path}')
    print(f'Saved {sweep_path}')
    print(f'Saved {PAPER_OUT / roc_name}')
    print(f'Saved {PAPER_OUT / sweep_name}')
    print(f'Saved {summary_path}')


if __name__ == '__main__':
    main()
