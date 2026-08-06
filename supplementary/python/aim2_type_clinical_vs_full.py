#!/usr/bin/env python3
"""Aim2-style clinical vs full (IED+slowing), stratified by epilepsy type.

Runs 1000 MC 80/20 for:
  - Generalized: Model1_clinical + Model4_IED_slow (Youden)
  - Focal:       Model1_clinical + Model4_IED_slow (Youden)

Same protocol as aim2_generalized_ml.py (lr/svm/sgd GridSearch; lock by mean MC AUROC).
lightSpike Youden thresholds via statistical_tests defaults.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import aim2_generalized_ml as a2
import statistical_tests as st

warnings.filterwarnings('ignore')

MODEL_CLINICAL = 'Model1_clinical'
MODEL_FULL = 'Model4_IED_slow'


def feature_sets_clinical_full() -> dict[str, list[str]]:
    return {
        MODEL_CLINICAL: a2.CLINICAL_COLS,
        MODEL_FULL: a2.CLINICAL_COLS + a2.IED_COLS + a2.SLOWING_COLS_BINARY,
    }


def run_one_type(
    epilepsy_type: str,
    out_dir: Path,
    n_mc: int,
    n_jobs: int,
    poor_cutoff: float,
    seed_start: int,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # aim2 helpers that write via OUT_DIR / save_table default
    a2.OUT_DIR = out_dir
    os.environ['ML_OUT_DIR'] = str(out_dir)

    feature_sets = feature_sets_clinical_full()
    a2.FEATURE_SETS = feature_sets
    a2.MODEL_ORDER = list(feature_sets.keys())
    model_keys = list(feature_sets.keys())
    # Restrict stratum loops to models we actually fit
    a2.PRIOR_STRATUM_MODELS = list(model_keys)
    a2.SEX_STRATUM_MODELS = list(model_keys)
    a2.IED_STRATUM_MODELS = [MODEL_FULL]
    a2.SUBTYPE_MODEL = MODEL_FULL

    print('=' * 80)
    print(f'Aim2 clinical vs full — {epilepsy_type} only (Youden lightSpike)')
    print(f'OUT_DIR: {out_dir}')
    print(f'N_MC={n_mc}  n_jobs={n_jobs}  poor_cutoff={poor_cutoff}/mo')
    print(
        f'Markers: IED>={st.SPIKE_YOUDEN_THRESH}, '
        f'slowing>={st.SLOWING_YOUDEN_THRESH_PCT}%'
    )
    print(f'Models: {model_keys}')
    print('=' * 80)

    patient_tables = st._load_combined_data()
    analysis_df = a2.build_analysis_df(
        patient_tables['sz_freq_class'],
        poor_cutoff=poor_cutoff,
        binary_markers=True,
        epilepsy_type=epilepsy_type,
    )
    analysis_df['ied_annotation'] = 'unknown'
    if epilepsy_type == 'Generalized':
        sub_ann = a2.load_patient_epilepsy_subtype(analysis_df['patient_id'])
        analysis_df = analysis_df.merge(
            sub_ann[['patient_id', 'epilepsy_subtype', 'subtype_raw']],
            on='patient_id', how='left',
        )
        analysis_df['epilepsy_subtype'] = analysis_df['epilepsy_subtype'].fillna('Unclassified')
        analysis_df['subtype_raw'] = analysis_df['subtype_raw'].fillna('')
    else:
        analysis_df['epilepsy_subtype'] = 'Unclassified'
        analysis_df['subtype_raw'] = ''

    print(
        f'\nCohort {epilepsy_type}: N={len(analysis_df)}  '
        f'prevalence={analysis_df["sz_freq_class"].mean():.3f}'
    )
    for mk, cols in feature_sets.items():
        print(f'  {mk}: {cols}')

    a2.save_table(
        analysis_df[[
            c for c in [
                'patient_id', 'sz_freq_class', 'ied_youden', 'sex_stratum',
                'age_at_eeg', 'gender', 'prior_sz_control',
                'prior_sz_poor', 'prior_sz_unavailable', 'prior_availability',
                'spikes', 'slowing', 'slowing_pct',
            ] if c in analysis_df.columns
        ]].copy(),
        'analysis_cohort.csv',
        out_dir,
    )

    print(f'\nRunning {n_mc} MC iterations...')
    seeds = range(seed_start, seed_start + n_mc)
    gen = Parallel(n_jobs=n_jobs, return_as='generator')(
        delayed(a2.run_mc_iteration)(seed, analysis_df, feature_sets)
        for seed in seeds
    )
    mc_results = list(tqdm(gen, total=n_mc, desc=f'MC {epilepsy_type}', unit='split'))
    n_ok = sum(r is not None for r in mc_results)
    print(f'  Kept {n_ok}/{n_mc}')

    selected, clf_rank = a2.select_clf_per_model(mc_results, model_keys)
    a2.save_table(clf_rank, 'algorithm_auc_ranking_by_model.csv', out_dir)
    print('\n=== Algorithm selection (mean MC AUROC) ===')
    for mk in model_keys:
        sub = clf_rank[clf_rank['model'] == mk]
        print(f'  {mk}:')
        for _, row in sub.iterrows():
            mark = ' ← SELECTED' if row['algorithm'] == selected[mk] else ''
            print(f"    {row['algorithm']:4s}  mean AUC={row['auc_mean']:.4f}{mark}")
    (out_dir / 'selected_algorithms_per_model.txt').write_text(
        ''.join(f'{mk}\t{selected[mk]}\n' for mk in model_keys), encoding='utf-8',
    )

    a2.apply_selected_clfs_per_model(mc_results, feature_sets, selected)

    # Core reporting (no subtype strata)
    a2.run_post_mc_analyses(
        mc_results, analysis_df,
        out_dir=out_dir,
        selection_label='per_model_best_of_lr_svm_sgd',
        selected_by_model=selected,
        bagged_model_keys=model_keys,
        prior_stratum_models=model_keys,
        pairwise_model_keys=model_keys,
    )

    # Compact summary
    summ = pd.read_csv(out_dir / 'mc_auc_auprc_summary.csv')
    pair = pd.read_csv(out_dir / 'pairwise_model_mc_splits.csv')
    lines = [
        f'Aim2 clinical vs full — {epilepsy_type} (Youden lightSpike)',
        f'N={len(analysis_df)}  N_MC={n_ok}',
        f'Selected: {selected}',
        '',
        'MC mean AUROC [bootstrap 95% CI of mean]:',
    ]
    for _, r in summ.iterrows():
        lines.append(
            f"  {r['model']}: AUROC={r['auc_mean']:.3f} "
            f"[{r['auc_ci_lo']:.3f}, {r['auc_ci_hi']:.3f}]  "
            f"AUPRC={r['auprc_mean']:.3f} [{r['auprc_ci_lo']:.3f}, {r['auprc_ci_hi']:.3f}]"
        )
    lines.append('')
    lines.append('Pairwise (MC sign test):')
    for _, r in pair.iterrows():
        lines.append(
            f"  {r['model_a']} vs {r['model_b']}: "
            f"Δ={r['diff']:+.3f} [{r['diff_ci_lo']:+.3f}, {r['diff_ci_hi']:+.3f}]  "
            f"p={r['p']:.4g}  wins {r['n_a_wins']}:{r['n_b_wins']}"
        )
    text = '\n'.join(lines) + '\n'
    (out_dir / 'RESULTS_SUMMARY.txt').write_text(text, encoding='utf-8')
    print('\n' + text)
    print(f'Done → {out_dir}')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--epilepsy-type', choices=['Generalized', 'Focal', 'both'], default='both')
    p.add_argument('--n-mc', type=int, default=1000)
    p.add_argument('--n-jobs', type=int, default=40)
    p.add_argument('--poor-cutoff', type=float, default=1.0)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument(
        '--out-root', type=Path,
        default=_SCRIPT_DIR / 'outputs' / 'by_type' / 'aim2_clinical_vs_full_youden_lightSpike',
    )
    args = p.parse_args()

    types = (
        ['Generalized', 'Focal'] if args.epilepsy_type == 'both'
        else [args.epilepsy_type]
    )
    for epi in types:
        tag = 'General' if epi == 'Generalized' else 'Focal'
        out = args.out_root / tag
        run_one_type(
            epilepsy_type=epi,
            out_dir=out,
            n_mc=args.n_mc,
            n_jobs=args.n_jobs,
            poor_cutoff=args.poor_cutoff,
            seed_start=args.seed_start,
        )


if __name__ == '__main__':
    main()
