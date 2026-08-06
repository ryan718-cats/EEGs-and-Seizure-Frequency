# Main analysis (follow in order)

Scripts retain their original internal paths in many places. To re-run, either:

1. Point paths at the CSVs under `../data/`, or  
2. Keep the original project layout and use this folder as a documented checklist of what was run for the paper.

Shared libraries used by multiple steps live in [`shared/`](shared/).

---

## 1. Light preprocessing

**Folder:** [`1_light_preprocessing/`](1_light_preprocessing/)

Remove zero-amplitude discontinuities; keep continuous segments ≥10 s for SpikeNet2 / MORGOTH.

| File | Role |
|---|---|
| `light_preprocessing.py` | Paper light-preprocessing used for detection inputs |
| `pipeline_02_eeg_preprocessing.py` | Related pipeline preprocessing |
| `run_light_preprocessed.sh` | Shell driver that runs MORGOTH on light-preprocessed EDFs |

**Outputs used later:** `../data/features/light_preprocessed_duration.csv` (no EDFs in this archive).

---

## 2. IED and slowing detection

**Folder:** [`2_ied_and_slowing_detection/`](2_ied_and_slowing_detection/)

| File | Role |
|---|---|
| `pipeline_03_feature_extractions.py` | Feature extraction / SpikeNet2-related pipeline step |
| `finetune_classification.py`, `EEG_level_head.py`, `task_model.py`, `backbone.py`, `data_provider.py`, `tokenizer.py`, `utils.py` | MORGOTH inference stack (slowing) |
| `slowing_stats.py` | Aggregate MORGOTH slowing predictions |
| `run_light_preprocessed.sh` | Runs slowing (+ spikes task wiring) on light EDFs |

**Weights:** `../data/model_weights/morgoth/`  
**Tabular outputs:** `../data/features/spike_counts.csv`, patient feature tables under `../data/features/`.

---

## 3. Youden threshold validation

**Folder:** [`3_youden_threshold_validation/`](3_youden_threshold_validation/)

Compare algorithm scores to clinician report present/absent; pick operating points with Youden’s J.

| File | Role |
|---|---|
| `ied_log_spike_threshold_sweep_light_duration.py` | SpikeNet2 rate (light duration) vs report IED |
| `slowing_verification.ipynb` | MORGOTH % slowing vs report_SLOWING |
| `results/` | Matched-session tables + Youden summaries |

Paper thresholds: **8.36 /hr** (IED), **13.87%** windows (slowing).

---

## 4. Statistical models (OR + LRT)

**Folder:** [`4_statistical_models_OR_and_LRT/`](4_statistical_models_OR_and_LRT/)

| File | Role |
|---|---|
| `statistical_tests.py` | Nested logistic models (clinical / main / interactions), bootstrap ORs, LRTs |
| `results/or_table_*.csv` | Odds ratios for Figure 2 / Table 2 |
| `results/lrt_results.csv` | Likelihood-ratio tests cited in Results |
| `results/*LGS*` | Sensitivity analysis excluding LGS |

---

## 5. Machine learning (Aim 1 pooled)

**Folder:** [`5_machine_learning_aim1/`](5_machine_learning_aim1/)

Clinical vs Interactions models; 1,000 patient-grouped 80/20 splits; inner 5-fold CV for SVM / LR / SGD; bagged OOB predictions.

| File | Role |
|---|---|
| `aim1_pooled_ml.py` | Main Aim1 MC training / evaluation |
| `make_fig_mc_mean_roc_pr_overlays.py` | Figure 3 ROC/PR overlays (MC-mean) |
| `make_mc_mean_confusion_matrices.py` | Figure 3C-style confusion matrices |
| `results/mc_auc_auprc_summary.csv` | MC-mean AUROC/AUPRC + CIs |
| `results/bagged_oos_predictions.csv` | Per-patient bagged OOB probabilities |
| `results/oob_paired_*` | Paired DeLong / AUPRC Δ clinical vs interactions |
| `results/cm_thr0p5_summary_lightSpike.txt` | Sens/spec at threshold 0.5 |

---

## 6. Subgroup OOB analyses (Figure 4)

**Folder:** [`6_subgroup_oob_analyses/`](6_subgroup_oob_analyses/)

| File | Role |
|---|---|
| `make_fig_aim1_interaction_subgroups.py` | Build subgroup AUROC panels |
| `results/*epilepsy_type*` | Focal vs generalized |
| `results/*ied*` | IED present vs absent |
| `results/*prior*` | Prior seizure-control available vs not |

---

## Suggested reading order for numbers in the PDF

1. Cohort: `../data/cohorts/aim1_analysis_cohort_N1358.csv` + `../tables/Table1_patient_characteristics.txt`  
2. Thresholds: step 3 `results/`  
3. ORs / LRT: step 4 `results/` + `../tables/`  
4. ML metrics: step 5 `results/mc_auc_auprc_summary.csv` and paired Δ CSVs  
5. Subgroups: step 6 `results/`
