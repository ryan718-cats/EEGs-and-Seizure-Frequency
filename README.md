# EEGs and Seizure Frequency

Archive of analyses linking **pathological slowing** and **IEDs** on routine outpatient EEG to **2-year seizure control**, including statistics, machine learning, and supplementary validation.

This repository is organized for reading and following along. Raw EDFs are **not** included (too large). Feature tables, cohorts, model weights used for slowing detection, scripts, figures, and key numeric results **are** included.

---

## Folder map

| Folder | What is inside |
|---|---|
| [`figures/`](figures/) | Main-text figure PNGs (Figure 2–4). Figure 1 (methods schematic) and Figure 5 (calculator) are omitted. |
| [`supplementary/`](supplementary/) | Supplementary figures (eFigures), Python/MATLAB code for those analyses, and related numeric summaries. |
| [`tables/`](tables/) | Table 1, Table 2 / OR tables, eTable 1, LRTs, univariate tables, plus table-building scripts. |
| [`main analysis/`](main%20analysis/) | Numbered end-to-end analysis steps (1→6) with scripts and key metric CSVs. |
| [`data/`](data/) | Analysis cohorts, spike/slowing feature tables, de-identified EEG reports, MORGOTH weights, references for SpikeNet2. |

Start here for a guided path: [`main analysis/README.md`](main%20analysis/README.md).

---

## Main figures → files

| Paper figure | File |
|---|---|
| Figure 2 (OR forest) | `figures/Figure2_odds_ratio_forest.png` |
| Figure 3A (ROC) | `figures/Figure3A_roc_clinical_vs_interactions.png` |
| Figure 3B (PR) | `figures/Figure3B_pr_clinical_vs_interactions.png` |
| Figure 3C (confusion matrix @ 0.5) | `figures/Figure3C_confusion_matrix_threshold_0p5.png` |
| Figure 4A (focal vs generalized) | `figures/Figure4A_auroc_by_epilepsy_type.png` |
| Figure 4B (IED±) | `figures/Figure4B_auroc_by_ied_status.png` |
| Figure 4C (prior control±) | `figures/Figure4C_auroc_by_prior_seizure_control.png` |

## Supplementary figures → files

| eFigure | File |
|---|---|
| eFigure 1 (auto vs clinician AUROC) | `supplementary/figures/eFigure1_auto_vs_clinician_report_auroc.png` |
| eFigure 2A (Youden slowing) | `supplementary/figures/eFigure2A_youden_slowing_threshold_sweep.png` |
| eFigure 2B (Youden IED) | `supplementary/figures/eFigure2B_youden_ied_threshold_sweep.png` |
| eFigure 3A–D (gen/focal ROC–PR) | `supplementary/figures/eFigure3*.png` |
| eFigure 4A–B (gen subtype AUC / % slowing) | `supplementary/figures/eFigure4*.png` |

---

## Key analysis facts (as reported)

- **Cohort:** N = 1,358 (Focal 868 / Generalized 490), one routine EEG per patient; poor control = >1 seizure/month over 2 years post index visit.
- **IED Youden threshold:** 8.36 spikes/hour (N = 1,882 matched sessions for calibration).
- **Slowing Youden threshold:** 13.87% of windows (N = 242 matched sessions for calibration).
- **Dual-report auto vs clinician subset (eFigure 1):** N = 92 patients in the Aim1 cohort.

---

## What is *not* included

- Raw / light-preprocessed **EDF** recordings
- Localization side projects, PDR bump notebooks, unused exploratory runs
- Clinical calculator website source (deployed separately)

## SpikeNet2 weights

SpikeNet2 checkpoints were **not** stored in the original working tree (only `spike_counts.csv` outputs). See [`data/references/MODEL_WEIGHTS.md`](data/references/MODEL_WEIGHTS.md) for how to obtain them from the original publication (Li et al., 2025). MORGOTH slowing weights **are** included under `data/model_weights/morgoth/`.

---

## Environment

Python dependencies for the ML/stats steps: [`main analysis/shared/requirements.txt`](main%20analysis/shared/requirements.txt).  
MORGOTH inference also needs [`data/model_weights/morgoth/requirements_morgoth.txt`](data/model_weights/morgoth/requirements_morgoth.txt) and a GPU-capable PyTorch install for re-running detection from EDFs.
