# Supplementary

Code and figures for **eFigures / eTables** (not main-text Figures 2–4).

## Layout

| Path | Contents |
|---|---|
| [`figures/`](figures/) | eFigure PNGs |
| [`python/`](python/) | Python scripts / notebooks for supplementary analyses |
| [`matlab/`](matlab/) | MATLAB scripts for supplementary figures |
| [`results/`](results/) | Numeric summaries cited with eFigures |

## Script → eFigure map

| Analysis | Code | Output figure |
|---|---|---|
| Auto Youden vs clinician report (N=92), paired DeLong | `python/aim1_report_vs_auto_markers_oob.py` | `figures/eFigure1_*.png` |
| Youden slowing sweep | `python/slowing_verification.ipynb` | `figures/eFigure2A_*.png` |
| Youden IED sweep | `python/ied_log_spike_threshold_sweep_light_duration.py` | `figures/eFigure2B_*.png` |
| Clinical vs full model by epilepsy type (ROC/PR) | `python/aim2_type_clinical_vs_full.py`, `python/make_fig_mc_mean_roc_pr_overlays.py` | `figures/eFigure3*.png` |
| Gen subtype OOB AUROC | `python/make_fig_aim2_subgroup_auroc_by_subtype.py` (and Aim1 subtype bagged metrics) | `figures/eFigure4A_*.png` |
| % slowing by gen subtype | `matlab/make_fig_general_slowing_by_subtype.m` | `figures/eFigure4B_*.png` |
| LGS exclusion OR check | `matlab/make_fig_general_slowing_by_outcome_excl_lgs.m` + `results/gen_main_effects_slowing_OR_by_LGS_subset_youden_lightSpike.txt` | (text result in Results/Supplement) |

Annotation availability helper: `python/count_slowing_ied_availability.py` (supports eTable 1 style counts).
