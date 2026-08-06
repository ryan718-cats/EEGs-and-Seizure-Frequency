# Data

De-identified tabular inputs used in the paper analyses. **No EDFs.**

## Cohorts

| File | Description |
|---|---|
| `cohorts/aim1_analysis_cohort_N1358.csv` | Pooled Aim1 analysis table (outcome + Youden binary markers + clinical covariates) |
| `cohorts/aim2_Focal_analysis_cohort.csv` | Focal-only ML cohort |
| `cohorts/aim2_General_analysis_cohort.csv` | Generalized-only ML cohort |

## Features

| File | Description |
|---|---|
| `features/spike_counts.csv` | SpikeNet2 spike counts / durations per session |
| `features/light_preprocessed_duration.csv` | Light-preprocessed usable duration per session |
| `features/*_patient.csv` | Patient-level feature tables (general / temporal / frontal) |

## Reports

| File | Description |
|---|---|
| `reports/Routineeegpec_Deidreport_DATA_LABELS.csv` | De-identified EEG report labels (IED / slowing annotations, etc.) |

## Youden validation intermediates

CSV dumps under `youden_validation/` from the light-duration IED threshold sweep.

## Model weights

See [`references/MODEL_WEIGHTS.md`](references/MODEL_WEIGHTS.md).
