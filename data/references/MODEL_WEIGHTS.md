# Model weights and external tools

## MORGOTH (included)

Path: `../model_weights/morgoth/`

| File | Use in this paper |
|---|---|
| `SLOWING.pth` | Event-level slowing classifier (10 s windows) |
| `FOC_SLOWING_EEGlevel.pth` | Focal slowing EEG-level head |
| `GEN_SLOWING_EEGlevel.pth` | Generalized slowing EEG-level head |
| `tokenizer/checkpoint.pth` | MORGOTH tokenizer checkpoint |

Citation: Sun et al., 2026 (MORGOTH 1.0; BDSP DOI in paper references).

Requirements note: `requirements_morgoth.txt` (full GPU stack may still need the upstream MORGOTH environment).

## SpikeNet2 (not bundled)

SpikeNet2 weights were **not** present as checkpoint files in the working project directory used for this archive. Session-level detections used in analysis are provided as:

- `../features/spike_counts.csv`

To re-detect spikes from EDFs, obtain SpikeNet2 from the original publication:

- Li et al., 2025. *Expert-Level Detection of Epilepsy Markers in EEG on Short and Long Timescales.* NEJM AI.  
  Threshold **0.46** was used for event detection; presence/absence for modeling used the paper Youden rate threshold **8.36 spikes/hour** (light-preprocessed duration).

Place downloaded SpikeNet2 weights under a local folder of your choosing and point your detection script at them (see `main analysis/2_ied_and_slowing_detection/`).
