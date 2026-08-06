"""
EEG preprocessing pipeline applied to cohort clips — saves as EDF.

Steps (mirroring eeg_psd_wt.py / Galer et al.):
  1. Read clip EDF
  2. Notch filter at 60 Hz harmonics; resample to 200 Hz
  3. Rename MCN channels to 10-20 (T7→T3, T8→T4, P7→T5, P8→T6)
  4. ICA artifact removal (EOG + ECG components)
  5. Select the 21 standard 10-20 channels
  6. Bandpass filter 0.5–70 Hz (2nd-order Butterworth, zero-phase)
  7. Save as EDF — same filename, same three-folder structure

Output:
  filtered_data/{General,Temporal,Frontal}/<same_filename_as_input>.edf

Usage:
  python eeg_preprocessing.py [--workers N] [--skip-existing]
"""

import os
import glob
import warnings
import argparse
import multiprocessing as mp

import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt
import mne

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CLIP_ROOT   = "/mnt/sauce/littlab/users/cho6/seizure_frequency_preprocessing/cohort_clips"
OUTPUT_ROOT = "/mnt/sauce/littlab/users/cho6/seizure_frequency_preprocessing/filtered_data"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

TARGET_SF    = 200
NOTCH_BASE   = 60      # Hz
LOW_CUTOFF   = 0.5     # Hz  bandpass low
HIGH_CUTOFF  = 70      # Hz  bandpass high
FILTER_ORDER = 2       # Butterworth order

SUB_CHANNELS = [
    'C3', 'C4', 'O1', 'O2', 'A1', 'A2', 'Cz',
    'F3', 'F4', 'F7', 'F8', 'Fz', 'Fp1', 'Fp2',
    'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6',
]

# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def _generate_harmonics(base, sf):
    harmonics, h = [], base
    while h < sf / 2:
        harmonics.append(h)
        h += base
    return harmonics


def _bandpass(data, low, high, sf, order=2):
    """Zero-phase Butterworth bandpass. data: (n_ch, n_samples)."""
    nyq = sf / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=1)


def _save_edf(data_uv, ch_names, sfreq, out_path):
    """Write (n_ch, n_samples) µV array to EDF via pyedflib."""
    n_ch = data_uv.shape[0]
    headers = []
    for i, ch in enumerate(ch_names):
        ch_min = float(data_uv[i].min())
        ch_max = float(data_uv[i].max())
        if ch_min == ch_max:
            ch_min -= 1.0
            ch_max += 1.0
        headers.append({
            "label":            ch,
            "dimension":        "uV",
            "sample_frequency": int(sfreq),
            "physical_min":     ch_min,
            "physical_max":     ch_max,
            "digital_min":      -32768,
            "digital_max":       32767,
            "transducer":       "",
            "prefilter":        "",
        })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pyedflib.EdfWriter(out_path, n_ch, file_type=pyedflib.FILETYPE_EDF) as f:
            f.setSignalHeaders(headers)
            f.writeSamples(data_uv)


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------

def _worker(args):
    edf_path, subtype, skip_existing = args
    fname    = os.path.basename(edf_path)
    out_path = os.path.join(OUTPUT_ROOT, subtype, fname)

    if skip_existing and os.path.exists(out_path):
        return fname, "skipped"

    try:
        # ── 1. Read ──────────────────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        sf = raw.info["sfreq"]

        # ── 2. Notch + resample to 200 Hz ────────────────────────────────────
        notch_freqs = _generate_harmonics(NOTCH_BASE, sf)
        raw.notch_filter(freqs=notch_freqs, verbose=False)
        if sf != TARGET_SF:
            if sf > TARGET_SF:
                raw.filter(l_freq=None, h_freq=TARGET_SF / 2 - 5,
                           method='iir', verbose=False)
            raw.resample(sfreq=TARGET_SF, verbose=False)
        sf = TARGET_SF

        # ── 3. MCN channel rename ────────────────────────────────────────────
        rename = {k: v for k, v in
                  {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}.items()
                  if k in raw.ch_names}
        if rename:
            raw.rename_channels(rename)

        # ── 4. Set channel types for ICA ─────────────────────────────────────
        sub_upper = {c.upper() for c in SUB_CHANNELS}
        type_map  = {}
        for ch in raw.ch_names:
            cu = ch.upper()
            if cu in ('LOC', 'ROC'):
                type_map[ch] = 'eog'
            elif cu in ('ECGL', 'ECGR', 'EKG'):
                type_map[ch] = 'ecg'
            elif cu not in sub_upper:
                type_map[ch] = 'misc'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type_map:
                raw.set_channel_types(type_map)

        # ── 5. ICA artifact removal ──────────────────────────────────────────
        eeg_picks = mne.pick_types(raw.info, eeg=True)
        if len(eeg_picks) >= 2:
            n_components = min(20, len(eeg_picks) - 1)

            # Fit ICA on high-pass filtered copy (MNE recommendation)
            raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
            ica = mne.preprocessing.ICA(
                n_components=n_components, random_state=42,
                max_iter='auto', verbose=False
            )
            ica.fit(raw_hp, picks='eeg', verbose=False)
            del raw_hp

            exclude = []
            eog_picks = mne.pick_types(raw.info, eog=True)
            if len(eog_picks):
                eog_names = [raw.ch_names[i] for i in eog_picks]
                idx, _ = ica.find_bads_eog(raw, ch_name=eog_names, verbose=False)
                exclude.extend(idx)
            ecg_picks = mne.pick_types(raw.info, ecg=True)
            if len(ecg_picks):
                ecg_name = raw.ch_names[ecg_picks[0]]
                idx, _ = ica.find_bads_ecg(raw, ch_name=ecg_name, verbose=False)
                exclude.extend(idx)

            ica.exclude = list(set(exclude))
            ica.apply(raw, verbose=False)

        # ── 6. Select 21 standard channels ───────────────────────────────────
        available = [c for c in SUB_CHANNELS if c in raw.ch_names]
        if len(available) < 2:
            return fname, "too_few_channels"
        raw.pick(available, verbose=False)

        # ── 7. Data to µV (MNE stores in Volts internally) ───────────────────
        data_uv  = raw.get_data() * 1e6    # (n_ch, n_samples)
        ch_names = raw.ch_names

        # ── 8. Bandpass filter ───────────────────────────────────────────────
        data_bp = _bandpass(data_uv, LOW_CUTOFF, HIGH_CUTOFF, sf, FILTER_ORDER)

        # ── 9. Save as EDF ───────────────────────────────────────────────────
        _save_edf(data_bp, ch_names, sf, out_path)
        return fname, "ok"

    except Exception as exc:
        return fname, f"error: {exc}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel worker processes (default: 16)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip clips that already have a preprocessed EDF")
    args = parser.parse_args()

    for subtype in ("General", "Temporal", "Frontal"):
        os.makedirs(os.path.join(OUTPUT_ROOT, subtype), exist_ok=True)

    work_items = []
    for subtype in ("General", "Temporal", "Frontal"):
        for edf in sorted(glob.glob(os.path.join(CLIP_ROOT, subtype, "*.edf"))):
            work_items.append((edf, subtype, args.skip_existing))

    print(f"Found {len(work_items)} clips — processing with {args.workers} workers.\n")

    done = errors = skipped = 0

    with mp.Pool(processes=args.workers) as pool:
        for fname, status in pool.imap_unordered(_worker, work_items):
            done += 1
            if status == "ok":
                print(f"  [{done}/{len(work_items)}] {fname}")
            elif status == "skipped":
                skipped += 1
                if done % 50 == 0:
                    print(f"  [{done}/{len(work_items)}] {skipped} skipped so far...")
            else:
                errors += 1
                print(f"  [{done}/{len(work_items)}] {fname}: {status}")

    print(f"\nDone. {done - errors - skipped} saved | {skipped} skipped | {errors} errors.")


if __name__ == "__main__":
    main()
