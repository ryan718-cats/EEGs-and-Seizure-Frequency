"""
EEG feature extraction from filtered clips — regional output.

For each 10-second window (50% overlap, 5-second step) per EDF clip:

  Per-channel features are first computed across all 19 channels, then
  averaged within each anatomical region:
    whole, frontal, temporal, central, parietal, occipital

  Per-channel features:
    • Relative bandpower via multitaper PSD (bandwidth=1.0 Hz)
        Delta 0.5-4 Hz, Theta 4-8 Hz, Alpha 8-13 Hz, Beta 13-25 Hz
        relative to total power in 0.5-70 Hz
    • Waveform length, Zero crossings
    • Hjorth Activity, Mobility, Complexity

  wPLI (delta band, all 171 pairs):
    Per-channel node strength = mean wPLI across all pairs for that channel.
    Regional wPLI = mean node strength of channels in that region.

Output: features/{general,frontal,temporal}.csv
  One row per window, 6 regions × 10 features = 60 feature columns.
  Saves after each file; skips files already in the CSV on re-run.

Usage:
  python feature_extractions.py                          # all subtypes, 8 workers
  python feature_extractions.py --subtype General
  python feature_extractions.py --workers 16
"""

import argparse
import glob
import os
from itertools import combinations
from multiprocessing import Pool

import mne
import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_multitaper
from scipy.signal import butter, filtfilt, hilbert

mne.set_log_level("WARNING")

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT   = "/mnt/sauce/littlab/users/cho6/seizure_frequency_preprocessing/filtered_data"
OUTPUT_ROOT = "/mnt/sauce/littlab/users/cho6/seizure_frequency_preprocessing/features"
SUBTYPES    = ["General", "Frontal", "Temporal"]

# ── channels ───────────────────────────────────────────────────────────────────
STANDARD_19 = [
    "FP1", "F3",  "C3",  "P3",  "F7", "T3", "T5", "O1",
    "FZ",  "CZ",  "PZ",
    "FP2", "F4",  "C4",  "P4",  "F8", "T4", "T6", "O2",
]
CH_PAIRS    = list(combinations(range(len(STANDARD_19)), 2))   # 171 pairs
_I_IDX      = np.array([i for i, j in CH_PAIRS])
_J_IDX      = np.array([j for i, j in CH_PAIRS])

# ── regional channel groupings (standard 10-20) ────────────────────────────────
REGIONS = {
    "frontal":   ["FP1", "FP2", "F3", "F4", "F7", "F8", "FZ"],
    "temporal":  ["T3", "T4", "T5", "T6"],
    "central":   ["C3", "C4", "CZ"],
    "parietal":  ["P3", "P4", "PZ"],
    "occipital": ["O1", "O2"],
}
# Index into STANDARD_19 for each region; "whole" = all 19
REGION_IDX = {reg: np.array([STANDARD_19.index(ch) for ch in chs])
              for reg, chs in REGIONS.items()}
REGION_IDX["whole"] = np.arange(len(STANDARD_19))

# For node strength: which pair indices involve each channel
_NODE_PAIR_IDX = {
    i: np.array([k for k, (a, b) in enumerate(CH_PAIRS) if a == i or b == i])
    for i in range(len(STANDARD_19))
}

# ── windowing ──────────────────────────────────────────────────────────────────
WIN_SEC  = 10
STEP_SEC = 5     # 50% overlap

# ── frequency bands ────────────────────────────────────────────────────────────
BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 25.0),
}
TOTAL_BAND = (0.5, 70.0)


# ── feature helpers ────────────────────────────────────────────────────────────

def compute_bandpowers_all(win, fs):
    """
    Multitaper PSD for all 19 channels in one call.
    win   : (19, n_samples)
    Returns dict {band: (19,) array of relative power}.
    bandwidth=1.0 Hz → NW=5 for 10 s window, 9 DPSS tapers.
    """
    psd, freqs = psd_array_multitaper(
        win, sfreq=fs, fmin=TOTAL_BAND[0], fmax=TOTAL_BAND[1],
        bandwidth=1.0, normalization="full", verbose=False,
    )  # psd: (19, n_freqs)

    total_mask = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])
    total = np.trapz(psd[:, total_mask], freqs[total_mask], axis=1)  # (19,)
    bad   = total < 1e-12

    result = {}
    for band, (fmin, fmax) in BANDS.items():
        mask  = (freqs >= fmin) & (freqs <= fmax)
        bp    = np.trapz(psd[:, mask], freqs[mask], axis=1)           # (19,)
        vals  = np.where(bad, np.nan, bp / total)
        result[band] = vals
    return result


def compute_wpli_vectorized(win, fs):
    """
    wPLI for all 171 pairs across all 4 frequency bands — vectorized.
    win : (19, n_samples)
    Returns dict {band: (171,) array} in CH_PAIRS order.
    """
    nyq = fs / 2.0
    result = {}
    for band, (fmin, fmax) in BANDS.items():
        b, a     = butter(4, [fmin / nyq, fmax / nyq], btype="band")
        filtered = filtfilt(b, a, win, axis=1)                        # (19, n_samples)
        analytic = hilbert(filtered, axis=1)                          # (19, n_samples)
        cross    = analytic[_I_IDX] * np.conj(analytic[_J_IDX])      # (171, n_samples)
        im       = np.imag(cross)                                     # (171, n_samples)
        mean_im  = np.mean(im, axis=1)                                # (171,)
        mean_abs = np.mean(np.abs(im), axis=1)                        # (171,)
        result[band] = np.where(mean_abs > 1e-12, np.abs(mean_im) / mean_abs, np.nan)
    return result


def _timedomain_all(win):
    """
    Compute waveform length, zero crossings, and Hjorth params for all channels.
    win : (19, n_samples)
    Returns dict of {metric: (19,) array}.
    """
    dx   = np.diff(win, axis=1)   # (19, n_samples-1)
    d2x  = np.diff(dx, axis=1)    # (19, n_samples-2)

    wl    = np.sum(np.abs(dx), axis=1)                                # (19,)
    zc    = np.sum(np.diff(np.sign(win), axis=1) != 0, axis=1)        # (19,)

    act   = np.var(win, axis=1)                                        # (19,)
    var_dx  = np.var(dx,  axis=1)
    var_d2x = np.var(d2x, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mob   = np.where(act  > 1e-12, np.sqrt(var_dx  / act),   np.nan)
        comp  = np.where(
            (var_dx > 1e-12) & (mob > 1e-12),
            np.sqrt(var_d2x / var_dx) / mob,
            np.nan,
        )
    act = np.where(act > 1e-12, act, np.nan)

    return {"wl": wl, "zc": zc.astype(float), "act": act, "mob": mob, "comp": comp}


# ── EDF loading ────────────────────────────────────────────────────────────────

def load_edf(path):
    """Returns (data_uV ndarray (19, n_samples), fs) or (None, None)."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.rename_channels({ch: ch.upper() for ch in raw.ch_names})
    raw.rename_channels({k: v for k, v in
                         {"T7": "T3", "P7": "T5", "T8": "T4", "P8": "T6"}.items()
                         if k in raw.ch_names})
    missing = set(STANDARD_19) - set(raw.ch_names)
    if missing:
        print(f"    missing {missing} — skip")
        return None, None
    raw.pick_channels(STANDARD_19)
    raw.reorder_channels(STANDARD_19)
    return raw.get_data(units="uV"), raw.info["sfreq"]


# ── per-window feature row ─────────────────────────────────────────────────────

def _build_row(fname, win_idx, win_start, bp, td, wpli_dict):
    """
    Assemble one regional feature row.
    bp        : {band: (19,) array}        — per-channel relative bandpower
    td        : {metric: (19,) array}      — per-channel time-domain features
    wpli_dict : {band: (171,) array}       — per-pair wPLI for each band

    Per-channel wPLI node strength = mean wPLI across all pairs for that channel.
    Regional value = nanmean of channels in that region.
    """
    # Node strength per band (19,): average wPLI across all pairs for each channel
    node_strength = {
        band: np.array([
            np.nanmean(wpli_dict[band][_NODE_PAIR_IDX[i]])
            for i in range(len(STANDARD_19))
        ])
        for band in BANDS
    }

    row = {
        "file_name":      fname,
        "window_idx":     win_idx,
        "window_start_s": win_start,
    }

    for reg, idx in REGION_IDX.items():
        for band in BANDS:
            row[f"{reg}_rel_{band}"]                          = float(np.nanmean(bp[band][idx]))
        row[f"{reg}_waveform_length"]                         = float(np.nanmean(td["wl"][idx]))
        row[f"{reg}_zero_crossings"]                          = float(np.nanmean(td["zc"][idx]))
        row[f"{reg}_hjorth_activity"]                         = float(np.nanmean(td["act"][idx]))
        row[f"{reg}_hjorth_mobility"]                         = float(np.nanmean(td["mob"][idx]))
        row[f"{reg}_hjorth_complexity"]                       = float(np.nanmean(td["comp"][idx]))
        for band in BANDS:
            row[f"{reg}_wpli_{band}_node_strength"]           = float(np.nanmean(node_strength[band][idx]))

    return row


# ── per-file processing ────────────────────────────────────────────────────────

def process_file(path):
    """
    Returns (fname, rows_list) where rows_list is a list of row dicts.
    Returns (fname, None) on failure or if no channels found.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")   # avoid thread over-subscription

    data, fs = load_edf(path)
    fname    = os.path.splitext(os.path.basename(path))[0]
    if data is None:
        return fname, None

    fs_int       = int(fs)
    win_samples  = int(WIN_SEC  * fs)
    step_samples = int(STEP_SEC * fs)
    n_samples    = data.shape[1]

    rows = []
    win_idx, start, pos = 0, 0.0, 0
    while pos + win_samples <= n_samples:
        win = data[:, pos : pos + win_samples]              # (19, win_samples)
        bp  = compute_bandpowers_all(win, fs_int)
        td  = _timedomain_all(win)
        wp  = compute_wpli_vectorized(win, fs_int)
        rows.append(_build_row(fname, win_idx, round(start, 3), bp, td, wp))
        pos     += step_samples
        start   += STEP_SEC
        win_idx += 1

    return fname, (rows if rows else None)


def _worker(path):
    """Top-level wrapper so Pool can pickle it."""
    try:
        return process_file(path)
    except Exception as e:
        fname = os.path.splitext(os.path.basename(path))[0]
        return fname, None, str(e)


# ── per-subtype loop ───────────────────────────────────────────────────────────

def process_subtype(subtype, workers=8, data_dir=None, out_file=None):
    edf_dir  = data_dir if data_dir else os.path.join(DATA_ROOT, subtype)
    out_path = out_file if out_file else os.path.join(OUTPUT_ROOT, f"{subtype.lower()}.csv")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    edfs = sorted(glob.glob(os.path.join(edf_dir, "*.edf")))
    print(f"\n{'='*64}")
    print(f"[{subtype}]  {len(edfs)} files  →  {out_path}  (workers={workers})")

    if os.path.exists(out_path):
        done        = set(pd.read_csv(out_path, usecols=["file_name"])["file_name"])
        write_header = False
        print(f"  Resuming — {len(done)} files already done")
    else:
        done        = set()
        write_header = True

    pending = [f for f in edfs
               if os.path.splitext(os.path.basename(f))[0] not in done]
    print(f"  {len(pending)} files to process\n")

    if not pending:
        return

    completed = 0
    with Pool(workers) as pool:
        for result in pool.imap_unordered(_worker, pending):
            # _worker may return 2 or 3 values depending on error path
            if len(result) == 3:
                fname, rows, err = result
                print(f"  ERROR {fname}: {err}")
                continue
            fname, rows = result
            completed += 1
            if rows is None:
                print(f"  [{completed:4d}/{len(pending)}] {fname}  skipped")
                continue
            df = pd.DataFrame(rows)
            df.to_csv(out_path, mode="a", header=write_header, index=False)
            write_header = False
            print(f"  [{completed:4d}/{len(pending)}] {fname}  {len(rows)} windows")

    total = sum(1 for _ in open(out_path)) - 1 if os.path.exists(out_path) else 0
    print(f"\n  Done — {total} total rows → {out_path}")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtype", choices=SUBTYPES + ["all"], default="all",
                    help="Subtype to process (default: all)")
    ap.add_argument("--data_dir",  default=None, help="Override EDF input directory")
    ap.add_argument("--out_file",  default=None, help="Override output CSV path")
    ap.add_argument("--workers", type=int, default=8,
                    help="Number of parallel EDF workers (default: 8)")
    args = ap.parse_args()

    for sub in (SUBTYPES if args.subtype == "all" else [args.subtype]):
        process_subtype(sub, workers=args.workers,
                        data_dir=args.data_dir, out_file=args.out_file)


if __name__ == "__main__":
    main()
