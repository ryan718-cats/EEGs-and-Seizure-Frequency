"""
Light preprocessing pipeline for IED and pathological slowing detection
(SpikeNet2 / MORGOTH inputs).

Per patient-session:
  1. Load raw EDF, crop to <= 4 hours
  2. Convert to µV
  3. MCN montage rename (T7→T3, T8→T4, P7→T5, P8→T6)
  4. Scan in 1-second windows: flag windows where ALL channels have
     max |amplitude| < FLAT_THRESHOLD_UV (default 1 µV) — zero/flat signal
  5. Split at flat windows into candidate continuous segments
  6. Discard segments shorter than MIN_SEG_SECONDS (default 10 s)
  7. Save each surviving segment as its own EDF

No filtering, no resampling, no ICA. All channels preserved.

Output filename:
  sub-Penn{pid}_ses-{ses}_seg-{nn:02d}_t{start_sec:.1f}-{end_sec:.1f}.edf
Output directory:
  light_preprocessed/{General,Temporal,Frontal}/

Usage:
  python light_preprocessing.py [--workers N] [--skip-existing]
"""

import glob
import os
import sys
import warnings
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
import mne
import pyedflib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SPIKE_COUNTS_CSV = (
    "/mnt/sauce/littlab/users/cho6/seizure_frequency_preprocessing/spike_counts.csv"
)
REPORT_CSV = (
    "/mnt/sauce/littlab/users/cho6/seizure_frequency_preprocessing/"
    "Routineeegpec-Deidreport_DATA_LABELS_2025-12-12_1339 (2).csv"
)
EDF_ROOT = (
    "/mnt/sauce/littlab/users/erinconr/projects/routine_eeg/data/routine_eegs"
)
OUTPUT_ROOT = (
    "/mnt/sauce/littlab/users/cho6/seizure_frequency_preprocessing/light_preprocessed"
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

MAX_ROUTINE_HOURS  = 4
FLAT_THRESHOLD_UV  = 1.0   # µV — below this across all channels → flat/disconnected
MIN_SEG_SECONDS    = 10.0  # discard segments shorter than this

# ---------------------------------------------------------------------------
# Cohort building  (mirrors 01_cohort_preprocessing.py)
# ---------------------------------------------------------------------------

def build_cohort(spike_csv: str, report_csv: str) -> pd.DataFrame:
    sc  = pd.read_csv(spike_csv)
    rep = pd.read_csv(report_csv)

    routine = sc[
        sc["Duration_sec"].notna() & (sc["Duration_sec"] <= MAX_ROUTINE_HOURS * 3600)
    ].copy()

    acq      = rep["acquired_on"].fillna("").str.lower().str.strip()
    cls      = rep["report_PATIENT_CLASS"].fillna("").str.lower().str.strip()
    jay      = rep["jay_in_or_out"].fillna("").str.lower().str.strip()
    is_outpt = (
        acq.str.contains("spe") | acq.str.contains("radnor")
        | (cls == "outpatient") | (jay == "out")
    )

    outpt_keys = (
        rep[is_outpt][["patient_id", "session_number"]]
        .drop_duplicates()
        .rename(columns={"patient_id": "Patient", "session_number": "Session"})
        .dropna()
        .astype(int)
    )

    filtered = routine.merge(outpt_keys, on=["Patient", "Session"])

    pat_type = (
        rep[["patient_id", "epilepsy_type", "epilepsy_specific"]]
        .dropna(subset=["patient_id"])
        .drop_duplicates(subset="patient_id")
        .copy()
    )
    pat_type["patient_id"] = pat_type["patient_id"].astype(float).astype(int)
    pat_type = pat_type.rename(columns={"patient_id": "Patient"})

    spec  = pat_type["epilepsy_specific"].fillna("").str.lower().str.strip()
    etype = pat_type["epilepsy_type"].fillna("").str.lower().str.strip()

    pat_type["EpiType3"] = ""
    pat_type.loc[spec.str.contains("temporal"), "EpiType3"] = "Temporal"
    pat_type.loc[spec.str.contains("frontal"),  "EpiType3"] = "Frontal"
    pat_type.loc[(pat_type["EpiType3"] == "") & (etype == "general"), "EpiType3"] = "General"

    canonical = pat_type[pat_type["EpiType3"].isin(["Temporal", "Frontal", "General"])][
        ["Patient", "EpiType3"]
    ]

    cohort = (
        filtered[["Patient", "Session"]]
        .merge(canonical, on="Patient")
        .drop_duplicates()
    )
    print(
        f"[Cohort] {len(cohort)} patient-sessions, "
        f"{cohort['Patient'].nunique()} unique patients"
    )
    return cohort


def edf_path_for(patient_id: int, session: int) -> str | None:
    pattern = (
        f"{EDF_ROOT}/sub-Penn{patient_id}/ses-{session}/eeg/"
        f"sub-Penn{patient_id}_ses-{session}_*_eeg.edf"
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# EDF save helper
# ---------------------------------------------------------------------------

def _save_edf_array(data_uv: np.ndarray, ch_names: list, sfreq: float, out_path: str):
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pyedflib.EdfWriter(out_path, n_ch, file_type=pyedflib.FILETYPE_EDF) as f:
            f.setSignalHeaders(headers)
            f.writeSamples(data_uv)


# ---------------------------------------------------------------------------
# Per-session worker
# ---------------------------------------------------------------------------

def _worker(args):
    pid, ses, subtype, skip_existing = args

    edf = edf_path_for(pid, ses)
    if edf is None:
        return pid, ses, 0, "no_edf"

    out_dir = os.path.join(OUTPUT_ROOT, subtype)

    if skip_existing:
        existing = glob.glob(
            os.path.join(out_dir, f"sub-Penn{pid}_ses-{ses}_seg-*.edf")
        )
        if existing:
            return pid, ses, len(existing), "skipped"

    try:
        mne.set_log_level("ERROR")

        # 1. Load EDF
        raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)

        # 2. Crop to <= 4 hours
        if raw.times[-1] > MAX_ROUTINE_HOURS * 3600 + 0.1:
            raw.crop(tmax=MAX_ROUTINE_HOURS * 3600 + 0.1)

        sf = raw.info["sfreq"]

        # 3. Convert to µV
        data_uv  = raw.get_data() * 1e6   # (n_ch, n_samples)
        ch_names = list(raw.ch_names)

        # 4. MCN rename
        rename_map = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
        ch_names = [rename_map.get(ch, ch) for ch in ch_names]

        # 5. Scan 1-second windows for flat/zero-amplitude signal
        win_samp = int(sf)
        n_wins   = data_uv.shape[1] // win_samp
        is_flat  = np.zeros(n_wins, dtype=bool)
        for w in range(n_wins):
            seg = data_uv[:, w * win_samp : (w + 1) * win_samp]
            if np.max(np.abs(seg)) < FLAT_THRESHOLD_UV:
                is_flat[w] = True

        # 6. Split into contiguous non-flat stretches
        segments = []
        in_seg   = False
        seg_start_win = 0
        for w in range(n_wins):
            if not is_flat[w] and not in_seg:
                seg_start_win = w
                in_seg = True
            elif (is_flat[w] or w == n_wins - 1) and in_seg:
                seg_end_win = w if is_flat[w] else w + 1
                in_seg = False
                s0 = seg_start_win * win_samp
                s1 = seg_end_win   * win_samp
                if (s1 - s0) / sf >= MIN_SEG_SECONDS:
                    segments.append((s0, s1))

        if not segments:
            return pid, ses, 0, "no_valid_segments"

        # 7. Save each segment as its own EDF
        os.makedirs(out_dir, exist_ok=True)
        for nn, (s0, s1) in enumerate(segments, start=1):
            t_start = s0 / sf
            t_end   = s1 / sf
            fname   = (
                f"sub-Penn{pid}_ses-{ses}_seg-{nn:02d}"
                f"_t{int(t_start)}-{int(t_end)}.edf"
            )
            _save_edf_array(data_uv[:, s0:s1], ch_names, sf,
                            os.path.join(out_dir, fname))

        return pid, ses, len(segments), "ok"

    except Exception as exc:
        return pid, ses, 0, f"error: {exc}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Light EEG preprocessing — per-segment EDF output"
    )
    parser.add_argument("--workers",       type=int, default=16,
                        help="Parallel worker processes (default: 16)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip sessions that already have segment EDFs")
    args = parser.parse_args()

    cohort = build_cohort(SPIKE_COUNTS_CSV, REPORT_CSV)

    for subtype in ("General", "Temporal", "Frontal"):
        os.makedirs(os.path.join(OUTPUT_ROOT, subtype), exist_ok=True)

    work_items = [
        (int(row["Patient"]), int(row["Session"]), row["EpiType3"], args.skip_existing)
        for _, row in cohort.iterrows()
    ]

    print(f"\nProcessing {len(work_items)} patient-sessions with {args.workers} workers...\n")

    total_segs   = 0
    total_missed = 0
    total_errors = 0
    total_skip   = 0

    with mp.Pool(processes=args.workers) as pool:
        for pid, ses, n_segs, status in pool.imap_unordered(_worker, work_items):
            if status == "ok":
                total_segs += n_segs
                print(f"  sub-Penn{pid} ses-{ses} → {n_segs} segment(s)")
            elif status == "skipped":
                total_skip += 1
                total_segs += n_segs
            elif status == "no_valid_segments":
                total_missed += 1
                print(f"  sub-Penn{pid} ses-{ses} — no valid segments")
            elif status == "no_edf":
                total_missed += 1
            else:
                total_errors += 1
                print(f"  sub-Penn{pid} ses-{ses} — {status}")

    print(f"\nDone.")
    print(f"  Total segments saved : {total_segs}")
    print(f"  Sessions skipped     : {total_skip}")
    print(f"  Sessions with no data: {total_missed}")
    print(f"  Errors               : {total_errors}")


if __name__ == "__main__":
    main()
