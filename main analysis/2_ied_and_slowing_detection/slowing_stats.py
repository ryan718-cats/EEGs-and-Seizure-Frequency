"""
Compute per-clip slowing statistics from morgoth Stage 1 event-level CSVs.

For each per-window CSV from finetune_classification.py (SLOWING dataset,
10-second snippets with 1-second step — pred_SLOWING_1sStep), computes:
  n_windows       : number of 10s snippets (1s step between starts)
  pct_no_slowing  : % of windows predicted as class 0 (no slowing)
  pct_foc_slowing : % of windows predicted as class 1 (focal slowing)
  pct_gen_slowing : % of windows predicted as class 2 (generalized slowing)

Saves to --output after every file so progress is visible mid-run.
Re-running skips files already present in the output CSV.

Usage:
  python slowing_stats.py \
      --results_dir test_data/one_clip_results/pred_SLOWING_1sStep \
      --output      slowing_stats.csv
"""

import argparse
import glob
import os

import pandas as pd


def process_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    n = len(df)
    counts = df["pred_class"].value_counts()
    return {
        "n_windows":       n,
        "pct_no_slowing":  round(counts.get(0, 0) / n * 100, 2),
        "pct_foc_slowing": round(counts.get(1, 0) / n * 100, 2),
        "pct_gen_slowing": round(counts.get(2, 0) / n * 100, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="Directory of morgoth Stage 1 per-second CSVs")
    ap.add_argument("--output", required=True,
                    help="Output CSV path (appended to on resume)")
    args = ap.parse_args()

    COLS = ["file_name", "n_windows", "pct_no_slowing", "pct_foc_slowing", "pct_gen_slowing"]

    if os.path.exists(args.output):
        out_df = pd.read_csv(args.output)
        done = set(out_df["file_name"].tolist())
        print(f"Resuming — {len(done)} files already done in {args.output}")
    else:
        out_df = pd.DataFrame(columns=COLS)
        done = set()

    csv_files = sorted(glob.glob(os.path.join(args.results_dir, "*.csv")))
    pending   = [f for f in csv_files
                 if os.path.splitext(os.path.basename(f))[0] not in done]

    print(f"{len(csv_files)} total CSVs found, {len(pending)} to process\n")

    for i, csv_path in enumerate(pending, 1):
        fname = os.path.splitext(os.path.basename(csv_path))[0]
        try:
            stats = process_csv(csv_path)
            row   = {"file_name": fname, **stats}
            out_df = pd.concat([out_df, pd.DataFrame([row])], ignore_index=True)
            out_df[COLS].to_csv(args.output, index=False)
            print(f"[{i}/{len(pending)}] {fname}")
            print(f"         windows={stats['n_windows']}  "
                  f"none={stats['pct_no_slowing']:.1f}%  "
                  f"focal={stats['pct_foc_slowing']:.1f}%  "
                  f"gen={stats['pct_gen_slowing']:.1f}%")
        except Exception as e:
            print(f"[{i}/{len(pending)}] ERROR {fname}: {e}")

    print(f"\nDone. {len(out_df)} total rows → {args.output}")


if __name__ == "__main__":
    main()
