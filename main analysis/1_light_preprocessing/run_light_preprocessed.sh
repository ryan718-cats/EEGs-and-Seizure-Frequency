#!/bin/bash
# Run MORGOTH slowing + spike detection on light-preprocessed EDFs.
#
# Input : light_preprocessed/{General,Temporal,Frontal}/
# Output: results/light_{SLOWING,SPIKES}_{General,Temporal,Frontal}/
#
# Runs subtypes sequentially within each task to avoid GPU contention.
# Set --rewrite_results no so individual files already processed are skipped.

set -euo pipefail

MORGOTH_DIR="/mnt/sauce/littlab/users/cho6/seizure_frequency_preprocessing/morgoth"
PYTHON="/mnt/sauce/littlab/users/cho6/.venv/bin/python3"
LIGHT_ROOT="/mnt/sauce/littlab/users/cho6/seizure_frequency_preprocessing/light_preprocessed"
export PYHEALTH_CACHE="/mnt/sauce/littlab/users/cho6/.cache/pyhealth/"

cd "${MORGOTH_DIR}"
mkdir -p logs

# ---------------------------------------------------------------------------
# SLOWING detection  (focal + generalized EEG-level heads)
# ---------------------------------------------------------------------------

echo "==================================================================="
echo "TASK: SLOWING"
echo "==================================================================="

for SUBTYPE in General Temporal Frontal; do
    INPUT_DIR="${LIGHT_ROOT}/${SUBTYPE}"
    RESULT_DIR="${MORGOTH_DIR}/results/light_SLOWING_${SUBTYPE}"
    STEP1_DIR="${RESULT_DIR}/pred_SLOWING_1sStep"
    mkdir -p "${STEP1_DIR}"

    n_done=$(find "${STEP1_DIR}" -name "*.csv" 2>/dev/null | wc -l)
    echo ""
    echo "[SLOWING / ${SUBTYPE}] Step 1: event-level (${n_done} already done, resuming)..."
    echo "  Input : ${INPUT_DIR}"
    echo "  Output: ${RESULT_DIR}"

    OMP_NUM_THREADS=1 ${PYTHON} -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=1 --master_port=29500 \
        finetune_classification.py \
        --batch_size 512 \
        --num_workers 16 \
        --predict \
        --model base_patch200_200 \
        --task_model checkpoints/SLOWING.pth \
        --abs_pos_emb \
        --dataset SLOWING \
        --data_format edf \
        --sampling_rate 0 \
        --already_format_channel_order no \
        --already_average_montage no \
        --allow_missing_channels no \
        --max_length_hour no \
        --leave_one_hemisphere_out no \
        --polarity 1 \
        --eval_sub_dir "${INPUT_DIR}" \
        --eval_results_dir "${STEP1_DIR}" \
        --prediction_slipping_step_second 1 \
        --rewrite_results no

    echo "[SLOWING / ${SUBTYPE}] Step 2: EEG-level heads..."
    for DATASET in FOC_SLOWING GEN_SLOWING; do
        OUT_CSV="${RESULT_DIR}/pred_EEG_level_${DATASET}.csv"
        if [ -f "${OUT_CSV}" ]; then
            echo "  ${DATASET}: already exists — skipping"
        else
            ${PYTHON} EEG_level_head.py \
                --mode predict \
                --dataset "${DATASET}" \
                --task_model "checkpoints/${DATASET}_EEGlevel.pth" \
                --test_csv_dir "${STEP1_DIR}" \
                --result_dir "${RESULT_DIR}"
        fi
    done
    echo "[SLOWING / ${SUBTYPE}] Done."
done

echo ""
echo "==================================================================="
echo "All subtypes complete."
echo "  Results: ${MORGOTH_DIR}/results/light_SLOWING_{General,Temporal,Frontal}/"
echo "==================================================================="
