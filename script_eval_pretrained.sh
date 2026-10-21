#!/bin/bash
#SBATCH --job-name=eval_EVREAL
#SBATCH --output=logs/%x.o%j
#SBATCH --error=logs/%x.e%j
#SBATCH --partition=R4090
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=vegi-01
#SBATCH --mem=16G

set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate evinr

cd /workspace/swcho/E2VID/EventINR

EVREAL=/workspace/swcho/E2VID/EVREAL
ECD=/workspace/swcho/E2VID/ECD_IJRR_memmap
HQF=/workspace/swcho/E2VID/HQF_memmap
MVSEC=/workspace/swcho/E2VID/MVSEC_memmap
# OUT="runs/eval_results/evreal_pretrained/baselines_clahe_${SLURM_JOB_ID:-manual}"
OUT="runs/eval_results/evreal_pretrain"

[[ -d "$EVREAL" && -d "$ECD" && -d "$HQF" ]] || { echo 'Check EVREAL / ECD / HQF paths.' >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Choose a new OUT: $OUT" >&2; exit 1; }

python -u eval_pretrained.py \
    --evreal_root "$EVREAL" --dataset ECD --data_root "$ECD" \
    --methods E2VID E2VID+ ET-Net HyperE2VID \
    --out "$OUT" --tone_mapping clahe --fps 25 --device cuda

python -u eval_pretrained.py \
    --evreal_root "$EVREAL" --dataset HQF --data_root "$HQF" \
    --methods E2VID E2VID+ ET-Net HyperE2VID \
    --out "$OUT" --tone_mapping clahe --fps 25 --device cuda

[[ -f "$EVREAL/config/dataset/MVSEC.json" && -d "$MVSEC" ]] || { echo 'Check EVREAL / MVSEC paths.' >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Choose a new OUT: $OUT" >&2; exit 1; }

# No CLAHE: preserve each model's native EVREAL output normalization.
python -u eval_pretrained.py \
    --evreal_root "$EVREAL" --dataset MVSEC --data_root "$MVSEC" \
    --methods E2VID E2VID+ ET-Net HyperE2VID \
    --out "$OUT/raw" --tone_mapping none --fps 25 --device cuda

# CLAHE: apply the same OpenCV recipe independently to prediction AND GT.
python -u eval_pretrained.py \
    --evreal_root "$EVREAL" --dataset MVSEC --data_root "$MVSEC" \
    --methods E2VID E2VID+ ET-Net HyperE2VID \
    --out "$OUT/clahe" --tone_mapping clahe --fps 25 --device cuda




EVREAL=../EVREAL
CUSTOM=../Custom_memmap
OUT=runs/eval_results/custom_benchmark/pretrained

python -u eval_pretrained.py \
    --evreal_root "$EVREAL" --dataset Custom --data_root "$CUSTOM" \
    --methods E2VID E2VID+ ET-Net HyperE2VID \
    --out "$OUT/raw" --tone_mapping none --fps 25 --device cuda

# CLAHE on: independently process prediction and GT before metrics.
python -u eval_pretrained.py \
    --evreal_root "$EVREAL" --dataset Custom --data_root "$CUSTOM" \
    --methods E2VID E2VID+ ET-Net HyperE2VID \
    --out "$OUT/clahe" --tone_mapping clahe --fps 25 --device cuda
