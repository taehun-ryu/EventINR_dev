#!/bin/bash
#SBATCH --job-name=eval_Ours
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
CUSTOM=/workspace/swcho/E2VID/Custom_memmap

EXP_NAME=local_lpips_fourier_sigmoid_5e-4
RUN="runs/train/ecoco_base/$EXP_NAME"

CKPT=last
CLAHE=clahe   # none 또는 clahe

OUT="runs/eval_results/${EXP_NAME}_${CKPT}_${CLAHE}_${SLURM_JOB_ID:-manual}"

python -u evaluate.py \
    --checkpoint "$RUN/$CKPT.pt" --out "$OUT" --tone_mapping "$CLAHE" \
    --dataset ECD --data_root ../ECD_IJRR_memmap --evreal_root "$EVREAL" \
    --video_fps 25 --device cuda

python -u evaluate.py \
    --checkpoint "$RUN/$CKPT.pt" --out "$OUT" --tone_mapping "$CLAHE" \
    --dataset HQF --data_root ../HQF_memmap --evreal_root "$EVREAL" \
    --video_fps 25 --device cuda

python -u evaluate.py \
    --checkpoint "$RUN/$CKPT.pt" --out "$OUT" --tone_mapping "$CLAHE" \
    --dataset MVSEC --data_root ../MVSEC_memmap --evreal_root "$EVREAL" \
    --video_fps 25 --device cuda



# CLAHE=none   # none 또는 clahe
OUT="runs/eval_results/custom_${EXP_NAME}_${CKPT}_${CLAHE}_${SLURM_JOB_ID:-manual}"
python -u evaluate.py \
    --checkpoint "$RUN/$CKPT.pt" --evreal_root "$EVREAL" \
    --dataset Custom --data_root "$CUSTOM" --out "$OUT" \
    --tone_mapping "$CLAHE" --video_fps 25 --device cuda
