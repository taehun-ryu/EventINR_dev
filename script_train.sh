#!/bin/bash
#SBATCH --job-name=train_EventINR
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

DATA=/data/swcho/ECOCO
OUT=runs/train/ecoco_fourier_lpips_v2

# Linear
EXP_NAME=local_lpips_fourier_5e-4_linear
RUN="$OUT/$EXP_NAME"

python -u train.py \
    --train "$DATA/train_list.txt" --out "$RUN" \
    --encoder etnet --decoder_type relu_fourier --intensity_head linear \
    --time_mode local --loss lpips --temporal_weight 0 --fourier_bands 8 \
    --voxel_normalization none --crop 112 --steps 40 --batch_size 2 --workers 4 \
    --lr 5e-4 --charbonnier_eps 0.001 --seed 42 --device cuda \
    --epochs 200 --save_every 10 --preview_every 1

# Sigmoid
EXP_NAME=local_lpips_fourier_5e-4_sigmoid
RUN="$OUT/$EXP_NAME"

python -u train.py \
    --train "$DATA/train_list.txt" --out "$RUN" \
    --encoder etnet --decoder_type relu_fourier --intensity_head sigmoid \
    --time_mode local --loss lpips --temporal_weight 0 --fourier_bands 8 \
    --voxel_normalization none --crop 112 --steps 40 --batch_size 2 --workers 4 \
    --lr 5e-4 --charbonnier_eps 0.001 --seed 42 --device cuda \
    --epochs 200 --save_every 10 --preview_every 1
