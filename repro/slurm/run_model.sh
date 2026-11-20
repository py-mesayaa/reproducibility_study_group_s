#!/bin/bash
# Run repro/run_inference.py on one GPU. Everything after the script name is passed on.
# Submit from ~/medchange (the log path below is relative to it).
#
# Qwen runs in bf16 -> also exclude GPUs without native bf16 (V100 cn3-6, RTX 8000 cn12;
# cn10's RTX 6000 is the Ada generation and does support bf16):
#   sbatch --job-name=qwen --exclude=desktop[1,6-8],cn[3-6,12,16-17] repro/slurm/run_model.sh --model qwen
# BioMistral runs in 8-bit -> any GPU with >= 16 GB works (the default exclude below is enough):
#   sbatch --job-name=biomistral repro/slurm/run_model.sh --model biomistral
#SBATCH --output=logs/%x.%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=desktop[1,6-8],cn[16-17]
#SBATCH --partition=scavenge

module load Python/3.12.3-GCCcore-13.3.0
source ~/medchange-env/bin/activate
export HF_HUB_OFFLINE=1

cd ~/medchange
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python repro/run_inference.py "$@"
