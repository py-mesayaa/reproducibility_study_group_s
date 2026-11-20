#!/bin/bash
# Which undocumented setting did the authors use for BioMistral?
# Runs 4 variants on the first 64 questions -> repro/diagnostics/outputs/.
# Submit from ~/medchange:   sbatch repro/diagnostics/biomistral_diagnostics.sh
# Then compare with the authors' answers on the laptop (their answer file is not on the HPC):
#   python repro/diagnostics/compare_authors.py repro/outputs/biomistral_8bit_table6.jsonl repro/diagnostics/outputs/*.jsonl
#SBATCH --job-name=bm_diag
#SBATCH --output=logs/%x.%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=desktop[1,6-8],cn[16-17]
#SBATCH --partition=scavenge

module load Python/3.12.3-GCCcore-13.3.0
source ~/medchange-env/bin/activate
export HF_HUB_OFFLINE=1
cd ~/medchange
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv

OUT=repro/diagnostics/outputs
RUN="python repro/run_inference.py --model biomistral --limit 64"
$RUN --prompt table6_indent              --out $OUT/biomistral_8bit_table6_indent_limit64.jsonl       # indented prompt
$RUN --prompt table6 --double-bos        --out $OUT/biomistral_8bit_table6_2bos_limit64.jsonl         # <s><s>[INST] ...
$RUN --prompt table6_indent --double-bos --out $OUT/biomistral_8bit_table6_indent_2bos_limit64.jsonl  # both
$RUN --prompt table6 --sample            --out $OUT/biomistral_8bit_table6_sample_limit64.jsonl       # sampling, not greedy
