#!/bin/bash
#SBATCH -J krs
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%A_%a.out
#
# One run:        sbatch scripts/train.sbatch configs/vanilla.yaml
# Three seeds:    sbatch --array=0-2 scripts/train.sbatch configs/vanilla.yaml
# With overrides: sbatch scripts/train.sbatch configs/vanilla.yaml train.lr=1e-4
#
# The array index is the seed. Resubmitting the same command resumes from the last checkpoint.
# Run from the repo root, and create logs/ first (Slurm will not create it).
set -euo pipefail

# TODO(team): load modules / activate the shared environment here.

CONFIG=$1
shift
SEED=${SLURM_ARRAY_TASK_ID:-0}
NAME=$(basename "$CONFIG" .yaml)-s$SEED

python -m krs.train --config "$CONFIG" --set seed="$SEED" run_name="$NAME" "$@"
python -m krs.evaluate --ckpt "runs/$NAME/last.pt"
