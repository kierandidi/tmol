#!/bin/bash
#SBATCH --job-name=full_gpu
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --array=0-47%4
#SBATCH --output=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261029-figure1-fixed50-min/logs/full_gpu-%A_%a.log
#SBATCH --gres=gpu:h200:1
set -euo pipefail
python3 /mnt/home/kdidi/tmol-paper-benchmark/runs/20261029-figure1-fixed50-min/src/dispatch.py matrix.json gpu results/raw
