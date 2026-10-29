#!/bin/bash
#SBATCH --job-name=x_cpu
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261029-figure1-fixed50-min/logs/x_cpu-%A_%a.log

set -euo pipefail
python3 /mnt/home/kdidi/tmol-paper-benchmark/runs/20261029-figure1-fixed50-min/src/dispatch.py matrix_extra.json cpu results/raw
