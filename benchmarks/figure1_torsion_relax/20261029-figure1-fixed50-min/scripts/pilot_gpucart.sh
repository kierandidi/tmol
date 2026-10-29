#!/bin/bash
#SBATCH --job-name=pilot_gpucart
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261029-figure1-fixed50-min/logs/pilot_gpucart-%A_%a.log
#SBATCH --gres=gpu:h200:1
set -euo pipefail
python3 /mnt/home/kdidi/tmol-paper-benchmark/runs/20261029-figure1-fixed50-min/src/dispatch.py matrix_pilot.json gpu_cart results/pilot
