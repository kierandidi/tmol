#!/bin/bash
#SBATCH --job-name=pilot_cart2
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/logs/pilot_cart2-%A_%a.log
#SBATCH --gres=gpu:h200:1
set -euo pipefail
python3 /mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/src/dispatch.py matrix_pilot.json gpu_cart results/pilot2
