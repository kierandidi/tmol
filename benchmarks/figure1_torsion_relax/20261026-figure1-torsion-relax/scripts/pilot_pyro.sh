#!/bin/bash
#SBATCH --job-name=pilot_pyro
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/logs/pilot_pyro-%A_%a.log

set -euo pipefail
python3 /mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/src/dispatch.py matrix_pilot.json pyro results/pilot
