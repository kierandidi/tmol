#!/bin/bash
#SBATCH --job-name=relax_micro
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:h200:1
#SBATCH --output=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/logs/micro-%j.log
set -euo pipefail
export PYTHONPATH=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/overlay
apptainer exec --nv /mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif /mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/../env/bin/python /mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/micro.py $@
