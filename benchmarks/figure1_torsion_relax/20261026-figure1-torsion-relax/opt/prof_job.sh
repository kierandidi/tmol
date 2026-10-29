#!/bin/bash
#SBATCH --job-name=relax_prof
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --array=0-5
#SBATCH --output=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/logs/prof-%A_%a.log
set -euo pipefail
cfg=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" /mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/prof_list.txt)
apptainer exec --nv /mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif /mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/env/bin/python /mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/prof_stages.py $cfg
