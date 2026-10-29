#!/bin/bash
#SBATCH --job-name=relax_prof4
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h200:1
export PYTHONPATH=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/overlay
OPT=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt
for ch in 25 50 100; do
 echo "#### CHUNK=$ch"
 TMOL_PACK_MAX_POSES_PER_CHUNK=$ch apptainer exec --nv /mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif $OPT/../env/bin/python $OPT/prof_stages.py "$@" 2>&1 | rg 'pack_rotamers|TOTAL|rror|#'
done
