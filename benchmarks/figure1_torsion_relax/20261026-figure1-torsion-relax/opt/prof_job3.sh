#!/bin/bash
#SBATCH --job-name=relax_prof3
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:h200:1
export PYTHONPATH=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/overlay
OPT=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt
for ks in 0 1; do
 echo "#### TMOL_KIN_SCALE=$ks"
 TMOL_KIN_SCALE=$ks apptainer exec --nv /mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif $OPT/../env/bin/python $OPT/prof_stages.py "$@" 2>&1 | rg -v 'CCD|PDB_MIRROR|VAR_NAME|\.env|Inaccurate'
done
