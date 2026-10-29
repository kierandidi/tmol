#!/bin/bash
#SBATCH --job-name=relax_ab
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h200:1
set -uo pipefail
# usage: sbatch ab_job.sh <tag> <overlay-dir|none> "<mode ds batch [graph|nograph]>" ...
OPT=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt
tag=$1; ov=$2; shift 2
[ "$ov" != none ] && export PYTHONPATH=$ov
export TMOL_KIN_SCALE=${KS:-1}
for cfg in "$@"; do
  for rep in 1 2; do
    apptainer exec --nv /mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif $OPT/../env/bin/python $OPT/ab.py $cfg 2>&1 | rg '^AB|Traceback|Error' | sed "s/^AB /AB tag=$tag rep=$rep /"
  done
done
