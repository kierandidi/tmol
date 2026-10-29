#!/bin/bash
#SBATCH --job-name=relax_cold
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:h200:1
OPT=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt
export PYTHONPATH=$OPT/overlay TMOL_USE_JIT=1 TORCH_EXTENSIONS_DIR=$OPT/jitcache TORCH_CUDA_ARCH_LIST=9.0 TMOL_KIN_SCALE=0 TMOL_PACK_SA_HITEMP_TRAJ=100
for i in 1 2; do
 export NUMBA_CACHE_DIR=$(mktemp -d)   # cold numba cache for each run
 apptainer exec --nv /mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif $OPT/../env/bin/python $OPT/ab.py torsion 5uoi 1 2>&1 | rg '^AB' | cut -c1-200
done
