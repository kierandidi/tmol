#!/bin/bash
#SBATCH --job-name=relax_jit
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h200:1
OPT=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt
export PYTHONPATH=$OPT/overlay TMOL_USE_JIT=1 TORCH_EXTENSIONS_DIR=$OPT/jitcache TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 TMOL_KIN_SCALE=0
time apptainer exec --nv /mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif $OPT/../env/bin/python $OPT/ab.py "$@" 2>&1 | rg 'AB|rror|Traceback|ninja|Loading|Building' | tail -20
