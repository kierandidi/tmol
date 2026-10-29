#!/bin/bash
#SBATCH --job-name=opt2_gpu_cart
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=05:00:00
#SBATCH --array=0-39%4
#SBATCH --gres=gpu:h200:1
set -uo pipefail
export PYTHONPATH=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/overlay TMOL_USE_JIT=1 TORCH_EXTENSIONS_DIR=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/jitcache TORCH_CUDA_ARCH_LIST=9.0 TMOL_KIN_SCALE=0 TMOL_PACK_SA_HITEMP_TRAJ=100
python3 /mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/src/dispatch.py matrix_opt2.json gpu_cart results/opt2_gpu_cart
