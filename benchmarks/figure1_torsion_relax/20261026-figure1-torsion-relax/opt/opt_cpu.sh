#!/bin/bash
#SBATCH --job-name=opt_cpu
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --array=0-1%2
set -uo pipefail
export PYTHONPATH=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/overlay TMOL_USE_JIT=1 TORCH_EXTENSIONS_DIR=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/jitcache TORCH_CUDA_ARCH_LIST=9.0 TMOL_KIN_SCALE=0
export TMOL_PACK_SA_HITEMP_TRAJ=${HI:-100} TMOL_PACK_SA_EXPANSIONS=${EXP:-10}
python3 /mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/src/dispatch.py matrix_opt.json cpu results/opt_${TAG:-sa100}
