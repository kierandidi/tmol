#!/bin/bash
#SBATCH --job-name=opt_tests
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h200:1
export PYTHONPATH=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/overlay TMOL_USE_JIT=1 TORCH_EXTENSIONS_DIR=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/jitcache TORCH_CUDA_ARCH_LIST=9.0 TMOL_KIN_SCALE=0 MAX_JOBS=16
cd /tmp
apptainer exec --nv /mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif /mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/env/bin/python -m pytest -x -q -p no:cacheprovider --rootdir=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/overlay /mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/overlay/tmol/tests/optimization /mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/overlay/tmol/tests/relax /mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt/overlay/tmol/tests/kinematics 2>&1 | tail -40
