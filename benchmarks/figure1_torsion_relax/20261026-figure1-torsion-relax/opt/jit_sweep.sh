#!/bin/bash
#SBATCH --job-name=relax_jsw
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h200:1
OPT=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt
export PYTHONPATH=$OPT/overlay TMOL_USE_JIT=1 TORCH_EXTENSIONS_DIR=$OPT/jitcache TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16 TMOL_KIN_SCALE=0
run(){ apptainer exec --nv /mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif $OPT/../env/bin/python $OPT/ab.py "$@" 2>&1 | rg '^AB|Traceback|rror' | sed "s/^AB /AB hi=$TMOL_PACK_SA_HITEMP_TRAJ exp=$TMOL_PACK_SA_EXPANSIONS /"; }
export TMOL_PACK_SA_HITEMP_TRAJ=500 TMOL_PACK_SA_EXPANSIONS=10; run torsion 5yzf 10 >/dev/null   # build / warm
for cfg in "500 10" "500 10" "100 10" "100 4" "50 4" "25 4"; do set -- $cfg
  export TMOL_PACK_SA_HITEMP_TRAJ=$1 TMOL_PACK_SA_EXPANSIONS=$2
  for b in "torsion 5yzf 10" "cartesian 5yzf 10"; do run $b; done
done
