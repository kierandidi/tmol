#!/bin/bash
#SBATCH --job-name=relax_master
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h200:1
OPT=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt
export PYTHONPATH=$OPT/overlay_master TMOL_USE_JIT=1 TORCH_EXTENSIONS_DIR=$OPT/jitcache_master TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 TMOL_KIN_SCALE=0
run(){ apptainer exec --nv /mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif $OPT/../env/bin/python $OPT/ab.py "$@" 2>&1 | rg '^AB|Traceback|rror' | sed "s/^AB /AB tag=$TAG /"; }
TAG=warm run torsion 5yzf 10 > /dev/null
for cfg in "m_default:500:0" "m_valueonly:500:1" "m_sa100:100:0" "m_sa100_valueonly:100:1"; do IFS=: read t hi vo <<< "$cfg"
  export TMOL_PACK_SA_HITEMP_TRAJ=$hi TMOL_VALUE_ONLY_LS=$vo
  for b in "torsion 5uoi 1" "torsion 5yzf 10" "torsion 5yzf 100"; do TAG=$t run $b; done
done
