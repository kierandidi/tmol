#!/usr/bin/env bash
#SBATCH --job-name=tmol-ncu-profile
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail

harness=${TMOL_PROFILE_HARNESS:?set TMOL_PROFILE_HARNESS}
source_dir=${TMOL_PROFILE_SOURCE:?set TMOL_PROFILE_SOURCE}
environment=${TMOL_PROFILE_ENV:?set TMOL_PROFILE_ENV}
image=${TMOL_PROFILE_IMAGE:?set TMOL_PROFILE_IMAGE}
output=${TMOL_PROFILE_OUTPUT:?set TMOL_PROFILE_OUTPUT}

mkdir -p "${output}/ncu"
binds="${harness}:/harness,${source_dir}:/source,${output}:/artifacts"

apptainer exec --nv --bind "${binds}" --pwd /harness "${image}" \
    env TMOL_PROFILE_SOURCE=/source TMOL_USE_JIT=0 \
    "${environment}/bin/python" -c \
    'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'

apptainer exec --nv --bind "${binds}" --pwd /harness "${image}" \
    env TMOL_PROFILE_SOURCE=/source TMOL_USE_JIT=0 \
    ncu --target-processes all --profile-from-start off \
    --kernel-name-base demangled \
    --kernel-name 'regex:.*Annealer.*run_simulated_annealing.*' \
    --launch-count 1 --set full --force-overwrite \
    --export /artifacts/ncu/fast-relax-protein100-b1 \
    "${environment}/bin/python" dev/profiling/workflows.py \
    --workflow fast_relax --system protein100 --batch 1 \
    --warmup 0 --iterations 1 --capture \
    --output /artifacts/ncu/fast-relax-protein100-b1.json

apptainer exec --bind "${output}:/artifacts" "${image}" \
    ncu --import /artifacts/ncu/fast-relax-protein100-b1.ncu-rep \
    --page details --csv > "${output}/ncu/fast-relax-protein100-b1-details.csv"

python3 "${harness}/dev/profiling/summarize_ncu.py" \
    "${output}/ncu/fast-relax-protein100-b1-details.csv" \
    --output "${output}/ncu/fast-relax-protein100-b1-summary.md"
