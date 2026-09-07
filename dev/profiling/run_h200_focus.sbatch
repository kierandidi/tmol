#!/usr/bin/env bash
#SBATCH --job-name=tmol-h200-profile
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -euo pipefail

harness=${TMOL_PROFILE_HARNESS:?set TMOL_PROFILE_HARNESS}
source_tree=${TMOL_PROFILE_SOURCE:?set TMOL_PROFILE_SOURCE}
environment=${TMOL_PROFILE_ENV:?set TMOL_PROFILE_ENV}
image=${TMOL_PROFILE_IMAGE:?set TMOL_PROFILE_IMAGE}
output=${TMOL_PROFILE_OUTPUT:?set TMOL_PROFILE_OUTPUT}

mkdir -p "${output}" "${output}/logs"

run_matrix() {
    local mode=$1
    local destination=$2
    local trace_args=()
    if [[ ${mode} == trace ]]; then
        trace_args=(--trace)
    fi
    apptainer exec --nv \
        --bind "${harness}:/harness,${source_tree}:/candidate,${output}:/artifacts" \
        --pwd /harness "${image}" \
        env TMOL_PROFILE_SOURCE=/candidate TMOL_USE_JIT=0 \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "${environment}/bin/python" /harness/dev/profiling/matrix.py \
        --output-dir "/artifacts/${destination}" "${trace_args[@]}" \
        --case score_grad-protein100-b1 \
        --case score_grad-protein100-b16 \
        --case score_grad-dna24-b1 \
        --case score_grad-dna24-b32 \
        --case score_grad-protein_ligand-b1 \
        --case score_grad-protein_ligand-b8 \
        --case fast_relax-protein100-b1 \
        --case fast_relax-protein100-b4
}

run_matrix timing timing
run_matrix trace nsys

apptainer exec \
    --bind "${harness}:/harness,${output}:/artifacts" \
    --pwd /harness "${image}" \
    "${environment}/bin/python" /harness/dev/profiling/analyze.py \
    /artifacts/timing
apptainer exec \
    --bind "${harness}:/harness,${output}:/artifacts" \
    --pwd /harness "${image}" \
    "${environment}/bin/python" /harness/dev/profiling/analyze.py \
    /artifacts/nsys --baseline /artifacts/timing
