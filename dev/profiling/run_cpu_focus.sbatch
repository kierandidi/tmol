#!/usr/bin/env bash
#SBATCH --job-name=tmol-cpu-profile
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail

harness=${TMOL_PROFILE_HARNESS:?set TMOL_PROFILE_HARNESS}
source_dir=${TMOL_PROFILE_SOURCE:?set TMOL_PROFILE_SOURCE}
environment=${TMOL_PROFILE_ENV:?set TMOL_PROFILE_ENV}
image=${TMOL_PROFILE_IMAGE:?set TMOL_PROFILE_IMAGE}
output=${TMOL_PROFILE_OUTPUT:?set TMOL_PROFILE_OUTPUT}

mkdir -p "${output}/results" "${output}/operators"
binds="${harness}:/harness,${source_dir}:/source,${output}:/artifacts"

run_case() {
    local system=$1 threads=$2
    local pdb_candidates=(
        "${source_dir}"/tmol/tests/data/pdb/bysize_${system}_res_*.pdb
    )
    [[ -f ${pdb_candidates[0]} ]] || {
        echo "No source PDB matched size ${system}" >&2
        exit 2
    }
    local pdb=/source/tmol/tests/data/pdb/$(basename "${pdb_candidates[0]}")
    local slug=score-grad-protein${system}-t${threads}
    apptainer exec --bind "${binds}" --pwd /harness "${image}" \
        env PYTHONPATH=/source TMOL_USE_JIT=0 \
        OMP_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}" \
        OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        "${environment}/bin/python" dev/benchmarks/rosetta_comparison.py \
        --engine tmol --workflow score-grad --device cpu --threads "${threads}" \
        --pdb "${pdb}" --warmup 5 --repeats 15 --profile-repeats 3 \
        --profile-dir "/artifacts/operators/${slug}" \
        --output "/artifacts/results/${slug}.json"
}

for system in 100 400; do
    run_case "${system}" 1
    run_case "${system}" 16
done
