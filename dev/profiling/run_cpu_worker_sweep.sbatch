#!/usr/bin/env bash
#SBATCH --job-name=tmol-cpu-workers
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -euo pipefail

harness=${TMOL_PROFILE_HARNESS:?set TMOL_PROFILE_HARNESS}
source_dir=${TMOL_PROFILE_SOURCE:?set TMOL_PROFILE_SOURCE}
binary_source=${TMOL_PROFILE_BINARY_SOURCE:-${source_dir}}
environment=${TMOL_PROFILE_ENV:?set TMOL_PROFILE_ENV}
image=${TMOL_PROFILE_IMAGE:?set TMOL_PROFILE_IMAGE}
output=${TMOL_PROFILE_OUTPUT:?set TMOL_PROFILE_OUTPUT}

mkdir -p "${output}/results" "${output}/metadata"
binary_candidates=("${binary_source}"/tmol/_C*.so)
[[ -f ${binary_candidates[0]} ]] || {
    echo "No compiled tmol/_C extension under ${binary_source}" >&2
    exit 2
}
binary_so=${binary_candidates[0]}
binary_name=$(basename "${binary_so}")
binds="${harness}:/harness,${source_dir}:/source,${binary_so}:/source/tmol/${binary_name},${output}:/artifacts"

git -C "${source_dir}" rev-parse HEAD > "${output}/metadata/source_revision.txt"
git -C "${source_dir}" status --short > "${output}/metadata/source_status.txt"
lscpu > "${output}/metadata/lscpu.txt"
sha256sum "${binary_so}" > "${output}/metadata/binary_sha256.txt"

run_case() {
    local system=$1 threads=$2 workers=$3
    local pdb_candidates=(
        "${source_dir}"/tmol/tests/data/pdb/bysize_${system}_res_*.pdb
    )
    [[ -f ${pdb_candidates[0]} ]] || {
        echo "No source PDB matched size ${system}" >&2
        exit 2
    }
    local pdb=/source/tmol/tests/data/pdb/$(basename "${pdb_candidates[0]}")
    local slug=score-grad-protein${system}-t${threads}-w${workers}
    apptainer exec --bind "${binds}" --pwd /harness "${image}" \
        env PYTHONPATH=/source TMOL_USE_JIT=0 \
        TMOL_CPU_SCORE_TERM_WORKERS="${workers}" \
        OMP_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}" \
        OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        "${environment}/bin/python" dev/benchmarks/rosetta_comparison.py \
        --engine tmol --workflow score-grad --device cpu --threads "${threads}" \
        --pdb "${pdb}" --warmup 5 --repeats 15 \
        --output "/artifacts/results/${slug}.json"
}

for system in 100 400; do
    for threads in 2 4 8 16 32; do
        for workers in 1 2 4 8; do
            run_case "${system}" "${threads}" "${workers}"
        done
    done
done

apptainer exec --bind "${binds}" --pwd /source "${image}" \
    "${environment}/bin/python" \
    dev/profiling/summarize_cpu_worker_sweep.py \
    --input /artifacts/results --output /artifacts/summary.csv
