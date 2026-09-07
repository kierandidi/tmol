#!/usr/bin/env bash
#SBATCH --job-name=tmol-cpu-shard-ab
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00

set -euo pipefail

harness=${TMOL_PROFILE_HARNESS:?set TMOL_PROFILE_HARNESS}
source_dir=${TMOL_PROFILE_SOURCE:?set TMOL_PROFILE_SOURCE}
binary_source=${TMOL_PROFILE_BINARY_SOURCE:-${source_dir}}
environment=${TMOL_PROFILE_ENV:?set TMOL_PROFILE_ENV}
image=${TMOL_PROFILE_IMAGE:?set TMOL_PROFILE_IMAGE}
output=${TMOL_PROFILE_OUTPUT:?set TMOL_PROFILE_OUTPUT}

mkdir -p "${output}/results" "${output}/metadata"
binds="${harness}:/harness,${source_dir}:/source,${output}:/artifacts"
while IFS= read -r binary_so; do
    relative_binary=${binary_so#"${binary_source}"/}
    binds+=",${binary_so}:/source/${relative_binary}"
done < <(find "${binary_source}/tmol" -type f -name '*.so' | sort)

git -C "${source_dir}" rev-parse HEAD > "${output}/metadata/source_revision.txt"
git -C "${source_dir}" status --short > "${output}/metadata/source_status.txt"
lscpu > "${output}/metadata/lscpu.txt"
find "${binary_source}/tmol" -type f -name '*.so' -exec sha256sum {} + \
    > "${output}/metadata/binary_sha256.txt"

pdb_candidates=("${source_dir}"/tmol/tests/data/pdb/bysize_400_res_*.pdb)
[[ -f ${pdb_candidates[0]} ]] || exit 2
pdb=/source/tmol/tests/data/pdb/$(basename "${pdb_candidates[0]}")

run_case() {
    local threads=$1 shards=$2 replicate=$3
    local slug=score-grad-protein400-t${threads}-shards${shards}-r${replicate}
    apptainer exec --bind "${binds}" --pwd /harness "${image}" \
        env PYTHONPATH=/source TMOL_USE_JIT=0 \
        TMOL_FUSED_CPU_SHARDS="${shards}" \
        OMP_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}" \
        OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        "${environment}/bin/python" dev/benchmarks/rosetta_comparison.py \
        --engine tmol --workflow score-grad --device cpu --threads "${threads}" \
        --pdb "${pdb}" --warmup 7 --repeats 51 \
        --output "/artifacts/results/${slug}.json"
}

for replicate in 1 2; do
    order4=(2 4)
    order8=(4 8)
    if [[ ${replicate} == 2 ]]; then
        order4=(4 2)
        order8=(8 4)
    fi
    for shards in "${order4[@]}"; do
        run_case 4 "${shards}" "${replicate}"
    done
    for shards in "${order8[@]}"; do
        run_case 8 "${shards}" "${replicate}"
    done
done
