# CPU scoring concurrency experiments

These Slurm jobs compare three CPU execution choices without changing score
semantics:

- `run_cpu_worker_sweep.sbatch` compares the current fused-neighbor sharding
  path with independent score-term workers at 2, 4, 8, 16, and 32 cores.
- `run_cpu_shard_sweep.sbatch` scans 2, 4, 8, and 16 fused-neighbor shards.
- `run_cpu_shard_ab.sbatch` repeats the only promising shard-count comparisons
  in reversed order on one allocation, so node load and ordering are not
  mistaken for a speedup.

All jobs require `TMOL_PROFILE_HARNESS`, `TMOL_PROFILE_SOURCE`,
`TMOL_PROFILE_ENV`, `TMOL_PROFILE_IMAGE`, and `TMOL_PROFILE_OUTPUT`.
`TMOL_PROFILE_BINARY_SOURCE` may point to a clean, identically based Release
build when the experiment changes Python dispatch only. Each job records its
source revision, source status, CPU description, and hashes of every mounted
extension module.

The September 2026 H200-host CPU runs are under:

- `/mnt/data/kdidi/tmol-pr468-cpu-worker-sweep-v2`
- `/mnt/data/kdidi/tmol-pr468-cpu-shard-sweep`
- `/mnt/data/kdidi/tmol-pr468-cpu-shard-ab`

The corrected worker sweep shows that fused sharding is substantially faster
than term-only parallelism. The reversed shard A/B shows no benefit from
increasing the current adaptive shard count: at 370 residues, 4 versus 2
shards at 4 cores gives 0.987x throughput, and 8 versus 4 shards at 8 cores
gives 0.923x. Therefore these controls do not justify a production default
change.
