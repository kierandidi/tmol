# Benchmarking

Use this guide for implementation-regression benchmarks. For application
throughput, follow the batching recipe and tutorial instead.

> - **Prerequisites:** A development installation; see {doc}`Development
>   </user_guide/development>`.
> - **Deep tutorial:** {doc}`02 — GPU Batching with TMol
>   </tutorial/02_gpu_batching>` for application measurements.
> - **Related workflow:** {doc}`GPU batching </workflows/gpu_batching>`.
> - **API reference:** {doc}`Pose </api/pose>` and {doc}`Scoring </api/score>`.
> - **Rosetta mapping:** {doc}`GPU batching and external orchestration
>   </tutorial/rosetta_crosswalk>`.

Performance work happens at two different levels:

- Application-level GPU batching measures a real workload over a `PoseStack`;
  use the {doc}`GPU batching workflow </workflows/gpu_batching>` for batch
  construction, synchronized timing, memory measurement, and chunking.
- The developer benchmark harness runs pytest benchmark cases to detect kernel
  or implementation regressions across code revisions.

The harness is not an application scheduler and its microbenchmark results do
not choose a production batch size.

## Running Developer Benchmarks

Use `dev/bin/benchmark` with pytest selectors:

```bash
dev/bin/benchmark tmol/tests/score -k cuda-full-lk_ball
```

The wrapper enables pytest benchmarks, prints a summary, and writes JSON results
under `dev/benchmark/`.

## Comparing Revisions

`dev/bin/compare_benchmark` compares benchmark results across revisions. Put
pytest arguments first, then revisions after `--`.

```bash
dev/bin/compare_benchmark tmol/tests/score -k cuda-full-lk_ball -- origin/master
```

The meta-revision `TREE` means the current working tree:

```bash
dev/bin/compare_benchmark tmol/tests/score -k cuda-full-lk_ball -- TREE HEAD
```

Ancillary benchmark plots live near the tests as `plot_*.py` scripts.

## Comparing Rosetta-family CPU workflows

`dev/benchmarks/rosetta_comparison.py` measures application-level CPU latency
and throughput for pose construction, full scoring,
score-plus-coordinate-gradient evaluation, fixed-iteration Cartesian
minimization, and fixed-sequence repacking. Run each engine in a separate
process so native runtimes, allocators, and thread pools do not interfere:

```bash
OMP_NUM_THREADS=1 python dev/benchmarks/rosetta_comparison.py \
  --engine tmol --workflow score --threads 1 --output tmol-score.json

PYTHONPATH=/path/to/pyrosetta python \
  dev/benchmarks/rosetta_comparison.py \
  --engine pyrosetta --workflow score --threads 1 \
  --output pyrosetta-score.json
```

The score workload clears Rosetta's complete pose-associated energy graph
before every measurement so it cannot return cached work; TMol scoring is
stateless. PyRosetta score-plus-gradient measurements use Rosetta's native C++
`CartesianMultifunc`, not a Python atom loop. The TMol runner uses `beta2016`,
and the Rosetta runners use `beta_nov16_cart`. These are analogous all-atom
functions, but their numerical scores and protocol outcomes are not
interchangeable.

Use both single-pose latency and batched throughput cases. A fair CPU report
also states thread count, affinity, warm-up count, structure size, and whether
setup is included. The JSON output records each of these fields. PyRosetta's
minimization and packing calls exercise the Rosetta C++ core through Python;
when licensed Rosetta applications are installed, `--engine rosetta` can also
measure end-to-end `score_jd2` process latency with `--rosetta-bin-dir`.

Never infer batch scaling from a single-pose core-scaling result. In
particular, a batch-1 Cartesian minimization comparison says nothing about
TMol's segmented batched minimizer versus PyRosetta's serial pose loop. Sweep
batch size for each workflow and label both dimensions in summary tables.

Cartesian minimization has two intentionally separate TMol modes:

- The default runner measures the one-shot `run_cart_min` API, including
  topology-dependent scorer rendering on every timed call.
- `--reuse-topology` retains the rendered scorer between compatible calls and
  measures the steady-state path used by repeated inference protocols. On
  CUDA, combine it with `--cuda-graph` to amortize graph capture and reduce
  launch overhead. Do not use `--cuda-graph` for a one-off call without also
  reporting its larger cold-call cost.

The JSON distinguishes `workload_setup_seconds`, `first_call_seconds`, warm-up
samples, and steady-state timing. Throughput comparisons must say which of
these costs they include. For direct application code, reuse a minimizer when
pose topology is stable:

```python
from tmol.optimization import CartesianMinimizer

minimize = CartesianMinimizer(cuda_graph=pose_stack.device.type == "cuda")
for pose_stack in same_topology_pose_stacks:
    result = minimize(pose_stack, score_function)
```

The comparison runner requests `fixed_iterations=True` from TMol's
`LBFGS_Armijo`. This both enforces the protocol's stated iteration count and
skips convergence reductions and CPU/GPU synchronization that would otherwise
make a fixed-iteration benchmark data-dependent. Normal application
minimization retains convergence checking by default.

Collect separate JSON records at each allocated core count, then generate
latency, speedup, parallel-efficiency, and TMol-versus-competitor plots:

```bash
python dev/benchmarks/plot_rosetta_comparison.py results/scaling \
  --output-dir results/plots
```

The plotter emits PNG, SVG, and PDF figures plus the underlying summary CSVs.
Keep single-pose scaling separate from independent-pose batch throughput:
PyRosetta and many Rosetta applications do not parallelize one ordinary
`ScoreFunction` evaluation merely because more threads were allocated.

For a batch-size sweep, generate throughput, per-pose latency, total latency,
and cold-to-steady-state plots with:

```bash
python dev/benchmarks/plot_batch_scaling.py results/batches \
  --output-dir results/batch-plots
```

For TMol CUDA runs, pass `--device cuda`. The runner synchronizes the selected
device immediately around every timed iteration so the reported time includes
completed kernel work rather than asynchronous launch latency. Rosetta and
PyRosetta runs accept CPU only.

Add `--profile-dir profile/score-b32` to a TMol invocation for an operator CSV,
a readable top-100 table, metadata, and a Chrome trace. Profiling is performed
after normal timing so profiler instrumentation does not contaminate the
reported latency. Use `--profile-repeats` sparingly because shape and memory
recording are intentionally detailed.

### CUDA graphs and eager latency

A CUDA graph does not make individual score kernels intrinsically faster. It
records the fixed launch dependency graph once and replays it from CUDA,
amortizing Python, PyTorch dispatcher, autograd-engine, and CUDA launch setup.
It is therefore most valuable for a small, repeatedly executed workload. At a
large batch, kernel execution dominates total time and eager execution is much
closer to graph replay.

The eager alternatives are complementary rather than equivalent: reuse
topology-dependent renderings and optimizer scratch, skip convergence checks
for explicitly fixed-iteration protocols, and fuse sequences of small
optimizer or score operations where profiling demonstrates launch-bound work.
Improving arithmetic inside one kernel cannot remove the host overhead of the
other launches. Prefer graph replay when inputs have stable shapes and storage;
prefer eager mode for dynamic control flow, changing shapes, or one-off calls
where capture cost cannot be amortized.

Compare event counts from two or more Chrome traces with:

```bash
python dev/benchmarks/plot_profile_launches.py \
  --profile eager=profile/eager \
  --profile graph=profile/graph \
  --output-dir profile/launch-plots
```

The event-count figure explains launch structure; it is not a latency result.
Use the runner's synchronized timing JSON for latency and throughput claims.

When measuring throughput scaling for independent structures, run one
single-threaded worker per allocated core for both engines. Aggregate those
worker records separately from within-call scaling with:

```bash
python dev/benchmarks/plot_process_scaling.py \
  --tmol results/tmol-processes --pyrosetta results/pyrosetta-processes \
  --output-dir results/process-plots
```

For implementation work, collect at least two interleaved processes for the
unchanged baseline and candidate, then plot their median-of-process medians:

```bash
python dev/benchmarks/plot_tmol_ab.py \
  --baseline results/baseline-*.json \
  --candidate results/candidate-*.json \
  --candidate-label neighbor-scratch --output-dir results/ab-plots
```

## Profiling

`dev/bin/profile_benchmark` runs a short pytest benchmark under Nsight Systems
by default:

```bash
dev/bin/profile_benchmark --output profile/ljlk \
  tmol/tests/score -k cuda-full-ljlk
```

Use Nsight Compute when kernel-level counters are needed; arguments after `--`
are forwarded to the profiler:

```bash
dev/bin/profile_benchmark --tool ncu --output profile/ljlk-kernels \
  tmol/tests/score -k cuda-forward-ljlk-100 -- \
  --kernel-name regex:ljlk --launch-count 20
```

The output prefix defaults to `dev/profile/<host>/<UTC timestamp>`. Keep pytest
selectors narrow: profiling every parametrized benchmark produces a very large
trace and makes hardware-counter collection unnecessarily slow. If the wrapper
is not launched from the development environment, pass its interpreter with
`--python /path/to/venv/bin/python` or set `TMOL_PROFILE_PYTHON`.
