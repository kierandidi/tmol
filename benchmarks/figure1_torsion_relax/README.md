# Figure 1 FastRelax benchmarks: torsion-space (atom-tree) relax, 50-step minimization, all inputs

Snapshot of work in progress, taken while the last large-batch GPU jobs were still running (see "Status"). Benchmark
bundle only: no tmol library code is changed on this branch (the experimental tmol patches are included as `.patch`
files for reference, not applied).

Everything ran on CoreWeave east (Slurm `hpc-high`, H200 GPUs) with tmol v0.2.0 release wheels (sha256
`fe0d4661...` / `34d03dc6...`, matching the GitHub release digests), PyTorch 2.13, PyRosetta 2024.39. Absolute
`/mnt/home/kdidi/...` paths in the JSON/CSV files are the cluster provenance paths.

## Why

Earlier Figure 1 FastRelax runs (`runs/20261005-fastrelax-f92d`, tmol 0.1.58) used **Cartesian** relax on both engines.
These runs redo the FastRelax series with torsion-space relax and add a fixed-step variant.

## Directories

| directory | what |
|---|---|
| `20261026-figure1-torsion-relax/` | torsion relax vs the old Cartesian runs, 200-step default minimizer (87/87 points), same-version Cartesian control, and `opt/` (profiling + optimisation experiments, patches) |
| `20261029-figure1-fixed50-min/` | same benchmark with exactly 50 minimization steps for every method; extended to all 27 inputs; score+gradient panel for all 27 inputs |

Each run has `metadata/rerun_spec.json` (versions, settings, Slurm job ids), `summary/` (CSV in the old
`fastrelax_latest/timing_summary.csv` schema plus comparison tables), `results/` (raw per-run JSON incl. node/GPU),
`src/` + `scripts/` (harness and Slurm scripts) and `matrix*.json` (what was run). Inputs (`data/`), the venv, wheels
and logs are not included.

## Definitions

* **tmol torsion**: `tmol.relax.fast_relax(num_repeats=1)` with `KinematicMinimizer`/`DefaultKinematicMinimizer`,
  `MoveMap` with all named torsions + non-root jumps free, `FoldForest.reasonable_fold_forest`, score function beta2016.
* **tmol Cartesian**: default Cartesian minimizer (`CartesianMoveMap`), same score function.
* **PyRosetta torsion**: `FastRelax` + MonomerRelax2019 lines, `cartesian(False)`, bb/chi/jump free, `beta_nov16`
  (`beta_genpot` for protein-ligand). **PyRosetta Cartesian**: `cartesian(True)`, `*_cart` score functions. The two
  weight files differ only by `pro_close` 1.25 vs 0 and `cart_bonded` 0.5.
* **Timing contract** (unchanged): one seeded one-repeat FastRelax per point, input/scorer construction outside the
  interval, no warm-up (first-call costs included for both modes), GPU synchronised, one process per point.
* **50-step run**: tmol `LBFGS_Armijo(max_iter=50, fixed_iterations=True)` per minimization (verified: exactly 50
  iterations); PyRosetta `FastRelax.max_iter(50)` with the script's `min` tolerance set to 1e-12 (the cap, not
  convergence, ends the minimization; exactness of 50 cannot be guaranteed as for tmol).

## Results (median over the 12 original systems, seconds per structure)

200-step default minimizer (`20261026-figure1-torsion-relax/summary/comparison_old_cartesian_vs_torsion.md`):

| series | old Cartesian (tmol 0.1.58) | torsion (tmol 0.2.0) | torsion / Cartesian, same tmol version | PyRosetta speedup: old Cart. -> torsion |
|---|---:|---:|---:|---:|
| PyRosetta CPU | 350 | 116 | | |
| tmol CPU, 32 thr | 33 | 36 | 1.05 | 10.6x -> 3.2x |
| tmol GPU B1 | 8.0 | 9.3 | 1.53 | 42x -> 12x |
| tmol GPU B10 | 2.8 | 2.6 | 1.23 | 134x -> 45x |
| tmol GPU B100 | 2.4 | 1.9 | 1.10 | 170x -> 60x |
| tmol GPU B1000 | 1.5 | 2.0 | 1.15 | 140x -> 58x |

Fixed 50 steps (`20261029-figure1-fixed50-min/summary/comparison.md`, 12-system medians):

| series | torsion 50 | Cartesian 50 | PyRosetta speedup torsion / Cartesian |
|---|---:|---:|---:|
| PyRosetta CPU | 26.7 | 29.7 | |
| tmol CPU, 32 thr | 18.0 | 13.8 | 1.4x / 2.1x |
| tmol GPU B1 | 8.1 | 4.8 | 3.1x / 5.6x |
| tmol GPU B10 | 2.1 | 1.8 | 12x / 15x |
| tmol GPU B100 | 1.5 | 1.5 | 17x / 21x |
| tmol GPU B1000 | 1.5 | 1.4 | 18x / 22x |

Why torsion is slower than Cartesian in tmol (profiled on 5yzf, see `20261026-figure1-torsion-relax/opt/README.md`):
~3 s one-time numba JIT in the fold-forest validator at the first kinematic build; packing (identical in both modes,
GPU-bound, ~65% of FastRelax at B>=10) dominated by the annealer, which runs 500 high-temperature trajectories per
pose of which only the best is used; more line-search evaluations per iteration in the stiff torsion space.
Experimental patches (`opt/patches/`): numba cache, skipping re-validation of `reasonable_fold_forest`, annealer
trajectory-count env knobs, value-only line-search trials. Median speedups 1.2-1.5x on 12 systems (B1..B1000), with
final energies equivalent (`20261026-figure1-torsion-relax/summary/optimised_vs_baseline.md`). The patches are not
part of any Figure 1 number, and the annealer-budget reduction changes the algorithm and needs a decision.

## Figures (`20261029-figure1-fixed50-min/figure/`)

* `fastrelax_50steps.{png,svg,pdf}`: FastRelax panel, Cartesian | torsion, 50 steps, all inputs available at snapshot.
* `score_gradient_all_inputs.{png,svg,pdf}`: score+gradient panel with all 27 inputs (tmol 0.2.0 re-measured with the
  0.1.58 harness's timing contract; PyRosetta values reused from the earlier Figure 1 data).
* The score-term-agreement (Pearson) panel was not re-run; it still has the earlier 26 structures (hivrt missing).
  Plot scripts: `plot_fastrelax_50.py`, `plot_score_gradient_all.py` (`uv run --with matplotlib --with numpy`).

## Notes and caveats

* **hivrt**: tmol refuses the default load (chain gap between residues 537/538). hivrt was run with
  `missing_density_distance_threshold=0`, which keeps the declared long bond. This is a comparison control, not a
  physical repair; hivrt points are therefore flagged and can be dropped from plots.
* tmol 0.2.0 renamed loader functions; `src/loader020.py` ports the harness with identical options. Protein-ligand
  initial scores differ between 0.1.58 and 0.2.0 (e.g. hsp90 386 vs 313), so compare old-vs-new tmol energies only
  within the same version (the Cartesian control in `results/raw_cart020`).
* tmol and PyRosetta final energies are not directly comparable (tmol keeps `cart_*` bonded terms in torsion mode;
  PyRosetta torsion uses `pro_close`).
* The same-version 200-step Cartesian control ran out of GPU memory at B1000 for 1o7j and 6q1h, so the
  "200-step Cartesian B1000" median is over 10 systems; the 50-step runs completed them.
* Timings are n=1 per point (no uncertainty estimate), as in the earlier figure.
* The earlier Figure 1 FastRelax data snapshot was 68/87 complete; this comparison uses the later complete raw directory.

## Status at snapshot

* 200-step torsion run and 50-step run on the original 12 tmol systems / 27 PyRosetta inputs: complete (87 + 87 points).
* Extension of the 50-step run to the 15 additional tmol systems (`matrix_extra.json`) and the hivrt reruns
  (`matrix_hivrt.json`): 155/162 points per mode done, the remaining points are large-batch GPU jobs still running.
  Rerun `scripts/summarize.py` and the plot scripts in the run directory when they finish.
* Score+gradient for all 27 inputs: complete (135 tmol points).
