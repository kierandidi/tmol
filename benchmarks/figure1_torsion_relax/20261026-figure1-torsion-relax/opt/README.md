# Torsion-relax optimisation experiments (scratch; nothing committed or pushed)

Base: tmol v0.2.0 wheel python tree copied to `overlay/` (patch vs pristine: `patches/v0.2.0_experimental.patch`),
and current uw-ipd/tmol master 1fbd25f34 (includes #615) copied to `overlay_master/`
(`patches/master_1fbd25f34_experimental.patch`). Kernels were JIT-compiled (TMOL_USE_JIT=1, sm_90; JIT build
matches the AOT wheel's speed within noise). No branch of the fork was touched.

Diagnosis (5yzf, H200; scripts prof_stages.py, micro.py, noise.py, gradcheck.py, dbg_ls.py):
* One-time cost: ~3 s of numba JIT in `validate_fold_forest` at the first kinematic network build (B1: 4.4 s torsion vs 2.3 s Cartesian).
* Packing is mode-independent and GPU-bound; it is ~65% of FastRelax at B>=10 (0.1 s/pose/stage, linear in poses, chunk size irrelevant).
  74% of packing is the SA kernel: 500 hi-temp + 1250 lo-temp + 312 quench trajectories per pose, of which only the best is used.
* Steady-state kinematics are cheap (0.33 ms of 7.5 ms per fwd+bwd at B100); the torsion-vs-Cartesian gap at batch is more line-search
  evaluations per iteration (~2-3 vs 1.5): torsion space is stiff (gradient verified correct vs finite differences, but curvature huge),
  and a few poses fail the Armijo search after ~20 backtracks, each of which costs a whole-batch evaluation.
* Energy evaluations are non-deterministic at ~1e-3 (float32 atomics), same for Cartesian.

What helped (patches): numba cache + skip re-validating `reasonable_fold_forest` (-2..-3 s one-time);
runtime knobs TMOL_PACK_SA_HITEMP_TRAJ / TMOL_PACK_SA_EXPANSIONS for the annealer (100 trajectories: -31% at B100, -16% at B10, same final energies);
value-only line-search trials (own patch on v0.2.0: ~-6% at B100; on master with #615's `value_only_line_search`: within +-3%, no consistent gain).
What did not help: Jacobian diagonal DOF scaling (less progress per 200 iterations), capping line-search trials (<=3%), CUDA graph for torsion (no change), larger packing chunks (no change).
Results: `../summary/optimised_vs_baseline.{md,csv}` (results/opt2_gpu*, 80 points, all ok).
