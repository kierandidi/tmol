#!/usr/bin/env python3
"""Compare common TMol and Rosetta-family workflows.

The benchmark deliberately runs one engine per process.  PyRosetta and TMol
both load large native runtimes, and mixing them in one interpreter changes
startup, allocator, and thread-pool behavior.

Examples::

    python dev/benchmarks/rosetta_comparison.py --engine tmol --workflow score
    PYTHONPATH=/path/to/pyrosetta python \
      dev/benchmarks/rosetta_comparison.py --engine pyrosetta --workflow score

Scores are not compared numerically: TMol beta2016 and Rosetta
beta_nov16_cart are related implementations, not guaranteed score-unit parity.
The comparable quantities are workload latency and throughput.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDB = ROOT / "tmol/tests/data/pdb/1ubq.pdb"
WORKFLOWS = ("pose-build", "score", "score-grad", "cart-min", "repack")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _timing_summary(samples: list[float], batch: int) -> dict[str, float]:
    median = statistics.median(samples)
    return {
        "median_seconds": median,
        "p25_seconds": _percentile(samples, 0.25),
        "p75_seconds": _percentile(samples, 0.75),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "median_seconds_per_pose": median / batch,
        "poses_per_second": batch / median,
    }


def _git_revision(root: Path = ROOT) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _host_metadata() -> dict[str, Any]:
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    )
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    except OSError:
        pass
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "affinity": affinity,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
    }


@dataclass
class Workload:
    setup_seconds: float
    run: Callable[[int], Any]
    metadata: dict[str, Any]
    synchronize: Callable[[], None]


def _tmol_workload(args: argparse.Namespace) -> Workload:
    import attrs
    import torch
    import tmol

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    from tmol.database import ParameterDatabase
    from tmol.io import pose_stack_from_pdb
    from tmol.optimization import CartesianMinimizer, run_cart_min
    from tmol.pack import PackerPalette, PackerTask, pack_rotamers
    from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler
    from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
    from tmol.pose import PoseStackBuilder
    from tmol.score import beta2016_score_function

    pdb_text = args.pdb.read_text()
    device = torch.device(args.device)
    synchronize = (
        (lambda: torch.cuda.synchronize(device))
        if device.type == "cuda"
        else (lambda: None)
    )

    def build_pose():
        single = pose_stack_from_pdb(pdb_text, device)
        if args.batch == 1:
            return single
        return PoseStackBuilder.from_poses([single] * args.batch, device)

    setup_start = time.perf_counter()
    pose = build_pose()
    param_db = ParameterDatabase.get_default()
    score_function = beta2016_score_function(device, param_db=param_db)
    scorer = score_function.render_whole_pose_scoring_module(pose)
    dun_sampler = None
    if args.workflow == "repack":
        dun_sampler = create_dunbrack_sampler_from_database(param_db, device)
    setup_seconds = time.perf_counter() - setup_start

    if args.workflow == "pose-build":

        def run(_: int):
            return build_pose()

    elif args.workflow == "score":
        coords = pose.coords.clone()

        def run(_: int):
            # TMol scoring is stateless: every call performs a complete score.
            with torch.no_grad():
                return scorer(coords)

    elif args.workflow == "score-grad":
        coords = pose.coords.detach().clone().requires_grad_(True)

        def run(_: int):
            coords.grad = None
            score = scorer(coords).sum()
            score.backward()
            return coords.grad

    elif args.workflow == "cart-min":
        reusable_minimizer = (
            CartesianMinimizer(cuda_graph=args.cuda_graph)
            if args.reuse_topology
            else None
        )

        def run(_: int):
            kwargs = {
                "optimizer_kwargs": {
                    "max_iter": args.protocol_iterations,
                    "gradtol": 0.0,
                    "atol": 0.0,
                    "rtol": 0.0,
                    "fixed_iterations": True,
                }
            }
            if reusable_minimizer is not None:
                # Copy changing coordinates while retaining immutable topology
                # object identity, as repeated inference protocols normally do.
                input_pose = attrs.evolve(pose, coords=pose.coords.detach().clone())
                return reusable_minimizer(input_pose, score_function, **kwargs)
            return run_cart_min(
                pose.clone(),
                score_function,
                cuda_graph=args.cuda_graph,
                **kwargs,
            )

    elif args.workflow == "repack":
        assert dun_sampler is not None

        def run(_: int):
            input_pose = pose.clone()
            task = PackerTask(input_pose, PackerPalette())
            task.restrict_to_repacking()
            task.add_conformer_sampler(dun_sampler)
            task.add_conformer_sampler(FixedAAChiSampler())
            task.add_conformer_sampler(IncludeCurrentSampler())
            return pack_rotamers(input_pose, score_function, task)

    else:  # pragma: no cover - argparse enforces this
        raise AssertionError(args.workflow)

    return Workload(
        setup_seconds=setup_seconds,
        run=run,
        synchronize=synchronize,
        metadata={
            "tmol_version": tmol.__version__,
            "tmol_revision": _git_revision(Path(tmol.__file__).resolve().parents[1]),
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "score_function": "beta2016",
            "reuse_topology": args.reuse_topology,
            "cuda_graph": args.cuda_graph,
            "fixed_iterations": args.workflow == "cart-min",
            "n_residues": int(pose.n_res_per_pose[0]),
            "n_atoms": int(pose.coords.shape[1]),
        },
    )


def _pyrosetta_workload(args: argparse.Namespace) -> Workload:  # noqa: C901
    if args.device != "cpu":
        raise ValueError("PyRosetta supports --device cpu only")
    if args.pyrosetta_root is not None:
        sys.path.insert(0, str(args.pyrosetta_root))

    import pyrosetta

    init_start = time.perf_counter()
    pyrosetta.init(
        "-mute all -constant_seed -jran 111111 "
        "-corrections:beta_nov16 true "
        f"-multithreading:total_threads {args.threads}"
    )
    init_seconds = time.perf_counter() - init_start

    rosetta = pyrosetta.rosetta

    def build_poses():
        first = pyrosetta.pose_from_pdb(str(args.pdb))
        return [first] + [first.clone() for _ in range(args.batch - 1)]

    setup_start = time.perf_counter()
    poses = build_poses()
    score_function = pyrosetta.create_score_function("beta_nov16_cart")
    setup_seconds = init_seconds + time.perf_counter() - setup_start

    if args.workflow == "pose-build":

        def run(_: int):
            return build_poses()

    elif args.workflow == "score":

        def run(_: int):
            scores = []
            for pose in poses:
                # Rosetta caches an unchanged Pose's energies. Clear the full
                # energy graph so every timed call performs a complete rescore.
                pose.energies().clear_energies()
                scores.append(score_function(pose))
            return scores

    elif args.workflow == "score-grad":

        def make_cartesian_gradient_operation(pose):
            # This follows CartesianMinimizer setup, but times only the native
            # C++ objective and derivative evaluations (not a Python atom loop).
            score_function(pose)
            move_map = rosetta.core.kinematics.MoveMap()
            move_map.set_bb(True)
            move_map.set_chi(True)
            move_map.set_jump(True)
            min_map = rosetta.core.optimization.CartesianMinimizerMap()
            min_map.setup(pose, move_map)
            score_function.setup_for_minimizing(pose, min_map)
            rosetta.core.optimization.activate_dof_deriv_terms_for_cart_min(
                pose, score_function, min_map
            )
            variables = rosetta.utility.vector1_double(min_map.ndofs())
            min_map.copy_dofs_from_pose(pose, variables)
            derivatives = rosetta.utility.vector1_double(min_map.ndofs())
            objective = rosetta.core.optimization.CartesianMultifunc(
                pose, min_map, score_function
            )

            def operation():
                # CartesianMultifunc holds a C++ reference, not shared
                # ownership. Capturing the map keeps it alive for timing.
                _ = min_map, pose, score_function
                objective(variables)
                objective.dfunc(variables, derivatives)

            return operation, derivatives, min_map.ndofs()

        gradient_operations = [
            make_cartesian_gradient_operation(pose) for pose in poses
        ]

        def run(_: int):
            outputs = []
            for operation, derivatives, _ in gradient_operations:
                operation()
                outputs.append(derivatives)
            return outputs

    elif args.workflow == "cart-min":
        move_map = rosetta.core.kinematics.MoveMap()
        move_map.set_bb(True)
        move_map.set_chi(True)
        move_map.set_jump(True)
        mover = rosetta.protocols.minimization_packing.MinMover()
        mover.movemap(move_map)
        mover.score_function(score_function)
        mover.min_type("lbfgs_armijo_nonmonotone")
        mover.cartesian(True)
        mover.max_iter(args.protocol_iterations)
        mover.tolerance(0.0)

        def run(_: int):
            outputs = []
            for pose in poses:
                output = pose.clone()
                mover.apply(output)
                outputs.append(output)
            return outputs

    elif args.workflow == "repack":
        mover = rosetta.protocols.minimization_packing.PackRotamersMover(score_function)

        def run(_: int):
            outputs = []
            for pose in poses:
                output = pose.clone()
                task = pyrosetta.standard_packer_task(output)
                task.restrict_to_repacking()
                mover.task(task)
                mover.apply(output)
                outputs.append(output)
            return outputs

    else:  # pragma: no cover - argparse enforces this
        raise AssertionError(args.workflow)

    return Workload(
        setup_seconds=setup_seconds,
        run=run,
        synchronize=lambda: None,
        metadata={
            "pyrosetta_version": rosetta.utility.Version.version(),
            "score_function": "beta_nov16_cart",
            "n_residues": poses[0].total_residue(),
            "n_atoms": sum(
                poses[0].residue(i).natoms() for i in range(1, poses[0].size() + 1)
            ),
            "cartesian_gradient_dofs": (
                gradient_operations[0][2] if args.workflow == "score-grad" else None
            ),
            "note": "Rosetta C++ core invoked through PyRosetta bindings",
        },
    )


def _find_rosetta_score_binary(bin_dir: Path | None) -> Path:
    names = (
        "score_jd2.default.linuxgccrelease",
        "score_jd2.linuxgccrelease",
        "score_jd2",
    )
    if bin_dir is not None:
        for name in names:
            candidate = bin_dir / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    for name in names:
        candidate = shutil.which(name)
        if candidate:
            return Path(candidate)
    raise FileNotFoundError(
        "No licensed Rosetta score_jd2 executable found; pass --rosetta-bin-dir."
    )


def _rosetta_workload(args: argparse.Namespace) -> Workload:
    if args.device != "cpu":
        raise ValueError("standalone Rosetta supports --device cpu only")
    if args.workflow != "score":
        raise ValueError(
            "The standalone runner currently supports score only; fixed-iteration "
            "minimization and repacking are compared through the same Rosetta C++ "
            "core via PyRosetta."
        )
    score_binary = _find_rosetta_score_binary(args.rosetta_bin_dir)
    command = [
        str(score_binary),
        "-s",
        str(args.pdb),
        "-score:weights",
        "beta_nov16_cart",
        "-out:nooutput",
        "-mute",
        "all",
    ]

    def run(_: int):
        return subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    return Workload(
        setup_seconds=0.0,
        run=run,
        synchronize=lambda: None,
        metadata={
            "score_function": "beta_nov16_cart",
            "binary": str(score_binary),
            "note": "End-to-end process latency; includes Rosetta startup and PDB I/O",
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", choices=("tmol", "pyrosetta", "rosetta"), required=True
    )
    parser.add_argument("--workflow", choices=WORKFLOWS, required=True)
    parser.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--protocol-iterations", type=int, default=10)
    parser.add_argument(
        "--reuse-topology",
        action="store_true",
        help="reuse TMol's rendered Cartesian network between minimizations",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="capture TMol Cartesian score/gradient replay (CUDA only)",
    )
    parser.add_argument("--pyrosetta-root", type=Path)
    parser.add_argument("--rosetta-bin-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="write a TMol PyTorch operator profile and Chrome trace",
    )
    parser.add_argument("--profile-repeats", type=int, default=1)
    parser.add_argument(
        "--profile-stacks",
        action="store_true",
        help="record Python stacks in the optional operator profile",
    )
    args = parser.parse_args()
    if (
        min(
            args.batch,
            args.threads,
            args.repeats,
            args.protocol_iterations,
            args.profile_repeats,
        )
        < 1
    ):
        parser.error(
            "batch, threads, repeats, protocol-iterations, and profile-repeats "
            "must be positive"
        )
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if not args.pdb.is_file():
        parser.error(f"PDB does not exist: {args.pdb}")
    if args.reuse_topology and not (
        args.engine == "tmol" and args.workflow == "cart-min"
    ):
        parser.error("--reuse-topology is supported only for TMol cart-min")
    if args.cuda_graph and not (
        args.engine == "tmol" and args.workflow == "cart-min" and args.device == "cuda"
    ):
        parser.error("--cuda-graph requires TMol cart-min on CUDA")
    if args.profile_dir is not None and args.engine != "tmol":
        parser.error("--profile-dir currently supports TMol only")
    return args


def _profile_workload(workload: Workload, args: argparse.Namespace) -> None:
    """Write an operator-level CPU/CUDA profile for the measured workload."""
    import torch

    assert args.profile_dir is not None
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=args.profile_stacks,
    ) as profile:
        for iteration in range(args.profile_repeats):
            workload.synchronize()
            with torch.profiler.record_function("benchmark_iteration"):
                result = workload.run(iteration)
            workload.synchronize()
            if result is None:
                raise RuntimeError("profiled workload returned no result")

    profile.export_chrome_trace(str(args.profile_dir / "trace.json"))
    events = profile.key_averages(group_by_input_shape=True)
    fields = (
        "key",
        "count",
        "self_cpu_time_total",
        "cpu_time_total",
        "self_device_time_total",
        "device_time_total",
        "self_cpu_memory_usage",
        "self_device_memory_usage",
        "input_shapes",
    )
    with (args.profile_dir / "operators.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for event in events:
            writer.writerow({field: getattr(event, field, None) for field in fields})
    sort_by = (
        "self_device_time_total" if args.device == "cuda" else "self_cpu_time_total"
    )
    (args.profile_dir / "operators.txt").write_text(
        events.table(sort_by=sort_by, row_limit=100) + "\n"
    )
    (args.profile_dir / "metadata.json").write_text(
        json.dumps(
            {
                "engine": args.engine,
                "workflow": args.workflow,
                "device": args.device,
                "batch": args.batch,
                "threads": args.threads,
                "protocol_iterations": args.protocol_iterations,
                "profile_repeats": args.profile_repeats,
                "reuse_topology": args.reuse_topology,
                "cuda_graph": args.cuda_graph,
                "engine_metadata": workload.metadata,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    args = _parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)

    process_start = time.perf_counter()
    builders = {
        "tmol": _tmol_workload,
        "pyrosetta": _pyrosetta_workload,
        "rosetta": _rosetta_workload,
    }
    workload = builders[args.engine](args)
    ready_seconds = time.perf_counter() - process_start

    warmup_samples = []
    for iteration in range(args.warmup):
        workload.synchronize()
        warmup_start = time.perf_counter()
        workload.run(iteration)
        workload.synchronize()
        warmup_samples.append(time.perf_counter() - warmup_start)

    samples = []
    for iteration in range(args.repeats):
        workload.synchronize()
        start = time.perf_counter()
        result = workload.run(iteration + args.warmup)
        workload.synchronize()
        samples.append(time.perf_counter() - start)
        if result is None:
            raise RuntimeError("workload returned no result")

    payload = {
        "schema_version": 2,
        "benchmark_revision": _git_revision(),
        "engine": args.engine,
        "workflow": args.workflow,
        "pdb": str(args.pdb.resolve()),
        "batch": args.batch,
        "threads": args.threads,
        "device": args.device,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "protocol_iterations": args.protocol_iterations,
        "process_ready_seconds": ready_seconds,
        "workload_setup_seconds": workload.setup_seconds,
        "first_call_seconds": warmup_samples[0] if warmup_samples else None,
        "warmup_samples_seconds": warmup_samples,
        "timing": _timing_summary(samples, args.batch),
        "samples_seconds": samples,
        "host": _host_metadata(),
        "engine_metadata": workload.metadata,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    if args.profile_dir is not None:
        _profile_workload(workload, args)


if __name__ == "__main__":
    main()
