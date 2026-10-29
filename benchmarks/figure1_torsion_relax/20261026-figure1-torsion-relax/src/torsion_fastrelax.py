"""Figure 1 FastRelax point, with a selectable relax mode (torsion / cartesian).

Same timing contract as runs/20261005-fastrelax-f92d/latest_fastrelax.py: one
seeded one-repeat native FastRelax per matrix point; input construction, scorer
rendering set-up and move-map/fold-forest construction are outside the timed
interval; no warm-up; GPU synchronised immediately before the final clock read.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / 'src'))
import benchmark_tmol as bt
import benchmark_pyrosetta as bp
from common import SPEC, machine_metadata, sha256

PYROSETTA_PATH = Path('/mnt/home/kdidi/projects/pyrosetta-2024.39')

parser = argparse.ArgumentParser()
parser.add_argument('--engine', choices=['tmol', 'pyrosetta'], required=True)
parser.add_argument('--mode', choices=['torsion', 'cartesian'], required=True)
parser.add_argument('--dataset', required=True)
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--threads', type=int, default=1)
parser.add_argument('--cuda-graph', choices=['auto', 'on', 'off'], default='auto',
                    help='torsion mode only: CUDA-graph capture in the kinematic minimizer')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()


def replicated_pose_020(row, device, batch_size):
    """Same inputs / loader options as benchmark_tmol.load_pose of the 0.1.58 harness,
    ported to the tmol 0.2.0 loader names (pose_stack_from_pdb/_biotite -> *_file/_atom_array)."""
    from tmol.database import ParameterDatabase
    from tmol.io import pose_stack_from_file, pose_stack_from_atom_array
    from tmol.pose import PoseStackBuilder
    path = Path(row['structure_path'])
    if row['modality'] != 'protein_ligand':
        pose, database = pose_stack_from_file(path, device), ParameterDatabase.get_default()
    else:
        import biotite.structure as struc
        import biotite.structure.io
        structure = biotite.structure.io.load_structure(str(path), model=1, include_bonds=True)
        if isinstance(structure, struc.AtomArrayStack):
            structure = structure[0]
        pose, context = pose_stack_from_atom_array(
            structure, device, prepare_ligands=True, ligand_params_files=[row['ligand_tmol_params']],
            no_optH=True, param_db=ParameterDatabase.get_default(), return_context=True)
        database = context.parameter_database
    loaded_blocks = int((pose.block_type_ind64[0] >= 0).sum())
    assert loaded_blocks == int(row['expected_blocks'] if row.get('expected_blocks') not in (None, '') else row['residues']), loaded_blocks
    if batch_size == 1:
        return pose, database
    return PoseStackBuilder.from_poses([pose] * batch_size, device), database


def tmol_fastrelax(row, device_name, batch_size, cpu_threads, mode, cuda_graph_opt):
    import torch
    import tmol
    from tmol.score import beta2016_score_function
    from tmol.relax import fast_relax
    from tmol.relax._fast_relax import DefaultKinematicMinimizer
    from tmol.kinematics import CartesianMoveMap, FoldForest, MoveMap
    from tmol.pack import PackerPalette

    device = (torch.device('cuda', torch.cuda.current_device())
              if device_name == 'cuda' else torch.device('cpu'))
    if device_name == 'cpu':
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
    else:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    setup_start = time.perf_counter()
    pose, database = replicated_pose_020(row, device, batch_size)
    sfxn = beta2016_score_function(device, param_db=database)
    fold_forest = FoldForest.reasonable_fold_forest(pose)
    palette = PackerPalette()
    extra = {}
    kwargs = {}
    if mode == 'cartesian':
        move_map = CartesianMoveMap()
    else:
        # Internal-coordinate DOFs: all backbone/side-chain/ligand named torsions
        # plus all (non-root) jumps; bond lengths and angles stay fixed.
        # This is the recipe of tmol's own test_kin_fast_relax_ubq.
        move_map = MoveMap.from_pose_stack(pose)
        move_map.move_all_jumps = True
        move_map.move_all_named_torsions = True
        use_graph = (device_name == 'cuda') if cuda_graph_opt == 'auto' else cuda_graph_opt == 'on'
        kwargs['min_fn'] = DefaultKinematicMinimizer(cuda_graph=use_graph)
        extra['kinematic_cuda_graph'] = use_graph
    if device_name == 'cuda':
        torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_start
    initial = sfxn.render_whole_pose_scoring_module(pose)(pose.coords).detach()

    torch.manual_seed(SPEC['seed'])
    if device_name == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    relaxed = fast_relax(pose, sfxn, palette, move_map, fold_forest,
                         num_repeats=SPEC['fastrelax_repeats'], verbose=False, **kwargs)
    if device_name == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if relaxed is None:
        raise RuntimeError('fast_relax returned None')
    final = sfxn.render_whole_pose_scoring_module(relaxed)(relaxed.coords).detach().float()
    assert bool(torch.isfinite(final).all()), 'non-finite final score'
    gpu = None
    if device_name == 'cuda':
        p = torch.cuda.get_device_properties(device)
        gpu = dict(name=p.name, total_memory=p.total_memory, compute_capability=f'{p.major}.{p.minor}')
    return {
        'relax_mode': mode,
        'seconds_per_batch_samples': [elapsed],
        'seconds_per_structure_samples': [elapsed / batch_size],
        'iterations_per_sample': 1,
        'setup_seconds': setup_seconds,
        'score_terms': {},
        'peak_memory_bytes': torch.cuda.max_memory_allocated() if device_name == 'cuda' else None,
        'tmol_import_version': getattr(tmol, '__version__', None),
        'torch_version': torch.__version__,
        'torch_cuda_version': torch.version.cuda,
        'torch_num_threads': torch.get_num_threads(),
        'gpu_properties': gpu,
        'validation_score_mean': float(final.mean()),
        'validation_score_min': float(final.min()),
        'validation_score_max': float(final.max()),
        'initial_score_mean': float(initial.float().mean()),
        'final_scores_first10': [float(x) for x in final[:10]],
        **extra,
    }


def pyrosetta_fastrelax(row, mode):
    pyrosetta = bp.initialize(row, PYROSETTA_PATH)
    pose = pyrosetta.pose_from_file(row['structure_path'])
    ligand = row['modality'] == 'protein_ligand'
    base = 'beta_genpot' if ligand else 'beta_nov16'
    sfxn_name = base + ('_cart' if mode == 'cartesian' else '')
    sfxn = pyrosetta.create_score_function(sfxn_name)
    setup_start = time.perf_counter()
    relax = pyrosetta.rosetta.protocols.relax.FastRelax(sfxn, SPEC['fastrelax_repeats'])
    script = (Path(pyrosetta._rosetta_database_from_env())
              / 'sampling/relax_scripts/MonomerRelax2019.txt')
    lines = pyrosetta.rosetta.std.vector_std_string()
    for line in script.read_text().splitlines():
        line = line.replace('%%nrepeats%%', str(SPEC['fastrelax_repeats']))
        if line.strip():
            lines.append(line)
    relax.set_script_from_lines(lines)
    if mode == 'cartesian':
        relax.cartesian(True)
    else:
        relax.cartesian(False)
        mm = pyrosetta.rosetta.core.kinematics.MoveMap()
        mm.set_bb(True)
        mm.set_chi(True)
        mm.set_jump(True)
        relax.set_movemap(mm)
    setup_seconds = time.perf_counter() - setup_start
    working = pose.clone()
    initial = float(sfxn(working))
    start = time.perf_counter()
    relax.apply(working)
    elapsed = time.perf_counter() - start
    final = float(sfxn(working))
    return {
        'relax_mode': mode,
        'score_function': sfxn_name,
        'seconds_per_batch_samples': [elapsed],
        'seconds_per_structure_samples': [elapsed],
        'iterations_per_sample': 1,
        'setup_seconds': setup_seconds,
        'score_terms': {},
        'peak_memory_bytes': None,
        'initial_score_mean': initial,
        'validation_score_mean': final,
        'pyrosetta_version_string': pyrosetta.version().splitlines()[-1],
        'actual_rosetta_threads': pyrosetta.rosetta.basic.thread_manager.RosettaThreadManager.get_instance().total_threads(),
    }


row = bt.select(args.dataset)
result = dict(engine=args.engine, protocol='fastrelax', relax_mode=args.mode,
              dataset_id=args.dataset, modality=row['modality'], device=args.device,
              batch_size=args.batch, atoms=int(row['atoms']), residues=int(row['residues']),
              input_path=row['structure_path'], input_sha256=row['structure_sha256'],
              cpu_threads_requested=args.threads, seed=SPEC['seed'],
              num_repeats=SPEC['fastrelax_repeats'], timed_samples=1,
              engine_version='0.2.0' if args.engine == 'tmol' else '2024.39',
              engine_commit='uw-ipd/tmol v0.2.0 (release wheel; master 60b0d77df is v0.2.0+10 docs commits)'
              if args.engine == 'tmol' else '59628fbc5bc09f1221e1642f1f8d157ce49b1410',
              machine=machine_metadata(), cpu_affinity=sorted(os.sched_getaffinity(0)),
              source_harness=str(RUN / 'src'), status='running',
              started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.with_suffix('.running.json').write_text(json.dumps(result, indent=2) + '\n')
started = time.perf_counter()
try:
    assert sha256(Path(row['structure_path'])) == row['structure_sha256']
    if args.engine == 'tmol':
        import tmol
        assert tmol.__version__.split('+', 1)[0] == '0.2.0', tmol.__version__
        if args.device == 'cuda':
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        result.update(tmol_fastrelax(row, args.device, args.batch, args.threads,
                                     args.mode, args.cuda_graph))
    else:
        assert args.device == 'cpu' and args.batch == 1 and args.threads == 1
        result.update(pyrosetta_fastrelax(row, args.mode))
    assert len(result['seconds_per_structure_samples']) == 1
    assert result['seconds_per_structure_samples'][0] > 0
    assert math.isfinite(result['validation_score_mean'])
    assert math.isfinite(result['initial_score_mean'])
    result['status'] = 'ok'
except Exception as exc:
    result.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
result['elapsed_seconds'] = time.perf_counter() - started
result['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
tmp = args.output.with_suffix('.tmp')
tmp.write_text(json.dumps(result, indent=2) + '\n')
tmp.replace(args.output)
args.output.with_suffix('.running.json').unlink()
print(json.dumps({k: result.get(k) for k in ['engine', 'relax_mode', 'dataset_id', 'device', 'batch_size',
                                             'status', 'error', 'elapsed_seconds',
                                             'initial_score_mean', 'validation_score_mean']}), flush=True)
if result['status'] != 'ok':
    raise SystemExit(2)
