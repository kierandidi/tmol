"""Experiment harness for optimising torsion FastRelax (not part of the Figure 1 contract)."""
import sys, time, json
from pathlib import Path
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN/'src'))
import torch
import benchmark_tmol as bt
from common import SPEC

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



def setup(row, device, batch, mode):
    from tmol.score import beta2016_score_function
    from tmol.kinematics import CartesianMoveMap, FoldForest, MoveMap
    from tmol.pack import PackerPalette
    pose, db = replicated_pose_020(row, device, batch)
    sfxn = beta2016_score_function(device, param_db=db)
    ff = FoldForest.reasonable_fold_forest(pose)
    if mode == 'cartesian':
        mm = CartesianMoveMap()
    else:
        mm = MoveMap.from_pose_stack(pose); mm.move_all_jumps = True; mm.move_all_named_torsions = True
    return pose, sfxn, ff, mm, PackerPalette()

def timed(fn):
    torch.cuda.synchronize(); t = time.perf_counter(); out = fn(); torch.cuda.synchronize()
    return out, time.perf_counter() - t
