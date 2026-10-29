"""tmol 0.2.0 loader, same inputs/options as the 0.1.58 harness."""
from pathlib import Path


def replicated_pose_020(row, device, batch_size):
    """Same inputs / loader options as benchmark_tmol.load_pose of the 0.1.58 harness,
    ported to the tmol 0.2.0 loader names (pose_stack_from_pdb/_biotite -> *_file/_atom_array)."""
    from tmol.database import ParameterDatabase
    from tmol.io import pose_stack_from_file, pose_stack_from_atom_array
    from tmol.pose import PoseStackBuilder
    path = Path(row['structure_path'])
    if row['modality'] != 'protein_ligand':
        # hivrt has a chain gap between residues 537/538 that tmol refuses to bridge by default;
        # missing_density_distance_threshold=0 keeps the declared (long) bond -- a comparison control, not a repair.
        pose, database = pose_stack_from_file(path, device), ParameterDatabase.get_default()
    else:
        import biotite.structure as struc
        import biotite.structure.io
        structure = biotite.structure.io.load_structure(str(path), model=1, include_bonds=True)
        if isinstance(structure, struc.AtomArrayStack):
            structure = structure[0]
        extra = {'missing_density_distance_threshold': 0.0} if row['dataset_id'] == 'hivrt' else {}
        pose, context = pose_stack_from_atom_array(
            structure, device, **extra, prepare_ligands=True, ligand_params_files=[row['ligand_tmol_params']],
            no_optH=True, param_db=ParameterDatabase.get_default(), return_context=True)
        database = context.parameter_database
    loaded_blocks = int((pose.block_type_ind64[0] >= 0).sum())
    assert loaded_blocks == int(row['expected_blocks'] if row.get('expected_blocks') not in (None, '') else row['residues']), loaded_blocks
    if batch_size == 1:
        return pose, database
    return PoseStackBuilder.from_poses([pose] * batch_size, device), database
