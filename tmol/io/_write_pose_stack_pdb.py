import numpy
import torch

from tmol.io._pdb_parsing import atom_record_dtype, _to_pdb
from tmol.pose._pose_stack import PoseStack
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.types._array import NDArray
from tmol.types._torch import Tensor
from tmol.types._functional import _validate_args
from typing import Optional
from tmol_core.io.atom_records import prepare_atom_records


@_validate_args
def _write_pose_stack_pdb(
    pose_stack: PoseStack,
    fname_out: str,
    merge_fragments: bool = True,
    **kwargs,
):
    """Write a PDB-formatted file to disk given an input PoseStack.
    Optionally, additional arguments may be passed to the inner function
    "_atom_records_from_pose_stack." Fragmented ligands use their original
    residue identity by default; pass ``merge_fragments=False`` to keep
    fragment residues separate.
    """
    atom_records = _atom_records_from_pose_stack(
        pose_stack, merge_fragments=merge_fragments, **kwargs
    )
    pdbstring = _to_pdb(atom_records)
    with open(fname_out, "w") as fid:
        fid.write(pdbstring)


@_validate_args
def _atom_records_from_pose_stack(
    pose_stack: PoseStack,
    merge_fragments: bool = True,
) -> NDArray[atom_record_dtype][:]:
    """Create a numpy array holding the atom records needed to write a
    PDB file from a PoseStack.

    Fragmented ligands use their original residue identity by default. Pass
    ``merge_fragments=False`` to retain the separate fragment residue numbers.
    """

    return _atom_records_from_coords(
        pose_stack.packed_block_types,
        pose_stack.chain_id64,
        pose_stack.block_type_ind64,
        pose_stack.coords,
        pose_stack.block_coord_offset,
        pose_stack.pdb_info.residue_labels,
        pose_stack.pdb_info.residue_insertion_codes,
        pose_stack.pdb_info.chain_labels,
        pose_stack.pdb_info.atom_occupancy,
        pose_stack.pdb_info.atom_b_factor,
        split_block_mapping=(
            getattr(pose_stack, "split_block_mapping", None)
            if merge_fragments
            else None
        ),
    )


@_validate_args
def _atom_records_from_coords(
    pbt: "PackedBlockTypes",
    chain_ind_for_block: Tensor[torch.int64][:, :],
    block_types64: Tensor[torch.int64][:, :],
    pose_like_coords: Tensor[torch.float32][:, :, 3],
    block_coord_offset: Tensor[torch.int32][:, :],
    residue_labels: Optional[NDArray[int][:, :]],
    residue_insertion_codes: Optional[NDArray[object][:, :]],
    chain_labels: Optional[NDArray[object][:, :]],
    atom_occupancy: NDArray[numpy.float32][:, :],
    atom_b_factor: NDArray[numpy.float32][:, :],
    *,
    split_block_mapping=None,
) -> NDArray[atom_record_dtype][:]:
    """Create a numpy array holding the atom records needed to write a
    PDB file from the coordinates and block types of a stack of structures,
    laid out in pose-stack form.
    """

    assert pose_like_coords.shape[0] == chain_ind_for_block.shape[0]
    assert pose_like_coords.shape[0] == block_types64.shape[0]
    assert pose_like_coords.shape[0] == block_coord_offset.shape[0]

    assert block_types64.shape[1] == block_coord_offset.shape[1]
    assert block_types64.shape[1] == chain_ind_for_block.shape[1]

    n_poses = block_coord_offset.shape[0]
    max_n_blocks = block_coord_offset.shape[1]
    max_n_pose_atoms = pose_like_coords.shape[1]

    is_real_block = block_types64 != -1
    real_block_pose, real_block_block = torch.nonzero(is_real_block, as_tuple=True)

    n_block_atoms = torch.zeros(
        (n_poses, max_n_blocks), dtype=torch.int32, device=pbt.device
    )
    n_block_atoms[is_real_block] = pbt.n_atoms[block_types64[is_real_block]]
    block_coord_offset64 = block_coord_offset.to(torch.int64)
    block_for_pose_atom = torch.zeros(
        (n_poses, max_n_pose_atoms), dtype=torch.int64, device=pbt.device
    )
    block_for_pose_atom[
        real_block_pose, block_coord_offset64[real_block_pose, real_block_block]
    ] = 1
    block_for_pose_atom = torch.cumsum(block_for_pose_atom, dim=1) - 1
    pose_for_pose_atom = (
        torch.arange(n_poses, dtype=torch.int64, device=pbt.device)
        .repeat_interleave(max_n_pose_atoms)
        .view(n_poses, max_n_pose_atoms)
    )

    n_pose_atoms = torch.sum(n_block_atoms, dim=1)

    # so we can index just the atoms out of coords tensor, e.g.
    # we need to know which atoms are real and which are padding
    atom_is_real = torch.tile(
        torch.arange(max_n_pose_atoms, dtype=torch.int64, device=pbt.device),
        (n_poses, 1),
    ) < n_pose_atoms.unsqueeze(dim=1)

    pose_for_real_atom = torch.arange(
        n_poses,
        dtype=torch.int64,
        device=pbt.device,
    ).repeat_interleave(n_pose_atoms, dim=0)
    block_for_atom = torch.zeros(
        (n_poses, max_n_pose_atoms), dtype=torch.int64, device=pbt.device
    )
    block_for_atom[
        real_block_pose, block_coord_offset64[real_block_pose, real_block_block]
    ] = 1
    block_for_atom = torch.cumsum(block_for_atom, dim=1) - 1
    block_for_real_atom = block_for_atom[atom_is_real]

    block_local_atom_index_for_pose_atom = (
        torch.arange(max_n_pose_atoms, dtype=torch.int32, device=pbt.device).repeat(
            (n_poses, 1)
        )
        - block_coord_offset[pose_for_pose_atom, block_for_pose_atom]
    )
    block_local_atom_index_for_real_atom = block_local_atom_index_for_pose_atom[
        atom_is_real
    ]

    # BEGIN BLOCK OF CODE FOR CREATING FROM-1 RESIDUE INDICES FROM CHAIN INDICES
    max_n_chains = torch.max(chain_ind_for_block) + 1
    global_chain_ind_for_block = (
        chain_ind_for_block
        + (
            torch.arange(n_poses, dtype=torch.int64, device=pbt.device).unsqueeze(1)
            * max_n_chains
        )
    ).view(-1)
    n_blocks_per_global_chain = torch.zeros(
        (n_poses * max_n_chains), dtype=torch.int64, device=pbt.device
    )

    is_not_real_block = torch.logical_not(is_real_block)
    global_chain_ind_for_block[is_not_real_block.view(-1)] = (
        0  # avoid index add to invalid locations
    )
    n_blocks_per_global_chain.index_add_(
        0,
        global_chain_ind_for_block,
        is_real_block.to(torch.int64).view(-1),
    )
    n_blocks_per_chain = n_blocks_per_global_chain.view(n_poses, max_n_chains)

    block_offset_for_chain = torch.zeros(
        (n_poses, max_n_chains), dtype=torch.int64, device=pbt.device
    )
    block_offset_for_chain[:, 1:] = torch.cumsum(n_blocks_per_chain[:, :-1], dim=1)
    # END BLOCK OF CODE FOR CREATING FROM-1 RESIDUE INDICES FROM CHAIN INDICES

    block_chain_local_index_for_real_atom = (
        block_for_atom
        - block_offset_for_chain[
            pose_for_pose_atom, chain_ind_for_block[pose_for_pose_atom, block_for_atom]
        ]
    )

    # ok, let's move everything to the cpu/numpy from here forward
    block_types64 = block_types64.cpu().numpy()
    pose_like_coords = pose_like_coords.cpu().detach().numpy()
    pose_for_real_atom = pose_for_real_atom.cpu().numpy()
    atom_is_real = atom_is_real.cpu().numpy()
    block_for_real_atom = block_for_real_atom.cpu().numpy()
    block_chain_local_index_for_real_atom = (
        block_chain_local_index_for_real_atom.cpu().numpy()
    )
    block_local_atom_index_for_real_atom = (
        block_local_atom_index_for_real_atom.cpu().numpy()
    )
    chain_ind_for_block = chain_ind_for_block.cpu().numpy()

    chain_ind_for_real_atom = chain_ind_for_block[
        pose_for_real_atom, block_for_real_atom
    ]

    if residue_labels is None:
        labels = block_chain_local_index_for_real_atom[atom_is_real] + 1
    else:
        labels = residue_labels[pose_for_real_atom, block_for_real_atom]
    if chain_labels is None:
        # Preserve the existing automatic chain-label convention.
        assert max_n_chains <= 26 * 27
        chain_letter_by_ind = numpy.array(
            [
                (
                    chr(i + ord("A"))
                    if i < 26
                    else chr((i // 26) + ord("A")) + chr((i % 26) + ord("A"))
                )
                for i in range(max_n_chains)
            ],
            dtype=object,
        )
        chains = chain_letter_by_ind[chain_ind_for_real_atom]
    else:
        chains = chain_labels[pose_for_real_atom, block_for_real_atom]

    return prepare_atom_records(
        pbt.active_block_types,
        block_types64[pose_for_real_atom, block_for_real_atom],
        pose_for_real_atom,
        block_for_real_atom,
        block_local_atom_index_for_real_atom,
        pose_like_coords[atom_is_real],
        # Retain the historical NumPy string-width convention of this API.
        record_names=numpy.full(len(pose_for_real_atom), "ATOM  ", dtype=str),
        chain_indices=chain_ind_for_real_atom,
        residue_labels=labels,
        chain_labels=chains,
        insertion_codes=(
            residue_insertion_codes[pose_for_real_atom, block_for_real_atom]
            if residue_insertion_codes is not None
            else numpy.full(len(pose_for_real_atom), " ")
        ),
        occupancy=atom_occupancy[atom_is_real],
        b_factor=atom_b_factor[atom_is_real],
        split_block_mapping=split_block_mapping,
    )
