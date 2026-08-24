import biotite.structure as struc
from biotite.structure.io.pdbx import CIFFile, get_structure
import numpy as np
import pytest
import torch

from tmol.io import biotite_from_pose_stack, pose_stack_from_biotite
from tmol.tests.data import data_path


def _glycan_fixture():
    cif = CIFFile.read(data_path("cif", "4BYH_chainC_glycan.cif"))
    return get_structure(cif, model=1, include_bonds=True)


def _pose_links(pose):
    pbt = pose.packed_block_types
    result = set()
    for block1 in range(pose.max_n_blocks):
        type1 = int(pose.block_type_ind64[0, block1])
        if type1 < 0:
            continue
        bt1 = pbt.active_block_types[type1]
        for conn1, (block2, conn2) in enumerate(
            pose.inter_residue_connections64[0, block1].tolist()
        ):
            if block2 <= block1:
                continue
            bt2 = pbt.active_block_types[int(pose.block_type_ind64[0, block2])]
            endpoint1 = (
                str(pose.pdb_info.chain_labels[0, block1]),
                int(pose.pdb_info.residue_labels[0, block1]),
                bt1.connections[conn1].atom,
            )
            endpoint2 = (
                str(pose.pdb_info.chain_labels[0, block2]),
                int(pose.pdb_info.residue_labels[0, block2]),
                bt2.connections[conn2].atom,
            )
            result.add((endpoint1, endpoint2))
    return result


def _pose_link_distances(pose):
    pbt = pose.packed_block_types
    distances = []
    for block1 in range(pose.max_n_blocks):
        type1 = int(pose.block_type_ind64[0, block1])
        if type1 < 0:
            continue
        bt1 = pbt.active_block_types[type1]
        for conn1, (block2, conn2) in enumerate(
            pose.inter_residue_connections64[0, block1].tolist()
        ):
            if block2 <= block1:
                continue
            bt2 = pbt.active_block_types[int(pose.block_type_ind64[0, block2])]
            atom1 = int(pose.block_coord_offset64[0, block1]) + int(
                bt1.connection_to_idx[bt1.connections[conn1].name]
            )
            atom2 = int(pose.block_coord_offset64[0, block2]) + int(
                bt2.connection_to_idx[bt2.connections[conn2].name]
            )
            distances.append(
                float(
                    torch.linalg.vector_norm(
                        pose.coords[0, atom1] - pose.coords[0, atom2]
                    )
                )
            )
    return distances


def _virtual_atom_indices(pose):
    indices = []
    for block, type_ind in enumerate(pose.block_type_ind64[0]):
        block_type = pose.packed_block_types.active_block_types[int(type_ind)]
        for atom_name in block_type.properties.virtual:
            indices.append(
                int(pose.block_coord_offset64[0, block])
                + int(block_type.atom_to_idx[atom_name])
            )
    return indices


def _input_atom_mask(pose, structure):
    starts = struc.get_residue_starts(structure)
    ends = np.append(starts[1:], structure.array_length())
    assert len(starts) == pose.max_n_blocks
    mask = torch.zeros_like(pose.real_atoms)
    for block, (start, end) in enumerate(zip(starts, ends)):
        block_type = pose.packed_block_types.active_block_types[
            int(pose.block_type_ind64[0, block])
        ]
        input_names = set(structure.atom_name[start:end])
        for atom_name in input_names:
            mask[
                0,
                int(pose.block_coord_offset64[0, block])
                + block_type.atom_to_idx[atom_name],
            ] = True
    return mask


EXPECTED_4BYH_LINKS = {
    (("A", 297, "ND2"), ("C", 1, "C1")),
    (("C", 1, "O4"), ("C", 2, "C1")),
    (("C", 1, "O6"), ("C", 10, "C1")),
    (("C", 2, "O4"), ("C", 3, "C1")),
    (("C", 3, "O6"), ("C", 4, "C1")),
    (("C", 3, "O3"), ("C", 8, "C1")),
    (("C", 4, "O2"), ("C", 5, "C1")),
    (("C", 5, "O4"), ("C", 6, "C1")),
    (("C", 6, "O6"), ("C", 7, "C2")),
    (("C", 8, "O2"), ("C", 9, "C1")),
}


def test_branched_n_glycan_topology_score_and_round_trip(torch_device):
    structure = _glycan_fixture()
    pose, context = pose_stack_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        no_optH=True,
        return_context=True,
    )

    assert _pose_links(pose) == EXPECTED_4BYH_LINKS
    assert len(_pose_link_distances(pose)) == 10
    assert all(1.40 < distance < 1.48 for distance in _pose_link_distances(pose))
    names = [
        pose.packed_block_types.active_block_types[int(ind)].name
        for ind in pose.block_type_ind64[0]
    ]
    assert names[0].endswith(":covalent_ND2")
    assert names[1].endswith(":covalent_C1,O4,O6")
    assert names[3].endswith(":covalent_C1,O3,O6")

    from tmol import beta2016_score_function

    coords = pose.coords.detach().clone().requires_grad_(True)
    scorer = beta2016_score_function(
        torch_device, param_db=context.parameter_database
    ).render_whole_pose_scoring_module(pose)
    score = scorer(coords).sum()
    score.backward()
    assert torch.isfinite(score)
    assert torch.all(torch.isfinite(coords.grad))

    virtual_indices = _virtual_atom_indices(pose)
    for block, type_ind in enumerate(pose.block_type_ind64[0]):
        block_type = pose.packed_block_types.active_block_types[int(type_ind)]
        for atom_name in block_type.properties.virtual:
            atom_index = block_type.atom_to_idx[atom_name]
            assert block_type.atoms[atom_index].atom_type == "Vrt"
            assert any(atom_index in bond for bond in block_type.bond_indices)
    assert len(virtual_indices) == 20
    moved = pose.coords.detach().clone()
    moved[0, virtual_indices] += 50.0
    assert torch.allclose(scorer(moved), scorer(pose.coords), atol=1e-4, rtol=0)

    exported = biotite_from_pose_stack(pose, co=context.canonical_ordering)
    rebuilt = pose_stack_from_biotite(
        exported, torch_device, context=context, no_optH=True
    )
    assert _pose_links(rebuilt) == EXPECTED_4BYH_LINKS


def test_branched_glycan_context_reuse(torch_device):
    structure = _glycan_fixture()
    first, context = pose_stack_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        no_optH=True,
        return_context=True,
    )
    shifted = structure.copy()
    shifted.coord += np.float32(0.1)
    second = pose_stack_from_biotite(
        shifted, torch_device, context=context, no_optH=True
    )
    assert torch.equal(
        first.inter_residue_connections, second.inter_residue_connections
    )
    delta = second.coords - first.coords
    deposited = _input_atom_mask(first, structure)
    assert torch.allclose(
        delta[deposited],
        torch.full_like(delta[deposited], 0.1),
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.parametrize(("residue_name", "link_atom"), (("SER", "OG"), ("THR", "OG1")))
def test_o_linked_nag_uses_sidechain_oxygen(
    biotite_1ubq, torch_device, residue_name, link_atom
):
    starts = struc.get_residue_starts(biotite_1ubq)
    ends = np.append(starts[1:], biotite_1ubq.array_length())
    protein_res = next(
        i
        for i, start in enumerate(starts)
        if biotite_1ubq.res_name[start] == residue_name
    )
    protein = biotite_1ubq[starts[protein_res] : ends[protein_res]].copy()

    glycan = _glycan_fixture()
    glycan_starts = struc.get_residue_starts(glycan)
    nag = glycan[glycan_starts[1] : glycan_starts[2]].copy()
    nag.chain_id[:] = "G"
    nag.res_id[:] = 1

    structure = protein + nag
    rows = []
    if protein.bonds is not None:
        rows.extend(protein.bonds.as_array().tolist())
    if nag.bonds is not None:
        rows.extend(
            (int(a) + len(protein), int(b) + len(protein), int(t))
            for a, b, t in nag.bonds.as_array()
        )
    attachment = next(
        i for i, name in enumerate(protein.atom_name) if name == link_atom
    )
    c1 = len(protein) + next(i for i, name in enumerate(nag.atom_name) if name == "C1")
    rows.append((attachment, c1, int(struc.BondType.SINGLE)))
    structure.bonds = struc.BondList(
        structure.array_length(), np.asarray(rows, dtype=np.int32)
    )

    pose = pose_stack_from_biotite(
        structure, torch_device, prepare_ligands=True, no_optH=True
    )
    names = [
        pose.packed_block_types.active_block_types[int(ind)].name
        for ind in pose.block_type_ind64[0]
    ]
    assert names[0].endswith(f":covalent_{link_atom}")
    assert names[1].endswith(":covalent_C1")
    assert len(_pose_links(pose)) == 1
    ser_type = pose.packed_block_types.active_block_types[
        int(pose.block_type_ind64[0, 0])
    ]
    assert len(ser_type.properties.virtual) == 1
    virtual_name = ser_type.properties.virtual[0]
    assert ser_type.atoms[ser_type.atom_to_idx[virtual_name]].atom_type == "Vrt"
