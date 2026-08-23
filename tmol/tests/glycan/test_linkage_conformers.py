import numpy
import torch
from biotite.structure.io.pdbx import CIFFile, get_structure

from tmol.glycan import (
    get_glycan_linkage_torsions,
    linkage_conformers_for_pose,
    sample_glycan_linkage,
    sample_linkage_angles,
    set_glycan_linkage_torsions,
)
from tmol.io import pose_stack_from_biotite
from tmol.tests.data import data_path


def _glycan_pose(device):
    structure = get_structure(
        CIFFile.read(data_path("cif", "4BYH_chainC_glycan.cif")),
        model=1,
        include_bonds=True,
    )
    return pose_stack_from_biotite(
        structure,
        device,
        prepare_ligands=True,
        no_optH=True,
        return_context=True,
    )[0]


def _block_coords(pose, block):
    start = int(pose.block_coord_offset64[0, block])
    count = len(pose.block_type(0, block).atoms)
    return pose.coords[0, start : start + count]


def _periodic_error(observed, expected):
    observed, expected = numpy.asarray(observed), numpy.asarray(expected)
    return numpy.abs((observed - expected + 180.0) % 360.0 - 180.0)


def test_rosetta_linkage_table_lookup_and_sampling(torch_device):
    pose = _glycan_pose(torch_device)
    root = linkage_conformers_for_pose(pose, 0, 1)
    nag4 = linkage_conformers_for_pose(pose, 0, 2)

    assert len(root) == 9
    assert root[0].means == (261.204, 176.582, 173.004, 190.375)
    assert root[0].standard_deviations == (18.738, 12.940, 42.546, 12.882)
    assert len(nag4) == 2
    assert nag4[0].population == 0.9876

    index, angles = sample_linkage_angles(
        nag4, numpy.random.default_rng(7), idealize=True
    )
    assert index == 0
    assert angles == nag4[0].means

    sampled, index, angles = sample_glycan_linkage(
        pose, 0, 2, numpy.random.default_rng(7), idealize=True
    )
    assert index == 0
    assert angles == nag4[0].means
    assert (
        _periodic_error(get_glycan_linkage_torsions(sampled, 0, 2)[:2], angles) < 1e-3
    ).all()

    for child_block in range(1, 11):
        conformer = linkage_conformers_for_pose(pose, 0, child_block)[0]
        moved = set_glycan_linkage_torsions(pose, 0, child_block, conformer.means)
        observed = get_glycan_linkage_torsions(moved, 0, child_block)
        assert (
            _periodic_error(observed[: len(conformer.means)], conformer.means) < 1e-3
        ).all()


def test_set_linkage_torsions_moves_only_downstream_tree(torch_device):
    pose = _glycan_pose(torch_device)
    conformer = linkage_conformers_for_pose(pose, 0, 3)[0]
    before = [_block_coords(pose, block).clone() for block in range(11)]
    moved = set_glycan_linkage_torsions(pose, 0, 3, conformer.means)

    assert (
        _periodic_error(get_glycan_linkage_torsions(moved, 0, 3)[:2], conformer.means)
        < 2e-4
    ).all()
    for block in range(11):
        torch.testing.assert_close(_block_coords(pose, block), before[block])
    for block in (0, 1, 2, 10):
        block_type = moved.block_type(0, block)
        nonvirtual = [
            i for i, atom in enumerate(block_type.atoms) if atom.atom_type != "Vrt"
        ]
        torch.testing.assert_close(
            _block_coords(moved, block)[nonvirtual], before[block][nonvirtual]
        )
    for block in (3, 4, 5, 6, 7, 8, 9):
        assert not torch.equal(_block_coords(moved, block), before[block])
        moved_xyz = _block_coords(moved, block).double()
        before_xyz = before[block].double()
        torch.testing.assert_close(
            torch.cdist(moved_xyz - moved_xyz[:1], moved_xyz - moved_xyz[:1]),
            torch.cdist(before_xyz - before_xyz[:1], before_xyz - before_xyz[:1]),
            atol=3e-5,
            rtol=1e-5,
        )


def test_asn_linkage_uses_all_four_rosetta_torsions(torch_device):
    pose = _glycan_pose(torch_device)
    conformer = linkage_conformers_for_pose(pose, 0, 1)[0]
    protein_before = _block_coords(pose, 0).clone()
    moved = set_glycan_linkage_torsions(pose, 0, 1, conformer.means)

    assert (
        _periodic_error(get_glycan_linkage_torsions(moved, 0, 1), conformer.means)
        < 2e-4
    ).all()
    protein = moved.block_type(0, 0)
    for atom in ("N", "CA", "C", "O"):
        atom_index = protein.atom_to_idx[atom]
        torch.testing.assert_close(
            _block_coords(moved, 0)[atom_index], protein_before[atom_index]
        )
