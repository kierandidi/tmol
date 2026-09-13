"""Regressions from AtomWorks' modified-component structures, without metals."""

from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest
import torch

from tmol.io import (
    atom_array_from_cif,
    build_context_from_biotite,
    pose_stack_from_biotite,
)
from tmol.ligand import chem_comp_types_from_cif
from tmol.ligand._registry import _applied_patch
from tmol.score.elec._params import ElecParamResolver
from tmol.tests.io.test_atomworks_corpus_regressions import _score_and_minimize

DATA = Path(__file__).parents[1] / "data" / "atomworks_regressions"


@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
def test_missing_ligand_carbon_reconstructs_and_backpropagates(reader, torch_device):
    from tmol.io import canonical_form_from_biotite, pose_stack_from_canonical_form

    array = atom_array_from_cif(
        DATA / "missing_ligand_carbon_5hs6.cif.gz", reader=reader
    )
    array = array[(np.char.upper(array.element) != "NA") & (array.res_name != "HOH")]
    ligand = array[array.res_name == "J3Z"]
    assert int((ligand.atom_name == "C6").sum()) == 1
    assert np.isnan(ligand.coord[ligand.atom_name == "C6"]).all()
    context = build_context_from_biotite(
        array, torch_device, prepare_ligands=True, ligand_seed=20260909
    )
    pose = pose_stack_from_biotite(array, torch_device, context=context, no_optH=True)
    types = pose.packed_block_types.active_block_types
    block = next(
        i for i, t in enumerate(pose.block_type_ind[0]) if types[int(t)].name == "J3Z"
    )
    residue = types[int(pose.block_type_ind[0, block])]
    observed = np.isfinite(ligand.coord).all(axis=-1)
    indices = [residue.atom_to_idx[name] for name in ligand.atom_name[observed]]
    _score_and_minimize(pose, context)

    # The tensor constructor must route gradients through the carbon and then
    # its dependent hydrogens, including passes that initially leave them NaN.
    canonical = canonical_form_from_biotite(
        ligand, torch_device, co=context.canonical_ordering
    )
    args = list(canonical)
    source = canonical.coords.clone().requires_grad_()
    targets = [residue.atom_to_idx[name] for name in ("C6", "HC4", "HC5", "HC6")]

    def rebuild(coords):
        args[2] = coords
        return pose_stack_from_canonical_form(
            context.canonical_ordering, context.packed_block_types, *args
        )

    rebuilt = rebuild(source)
    assert torch.isfinite(rebuilt.coords[rebuilt.real_atoms]).all()
    torch.testing.assert_close(
        rebuilt.coords[0, indices],
        torch.as_tensor(ligand.coord[observed], device=torch_device),
        rtol=0,
        atol=0,
    )
    rebuilt.coords[0, targets].sum().backward()
    assert torch.isfinite(source.grad).all()
    assert torch.count_nonzero(source.grad) > 0
    assert torch.count_nonzero(source.grad[torch.isnan(source)]) == 0
    torch.testing.assert_close(source.grad.sum(dim=(0, 1, 2)), source.new_full((3,), 4))
    direction = torch.zeros_like(source)
    # Perturb the strongest source derivative to avoid a vacuous zero comparison.
    direction.flatten()[source.grad.abs().argmax()] = 1
    epsilon = 0.01
    with torch.no_grad():
        positive = rebuild(source + epsilon * direction).coords[0, targets].sum()
        negative = rebuild(source - epsilon * direction).coords[0, targets].sum()
    torch.testing.assert_close(
        (positive - negative) / (2 * epsilon),
        (source.grad * direction).sum(),
        rtol=0.005,
        atol=0.005,
    )
    with pytest.raises(ValueError, match="missing non-leaf atom"):
        rebuild(torch.full_like(source, float("nan")))


@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
def test_single_atom_plp_backbone_packs_and_preserves_chirality(
    reader, torch_device, monkeypatch
):
    from tmol.pack.rotamer import create_mainchain_fingerprint
    from tmol.tests.io.test_atomworks_corpus_regressions import (
        _assert_all_source_connections,
    )

    array = atom_array_from_cif(DATA / "plp_cap_5t4j.cif.gz", reader=reader)
    array = array[array.res_name != "HOH"]
    context = build_context_from_biotite(
        array, torch_device, prepare_ligands=True, ligand_seed=20260909
    )
    # Missing sidechain atoms trigger packing and fingerprint the PLP cap.
    assert np.isnan(array.coord).any()
    pose = pose_stack_from_biotite(array, torch_device, context=context, no_optH=True)
    # The author chain A also names the separate ligand entities. The protein
    # still ends at histidine and needs its terminal oxygen through both readers.
    assert (
        sum(
            pose.packed_block_types.active_block_types[int(i)].name == "HIS:cterm"
            for i in pose.block_type_ind[0]
            if i >= 0
        )
        == 1
    )
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    resolved = np.array(
        [np.isfinite(array.coord[a:b]).any() for a, b in zip(starts[:-1], starts[1:])]
    )
    # AtomWorks additionally carries five completely unresolved protein residues.
    assert int((~resolved).sum()) == (5 if reader == "atomworks" else 0)
    _assert_all_source_connections(pose, array[np.repeat(resolved, np.diff(starts))])
    plp = next(rt for rt in context.restype_set.residue_types if rt.name == "PLP")
    assert plp.properties.polymer.mainchain_atoms == ("C4A",)
    original = create_mainchain_fingerprint(
        plp, (), context.parameter_database.chemical
    )[1]
    assert len(set(original)) == plp.n_atoms
    assert {original[plp.atom_to_idx[name]].chirality for name in ("HC4", "HC5")} == {
        1,
        2,
    }
    with monkeypatch.context() as patch:
        patch.setattr(plp, "ideal_coords", plp.ideal_coords * [-1, 1, 1])
        mirrored = create_mainchain_fingerprint(
            plp, (), context.parameter_database.chemical
        )[1]
    for first, reflected in zip(original, mirrored):
        expected = 3 - first.chirality if first.chirality in (1, 2) else first.chirality
        assert reflected.chirality == expected
    _score_and_minimize(pose, context)


@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
def test_chromophore_with_one_terminal_patch_constructs_and_minimizes(
    reader, torch_device
):
    array = atom_array_from_cif(DATA / "chromophore_3nez.cif.gz", reader=reader)
    array = array[array.res_name != "HOH"]
    context = build_context_from_biotite(
        array, torch_device, prepare_ligands=True, ligand_seed=20260909
    )
    assert context.canonical_ordering.restypes_default_termini_mapping["NRQ"] == (
        None,
        "cterm",
    )
    pose = pose_stack_from_biotite(array, torch_device, context=context, no_optH=True)
    types = pose.packed_block_types.active_block_types
    chromophores = [
        (i, types[int(t)])
        for i, t in enumerate(pose.block_type_ind[0])
        if t >= 0 and types[int(t)].base_name == "NRQ"
    ]
    assert (
        len(chromophores)
        == int((array.res_name[struc.get_residue_starts(array)] == "NRQ").sum())
        == 4
    )
    for block, restype in chromophores:
        for port in (restype.down_connection_ind, restype.up_connection_ind):
            assert port >= 0
            assert pose.inter_residue_connections[0, block, port, 0] >= 0
    _score_and_minimize(pose, context)


def test_aromatic_acyl_cap_keeps_every_heavy_atom_in_its_tree():
    path = DATA / "modified_components_6q9t.cif"
    array = atom_array_from_cif(path)
    # Retain 4SO and its directly attached A1IJ4 partner. The complete source
    # also contains zinc; this regression concerns the organic atom tree.
    indices = struc.get_residue_positions(array, np.arange(len(array)))
    selected = set(indices[array.res_name == "4SO"])
    bonds = array.bonds.as_array()
    linked = bonds[np.isin(indices[bonds[:, :2]], list(selected)).any(axis=1), :2]
    selected.update(indices[linked].flatten())
    array = array[np.isin(indices, list(selected))]
    assert set(array.res_name) == {"4SO", "A1IJ4"}
    context = build_context_from_biotite(
        array,
        torch.device("cpu"),
        prepare_ligands=True,
        ligand_seed=20260909,
        chem_comp_types=chem_comp_types_from_cif(path),
    )
    residue = next(r for r in context.restype_set.residue_types if r.name == "4SO")
    observed = set(array.atom_name[array.res_name == "4SO"])
    assert observed <= set(residue.atom_to_idx)
    assert observed <= {ic.name for ic in residue.icoors}
    assert np.isfinite(residue.compute_ideal_coords()).all()


@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
def test_plp_enzyme_completion_preserves_residues_and_terminal_chemistry(
    reader, torch_device
):
    path = DATA / "plp_enzyme_7mkv.cif"
    array = atom_array_from_cif(path, reader=reader)
    starts = struc.get_residue_starts(array)
    terminal_arg = (array.chain_id == "B") & (array.res_id == 437)
    assert terminal_arg[starts].sum() == 1
    assert {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"} <= set(
        array.atom_name[terminal_arg]
    )
    context = build_context_from_biotite(
        array,
        torch_device,
        prepare_ligands=True,
        ligand_seed=20260909,
        chem_comp_types=chem_comp_types_from_cif(path),
    )
    db = context.parameter_database
    base = next(r for r in db.chemical.residues if r.name == "LLP")
    terminal = next(r for r in db.chemical.residues if r.name == "LLP:cterm")
    patch = _applied_patch(db.chemical, base, terminal)
    assert patch.display_name == "cterm"
    assert patch.applies_to.matches(base)
    charges = {
        q.atom: q.charge
        for q in db.scoring.elec.atom_charge_parameters
        if q.res == "LLP:cterm"
    }
    assert np.isfinite(charges["OXT"])
    resolver = ElecParamResolver.from_database(db.scoring.elec, torch_device)
    for residue in context.restype_set.residue_types:
        if residue.name.split(":")[0] == "LLP":
            assert np.isfinite(resolver.get_partial_charges_for_block(residue)).all()
    pose = pose_stack_from_biotite(array, torch_device, context=context, no_optH=True)
    _score_and_minimize(pose, context)
