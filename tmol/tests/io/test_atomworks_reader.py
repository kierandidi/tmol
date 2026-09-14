"""Exercise shared parsing through chemical preparation and minimization."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tmol.io import atom_array_from_cif, pose_stack_from_cif
from tmol.score import beta2016_score_function

pytest.importorskip("atomworks")

DATA = Path(__file__).parents[1] / "data"


def test_file_hydrogen_policy_through_scoring(tmp_path, ubq_pdb):
    from biotite.structure.io import pdbx
    from tmol.io import biotite_from_pose_stack, pose_stack_from_pdb

    device = torch.device("cpu")
    source = biotite_from_pose_stack(pose_stack_from_pdb(ubq_pdb, device))
    index = np.flatnonzero((source.res_name == "ALA") & (source.element == "H"))[0]
    source.coord[index] += [0.2, 0.1, -0.1]
    file = pdbx.CIFFile()
    pdbx.set_structure(file, source)
    path = tmp_path / "hydrogens.cif"
    file.write(path)
    array = atom_array_from_cif(path)
    selected = (
        (array.res_id == source.res_id[index])
        & (array.atom_name == source.atom_name[index])
        & (array.chain_id == source.chain_id[index])
    )
    np.testing.assert_allclose(
        array.coord[selected], source.coord[index][None], atol=0.001
    )
    pose = pose_stack_from_cif(path, device, no_optH=True)
    restored = biotite_from_pose_stack(pose)
    selected = (
        (restored.res_id == source.res_id[index])
        & (restored.atom_name == source.atom_name[index])
        & (restored.chain_id == source.chain_id[index])
    )
    np.testing.assert_allclose(
        restored.coord[selected], source.coord[index][None], atol=0.001
    )
    coords = pose.coords.detach().clone().requires_grad_()
    energy = beta2016_score_function(device).render_whole_pose_scoring_module(pose)(
        coords
    )
    energy.sum().backward()
    assert torch.isfinite(energy).all()
    assert torch.isfinite(coords.grad).all()


@pytest.mark.parametrize(
    "fixture",
    [
        "ncaa_fixtures/capped_peptide_ace_nh2.cif",
        "ncaa_fixtures/beta_peptide_3c3g.cif",
        "ncaa_fixtures/na_dna_8og_183d.cif",
        "ncaa_fixtures/na_dna_5mc_1d17.cif",
        "ncaa_fixtures/na_rna_2ome_310d.cif",
        "ncaa_fixtures/na_dna_ttd_1ttd.cif",
        "atomworks_regressions/hydrolase_intermediate_1tqh.cif.gz",
        "atomworks_regressions/phosphate_charge_4js1.cif.gz",
        "atomworks_regressions/chloride_complex_4hbt.cif.gz",
        "atomworks_regressions/triphosphate_rna_4gxy.cif",
        "atomworks_regressions/unresolved_modified_polymer_1xj9.cif.gz",
        "atomworks_regressions/missing_phosphate_rna_5w1i.cif",
    ],
)
def test_shared_parser_builds_and_scores_general_chemistry(fixture, torch_device):
    import biotite.structure as struc
    from tmol.tests.io.test_atomworks_corpus_regressions import (
        _assert_all_source_connections,
        _score_and_minimize,
    )

    pose, context = pose_stack_from_cif(
        DATA / fixture,
        torch_device,
        prepare_ligands=True,
        ligand_seed=20260909,
        no_optH=True,
        return_context=True,
    )
    array = atom_array_from_cif(DATA / fixture)
    array = array[array.res_name != "HOH"]
    if "1tqh" in fixture or "4hbt" in fixture:
        residues = list(struc.residue_iter(array))
        unresolved = np.array([not np.isfinite(r.coord).any() for r in residues])
        # AtomWorks also restores wholly unresolved protein residues.
        # Their absence from the constructed pose must not hide observed atoms.
        expected = 0
        assert int(unresolved.sum()) == expected
        keep = np.repeat(~unresolved, [len(r) for r in residues])
        assert not array.hetero[~keep].any()
        array = array[keep]
    if "1xj9" in fixture:
        from collections import Counter

        residues = list(struc.residue_iter(array))
        excluded = [
            r.res_name[0] == "LYS" or not np.isfinite(r.coord).any() for r in residues
        ]
        expected = {"LYS": 2}
        assert (
            Counter(r.res_name[0] for r, skip in zip(residues, excluded) if skip)
            == expected
        )
        keep = np.repeat(np.logical_not(excluded), [len(r) for r in residues])
        # Only the two isolated lysine nitrogens have observed coordinates.
        assert np.isfinite(array.coord[~keep]).all(-1).sum() == 2
        array = array[keep]
        assert len(list(struc.residue_iter(array))) == 16
    if "5w1i" in fixture:
        residues = list(struc.residue_iter(array))
        assert [r.res_name[0] for r in residues] == ["A", "G", "C", "C"]
        assert not np.isfinite(residues[1].coord[residues[1].atom_name == "O3'"]).any()
        # The backbone-incomplete G is excluded; C retains its phosphate
        # across that gap and builds it from its own resolved sugar frame.
        array = array[np.repeat([True, False, True, True], list(map(len, residues)))]
        cytidine = residues[2]
        # The author reader can also retain the template's unresolved
        # leaving oxygen; it is not part of an internal nucleotide type.
        leaving = cytidine.atom_name == "OP3"
        assert not np.isfinite(cytidine.coord[leaving]).any()
        cytidine = cytidine[~leaving]
        missing = ~np.isfinite(cytidine.coord).all(-1)
        assert set(cytidine.atom_name[missing]) == {"P", "OP1", "OP2"}
        bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, 1])]
        offset = int(pose.block_coord_offset[0, 1])
        indices = [offset + bt.atom_to_idx[str(name)] for name in cytidine.atom_name]
        actual = pose.coords[0, indices].detach().cpu().numpy()
        np.testing.assert_array_equal(actual[~missing], cytidine.coord[~missing])
        assert np.isfinite(actual[missing]).all()
    _assert_all_source_connections(pose, array)
    if "/na_" in fixture:
        # Capping must displace only terminal oxygen, preserving both retained
        # phosphate oxygens and their supplied coordinates in the final pose.
        for i, residue in enumerate(struc.residue_iter(array)):
            bt = pose.packed_block_types.active_block_types[
                int(pose.block_type_ind[0, i])
            ]
            offset = int(pose.block_coord_offset[0, i])
            for name in ("OP1", "OP2"):
                observed = residue[
                    (residue.atom_name == name) & np.isfinite(residue.coord).all(-1)
                ]
                if len(observed):
                    assert name in bt.atom_to_idx
                    np.testing.assert_allclose(
                        pose.coords[0, offset + bt.atom_to_idx[name]].detach().cpu(),
                        observed.coord[0],
                        atol=1e-6,
                    )
        if "8og" in fixture:
            nucleotide = array[array.res_name == "8OG"]
            assert "OP2" in nucleotide.atom_name and "OP3" not in nucleotide.atom_name
    if "1tqh" in fixture:
        # The observed tetrahedral intermediate has four single bonds at CAI:
        # restoring the free component's carbonyl would overfill that carbon.
        ligand = array.res_name == "4PA"
        carbon = int(np.flatnonzero(ligand & (array.atom_name == "CAI"))[0])
        oxygen = int(np.flatnonzero(ligand & (array.atom_name == "OAD"))[0])
        neighbors, orders = array.bonds.get_bonds(carbon)
        assert len(neighbors) == 4 and np.all(orders == struc.BondType.SINGLE)
        assert oxygen in neighbors and array.charge[oxygen] == -1
        assert (
            np.count_nonzero(
                (array.res_name[neighbors] == "SER")
                & (array.atom_name[neighbors] == "OG")
            )
            == 1
        )
        for bi, residue in enumerate(struc.residue_iter(array)):
            if residue.res_name[0] == "4PA":
                bt = pose.packed_block_types.active_block_types[
                    int(pose.block_type_ind[0, bi])
                ]
                assert "conj_CAI" in bt.connection_to_cidx
                offset = int(pose.block_coord_offset[0, bi])
                indices = [offset + bt.atom_to_idx[str(n)] for n in residue.atom_name]
                np.testing.assert_array_equal(
                    pose.coords[0, indices].detach().cpu(), residue.coord
                )
    if "4js1" in fixture or "4hbt" in fixture:
        from tmol.tests.ligand.test_local_conjugate_params import _charges

        ion, names, charge = (
            ("PO4", {"P", "O1", "O2", "O3", "O4"}, -3)
            if "4js1" in fixture
            else ("CL", {"CL"}, -1)
        )
        for bi, residue in enumerate(struc.residue_iter(array)):
            if residue.res_name[0] != ion:
                continue
            bt = pose.packed_block_types.active_block_types[
                int(pose.block_type_ind[0, bi])
            ]
            assert set(bt.atom_to_idx) == names
            assert sum(
                _charges(context.parameter_database, bt).values()
            ) == pytest.approx(charge, abs=1e-8)
            offset = int(pose.block_coord_offset[0, bi])
            indices = [offset + bt.atom_to_idx[str(n)] for n in residue.atom_name]
            np.testing.assert_array_equal(
                pose.coords[0, indices].detach().cpu(), residue.coord
            )
    if "4gxy" in fixture:
        nucleotide = next(struc.residue_iter(array))
        bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, 0])]
        assert bt.base_name == "GTP"
        assert {connection.name for connection in bt.connections} == {"up"}
        aliases = {alias.alt_name: alias.name for alias in bt.atom_aliases}
        elements = {
            at.name: at.element for at in context.parameter_database.chemical.atom_types
        }
        assert sum(elements[atom.atom_type] == "P" for atom in bt.atoms) == 3
        missing = ~np.isfinite(nucleotide.coord).all(-1)
        assert set(nucleotide.atom_name[missing]) == {"PG", "O1G", "O2G", "O3G"}
        indices = [
            bt.atom_to_idx[aliases.get(str(name), str(name))]
            for name in nucleotide.atom_name
        ]
        actual = pose.coords[0, indices].detach().cpu().numpy()
        np.testing.assert_array_equal(actual[~missing], nucleotide.coord[~missing])
        assert np.isfinite(actual[missing]).all()
    _score_and_minimize(pose, context, max_iter=100)


@pytest.mark.parametrize("state", ["HD1", "HE2", "both", "none"])
def test_reader_preserves_observed_histidine_tautomer_evidence(tmp_path, state):
    from biotite.structure import info
    from biotite.structure.io import pdbx

    source = info.residue("HIS")
    source.chain_id[:] = "A"
    source.res_id[:] = 1
    removed = {"HD1": ["HE2"], "HE2": ["HD1"], "both": [], "none": ["HD1", "HE2"]}[
        state
    ]
    source = source[~np.isin(source.atom_name, removed)]
    file = pdbx.CIFFile()
    pdbx.set_structure(file, source)
    path = tmp_path / "histidine.cif"
    file.write(path)
    parsed = atom_array_from_cif(path)
    for name in ("HD1", "HE2"):
        observed = source.atom_name == name
        retained = parsed.atom_name == name
        assert bool(retained.any()) == bool(observed.any())
        if observed.any():
            np.testing.assert_allclose(
                parsed.coord[retained], source.coord[observed], atol=0.001
            )


def test_pdb_modified_polymer_preserves_chain_and_scores(tmp_path, torch_device):
    from biotite.structure.io import pdb
    from tmol.io import atom_array_from_file, pose_stack_from_pdb
    from tmol.tests.io.test_atomworks_corpus_regressions import (
        _assert_all_source_connections,
        _score_and_minimize,
    )

    source = atom_array_from_file(DATA / "ncaa_fixtures/phosphopeptide_5ema.cif")
    source = source[
        np.isfinite(source.coord).all(-1) & ~np.isin(source.element, ["H", "D"])
    ]
    file = pdb.PDBFile()
    file.set_structure(source)
    path = tmp_path / "phosphopeptide.pdb"
    file.write(path)
    observed = file.get_structure(model=1)
    raw = atom_array_from_file(path)
    np.testing.assert_array_equal(raw.coord, observed.coord)
    assert raw.bonds is not None  # Preserve any authored CONECT records.
    array = atom_array_from_file(path)
    assert set(array.chain_id) == set(source.chain_id)
    assert np.all(array.tmol_polymer_entity)
    assert set(array.res_name) == set(source.res_name)
    pose, context = pose_stack_from_pdb(
        path, torch_device, prepare_ligands=True, return_context=True, ligand_seed=17
    )
    charges = context.parameter_database.scoring.elec.atom_charge_parameters
    # Database partial charges are rounded; their sum retains phosphate's -2 state.
    assert sum(row.charge for row in charges if row.res == "SEP") == pytest.approx(
        -2, abs=1e-3
    )
    _assert_all_source_connections(pose, array)
    for start in np.flatnonzero(
        np.r_[True, observed.res_id[1:] != observed.res_id[:-1]]
    ):
        resid = observed.res_id[start]
        block = np.flatnonzero(pose.pdb_info.residue_labels[0] == resid)[0]
        bt = pose.packed_block_types.active_block_types[
            int(pose.block_type_ind[0, block])
        ]
        selected = observed.res_id == resid
        offset = int(pose.block_coord_offset[0, block])
        actual = pose.coords[
            0, [offset + bt.atom_to_idx[name] for name in observed.atom_name[selected]]
        ]
        np.testing.assert_array_equal(
            actual.detach().cpu().numpy(), observed.coord[selected]
        )
    _score_and_minimize(pose, context, max_iter=20)


def test_known_and_unknown_chemistry_share_file_contract(
    tmp_path, monkeypatch, torch_device
):
    """MOL2 preparation makes atom-only PDB/CIF sufficient for scoring."""
    from biotite.structure.io import pdb, pdbx
    from tmol.io import (
        atom_array_from_file,
        pose_stack_from_file,
        pose_stack_from_biotite,
    )
    from tmol.ligand import prepare_ligand_from_mol2, write_params_from_mol2
    from tmol.ligand._detect import nonstandard_residue_info_from_mol2
    from tmol.ligand._preparation import LigandPreparationError
    from tmol.tests.io.test_atomworks_corpus_regressions import _score_and_minimize
    import atomworks.io.utils.ccd as ccd
    import tmol.ligand._preparation as preparation

    mol2 = DATA / "protein_ligand_test/ace.lig.mol2"
    source = nonstandard_residue_info_from_mol2(mol2, res_name="ZZQ").atom_array
    params_path = tmp_path / "ligand.tmol"
    param_db, _ = prepare_ligand_from_mol2(mol2, res_name="ZZQ", seed=17)
    write_params_from_mol2(mol2, params_path, res_name="ZZQ", seed=17, format="tmol")
    declared_bonds = source.bonds
    source.bonds = None

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "Known chemistry must not regenerate or consult disabled CCD"
        )

    for extension in ("pdb", "cif"):
        file = pdb.PDBFile() if extension == "pdb" else pdbx.CIFFile()
        if extension == "pdb":
            file.set_structure(source)
        else:
            pdbx.set_structure(file, source)
            for category in list(file.block):
                if category != "atom_site":
                    del file.block[category]
        path = tmp_path / f"coordinates.{extension}"
        file.write(path)
        if extension == "pdb":
            connected = source.copy()
            connected.bonds = declared_bonds
            file.set_structure(connected)
            connected_path = tmp_path / "connectivity.pdb"
            file.write(connected_path)
            for use_ccd in (False, True):
                untyped = atom_array_from_file(connected_path, use_ccd=use_ccd)
                assert untyped.bonds.get_bond_count() > 0
                assert np.all(untyped.bonds.as_array()[:, 2] == 0)
                with pytest.raises(
                    LigandPreparationError, match="ZZQ.*chemical bond orders"
                ):
                    pose_stack_from_biotite(
                        untyped, torch_device, prepare_ligands=True, use_ccd=False
                    )
        for use_ccd in (False, True):
            with monkeypatch.context() as patch:
                if not use_ccd:
                    patch.setattr(ccd, "atom_array_from_bundled_ccd_code", unexpected)
                patch.setattr(preparation, "_prepare_ligand_via_smiles", unexpected)
                array = atom_array_from_file(path, use_ccd=use_ccd)
                np.testing.assert_allclose(array.coord, source.coord, atol=0.001)
                assert list(array.atom_name) == list(source.atom_name)
                pose, context = pose_stack_from_file(
                    path,
                    torch_device,
                    use_ccd=use_ccd,
                    prepare_ligands=True,
                    param_db=param_db,
                    no_optH=True,
                    return_context=True,
                )
                _score_and_minimize(pose, context, max_iter=10)
                reloaded = pose_stack_from_file(
                    path,
                    torch_device,
                    use_ccd=use_ccd,
                    ligand_params_files=[str(params_path)],
                    no_optH=True,
                )
                assert torch.isfinite(reloaded.coords[reloaded.real_atoms]).all()
            with pytest.raises(
                LigandPreparationError, match="ZZQ.*chemical bond orders"
            ):
                pose_stack_from_biotite(
                    array, torch_device, prepare_ligands=True, use_ccd=False
                )
