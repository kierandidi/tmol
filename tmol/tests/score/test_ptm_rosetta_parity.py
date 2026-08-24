"""Mapped beta_nov16 parity for Rosetta PTM chemical variants."""

from io import StringIO

from biotite.structure.io.pdb import PDBFile, get_structure
import pytest
import torch

from tmol import beta2016_score_function
from tmol.io import (
    extended_pose_stack_from_sequences,
    pose_stack_from_biotite,
    pose_stack_to_pdb_string,
)
from tmol.score.elec import ElecEnergyTerm


# PyRosetta 2026.33, initialized with ``-corrections::beta_nov16 true``.
# Each value is score(AX[modified]A) - score(AX[base]A), using PDB-rounded
# coordinates exported by TMol. Rosetta intra-residue xover4 components are
# added to their corresponding TMol combined score type.
PYROSETTA_PTM_DELTAS = {
    "SEP": (-0.570231322, -0.287998474, 1.438678712),
    "TPO": (-1.382267420, -0.173295013, 3.215433434),
    "PTR": (-0.406762273, 0.001606823, 0.550248011),
    "MLZ": (-3.249998326, 475.174291283, -4.341895811),
    "MLY": (-5.998207843, 993.031300600, -4.594656962),
    "M3L": (-8.973820761, 1926.003048825, -8.841849691),
}

PTMS = (
    ("SER", "phosphorylated", "SEP"),
    ("THR", "phosphorylated", "TPO"),
    ("TYR", "phosphorylated", "PTR"),
    ("LYS", "monomethylated", "MLZ"),
    ("LYS", "dimethylated", "MLY"),
    ("LYS", "trimethylated", "M3L"),
)


def _pdb_rounded_pose(sequence, device):
    pose = extended_pose_stack_from_sequences(sequence, device=torch.device("cpu"))
    structure = get_structure(
        PDBFile.read(StringIO(pose_stack_to_pdb_string(pose))),
        model=1,
        include_bonds=True,
    )
    return pose_stack_from_biotite(
        structure, device, no_optH=True, return_context=True
    )


def _score_map(sequence, device):
    pose, context = _pdb_rounded_pose(sequence, device)
    score_function = beta2016_score_function(
        device, param_db=context.parameter_database
    )
    values = score_function.render_whole_pose_scoring_module(pose)(
        pose.coords, sum_terms=False
    )[:, 0]
    return {
        score_type.name: values[index]
        for index, score_type in enumerate(score_function.all_score_types())
    }


@pytest.mark.parametrize("base,variant,name3", PTMS)
def test_ptm_nonbonded_deltas_match_pyrosetta_beta_nov16(
    torch_device, base, variant, name3
):
    modified = _score_map(f"AX[{base}:{variant}]A", torch_device)
    unmodified = _score_map(f"AX[{base}]A", torch_device)
    expected = PYROSETTA_PTM_DELTAS[name3]

    for score_type, reference in zip(
        ("fa_ljatr", "fa_ljrep", "fa_lk"), expected
    ):
        actual = modified[score_type] - unmodified[score_type]
        torch.testing.assert_close(
            actual,
            torch.tensor(reference, device=torch_device),
            rtol=0.03 if score_type == "fa_lk" else 0.003,
            atol=5e-4,
        )

    # Rosetta variants inherit the base residue's backbone and Dunbrack terms.
    for score_type in (
        "omega",
        "rama",
        "dunbrack_rot",
        "dunbrack_rotdev",
        "dunbrack_semirot",
        "ref",
    ):
        torch.testing.assert_close(
            modified[score_type] - unmodified[score_type],
            torch.zeros((), device=torch_device),
            rtol=0,
            atol=2e-5,
        )


@pytest.mark.parametrize("base,variant,name3", PTMS)
def test_ptm_atom_types_match_rosetta_fa_standard(
    torch_device, base, variant, name3
):
    pose = extended_pose_stack_from_sequences(
        f"AX[{base}:{variant}]A", device=torch_device
    )
    block_type = pose.packed_block_types.active_block_types[
        int(pose.block_type_ind64[0, 1])
    ]
    atom_types = {atom.name: atom.atom_type for atom in block_type.atoms}
    if name3 in {"SEP", "TPO", "PTR"}:
        assert {atom_types[name] for name in ("O1P", "O2P", "O3P")} == {
            "OOC"
        }
        assert atom_types["P"] == "Phos"
        if name3 == "PTR":
            assert atom_types["OH"] == "OH"
    else:
        expected_nitrogen = {"MLZ": "Narg", "MLY": "Ntrp", "M3L": "Npro"}
        assert atom_types["NZ"] == expected_nitrogen[name3]
        assert {
            atom_type
            for atom_name, atom_type in atom_types.items()
            if atom_name.startswith(("CM", "CH"))
        } == {"CH3"}


@pytest.mark.parametrize(
    "variant,methyls",
    [
        ("monomethylated", {"CM": ("HCM1", "HCM2", "HCM3")}),
        (
            "dimethylated",
            {
                "CH1": ("HH11", "HH12", "HH13"),
                "CH2": ("HH21", "HH22", "HH23"),
            },
        ),
        (
            "trimethylated",
            {
                "CM1": ("HM11", "HM12", "HM13"),
                "CM2": ("HM21", "HM22", "HM23"),
                "CM3": ("HM31", "HM32", "HM33"),
            },
        ),
    ],
)
def test_methyl_carbons_use_terminal_hydrogen_count_pair_representative(
    torch_device, default_database, variant, methyls
):
    pose = extended_pose_stack_from_sequences(
        f"AX[LYS:{variant}]A", device=torch_device
    )
    block_type = pose.packed_block_types.active_block_types[
        int(pose.block_type_ind64[0, 1])
    ]
    term = ElecEnergyTerm(default_database, torch_device)
    representatives = term.param_resolver.get_bonded_path_length_mapping_for_block(
        block_type
    )
    for carbon, hydrogens in methyls.items():
        carbon_index = block_type.atom_to_idx[carbon]
        assert representatives[carbon_index] == block_type.atom_to_idx[hydrogens[-1]]
