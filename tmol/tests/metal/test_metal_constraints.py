"""Numerical parity tests for Rosetta-style metal geometry restraints."""

import pytest
import torch

from tmol import run_cart_min
from tmol.io import pose_stack_from_biotite
from tmol.metal import setup_metal_constraints
from tmol.score import ScoreFunction, ScoreType
from tmol.tests.io.test_metal_bonds import _CA_SITE, _ZN_SITE, _structure


def _without_water(pdb_text):
    water_serials = {
        int(line[6:11])
        for line in pdb_text.splitlines()
        if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() == "HOH"
    }
    result = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and int(line[6:11]) in water_serials:
            continue
        if line.startswith("CONECT"):
            serials = [int(line[i : i + 5]) for i in range(6, len(line), 5)]
            serials = [serial for serial in serials if serial not in water_serials]
            if len(serials) < 2:
                continue
            line = "CONECT" + "".join(f"{serial:5d}" for serial in serials)
        result.append(line)
    return "\n".join(result) + "\n"


def _constraint_score(pose_stack, param_db):
    score_function = ScoreFunction(param_db, pose_stack.device)
    score_function.set_weight(ScoreType.constraint, 1.0)
    return score_function.render_whole_pose_scoring_module(pose_stack)(
        pose_stack.coords
    ).sum()


def _site_atom_refs(pose_stack):
    pbt = pose_stack.packed_block_types
    for block_index in range(pose_stack.max_n_blocks):
        type_index = int(pose_stack.block_type_ind64[0, block_index])
        if type_index < 0:
            continue
        block_type = pbt.active_block_types[type_index]
        if block_type.name3 not in {"ZN", "CA", "MG"}:
            continue
        connection_index = next(
            index
            for index, connection in enumerate(block_type.connections)
            if connection.name.startswith("metal_")
        )
        donor_block, donor_connection_index = pose_stack.inter_residue_connections64[
            0, block_index, connection_index
        ].tolist()
        donor_type = pbt.active_block_types[
            int(pose_stack.block_type_ind64[0, donor_block])
        ]
        donor_name = donor_type.connections[donor_connection_index].atom
        metal_name = block_type.connections[connection_index].atom
        return (
            int(pose_stack.block_coord_offset64[0, donor_block])
            + donor_type.atom_to_idx[donor_name],
            int(pose_stack.block_coord_offset64[0, block_index])
            + block_type.atom_to_idx[metal_name],
        )
    raise AssertionError("fixture has no metal site")


@pytest.mark.parametrize(
    "pdb_text,n_constraints,donor_reference,metal_reference",
    [
        pytest.param(
            _ZN_SITE,
            12,
            0.252283895876688,
            0.506329562740481,
            id="1CA2-zinc",
        ),
        pytest.param(
            _CA_SITE,
            33,
            0.484668200114718,
            0.922061032062049,
            id="1CLL-calcium",
        ),
    ],
)
def test_metal_constraints_match_pyrosetta_deposited_site_scores(
    torch_device, pdb_text, n_constraints, donor_reference, metal_reference
):
    pose, context = pose_stack_from_biotite(
        _structure(_without_water(pdb_text)),
        torch_device,
        no_optH=True,
        return_context=True,
    )
    pose = setup_metal_constraints(pose)
    assert pose.constraint_set.constraint_atoms.shape[0] == n_constraints
    torch.testing.assert_close(
        _constraint_score(pose, context.parameter_database),
        torch.zeros((), device=torch_device),
        atol=2e-5,
        rtol=0,
    )

    donor_index, metal_index = _site_atom_refs(pose)
    for atom_index, reference in (
        (donor_index, donor_reference),
        (metal_index, metal_reference),
    ):
        perturbed = pose.coords.clone()
        perturbed[0, atom_index, 0] += 0.05
        pose.coords = perturbed
        torch.testing.assert_close(
            _constraint_score(pose, context.parameter_database),
            torch.tensor(reference, device=torch_device),
            atol=3e-4,
            rtol=3e-4,
        )
        pose.coords = perturbed.clone()
        pose.coords[0, atom_index, 0] -= 0.05


def test_metal_constraint_multipliers_and_gradients(torch_device):
    pose, context = pose_stack_from_biotite(
        _structure(_without_water(_ZN_SITE)),
        torch_device,
        no_optH=True,
        return_context=True,
    )
    donor_index, _metal_index = _site_atom_refs(pose)
    scores = []
    for multiplier in (1.0, 2.0):
        constrained = setup_metal_constraints(
            pose,
            distance_multiplier=multiplier,
            angle_multiplier=multiplier,
        )
        coords = constrained.coords.clone().requires_grad_(True)
        coords.data[0, donor_index, 0] += 0.05
        constrained.coords = coords
        score = _constraint_score(constrained, context.parameter_database)
        score.backward()
        assert torch.isfinite(coords.grad).all()
        scores.append(score.detach())
    torch.testing.assert_close(scores[1], 2 * scores[0])


def test_setup_metal_constraints_rejects_negative_multipliers(torch_device):
    pose = pose_stack_from_biotite(
        _structure(_without_water(_ZN_SITE)), torch_device, no_optH=True
    )
    with pytest.raises(ValueError, match="non-negative"):
        setup_metal_constraints(pose, distance_multiplier=-1)


@pytest.mark.parametrize("pdb_text", [_ZN_SITE, _CA_SITE], ids=["zinc", "calcium"])
def test_metal_site_recovers_after_cartesian_minimization(torch_device, pdb_text):
    """A perturbed deposited site is usable in the normal minimization stack."""

    pose, context = pose_stack_from_biotite(
        _structure(_without_water(pdb_text)),
        torch_device,
        no_optH=True,
        return_context=True,
    )
    pose = setup_metal_constraints(pose)
    donor_index, _metal_index = _site_atom_refs(pose)
    deposited = pose.coords[0, donor_index].clone()
    pose.coords = pose.coords.clone()
    pose.coords[0, donor_index, 0] += 0.15

    score_function = ScoreFunction(context.parameter_database, torch_device)
    score_function.set_weight(ScoreType.constraint, 1.0)
    scorer = score_function.render_whole_pose_scoring_module(pose)
    score_before = scorer(pose.coords).sum()

    movable = torch.zeros(pose.coords.shape[:-1], dtype=torch.bool, device=pose.device)
    movable[0, donor_index] = True
    minimized = run_cart_min(
        pose,
        score_function,
        coord_mask=movable,
        optimizer_kwargs={"max_iter": 50},
    )
    score_after = scorer(minimized.coords).sum()

    assert score_after < score_before
    assert score_after < score_before * 0.05
    assert torch.linalg.vector_norm(minimized.coords[0, donor_index] - deposited) < 0.03
