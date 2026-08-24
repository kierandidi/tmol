from biotite.structure.io.pdbx import CIFFile, get_structure
import pytest
import torch

from tmol import (
    ScoreType,
    beta2016_score_function,
    carbohydrate_beta2016_score_function,
)
from tmol.io import pose_stack_from_biotite
from tmol.tests.data import data_path


PYROSETTA_2026_33_SUGAR_BB = 12.252923544755266


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
    )


def test_4byh_sugar_bb_matches_pyrosetta(torch_device):
    pose, context = _glycan_pose(torch_device)
    score_function = carbohydrate_beta2016_score_function(
        torch_device, context.parameter_database
    )
    scorer = score_function.render_whole_pose_scoring_module(pose)
    score_types = score_function.all_score_types()
    sugar_bb = scorer.unweighted_scores(pose.coords)[
        score_types.index(ScoreType.sugar_bb)
    ]

    torch.testing.assert_close(
        sugar_bb,
        torch.tensor([PYROSETTA_2026_33_SUGAR_BB], device=torch_device),
        atol=3e-5,
        rtol=0,
    )

    coords = pose.coords.detach().clone().requires_grad_(True)
    scorer(coords).sum().backward()
    assert torch.isfinite(coords.grad).all()


def test_sugar_bb_block_pair_decomposition(torch_device):
    pose, context = _glycan_pose(torch_device)
    score_function = carbohydrate_beta2016_score_function(
        torch_device, context.parameter_database
    )
    score_function.pre_work_initialization(pose)
    term = next(
        term
        for term in score_function.all_terms()
        if ScoreType.sugar_bb in term.score_types()
    )
    whole = term.render_whole_pose_scoring_module(pose)(pose.coords)
    block_pair = term.render_block_pair_scoring_module(pose)(pose.coords)

    torch.testing.assert_close(block_pair.sum(dim=(2, 3)), whole)
    assert torch.count_nonzero(block_pair) == 10


def test_carbohydrate_score_function_is_explicit():
    beta = beta2016_score_function(torch.device("cpu"))
    carbohydrate = carbohydrate_beta2016_score_function(torch.device("cpu"))

    assert beta.get_weight(ScoreType.sugar_bb) == 0
    assert carbohydrate.get_weight(ScoreType.sugar_bb) == 0.5


def test_sugar_bb_cuda_graph_matches_eager(torch_device):
    if torch_device.type != "cuda":
        pytest.skip("CUDA Graphs require a CUDA device")

    pose, context = _glycan_pose(torch_device)
    score_function = carbohydrate_beta2016_score_function(
        torch_device, context.parameter_database
    )
    scorer = score_function.render_whole_pose_scoring_module(pose)
    sugar = next(term for term in scorer.term_modules if term.classname == "SugarBB")

    eager_coords = pose.coords.detach().clone().requires_grad_(True)
    eager_score = sugar(eager_coords)
    eager_score.sum().backward()
    expected_score = eager_score.detach().clone()
    expected_grad = eager_coords.grad.detach().clone()
    del eager_score, eager_coords

    sample = pose.coords.detach().clone().requires_grad_(True)
    graphed = torch.cuda.make_graphed_callables(
        sugar, (sample,), allow_unused_input=True
    )
    graph_coords = pose.coords.detach().clone().requires_grad_(True)
    graph_score = graphed(graph_coords)
    graph_score.sum().backward()

    torch.testing.assert_close(graph_score, expected_score, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(graph_coords.grad, expected_grad, rtol=1e-4, atol=1e-3)
