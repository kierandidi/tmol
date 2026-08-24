import pytest
from biotite.structure.io.pdbx import CIFFile, get_structure
import torch

from tmol import (
    ScoreType,
    beta2016_score_function,
    carbohydrate_beta2016_score_function,
)
from tmol.io import pose_stack_from_biotite
from tmol.pose import PoseStackBuilder
from tmol.score import ScoreFunction
from tmol.tests.data import data_path


def _glycan_pose(device, batch):
    structure = get_structure(
        CIFFile.read(data_path("cif", "4BYH_chainC_glycan.cif")),
        model=1,
        include_bonds=True,
    )
    pose, context = pose_stack_from_biotite(
        structure,
        device,
        prepare_ligands=True,
        no_optH=True,
        return_context=True,
    )
    if batch > 1:
        pose = PoseStackBuilder.from_poses([pose] * batch, device)
    return pose, context


@pytest.mark.benchmark(group="sugar_bb_forward_backward")
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("full_score", [False, True], ids=["sugar", "full"])
def test_sugar_bb_forward_backward_benchmark(
    benchmark, torch_device, batch, full_score
):
    pose, context = _glycan_pose(torch_device, batch)
    if full_score:
        score_function = carbohydrate_beta2016_score_function(
            torch_device, context.parameter_database
        )
    else:
        score_function = ScoreFunction(context.parameter_database, torch_device)
        score_function.set_weight(ScoreType.sugar_bb, 1.0)
    scorer = score_function.render_whole_pose_scoring_module(pose)
    coords = pose.coords.detach().clone().requires_grad_(True)

    @benchmark
    def forward_backward():
        score = scorer(coords).sum()
        score.backward()
        coords.grad = None
        return score.cpu()

    forward_backward


@pytest.mark.benchmark(group="sugar_bb_compiled_hybrid")
@pytest.mark.parametrize("batch", [1, 32])
def test_sugar_bb_compiled_hybrid_benchmark(benchmark, torch_device, batch):
    if torch_device.type != "cuda":
        pytest.skip("CUDA benchmark")

    pose, context = _glycan_pose(torch_device, batch)
    eager = carbohydrate_beta2016_score_function(
        torch_device, context.parameter_database
    ).render_whole_pose_scoring_module(pose)
    base = beta2016_score_function(
        torch_device, context.parameter_database
    ).render_whole_pose_scoring_module(pose)
    sugar_function = ScoreFunction(context.parameter_database, torch_device)
    sugar_function.set_weight(ScoreType.sugar_bb, 1.0)
    sugar = sugar_function.render_whole_pose_scoring_module(pose)
    compiled_sugar = torch.compile(sugar, fullgraph=True, mode="reduce-overhead")

    def hybrid(coords):
        return base(coords) + 0.5 * compiled_sugar(coords)

    eager_coords = pose.coords.detach().clone().requires_grad_(True)
    hybrid_coords = pose.coords.detach().clone().requires_grad_(True)
    eager_score = eager(eager_coords).sum()
    hybrid_score = hybrid(hybrid_coords).sum()
    eager_score.backward()
    hybrid_score.backward()
    torch.testing.assert_close(hybrid_score, eager_score, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(
        hybrid_coords.grad, eager_coords.grad, rtol=1e-4, atol=1e-3
    )

    coords = pose.coords.detach().clone().requires_grad_(True)

    @benchmark
    def forward_backward():
        score = hybrid(coords).sum()
        score.backward()
        coords.grad = None
        return score.cpu()

    forward_backward
