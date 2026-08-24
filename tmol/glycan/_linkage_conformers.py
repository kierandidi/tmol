"""Rosetta-style statistical glycan-linkage proposals.

Unlike an amino-acid side-chain rotamer, a glycosidic torsion moves the whole
glycan subtree on the non-reducing side of the linkage. The coordinate mover
below finds that component from the chemical graph and rotates it as a unit.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy
import torch

from tmol.pose import PoseStack


@dataclass(frozen=True)
class LinkageConformer:
    """One population-weighted row of Rosetta's linkage-conformer table."""

    bins: str
    population: float
    observations: int
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]


def _conformer(bins, population, observations, *mean_sd):
    return LinkageConformer(
        bins,
        population,
        observations,
        tuple(mean_sd[::2]),
        tuple(mean_sd[1::2]),
    )


# Exact rows from Rosetta's 2017 default.table (Git blob c38bfc306e3e0729) for
# the linkages exercised by TMol's PDB glycan fixtures. Keys are
# (non-reducing CCD name, reducing-end
# family/name, reducing-end connection atom). The right-hand sugar anomer does
# not affect Rosetta's lookup, so MAN/BMA and NAG/NDG share parent families.
_CONFORMERS = {
    ("NAG", "ASN", "ND2"): (
        _conformer(
            "2.1.1.2",
            0.4755,
            2116,
            261.204,
            18.738,
            176.582,
            12.940,
            173.004,
            42.546,
            190.375,
            12.882,
        ),
        _conformer(
            "2.1.1.3",
            0.3218,
            1432,
            266.260,
            19.488,
            175.113,
            11.793,
            164.505,
            40.855,
            289.290,
            9.885,
        ),
        _conformer(
            "2.1.1.1",
            0.1613,
            718,
            260.329,
            21.021,
            178.349,
            15.474,
            193.810,
            39.885,
            63.684,
            8.287,
        ),
        _conformer(
            "1.1.1.1",
            0.0169,
            75,
            63.320,
            13.298,
            183.561,
            16.069,
            189.020,
            22.727,
            66.313,
            9.523,
        ),
        _conformer(
            "1.1.1.3",
            0.0092,
            41,
            61.052,
            28.933,
            187.818,
            27.368,
            150.766,
            60.632,
            293.143,
            16.627,
        ),
        _conformer(
            "1.1.1.2",
            0.0063,
            28,
            66.268,
            38.218,
            198.794,
            46.773,
            164.958,
            61.471,
            195.240,
            21.508,
        ),
        _conformer(
            "2.2.1.2",
            0.0058,
            26,
            276.963,
            7.572,
            356.389,
            6.746,
            162.310,
            7.754,
            193.407,
            3.919,
        ),
        _conformer(
            "2.2.1.1",
            0.0020,
            9,
            250.838,
            41.359,
            10.789,
            21.143,
            152.733,
            27.057,
            63.676,
            18.049,
        ),
        _conformer(
            "1.2.1.2",
            0.0009,
            4,
            62.445,
            16.943,
            0.374,
            2.137,
            178.520,
            9.758,
            201.426,
            4.712,
        ),
    ),
    ("NAG", "NAG", "O4"): (
        _conformer("2.1", 0.9876, 2315, 280.880, 14.334, 235.258, 20.313),
        _conformer("1.1", 0.0124, 29, 67.481, 23.527, 245.531, 25.986),
    ),
    ("BMA", "NAG", "O4"): (
        _conformer("1.1", 1.0, 794, 272.910, 25.183, 228.055, 23.562),
    ),
    ("MAN", "MAN", "O3"): (
        _conformer("1.1", 1.0, 628, 76.504, 25.055, 125.953, 23.759),
    ),
    ("MAN", "MAN", "O6"): (
        _conformer(
            "1.1.2", 0.5599, 299, 71.152, 24.784, 173.040, 33.093, 296.683, 11.407
        ),
        _conformer(
            "1.1.1", 0.4401, 235, 75.813, 38.077, 161.378, 38.128, 61.404, 27.884
        ),
    ),
    ("NAG", "MAN", "O2"): (
        _conformer("2.3", 0.8416, 85, 279.370, 12.638, 152.356, 10.516),
        _conformer("2.2", 0.1089, 11, 272.658, 11.632, 112.072, 3.013),
        _conformer("1.3", 0.0198, 2, 94.100, 18.254, 142.850, 7.159),
    ),
    ("GAL", "NAG", "O4"): (
        _conformer("1.2", 0.9864, 218, 286.858, 19.942, 245.149, 16.571),
        _conformer("1.1", 0.0136, 3, 295.361, 11.097, 83.668, 3.895),
    ),
    ("SIA", "GAL", "O6"): (
        _conformer("1.2.1", 0.5000, 10, 67.001, 10.001, 198.226, 13.909, 62.239, 6.524),
        _conformer("1.1.1", 0.3000, 6, 60.800, 1.739, 104.299, 3.675, 44.349, 2.180),
        _conformer("2.2.1", 0.1500, 3, 111.976, 8.704, 219.293, 10.601, 40.294, 5.870),
    ),
    ("FUC", "NAG", "O6"): (
        _conformer(
            "1.1.2", 0.8187, 149, 287.067, 23.079, 180.443, 35.526, 297.074, 18.501
        ),
        _conformer(
            "1.1.1", 0.1813, 33, 287.514, 45.620, 137.790, 77.920, 65.935, 21.670
        ),
    ),
}

_SUGAR_FAMILY = {
    "NAG": "NAG",
    "NDG": "NAG",
    "MAN": "MAN",
    "BMA": "MAN",
    "GAL": "GAL",
    "GLA": "GAL",
    "GLC": "GLC",
    "BGC": "GLC",
    "FUC": "FUC",
    "FUL": "FUC",
    "SIA": "SIA",
}
_ANOMERIC_ATOMS = {
    name: ("C2", "O6") if name == "SIA" else ("C1", "O5") for name in _SUGAR_FAMILY
}
_PROTEIN_PATHS = {
    "ASN": ("ND2", "CG", "CB", "CA", "N"),
    "SER": ("OG", "CB", "CA", "N"),
    "THR": ("OG1", "CB", "CA", "N"),
}


def get_linkage_conformers(
    child_name: str, parent_name: str, parent_atom: str
) -> tuple[LinkageConformer, ...]:
    """Return Rosetta conformers for a CCD-name linkage, or an empty tuple."""

    parent = _SUGAR_FAMILY.get(parent_name, parent_name)
    return _CONFORMERS.get((child_name, parent, parent_atom), ())


def _linkage(pose_stack: PoseStack, pose: int, child_block: int):
    child = pose_stack.block_type(pose, child_block)
    if child.base_name not in _ANOMERIC_ATOMS:
        raise ValueError(
            f"block {child_block} ({child.base_name}) is not a recognized sugar"
        )
    anomeric, ring_oxygen = _ANOMERIC_ATOMS[child.base_name]
    child_conn = next(
        (
            i
            for i, connection in enumerate(child.connections)
            if connection.atom == anomeric
            and int(pose_stack.inter_residue_connections64[pose, child_block, i, 0])
            >= 0
        ),
        None,
    )
    if child_conn is None:
        raise ValueError(f"block {child_block} has no parent linkage at {anomeric}")
    parent_block, parent_conn = map(
        int,
        pose_stack.inter_residue_connections64[pose, child_block, child_conn],
    )
    parent = pose_stack.block_type(pose, parent_block)
    parent_atom = parent.connections[parent_conn].atom
    return child, parent, parent_block, parent_atom, anomeric, ring_oxygen


def linkage_conformers_for_pose(
    pose_stack: PoseStack, pose: int, child_block: int
) -> tuple[LinkageConformer, ...]:
    """Return statistical conformers for one pose linkage."""

    child, parent, _, parent_atom, _, _ = _linkage(pose_stack, pose, child_block)
    return get_linkage_conformers(child.base_name, parent.base_name, parent_atom)


def sample_linkage_angles(
    conformers: Sequence[LinkageConformer],
    rng: numpy.random.Generator | None = None,
    *,
    idealize: bool = False,
    gaussian: bool = True,
) -> tuple[int, tuple[float, ...]]:
    """Select a conformer by population and sample its torsions as Rosetta does."""

    if not conformers:
        raise ValueError("no linkage conformers are available")
    rng = numpy.random.default_rng() if rng is None else rng
    populations = numpy.asarray([c.population for c in conformers], dtype=float)
    index = int(rng.choice(len(conformers), p=populations / populations.sum()))
    conformer = conformers[index]
    means = numpy.asarray(conformer.means)
    if idealize:
        angles = means
    elif gaussian:
        angles = rng.normal(means, conformer.standard_deviations)
    else:
        widths = numpy.asarray(conformer.standard_deviations)
        angles = rng.uniform(means - widths, means + widths)
    return index, tuple(float(x % 360.0) for x in angles)


def _atom_index(block, name, pose_stack, pose):
    block_type = pose_stack.block_type(pose, block)
    if name not in block_type.atom_to_idx:
        raise ValueError(f"block {block} ({block_type.base_name}) has no atom {name}")
    return (
        int(pose_stack.block_coord_offset64[pose, block]) + block_type.atom_to_idx[name]
    )


def _torsion_quartets(pose_stack, pose, child_block):
    child, parent, parent_block, parent_atom, anomeric, ring_oxygen = _linkage(
        pose_stack, pose, child_block
    )
    if parent.base_name in _SUGAR_FAMILY:
        position = int(parent_atom[1:])
        parent_path = [
            parent_atom,
            f"C{position}",
            f"C{position - 1}",
        ]
        if position > 5:
            parent_path.append(f"C{position - 2}")
    elif parent.base_name in _PROTEIN_PATHS:
        parent_path = _PROTEIN_PATHS[parent.base_name]
    else:
        raise ValueError(f"no linkage atom path is defined for {parent.base_name}")
    chain = [
        _atom_index(child_block, ring_oxygen, pose_stack, pose),
        _atom_index(child_block, anomeric, pose_stack, pose),
        *(_atom_index(parent_block, atom, pose_stack, pose) for atom in parent_path),
    ]
    return tuple(tuple(chain[i : i + 4]) for i in range(len(chain) - 3))


def _pose_graph(pose_stack, pose):
    adjacency = [set() for _ in range(pose_stack.max_n_pose_atoms)]
    n_blocks = int(pose_stack.n_res_per_pose[pose])
    for block in range(n_blocks):
        bt = pose_stack.block_type(pose, block)
        offset = int(pose_stack.block_coord_offset64[pose, block])
        for atom1, atom2 in bt.bond_indices:
            atom1, atom2 = offset + int(atom1), offset + int(atom2)
            adjacency[atom1].add(atom2)
            adjacency[atom2].add(atom1)
        for conn, connection in enumerate(bt.connections):
            partner_block, partner_conn = map(
                int, pose_stack.inter_residue_connections64[pose, block, conn]
            )
            if partner_block <= block:
                continue
            partner = pose_stack.block_type(pose, partner_block)
            atom1 = offset + bt.atom_to_idx[connection.atom]
            atom2 = (
                int(pose_stack.block_coord_offset64[pose, partner_block])
                + partner.atom_to_idx[partner.connections[partner_conn].atom]
            )
            adjacency[atom1].add(atom2)
            adjacency[atom2].add(atom1)
    return adjacency


def _component(adjacency, start, cut1, cut2):
    seen = {start}
    todo = [start]
    while todo:
        atom = todo.pop()
        for neighbor in adjacency[atom]:
            if {atom, neighbor} == {cut1, cut2} or neighbor in seen:
                continue
            seen.add(neighbor)
            todo.append(neighbor)
    if cut2 in seen:
        raise ValueError("cannot rotate a bond that is part of a chemical ring")
    return sorted(seen)


def _dihedral(points):
    p0, p1, p2, p3 = points.double()
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 * torch.rsqrt((b1 * b1).sum().clamp_min(1e-18))
    v = b0 - (b0 * b1).sum() * b1
    w = b2 - (b2 * b1).sum() * b1
    return torch.rad2deg(
        torch.atan2((torch.cross(b1, v, dim=-1) * w).sum(), (v * w).sum())
    )


def _set_dihedral(coords, adjacency, atoms, target):
    current = float(_dihedral(coords[list(atoms)]))
    delta = ((target - current + 180.0) % 360.0) - 180.0
    moving = _component(adjacency, atoms[0], atoms[1], atoms[2])
    origin = coords[atoms[1]]
    axis = coords[atoms[2]] - origin
    axis = axis * torch.rsqrt((axis * axis).sum().clamp_min(1e-18))
    angle = torch.as_tensor(
        -numpy.deg2rad(delta), dtype=coords.dtype, device=coords.device
    )
    xyz = coords[moving] - origin
    rotated = (
        xyz * torch.cos(angle)
        + torch.cross(axis.expand_as(xyz), xyz, dim=-1) * torch.sin(angle)
        + axis * (xyz * axis).sum(-1, keepdim=True) * (1.0 - torch.cos(angle))
    )
    coords[moving] = rotated + origin


def get_glycan_linkage_torsions(
    pose_stack: PoseStack, pose: int, child_block: int
) -> tuple[float, ...]:
    """Measure the IUPAC phi/psi/omega torsions for one glycan linkage."""

    quartets = _torsion_quartets(pose_stack, pose, child_block)
    return tuple(
        float(_dihedral(pose_stack.coords[pose, list(atoms)])) % 360.0
        for atoms in quartets
    )


def set_glycan_linkage_torsions(
    pose_stack: PoseStack,
    pose: int,
    child_block: int,
    angles: Sequence[float],
) -> PoseStack:
    """Return a pose with linkage torsions set by rotating downstream atoms."""

    quartets = _torsion_quartets(pose_stack, pose, child_block)
    if len(angles) > len(quartets):
        raise ValueError(f"linkage defines {len(quartets)} torsions, got {len(angles)}")
    moved = pose_stack.clone()
    graph = _pose_graph(moved, pose)
    with torch.no_grad():
        for atoms, target in zip(quartets, angles):
            _set_dihedral(moved.coords[pose], graph, atoms, float(target) % 360.0)
    return moved


def apply_linkage_conformer(
    pose_stack: PoseStack,
    pose: int,
    child_block: int,
    conformer: LinkageConformer,
    rng: numpy.random.Generator | None = None,
    *,
    idealize: bool = False,
    gaussian: bool = True,
) -> tuple[PoseStack, tuple[float, ...]]:
    """Apply one specified conformer and return the moved pose and angles."""

    _, angles = sample_linkage_angles(
        (conformer,), rng, idealize=idealize, gaussian=gaussian
    )
    return set_glycan_linkage_torsions(pose_stack, pose, child_block, angles), angles


def sample_glycan_linkage(
    pose_stack: PoseStack,
    pose: int,
    child_block: int,
    rng: numpy.random.Generator | None = None,
    *,
    idealize: bool = False,
    gaussian: bool = True,
) -> tuple[PoseStack, int, tuple[float, ...]]:
    """Population-sample and apply a Rosetta linkage conformer."""

    conformers = linkage_conformers_for_pose(pose_stack, pose, child_block)
    index, angles = sample_linkage_angles(
        conformers, rng, idealize=idealize, gaussian=gaussian
    )
    moved = set_glycan_linkage_torsions(pose_stack, pose, child_block, angles)
    return moved, index, angles
