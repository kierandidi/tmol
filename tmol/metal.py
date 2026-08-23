"""Rosetta-compatible geometry restraints for deposited metal sites."""

from itertools import combinations
import math

import attrs
import torch

from tmol.io._covalent_bonds import METAL_RESIDUE_NAMES
from tmol.pose import ConstraintSet, PoseStack
from tmol.score.constraint import ConstraintEnergyTerm


def _atom_ref(pose_index, block_index, atom_index):
    return (pose_index, block_index, atom_index)


def _coord(pose_stack, atom_ref):
    pose_index, block_index, atom_index = atom_ref
    offset = pose_stack.block_coord_offset64[pose_index, block_index]
    return pose_stack.coords[pose_index, offset + atom_index]


def _distance(pose_stack, first, second):
    return torch.linalg.vector_norm(
        _coord(pose_stack, first) - _coord(pose_stack, second)
    ).item()


def _angle(pose_stack, first, center, last):
    vector1 = _coord(pose_stack, first) - _coord(pose_stack, center)
    vector2 = _coord(pose_stack, last) - _coord(pose_stack, center)
    return torch.atan2(
        torch.linalg.vector_norm(torch.linalg.cross(vector1, vector2, dim=-1)),
        torch.sum(vector1 * vector2),
    ).item()


def _add_constraints(constraint_set, function, atoms, params, device):
    if not atoms:
        return constraint_set
    return constraint_set.add_constraints(
        function,
        torch.tensor(atoms, dtype=torch.int32, device=device),
        torch.tensor(params, dtype=torch.float32, device=device),
    )


def setup_metal_constraints(
    pose_stack: PoseStack,
    distance_multiplier: float = 1.0,
    angle_multiplier: float = 1.0,
) -> PoseStack:
    """Add Rosetta SetupMetalsMover restraints at deposited geometry.

    For every explicitly connected metal site this adds harmonic restraints for
    proxy--donor, metal--proxy, and all proxy--proxy distances, plus circular
    harmonic restraints for metal--donor--donor-parent angles. The Rosetta
    standard deviations are 0.1 Angstrom and 0.05 radians. Multipliers scale
    the corresponding energies; non-positive multipliers disable that family.

    Donor-orientation restraints are omitted when the donor parent is virtual,
    as deposited waters do not define an experimental orientation.
    """

    if distance_multiplier < 0 or angle_multiplier < 0:
        raise ValueError("metal constraint multipliers must be non-negative")

    distance_atoms = []
    distance_params = []
    angle_atoms = []
    angle_params = []
    pbt = pose_stack.packed_block_types

    for pose_index in range(pose_stack.n_poses):
        for metal_block in range(pose_stack.max_n_blocks):
            type_index = int(pose_stack.block_type_ind64[pose_index, metal_block])
            if type_index < 0:
                continue
            metal_type = pbt.active_block_types[type_index]
            if metal_type.name3 not in METAL_RESIDUE_NAMES:
                continue

            site = []
            metal_connections = [
                (connection_index, connection)
                for connection_index, connection in enumerate(metal_type.connections)
                if connection.name.startswith("metal_")
            ]
            for proxy_ordinal, (connection_index, connection) in enumerate(
                metal_connections, start=1
            ):
                donor_block, donor_connection_index = (
                    pose_stack.inter_residue_connections64[
                        pose_index, metal_block, connection_index
                    ].tolist()
                )
                if donor_block < 0:
                    continue
                donor_type_index = int(
                    pose_stack.block_type_ind64[pose_index, donor_block]
                )
                donor_type = pbt.active_block_types[donor_type_index]
                donor_connection = donor_type.connections[donor_connection_index]
                metal_ref = _atom_ref(
                    pose_index, metal_block, metal_type.atom_to_idx[connection.atom]
                )
                proxy_ref = _atom_ref(
                    pose_index,
                    metal_block,
                    metal_type.atom_to_idx[f"V{proxy_ordinal}"],
                )
                donor_ref = _atom_ref(
                    pose_index,
                    donor_block,
                    donor_type.atom_to_idx[donor_connection.atom],
                )
                parent_name = next(
                    icoor.parent
                    for icoor in donor_type.icoors
                    if icoor.name == donor_connection.atom
                )
                parent_ref = _atom_ref(
                    pose_index, donor_block, donor_type.atom_to_idx[parent_name]
                )
                parent_is_virtual = (
                    parent_ref[2] == donor_ref[2]
                    or parent_name in donor_type.properties.virtual
                )
                site.append(
                    (metal_ref, proxy_ref, donor_ref, parent_ref, parent_is_virtual)
                )

            if distance_multiplier > 1e-10:
                distance_sd = 0.1 / math.sqrt(distance_multiplier)
                for metal_ref, proxy_ref, donor_ref, _parent_ref, _donor_type in site:
                    for atom_pair in ((proxy_ref, donor_ref), (metal_ref, proxy_ref)):
                        distance_atoms.append(atom_pair)
                        distance_params.append(
                            [
                                _distance(pose_stack, *atom_pair),
                                distance_sd,
                            ]
                        )
                for first, second in combinations(site, 2):
                    atom_pair = (first[1], second[1])
                    distance_atoms.append(atom_pair)
                    distance_params.append(
                        [
                            _distance(pose_stack, *atom_pair),
                            distance_sd,
                        ]
                    )

            if angle_multiplier > 1e-10:
                angle_sd = 0.05 / math.sqrt(angle_multiplier)
                for metal_ref, _proxy_ref, donor_ref, parent_ref, parent_virtual in site:
                    if parent_virtual:
                        continue
                    angle_atoms.append((metal_ref, donor_ref, parent_ref))
                    angle_params.append(
                        [
                            _angle(
                                pose_stack, metal_ref, donor_ref, parent_ref
                            ),
                            angle_sd,
                        ]
                    )

    constraint_set = pose_stack.constraint_set or ConstraintSet.create_empty(
        pose_stack.device, pose_stack.n_poses
    )
    constraint_set = _add_constraints(
        constraint_set,
        ConstraintEnergyTerm.harmonic,
        distance_atoms,
        distance_params,
        pose_stack.device,
    )
    constraint_set = _add_constraints(
        constraint_set,
        ConstraintEnergyTerm.circularharmonic_angle,
        angle_atoms,
        angle_params,
        pose_stack.device,
    )
    if not distance_atoms and not angle_atoms:
        return pose_stack
    return attrs.evolve(pose_stack, constraint_set=constraint_set)
