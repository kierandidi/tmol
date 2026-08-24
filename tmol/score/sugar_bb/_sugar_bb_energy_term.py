"""Rosetta's carbohydrate intrinsic (CHI) linkage energy."""

import numpy
import torch

from tmol.chemical import RefinedResidueType
from tmol.database import ParameterDatabase
from tmol.pose import PackedBlockTypes, PoseStack
from tmol.score import EnergyTerm


# CHIEnergyFunctionLinkageType order in Rosetta: alpha, beta, the two
# axial/equatorial psi classes, alpha-1,6, beta-1,6. These are the published
# CHI parameters shipped in Rosetta's scoring/score_functions/carbohydrates.
_CHI_ROWS = (
    (
        (2.977, 102.25, 10.745, 3.6735, 2.061, 6.1939, -2.1115, -98.001),
        (-199.49, 170.6, -105.31, 6.2012, 91.655, -22.979, 83.602, 170.01),
        (677.81, 1696.8, 4724.6, 1347.7, 1500.0, 2122.3, 1254.1, 1598.7),
        0.0,
    ),
    (
        (450.54, 23.712, 5.9353, 22.467, 10.036, -18.141, 5.8823),
        (-330.77, 304.63, -152.08, -23.516, 120.96, -24.268, 19.632),
        (4449.8, 8375.2, 6049.8, 606.9, 4038.0, 543.05, 897.93),
        -2.1283,
    ),
    (
        (4.6237, 4.6139, 4.9419, 0.4029, 0.79888, 0.22299),
        (5.0456, 362.49, 121.2, 241.43, 68.425, 192.93),
        (5005.8, 2090.6, 2093.8, 456.83, 678.81, 347.25),
        -0.12565,
    ),
    (
        (4.4681, 4.382, 284.95, 4.7613, -169.2, -118.44),
        (1.0e-30, 357.77, 146.64, 220.68, 147.37, 146.06),
        (1279.6, 6050.1, 1551.8, 5892.9, 1742.5, 1359.8),
        1.022,
    ),
    (
        (67.943, 6.134, 3.276, 0.728, 2.574, 5.760, 3.475, -0.741),
        (-59.539, 10.479, 54.296, 131.068, 245.103, 360.0, 321.678, 199.107),
        (993.324, 945.771, 851.528, 1037.412, 2012.995, 1153.397, 2080.971, 522.180),
        0.0,
    ),
    (
        (7.249, 1.900, 0.741, 0.200, 0.287, 1.226, 7.411, -0.615, -0.350),
        (3.606, 96.593, 141.664, 162000.0, 228.171, 292.206, 369.868, 271.319, 183.0),
        (
            2459.239,
            2683.887,
            1150.048,
            400.0,
            272.201,
            1134.525,
            3499.160,
            532.437,
            100.0,
        ),
        0.0,
    ),
)

# name3: (anomer class [alpha=0, beta=1], is L, anomeric C, ring O,
#          axial acceptor positions). PDB CCD names distinguish the anomer.
_SUGARS = {
    "GLC": (0, False, "C1", "O5", ()),
    "BGC": (1, False, "C1", "O5", ()),
    "NAG": (1, False, "C1", "O5", ()),
    "NDG": (0, False, "C1", "O5", ()),
    "MAN": (0, False, "C1", "O5", (2,)),
    "BMA": (1, False, "C1", "O5", (2,)),
    "GLA": (0, False, "C1", "O5", (4,)),
    "GAL": (1, False, "C1", "O5", (4,)),
    "FUC": (0, True, "C1", "O5", (4,)),
    "FUL": (1, True, "C1", "O5", (4,)),
    "SIA": (0, False, "C2", "O6", ()),
}


def _dihedral(p0, p1, p2, p3, valid, reference):
    # Some packed entries are padding or linkages without psi/omega atoms.
    # Replace those coordinates before doing any vector algebra: masking only
    # the final energy still lets undefined norm gradients leak through as NaN.
    p0 = torch.where(valid.unsqueeze(-1), p0, reference[0])
    p1 = torch.where(valid.unsqueeze(-1), p1, reference[1])
    p2 = torch.where(valid.unsqueeze(-1), p2, reference[2])
    p3 = torch.where(valid.unsqueeze(-1), p3, reference[3])
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 * torch.rsqrt((b1 * b1).sum(-1, keepdim=True).clamp_min(1e-18))
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    y, x = (torch.cross(b1, v, dim=-1) * w).sum(-1), (v * w).sum(-1)
    degenerate = (x.abs() + y.abs()) < 1e-12
    x = torch.where(degenerate, torch.ones_like(x), x)
    y = torch.where(degenerate, torch.zeros_like(y), y)
    return torch.atan2(y, x) * (180.0 / torch.pi)


def _chi_energy(linkage_type, angle, a, b, c, d, mask):
    valid = linkage_type >= 0
    index = linkage_type.clamp_min(0).to(torch.int64)
    dtype = angle.dtype
    terms = a.to(dtype)[index] * torch.exp(
        -((angle.unsqueeze(-1) - b.to(dtype)[index]) ** 2) / c.to(dtype)[index]
    )
    energy = (terms * mask[index]).sum(-1) + d.to(dtype)[index]
    return torch.where(valid, energy, torch.zeros_like(energy))


def _omega_energy(preference, angle):
    valid = preference >= 0
    anti = preference == 0
    low, high = angle <= 120.0, angle > 240.0
    theta = torch.where(low, 60.0, torch.where(high, 300.0, 180.0))
    anti_offset = torch.where(low, 0.0, torch.where(high, 1.0, 0.3))
    gauche_offset = torch.where(low, 0.21, torch.where(high, 0.0, 1.39))
    offset = torch.where(anti, anti_offset, gauche_offset)
    energy = 0.0025 * (angle - theta) ** 2 + offset
    return torch.where(valid, energy, torch.zeros_like(energy))


def _block_type_params(block_type):
    meta = _SUGARS.get(block_type.base_name)
    child_atoms = numpy.full(2, -1, dtype=numpy.int32)
    child_conn = -1
    anomer = -1
    is_l = False
    if meta is not None:
        anomer, is_l, anomeric_c, ring_o, _ = meta
        for conn, connection in enumerate(block_type.connections):
            if connection.atom == anomeric_c and connection.name.startswith(
                "covalent_"
            ):
                child_conn = conn
                break
        if child_conn >= 0 and {anomeric_c, ring_o} <= block_type.atom_to_idx.keys():
            child_atoms[:] = (
                block_type.atom_to_idx[ring_o],
                block_type.atom_to_idx[anomeric_c],
            )

    n_conn = len(block_type.connections)
    psi_type = numpy.full(n_conn, -1, dtype=numpy.int32)
    exocyclic = numpy.zeros(n_conn, dtype=numpy.bool_)
    omega_pref = numpy.full(n_conn, -1, dtype=numpy.int32)
    parent_atoms = numpy.full((n_conn, 4), -1, dtype=numpy.int32)
    downstream = numpy.asarray(block_type.atom_downstream_of_conn, dtype=numpy.int32)
    parent_atoms[:, : min(4, downstream.shape[1])] = downstream[:, :4]
    if meta is not None:
        _, _, anomeric_c, _, axial_positions = meta
        is_aldose_hexopyranose = anomeric_c == "C1"
        for conn, connection in enumerate(block_type.connections):
            atom = connection.atom
            if not (atom.startswith("O") and atom[1:].isdigit()):
                continue
            position = int(atom[1:])
            carbon_path = (
                atom,
                f"C{position}",
                f"C{position - 1}",
                f"C{position - 2}",
            )
            for i, name in enumerate(carbon_path):
                if name in block_type.atom_to_idx:
                    parent_atoms[conn, i] = block_type.atom_to_idx[name]
            if position in (2, 3, 4) and is_aldose_hexopyranose:
                axial = position in axial_positions
                psi_type[conn] = 2 if axial == (position % 2 == 0) else 3
            elif position > 5:
                exocyclic[conn] = True
                if is_aldose_hexopyranose:
                    omega_pref[conn] = 0 if 4 in axial_positions else 1
    return (
        child_atoms,
        child_conn,
        anomer,
        is_l,
        psi_type,
        exocyclic,
        omega_pref,
        parent_atoms,
    )


class SugarBBEnergyTerm(EnergyTerm):
    """Differentiable Rosetta ``sugar_bb`` energy for recognized PDB sugars."""

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super().__init__(param_db=param_db, device=device)
        self.device = device
        n_terms = max(len(row[0]) for row in _CHI_ROWS)
        arrays = [
            numpy.zeros((len(_CHI_ROWS), n_terms)),
            numpy.zeros((len(_CHI_ROWS), n_terms)),
            # Padded Gaussian terms have zero amplitude, but still need a
            # finite width so their masked gradients remain well-defined.
            numpy.ones((len(_CHI_ROWS), n_terms)),
        ]
        mask = numpy.zeros((len(_CHI_ROWS), n_terms))
        intercept = numpy.zeros(len(_CHI_ROWS))
        for i, (aa, bb, cc, dd) in enumerate(_CHI_ROWS):
            for target, values in zip(arrays, (aa, bb, cc)):
                target[i, : len(values)] = values
            mask[i, : len(aa)] = 1.0
            intercept[i] = dd
        self.chi_params = tuple(
            torch.tensor(value, dtype=torch.float32, device=device)
            for value in (*arrays, intercept, mask)
        )
        # CUDA Graph capture cannot copy a newly constructed CPU constant to
        # the device from inside the score call.
        self.dihedral_reference = torch.tensor(
            (
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 1.0, 1.0),
            ),
            dtype=torch.float32,
            device=device,
        )

    @classmethod
    def class_name(cls):
        return "SugarBB"

    @classmethod
    def score_types(cls):
        from tmol.score.terms._sugar_bb_creator import SugarBBTermCreator

        return SugarBBTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super().setup_block_type(block_type)
        if not hasattr(block_type, "sugar_bb_params"):
            block_type.sugar_bb_params = _block_type_params(block_type)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super().setup_packed_block_types(packed_block_types)
        if hasattr(packed_block_types, "sugar_bb_params"):
            return
        block_params = [
            bt.sugar_bb_params for bt in packed_block_types.active_block_types
        ]
        n_types = len(block_params)
        max_conn = packed_block_types.max_n_conn

        child_atoms = numpy.full((n_types, 2), -1, dtype=numpy.int32)
        child_conn = numpy.full(n_types, -1, dtype=numpy.int32)
        anomer = numpy.full(n_types, -1, dtype=numpy.int32)
        is_l = numpy.zeros(n_types, dtype=numpy.bool_)
        psi_type = numpy.full((n_types, max_conn), -1, dtype=numpy.int32)
        exocyclic = numpy.zeros((n_types, max_conn), dtype=numpy.bool_)
        omega_pref = numpy.full((n_types, max_conn), -1, dtype=numpy.int32)
        parent_atoms = numpy.full((n_types, max_conn, 4), -1, dtype=numpy.int32)
        for i, params in enumerate(block_params):
            child_atoms[i], child_conn[i], anomer[i], is_l[i] = params[:4]
            n_conn = len(params[4])
            psi_type[i, :n_conn], exocyclic[i, :n_conn], omega_pref[i, :n_conn] = (
                params[4:7]
            )
            parent_atoms[i, :n_conn] = params[7]

        def to_tensor(value):
            return torch.tensor(value, device=self.device)

        packed_block_types.sugar_bb_params = tuple(
            to_tensor(value)
            for value in (
                child_atoms,
                child_conn,
                anomer,
                is_l,
                psi_type,
                exocyclic,
                omega_pref,
                parent_atoms,
            )
        )

    def setup_poses(self, poses: PoseStack):
        super().setup_poses(poses)

    def get_pose_score_term_function(self):
        return eval_sugar_bb_for_pose

    def get_rotamer_score_term_function(self):
        return eval_sugar_bb_for_rotamers

    def get_score_term_attributes(self, pose_stack):
        topology = _pose_linkage_metadata(
            pose_stack.block_type_ind,
            pose_stack.inter_residue_connections,
            pose_stack.block_coord_offset,
            pose_stack.coords.shape[1],
            *pose_stack.packed_block_types.sugar_bb_params,
        )
        return [
            *topology,
            *self.chi_params,
            self.dihedral_reference,
        ]


def _pose_linkage_metadata(
    block_type,
    inter_res_conn,
    block_coord_offset,
    max_n_atoms,
    child_atoms_by_bt,
    child_conn_by_bt,
    anomer_by_bt,
    is_l_by_bt,
    psi_by_conn,
    exocyclic_by_conn,
    omega_by_conn,
    parent_atoms_by_bt,
):
    """Precompute coordinate-independent linkage indices and masks."""

    n_poses, max_n_blocks = block_type.shape
    bt = block_type.to(torch.int64)
    real = bt >= 0
    safe = bt.clamp_min(0)
    child_conn = child_conn_by_bt.to(torch.int64)[safe]
    pose = torch.arange(n_poses, device=bt.device).view(n_poses, 1)
    block = torch.arange(max_n_blocks, device=bt.device).view(1, max_n_blocks)
    partner = inter_res_conn.to(torch.int64)[pose, block, child_conn.clamp_min(0)]
    parent_block, parent_conn = partner[..., 0], partner[..., 1]
    valid = real & (child_conn >= 0) & (parent_block >= 0)
    parent_bt = torch.gather(bt, 1, parent_block.clamp_min(0))
    valid = valid & (parent_bt >= 0)
    parent_safe = parent_bt.clamp_min(0)
    parent_conn_safe = parent_conn.clamp_min(0)

    anomer = anomer_by_bt.to(torch.int64)[safe]
    child_local = child_atoms_by_bt.to(torch.int64)[safe]
    parent_local = parent_atoms_by_bt.to(torch.int64)[parent_safe, parent_conn_safe]
    parent_exocyclic = exocyclic_by_conn[parent_safe, parent_conn_safe]
    psi_type = psi_by_conn.to(torch.int64)[parent_safe, parent_conn_safe]
    psi_type = torch.where(parent_exocyclic, anomer + 4, psi_type)
    omega_pref = omega_by_conn.to(torch.int64)[parent_safe, parent_conn_safe]
    child_index = block_coord_offset.to(torch.int64).unsqueeze(-1) + child_local
    parent_offset = torch.gather(
        block_coord_offset.to(torch.int64), 1, parent_block.clamp_min(0)
    )
    parent_index = parent_offset.unsqueeze(-1) + parent_local
    pose_offset = torch.arange(n_poses, device=bt.device).view(n_poses, 1, 1)
    pose_offset = pose_offset * max_n_atoms
    child_index = (pose_offset + child_index).clamp_min(0)
    parent_index = (pose_offset + parent_index).clamp_min(0)
    phi_valid = (
        valid & (child_local >= 0).all(-1) & (parent_local[..., :2] >= 0).all(-1)
    )
    psi_valid = phi_valid & (psi_type >= 0) & (parent_local[..., 2] >= 0)
    omega_valid = phi_valid & (omega_pref >= 0) & (parent_local[..., 3] >= 0)
    pose_index = torch.arange(n_poses, device=bt.device).view(n_poses, 1)
    child_block = torch.arange(max_n_blocks, device=bt.device).view(1, max_n_blocks)
    block_pair_index = (
        pose_index * max_n_blocks * max_n_blocks
        + child_block * max_n_blocks
        + parent_block.clamp_min(0)
    )
    return (
        phi_valid,
        psi_valid,
        omega_valid,
        anomer,
        is_l_by_bt[safe],
        psi_type,
        omega_pref,
        (psi_type < 4) & is_l_by_bt[parent_safe],
        child_index,
        parent_index,
        block_pair_index,
    )


def eval_sugar_bb_for_pose(
    rot_coords,
    _rot_coord_offset,
    _pose_ind_for_atom,
    _first_rot_for_block,
    _first_rot_block_type,
    _block_ind_for_rot,
    _pose_ind_for_rot,
    _block_type_ind_for_rot,
    _n_rots_for_pose,
    _rot_offset_for_pose,
    _n_rots_for_block,
    _rot_offset_for_block,
    _max_n_rots_per_pose,
    phi_valid,
    psi_valid,
    omega_valid,
    anomer,
    child_l,
    psi_type,
    omega_pref,
    invert_psi,
    child_index,
    parent_index,
    block_pair_index,
    chi_a,
    chi_b,
    chi_c,
    chi_d,
    chi_mask,
    dihedral_reference,
    output_block_pair_energies: bool,
):
    n_poses, max_n_blocks = phi_valid.shape
    child_xyz = rot_coords[child_index.reshape(-1)].reshape(*child_index.shape, 3)
    parent_xyz = rot_coords[parent_index.reshape(-1)].reshape(*parent_index.shape, 3)
    phi = _dihedral(
        child_xyz[..., 0, :],
        child_xyz[..., 1, :],
        parent_xyz[..., 0, :],
        parent_xyz[..., 1, :],
        phi_valid,
        dihedral_reference,
    )
    psi = _dihedral(
        child_xyz[..., 1, :],
        parent_xyz[..., 0, :],
        parent_xyz[..., 1, :],
        parent_xyz[..., 2, :],
        psi_valid,
        dihedral_reference,
    )
    omega = _dihedral(
        parent_xyz[..., 0, :],
        parent_xyz[..., 1, :],
        parent_xyz[..., 2, :],
        parent_xyz[..., 3, :],
        omega_valid,
        dihedral_reference,
    )
    phi = torch.where(child_l, -phi, phi)
    psi = psi % 360.0
    psi = torch.where(invert_psi, 360.0 - psi, psi)
    omega = omega % 360.0

    score = _chi_energy(anomer, phi, chi_a, chi_b, chi_c, chi_d, chi_mask)
    score = score + torch.where(
        psi_valid,
        _chi_energy(psi_type, psi, chi_a, chi_b, chi_c, chi_d, chi_mask),
        torch.zeros_like(score),
    )
    score = score + torch.where(
        omega_valid,
        _omega_energy(omega_pref, omega),
        torch.zeros_like(score),
    )
    score = torch.where(phi_valid, score, torch.zeros_like(score))

    if not output_block_pair_energies:
        return score.sum(-1).unsqueeze(0), None
    output = torch.zeros(
        n_poses * max_n_blocks * max_n_blocks,
        dtype=score.dtype,
        device=score.device,
    )
    output.index_add_(0, block_pair_index[phi_valid], score[phi_valid])
    return output.view(1, n_poses, max_n_blocks, max_n_blocks), None


def eval_sugar_bb_for_rotamers(*args, **kwargs):
    raise RuntimeError(
        "sugar_bb rotamer-pair scoring is not implemented; use it for Cartesian "
        "scoring/minimization and keep weight 0 during discrete packing"
    )
