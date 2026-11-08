"""Chemical atom records and PDB atom formatting, independent of execution."""

from dataclasses import dataclass
from functools import lru_cache

import numpy

from tmol_core.cache import AnnotationKey, PreparationCache

from .pdb_info import DEFAULT_ATOM_OCCUPANCY, DEFAULT_ATOM_B_FACTOR

atom_record_dtype = numpy.dtype(
    [
        ("record_name", str, 6),
        ("modeli", int),
        ("chaini", int),
        ("resi", int),
        ("atomi", int),
        ("model", str, 64),
        ("chain", str, 1),
        ("resn", str, 3),
        ("atomn", str, 4),
        ("x", float),
        ("y", float),
        ("z", float),
        ("location", str, 1),
        ("insert", str, 1),
        ("occupancy", float),
        ("b", float),
    ]
)
# Extended records retain author labels until fixed-width text formatting.
extended_atom_record_dtype = numpy.dtype(
    [
        (name, object if name in ("chain", "atomn", "insert") else dtype)
        for name, dtype in atom_record_dtype.descr
    ]
    + [("element", object)]
)


@dataclass(frozen=True, slots=True, eq=False)
class _ChemicalRecordKey:
    source: AnnotationKey
    block_types: object
    atom_types: object

    def matches(self, other):
        return (
            isinstance(other, _ChemicalRecordKey)
            and self.block_types is other.block_types
            and self.atom_types is other.atom_types
            and self.source.matches(other.source)
        )


def _readonly(values):
    result = numpy.array(values, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True, eq=False)
class AtomRecordPlan:
    """Prepared chemical selection; coordinates and author metadata stay live.

    Construct with ``prepare_atom_record_plan`` and reuse only while topology,
    chemical sources and output policies remain unchanged. Compact selected
    columns are owned snapshots, not a retained full record/coordinate buffer.
    """

    _n_input: int
    _block_types: object
    _keep: object
    _poses: numpy.ndarray
    _blocks: numpy.ndarray
    _serials: numpy.ndarray
    _names: numpy.ndarray
    _elements: object
    _residue_names: object
    _extended: bool

    def records(
        self,
        coords,
        *,
        chain_indices,
        residue_labels,
        chain_labels,
        insertion_codes,
        occupancy,
        b_factor,
        residue_names=None,
        record_names=None,
        split_block_mapping=None,
    ):
        coords = numpy.asarray(coords)
        if coords.shape != (self._n_input, 3):
            raise ValueError("Atom indices and coordinates must describe the same rows")
        if self._n_input == 0:
            return numpy.empty(
                0,
                dtype=(
                    extended_atom_record_dtype if self._extended else atom_record_dtype
                ),
            )
        if residue_names is None and self._residue_names is None:
            raise ValueError("This atom record plan requires residue names")
        keep, poses, blocks = self._keep, self._poses, self._blocks
        records = numpy.empty(
            len(poses),
            dtype=extended_atom_record_dtype if self._extended else atom_record_dtype,
        )
        records["record_name"] = (
            "ATOM  " if record_names is None else numpy.asarray(record_names)[keep]
        )
        records["modeli"] = poses
        records["model"] = poses + 1
        records["chaini"] = numpy.asarray(chain_indices)[keep]
        records["resi"] = numpy.asarray(residue_labels)[keep]
        records["atomi"] = self._serials
        records["chain"] = numpy.asarray(chain_labels)[keep]
        records["resn"] = (
            self._residue_names
            if residue_names is None
            else numpy.asarray(residue_names)[keep]
        )
        records["atomn"] = self._names
        for i, field in enumerate(("x", "y", "z")):
            records[field] = coords[keep, i]
        records["location"] = " " if self._extended else ""
        records["insert"] = numpy.asarray(insertion_codes)[keep]
        records["occupancy"] = numpy.asarray(occupancy)[keep]
        records["b"] = numpy.asarray(b_factor)[keep]
        if self._extended:
            records["element"] = self._elements

        if split_block_mapping is not None:
            for entry in split_block_mapping.entries:
                # Author numbering can repeat across chains and distinct ligands.
                selected = (poses == entry.pose_ind) & (blocks == entry.block_ind)
                records["resn"][selected] = self._block_types[
                    entry.orig_block_type_ind
                ].name
                records["resi"][selected] = entry.orig_residue_label
                records["chain"][selected] = entry.orig_chain_label
                records["insert"][selected] = entry.orig_ins_code
        return records


def prepare_atom_record_plan(
    block_types,
    type_indices,
    pose_indices,
    block_indices,
    atom_indices,
    *,
    atom_types=None,
    extended=False,
    include_virtual=False,
    chemistry_owner=None,
    default_residue_names=True,
):
    """Prepare one host atom selection without retaining dynamic molecular data.

    Rows are grouped by pose in output order. Unemitted chemical types and atoms
    are not visited. Set ``default_residue_names=False`` when each records call
    supplies its own residue names, including legacy aliases or custom labels.
    """
    type_indices = numpy.asarray(type_indices)
    pose_indices = numpy.asarray(pose_indices)
    block_indices = numpy.asarray(block_indices)
    atom_indices = numpy.asarray(atom_indices)
    n = len(type_indices)
    if any(
        values.shape != (n,) for values in (pose_indices, block_indices, atom_indices)
    ):
        raise ValueError("Atom indices must describe the same rows")

    if n == 0:
        empty = _readonly(numpy.empty(0, int))
        return AtomRecordPlan(
            n,
            block_types,
            slice(None),
            empty,
            empty,
            empty,
            empty,
            None,
            None,
            extended,
        )

    if type_indices.dtype.kind not in "iu" or atom_indices.dtype.kind not in "iu":
        raise TypeError("Atom type and local atom indices must have integer dtype")

    if extended and atom_types is None:
        raise ValueError("Extended records require chemical atom types")

    # Identify unique type/atom pairs without padding or inspecting unused types.
    # The temporary key preserves original palette indices when decoded below.
    first_atom = int(atom_indices.min())
    width = int(atom_indices.max()) - first_atom + 1
    keys, inverse = numpy.unique(
        type_indices.astype(numpy.int64, copy=False) * width
        + atom_indices.astype(numpy.int64, copy=False)
        - first_atom,
        return_inverse=True,
    )

    def chemical_identities():
        names = numpy.empty(len(keys), dtype=object)
        virtual = numpy.zeros(len(keys), dtype=bool)
        elements = numpy.empty(len(keys), dtype=object) if extended else None
        residue_name3 = (
            numpy.empty(len(keys), dtype=object) if default_residue_names else None
        )
        element_by_type = (
            {at.name: at.element for at in atom_types} if extended else None
        )
        for i, (type_index, atom_index) in enumerate(
            zip(keys // width, keys % width + first_atom)
        ):
            bt = block_types[type_index]
            atom = bt.atoms[atom_index]
            names[i] = atom.name
            if not include_virtual:
                virtual[i] = atom.name in bt.properties.virtual
            if extended:
                elements[i] = element_by_type[atom.atom_type]
            if default_residue_names:
                residue_name3[i] = bt.name3
        return names, virtual, elements, residue_name3

    chem_db = getattr(chemistry_owner, "chem_db", None)
    if (
        chem_db is not None
        and block_types is chemistry_owner.active_block_types
        and (not extended or atom_types is chem_db.atom_types)
    ):
        cache = getattr(chemistry_owner, "_export_preparations", None)
        if cache is None:
            cache = chemistry_owner._export_preparations = PreparationCache()
        key = AnnotationKey.from_sources(
            chem_db,
            settings=(
                keys.tobytes(),
                first_atom,
                width,
                extended,
                include_virtual,
                default_residue_names,
            ),
        )
        # Collections may be replaced on a live owner/database. Retain exact
        # referents so replacements invalidate without relying on recycled ids.
        key = _ChemicalRecordKey(key, block_types, atom_types if extended else None)
        names, virtual, elements, residue_name3 = cache.get_or_create(
            "pdb_atoms", key, chemical_identities
        )
    else:
        names, virtual, elements, residue_name3 = chemical_identities()

    keep = slice(None) if include_virtual or not virtual.any() else ~virtual[inverse]
    if not isinstance(keep, slice):
        keep = _readonly(keep)
    poses = _readonly(pose_indices[keep])
    blocks = _readonly(block_indices[keep])
    serials = _readonly(numpy.arange(len(poses)) - numpy.searchsorted(poses, poses) + 1)
    selected = inverse[keep]
    return AtomRecordPlan(
        n,
        block_types,
        keep,
        poses,
        blocks,
        serials,
        _readonly(names[selected]),
        _readonly(elements[selected]) if extended else None,
        _readonly(residue_name3[selected]) if default_residue_names else None,
        extended,
    )


def prepare_atom_records(
    block_types,
    type_indices,
    pose_indices,
    block_indices,
    atom_indices,
    coords,
    *,
    chain_indices,
    residue_labels,
    chain_labels,
    insertion_codes,
    occupancy,
    b_factor,
    residue_names=None,
    record_names=None,
    split_block_mapping=None,
    atom_types=None,
    extended=False,
    include_virtual=False,
    chemistry_owner=None,
):
    """Project aligned host atom arrays onto chemical and author identities.

    Rows must be grouped by pose, in output atom order. Indices identify each
    atom's type, pose, block and local atom; metadata arrays are already aligned
    to those rows by the execution adapter. Coordinates retain double precision
    in the records. The extended schema preserves full author labels and adds
    chemical elements; the legacy schema retains its historical fixed widths.
    An optional packed-block-types owner reuses chemical identities only when
    its immutable collections are supplied; coordinates and author data are
    always read anew.
    """
    plan = prepare_atom_record_plan(
        block_types,
        type_indices,
        pose_indices,
        block_indices,
        atom_indices,
        atom_types=atom_types,
        extended=extended,
        include_virtual=include_virtual,
        chemistry_owner=chemistry_owner,
        default_residue_names=residue_names is None,
    )
    return plan.records(
        coords,
        chain_indices=chain_indices,
        residue_labels=residue_labels,
        chain_labels=chain_labels,
        insertion_codes=insertion_codes,
        occupancy=occupancy,
        b_factor=b_factor,
        residue_names=residue_names,
        record_names=record_names,
        split_block_mapping=split_block_mapping,
    )


@lru_cache(maxsize=256)
def format_atom_name(name, *, strip=False):
    if strip:
        name = name.strip()
    if name.startswith(("H", "C", "N", "O", "S")) and len(name) < 4:
        return " {0:<3}".format(name)
    return "{0:<4}".format(name)


ATOM_RECORD_FORMAT = (
    "ATOM  {atomi:5d} {atomn:^4}{location:^1}{resn:3s} {chain:1}{resi:4d}{insert:1s}   "
    "{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b:6.2f}\n"
)


def format_atom_line(record, *, legacy=False):
    """Format an atom while retaining each public writer's text convention."""
    if legacy:
        fields = {
            name: record[name]
            for name in (
                "atomi",
                "atomn",
                "location",
                "resn",
                "chain",
                "resi",
                "insert",
                "x",
                "y",
                "z",
                "occupancy",
                "b",
            )
        }
        fields["atomn"] = format_atom_name(record["atomn"])
        return ATOM_RECORD_FORMAT.format(**fields)

    # Both mappings and structured records are accepted by the common boundary.
    fields = (
        record if isinstance(record, dict) else dict(zip(record.dtype.names, record))
    )
    name = str(fields.get("record_name", "ATOM  ")).ljust(6)[:6]
    atom = format_atom_name(str(fields.get("atomn", "")), strip=True)
    location = (str(fields.get("location", "")) or " ")[:1]
    residue = str(fields.get("resn", "UNK"))[:3].rjust(3)
    chain = (str(fields.get("chain", "A")) or "A")[:1]
    insertion = (str(fields.get("insert", "")) or " ")[:1]
    element = str(fields.get("element", "")).rjust(2)[:2]
    occupancy = float(fields.get("occupancy", DEFAULT_ATOM_OCCUPANCY))
    b_factor = float(fields.get("b", DEFAULT_ATOM_B_FACTOR))
    return (
        f"{name}{int(fields['atomi']) % 100000:>5d} {atom}{location:1s}"
        f"{residue:3s} {chain:1s}{int(fields['resi']):>4d}{insertion:1s}   "
        f"{float(fields['x']):>8.3f}{float(fields['y']):>8.3f}"
        f"{float(fields['z']):>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}"
        f"          {element:>2s}\n"
    )
