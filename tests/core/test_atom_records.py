"""Chemical identities survive projection independently of label collisions."""

from types import SimpleNamespace as Record

import numpy as np
import pytest

from tmol_core.io.atom_records import prepare_atom_records, format_atom_line


def projected(
    *, extended=False, include_virtual=False, unused_chemistry=False, index_dtype=None
):
    atom_types = [Record(name="metal", element="Zn"), Record(name="virt", element="X")]
    atoms = [
        Record(name="ZN-long", atom_type="metal"),
        Record(name="VRT", atom_type="virt"),
    ]
    blocks = [
        Record(
            name="FRA", name3="FRA", atoms=atoms, properties=Record(virtual=("VRT",))
        ),
        Record(
            name="ZIN-original", name3="ZIN", atoms=atoms, properties=Record(virtual=())
        ),
    ]
    mapping = Record(
        entries=[
            Record(
                pose_ind=1,
                block_ind=3,
                orig_block_type_ind=1,
                orig_residue_label=-17,
                orig_chain_label="author-chain",
                orig_ins_code="AB",
            )
        ]
    )
    active_type = 0
    local_atoms = [0, 1, 0, 0]
    if unused_chemistry:
        # Unemitted custom chemistry need not have output element parameters.
        unused = Record(
            name="CUSTOM",
            name3="CUS",
            atoms=[Record(name="Q", atom_type="unparameterized")],
            properties=Record(virtual=()),
        )
        blocks = [unused, unused, blocks[0], unused, blocks[1], unused]
        active_type = 2
        mapping.entries[0].orig_block_type_ind = 4
        blocks[active_type].atoms = unused.atoms + atoms + unused.atoms
        local_atoms = [1, 2, 1, 1]
    return prepare_atom_records(
        blocks,
        np.asarray([active_type] * 4, dtype=index_dtype),
        [1, 1, 1, 2],
        [2, 2, 3, 3],
        np.asarray(local_atoms, dtype=index_dtype),
        np.array([[1.123456789123, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]),
        chain_indices=[0, 0, 1, 0],
        residue_labels=[5, 5, 5, 5],
        chain_labels=["AA", "AA", "BB", "CC"],
        insertion_codes=[" "] * 4,
        occupancy=[0.5] * 4,
        b_factor=[12] * 4,
        record_names=["HETATM"] * 4,
        split_block_mapping=mapping,
        atom_types=atom_types,
        extended=extended,
        include_virtual=include_virtual,
    )


def test_fragment_identity_uses_pose_and_block_instead_of_author_number():
    records = projected()
    np.testing.assert_array_equal(records["resi"], [5, -17, 5])
    np.testing.assert_array_equal(records["resn"], ["FRA", "ZIN", "FRA"])
    np.testing.assert_array_equal(records["atomi"], [1, 2, 1])
    np.testing.assert_array_equal(records["model"], ["2", "2", "3"])
    assert records["x"][0] == 1.123456789123


@pytest.mark.parametrize("include_virtual", (False, True))
def test_extended_records_preserve_labels_and_chemical_elements(include_virtual):
    records = projected(extended=True, include_virtual=include_virtual)
    assert len(records) == (4 if include_virtual else 3)
    assert records["chain"][0] == "AA"
    assert records["atomn"][0] == "ZN-long"
    restored = records[records["resi"] == -17][0]
    assert restored["chain"] == "author-chain"
    assert restored["insert"] == "AB"
    assert restored["element"] == "Zn"
    # PDB formatting has a fixed-width author-label contract distinct from records.
    line = format_atom_line(dict(zip(records.dtype.names, restored)), legacy=False)
    assert line.startswith("HETATM")
    # A regular-width atom name gives the standard element column.
    record = dict(zip(records.dtype.names, restored))
    record["atomn"] = "ZN"
    line = format_atom_line(record)
    assert line[21] == "a" and line[22:26] == " -17" and line[26] == "A"
    assert line[76:78] == "Zn"


def test_legacy_atom_lines_accept_only_the_original_required_fields():
    record = dict(
        atomi=7,
        atomn="CA",
        location="",
        resn="ALA",
        chain="Z",
        resi=-2,
        insert="",
        x=1.25,
        y=-2.5,
        z=3.0,
        occupancy=1.0,
        b=30.0,
    )
    line = format_atom_line(record, legacy=True)
    assert line[:6] == "ATOM  "
    assert line[6:11] == "    7" and line[12:16] == " CA "
    assert line[17:20] == "ALA" and line[21:26] == "Z  -2"
    assert line[30:54] == "   1.250  -2.500   3.000"


def test_empty_projection_has_the_declared_record_schema():
    records = prepare_atom_records(
        [],
        [],
        [],
        [],
        [],
        np.empty((0, 3)),
        chain_indices=[],
        residue_labels=[],
        chain_labels=[],
        insertion_codes=[],
        occupancy=[],
        b_factor=[],
        atom_types=[],
        extended=True,
    )
    assert records.shape == (0,)
    assert "element" in records.dtype.names
    assert records.dtype["x"] == np.float64


@pytest.mark.parametrize("extended", (False, True))
@pytest.mark.parametrize("include_virtual", (False, True))
def test_unemitted_custom_chemistry_does_not_change_output(extended, include_virtual):
    expected = projected(extended=extended, include_virtual=include_virtual)
    actual = projected(
        extended=extended, include_virtual=include_virtual, unused_chemistry=True
    )
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_unsigned_atom_indices_preserve_chemical_identity():
    expected = projected(extended=True)
    actual = projected(extended=True, index_dtype=np.uint64)
    np.testing.assert_array_equal(actual, expected)


def owned_projection():
    class Source:
        pass

    class ObservedAtoms(list):
        reads = 0

        def __getitem__(self, index):
            self.reads += 1
            return super().__getitem__(index)

    owner = Source()
    owner.chem_db = Source()
    owner.chem_db.atom_types = [
        Record(name="metal", element="Zn"),
        Record(name="virt", element="X"),
    ]
    atoms = ObservedAtoms(
        [
            Record(name="ZN", atom_type="metal"),
            Record(name="VRT", atom_type="virt"),
        ]
    )
    # The absent properties on an unused type also prohibit full-palette scans.
    owner.active_block_types = [
        None,
        Record(
            name="ZINC",
            name3="ZIN",
            atoms=atoms,
            properties=Record(virtual=("VRT",)),
        ),
    ]
    args = [
        owner.active_block_types,
        np.array([1, 1, 1]),
        [0, 0, 1],
        [0, 0, 0],
        np.array([0, 1, 0]),
        np.array([[1.123456789123, 0, 0], [2, 0, 0], [3, 0, 0]]),
    ]
    kwargs = dict(
        chain_indices=[0, 0, 0],
        residue_labels=[5, 5, 5],
        chain_labels=["A", "A", "B"],
        insertion_codes=[""] * 3,
        occupancy=[1] * 3,
        b_factor=[0] * 3,
        atom_types=owner.chem_db.atom_types,
        extended=True,
    )
    return owner, args, kwargs, atoms


def test_owned_chemistry_reuse_keeps_coordinates_and_author_data_live():
    import gc
    import weakref

    owner, args, kwargs, atoms = owned_projection()
    first = prepare_atom_records(*args, **kwargs, chemistry_owner=owner)
    reads = atoms.reads
    args[-1][:, 0] += 0.125
    kwargs.update(
        chain_labels=["new", "new", "chain"],
        residue_labels=[-2] * 3,
        split_block_mapping=Record(
            entries=[
                Record(
                    pose_ind=1,
                    block_ind=0,
                    orig_block_type_ind=1,
                    orig_residue_label=-17,
                    orig_chain_label="author",
                    orig_ins_code="AB",
                )
            ]
        ),
    )
    actual = prepare_atom_records(*args, **kwargs, chemistry_owner=owner)
    assert atoms.reads == reads
    expected = prepare_atom_records(*args, **kwargs)
    np.testing.assert_array_equal(actual, expected)
    assert actual["x"][0] == first["x"][0] + 0.125
    assert list(actual["chain"]) == ["new", "author"]
    assert list(actual["resi"]) == [-2, -17]
    coordinates = weakref.ref(args[-1])
    source = weakref.ref(owner.chem_db)
    owner_ref = weakref.ref(owner)
    del args, kwargs
    gc.collect()
    assert coordinates() is None  # cache retains no full output or coordinates
    del owner
    gc.collect()
    assert owner_ref() is None and source() is None


def test_owned_projection_respects_policies_subsets_and_source_identity():
    owner, args, kwargs, atoms = owned_projection()
    for options in (
        {},
        {"include_virtual": True},
        {"residue_names": ["CUS"] * 3},
        {"extended": False},
        {},
    ):
        settings = dict(kwargs, **options)
        expected = prepare_atom_records(*args, **settings)
        actual = prepare_atom_records(*args, **settings, chemistry_owner=owner)
        np.testing.assert_array_equal(actual, expected)
    # Same compact keys can decode to different atoms after a range change.
    args[4] = np.array([1, 1, 1])
    np.testing.assert_array_equal(
        prepare_atom_records(*args, **kwargs, chemistry_owner=owner),
        prepare_atom_records(*args, **kwargs),
    )
    args[4] = np.array([0, 1, 0])
    # A replacement chemical source, even with the same collection identity,
    # invalidates the annotation.
    owner.chem_db = type(owner.chem_db)()
    owner.chem_db.atom_types = [
        Record(name="metal", element="Cu"),
        Record(name="virt", element="X"),
    ]
    kwargs["atom_types"] = owner.chem_db.atom_types
    actual = prepare_atom_records(*args, **kwargs, chemistry_owner=owner)
    assert list(actual["element"]) == ["Cu", "Cu"]
    # Independently supplied chemistry must bypass the owner's annotation.
    kwargs["atom_types"] = [
        Record(name="metal", element="Fe"),
        Record(name="virt", element="X"),
    ]
    actual = prepare_atom_records(*args, **kwargs, chemistry_owner=owner)
    assert list(actual["element"]) == ["Fe", "Fe"]
    args[0] = [
        None,
        Record(name="ALT", name3="ALT", atoms=atoms, properties=Record(virtual=())),
    ]
    np.testing.assert_array_equal(
        prepare_atom_records(*args, **kwargs, chemistry_owner=owner),
        prepare_atom_records(*args, **kwargs),
    )
