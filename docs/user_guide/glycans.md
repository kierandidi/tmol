# Covalently Linked Glycans

TMol imports explicit non-polymeric bonds from a Biotite bond table when
`prepare_ligands=True`. The input should be mmCIF with `_struct_conn` and
`_chem_comp_bond` records, or another format that supplies equivalent explicit
bonds and bond orders. This supports N-linked glycans at ASN ND2, O-linked
glycans at SER OG or THR OG1, and branched carbohydrate trees.

```python
from biotite.structure.io.pdbx import CIFFile, get_structure
import torch

from tmol.io import build_context_from_biotite, pose_stack_from_biotite

cif = CIFFile.read("glycoprotein.cif")
structure = get_structure(cif, model=1, include_bonds=True)
device = torch.device("cuda")

context = build_context_from_biotite(structure, device, prepare_ligands=True)
pose = pose_stack_from_biotite(
    structure, device, context=context, no_optH=True
)
```

Context construction includes ligand chemistry preparation and is deliberately
separate from per-structure pose construction. Reuse the context for an
ensemble or trajectory with the same residue and bond topology.

## Rosetta Comparison

Rosetta's specialized carbohydrate importer requires sugar support and its PDB
three-letter-code mapping. For an experimental PDB with `LINK` records, use:

```bash
python scripts/inspect_rosetta_glycan_topology.py 4BYH.pdb --score
```

The equivalent Rosetta initialization flags are:

```text
-include_sugars -alternate_3_letter_codes pdb_sugar
-auto_detect_glycan_connections -maintain_links
```

On the 4BYH chain-C fixture, TMol and Rosetta both recover the ASN ND2--NAG C1
root link and all nine sugar--sugar links, including the two three-way branch
residues. Compare topology and bond geometry, not total score values: TMol's
beta2016 score and Rosetta's carbohydrate-aware score function are different
models.

## Current Limits

- Bonds must be explicit. TMol does not infer glycosylation from distance.
- Input ligand atoms and bond orders must be complete enough for ligand
  preparation; PDB topology without chemistry-level bond orders is rejected.
- The generic bonded-neighbor graph is scored and differentiable, but TMol does
  not yet provide Rosetta-style carbohydrate torsion preferences, glycan-tree
  movers, or IUPAC glycan construction.
