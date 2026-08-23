# Covalently Linked Glycans

TMol imports explicit non-polymeric bonds from a Biotite bond table when
`prepare_ligands=True`. The input should be mmCIF with `_struct_conn` and
`_chem_comp_bond` records, or another format that supplies equivalent explicit
bonds and bond orders. This supports N-linked glycans at ASN ND2, O-linked
glycans at SER OG or THR OG1, and branched carbohydrate trees.

For a score/pack/minimize walkthrough, see {doc}`PTMs and covalently linked
glycans </tutorial/09_ptms_and_glycopeptides>`.

```python
from biotite.structure.io.pdbx import CIFFile, get_structure
import torch

from tmol import carbohydrate_beta2016_score_function
from tmol.io import build_context_from_biotite, pose_stack_from_biotite

cif = CIFFile.read("glycoprotein.cif")
structure = get_structure(cif, model=1, include_bonds=True)
device = torch.device("cuda")

context = build_context_from_biotite(structure, device, prepare_ligands=True)
pose = pose_stack_from_biotite(
    structure, device, context=context, no_optH=True
)
score_function = carbohydrate_beta2016_score_function(
    device, param_db=context.parameter_database
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
residues. TMol's differentiable `sugar_bb` term matches Rosetta 2026.33's raw
score for this branch to within 3e-5 score units. The explicit carbohydrate
preset gives it Rosetta's default weight of 0.5; ordinary beta2016 keeps it at
zero, matching Rosetta's `beta_nov16` weights.

`sugar_bb` supplies Rosetta's intrinsic CHI phi/psi and exocyclic-omega
preferences for the recognized PDB sugars. Use it for Cartesian scoring and
minimization. Keep its weight at zero during discrete packing: TMol does not
yet move an entire downstream glycan subtree as one linkage proposal.

## Current Limits

- Bonds must be explicit. TMol does not infer glycosylation from distance.
- Input ligand atoms and bond orders must be complete enough for ligand
  preparation; PDB topology without chemistry-level bond orders is rejected.
- The generic bonded-neighbor graph and `sugar_bb` are differentiable, but TMol
  does not yet provide Rosetta linkage-conformer sampling, glycan-tree movers,
  or IUPAC glycan construction.
