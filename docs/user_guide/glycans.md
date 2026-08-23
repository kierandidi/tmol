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

## Compiling Repeated Scoring

For repeated scoring or minimization on one static pose topology, compile only
the pure-PyTorch `sugar_bb` scorer and add it to ordinary beta2016. This avoids
requiring fake/meta implementations for the native score operators while
remaining numerically equivalent to the carbohydrate preset:

```python
from tmol import ScoreType, beta2016_score_function
from tmol.score import ScoreFunction

base = beta2016_score_function(
    device, param_db=context.parameter_database
).render_whole_pose_scoring_module(pose)
sugar_function = ScoreFunction(context.parameter_database, device)
sugar_function.set_weight(ScoreType.sugar_bb, 1.0)
sugar = sugar_function.render_whole_pose_scoring_module(pose)
compiled_sugar = torch.compile(
    sugar, fullgraph=True, mode="reduce-overhead"
)

def compiled_carbohydrate_score(coords):
    return base(coords) + 0.5 * compiled_sugar(coords)
```

Compilation is shape-specific and should be warmed before timing. On H200 the
one-time compile cost was about 10.8 seconds. Over 200 synchronized steady-state
iterations, this composition reduced batch-1 forward plus backward from 5.277
to 1.953 ms and batch-32 from 5.491 to 2.024 ms, with output and coordinate
gradients checked against the eager carbohydrate preset.

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
score it in the ordinary residue packer, whose rotamers do not move an entire
downstream glycan subtree.

For discrete linkage proposals, use the statistical conformer library instead
of the ordinary side-chain packer:

```python
import numpy as np

from tmol.glycan import sample_glycan_linkage

proposal, conformer_index, angles = sample_glycan_linkage(
    pose, pose=0, child_block=1,
    rng=np.random.default_rng(7),
)
```

This population-samples Rosetta's linkage means and standard deviations and
rotates the complete downstream glycan subtree. `idealize=True` sets the exact
library means; the default applies Rosetta's Gaussian sampling. Score several
proposals and accept or reject them at the protocol level. This is deliberately
separate from amino-acid rotamers, since a glycan linkage can move multiple
residues and entire branches.

## Current Limits

- Bonds must be explicit. TMol does not infer glycosylation from distance.
- Input ligand atoms and bond orders must be complete enough for ligand
  preparation; PDB topology without chemistry-level bond orders is rejected.
- The current conformer subset covers all linkages in the 4BYH validation
  glycan. Unknown linkages return no statistical conformers; there is no silent
  generic fallback.
- TMol does not yet provide a complete GlycanRelax Monte Carlo driver,
  automatic glycan-tree construction, or IUPAC glycan construction.
