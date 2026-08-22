# PTMs and covalently linked glycans

This tutorial imports, scores, repacks, minimizes, and exports modified
peptides. The built-in protein modifications are:

| PDB name | TMol residue type |
| --- | --- |
| `SEP`, `TPO`, `PTR` | phosphorylated SER, THR, TYR |
| `MLZ`, `MLY`, `M3L` | mono-, di-, trimethylated LYS |

Glycans are prepared from their deposited chemistry, so they are not limited
to a fixed list of sugar residue names. Their inter-residue bonds must be
present in the input bond table. For mmCIF, read both `_chem_comp_bond` and
`_struct_conn` with `include_bonds=True`.

## Import and inspect

```python
import torch
from biotite.structure.io.pdbx import CIFFile, get_structure
from tmol.io import pose_stack_from_biotite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cif = CIFFile.read("modified_peptide.cif")
structure = get_structure(cif, model=1, include_bonds=True)
pose, context = pose_stack_from_biotite(
    structure,
    device,
    prepare_ligands=True,
    no_optH=True,
    return_context=True,
)

block_types = [
    pose.packed_block_types.active_block_types[int(index)]
    for index in pose.block_type_ind64[0]
    if index >= 0
]
print([block_type.name for block_type in block_types])
```

`prepare_ligands=True` is safe for mixed structures: built-in PTM names are
recognized before unknown residues are sent through ligand preparation. A
5VVU O-GlcNAc site, for example, imports as `SER:covalent_OG` and
`NAG:covalent_C1`. A branched N-glycan similarly gives the ASN root and every
sugar a connection-capable residue type.

## Score and differentiate

```python
from tmol import beta2016_score_function

score_function = beta2016_score_function(
    device, param_db=context.parameter_database
)
scorer = score_function.render_whole_pose_scoring_module(pose)
coords = pose.coords.detach().clone().requires_grad_(True)
score = scorer(coords).sum()
score.backward()
print(float(score.detach()), torch.isfinite(coords.grad).all().item())
```

## Repack the peptide

TMol has Dunbrack rotamers for the modified amino-acid backbone/side-chain
degrees of freedom, but no carbohydrate rotamer library yet. Keep glycan
blocks fixed during the discrete packing step, then let Cartesian minimization
move the entire covalent system.

```python
from tmol.pack import PackerPalette, PackerTask, pack_rotamers
from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database

task = PackerTask(pose, PackerPalette())
task.restrict_to_repacking()
task.add_conformer_sampler(IncludeCurrentSampler())
task.add_conformer_sampler(
    create_dunbrack_sampler_from_database(
        context.parameter_database, device
    )
)
task.add_conformer_sampler(FixedAAChiSampler())

freeze_glycans = torch.tensor(
    [[
        block_type.properties.polymer.polymer_type != "amino_acid"
        for block_type in block_types
    ]],
    device=device,
)
task.disable_packing_by_block_mask(freeze_glycans)
packed = pack_rotamers(pose, score_function, task, verbose=False)
```

For a PTM-only peptide, omit `disable_packing_by_block_mask()` to repack every
residue.

## Minimize and export

```python
from tmol import run_cart_min
from tmol.io import biotite_from_pose_stack

minimized = run_cart_min(
    packed.clone(),
    score_function,
    optimizer_kwargs={"max_iter": 200},
)
output = biotite_from_pose_stack(
    minimized, co=context.canonical_ordering
)
```

The explicit glycosidic connection remains part of the pose topology through
packing, minimization, and export. Reuse `context` only for structures with the
same residue chemistry and covalent attachment pattern.

## Practical checks

Before using a deposited structure in production:

1. Confirm every expected PTM appears in `block_types` instead of being
   silently dropped.
2. Confirm the mmCIF supplies each glycosidic `_struct_conn` record.
3. Score once with gradients and require finite energy and coordinates.
4. Include the current rotamer so repacking cannot be forced to discard a
   deposited conformation.
5. Compare scores only within the same score function. Rosetta and TMol totals
   are not numerically interchangeable.

See {doc}`the glycan guide </user_guide/glycans>` for branched topology and
current carbohydrate-model limitations.
