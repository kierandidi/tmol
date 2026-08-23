# Deposited metal coordination

TMol can import explicit Zn, Mg, and Ca coordination from a Biotite bond table
and restrain the deposited site with the same geometry model as Rosetta's
`SetupMetalsMover`. This is a known-site workflow, not metal-site prediction.
For mmCIF input, retain `_struct_conn` records with `include_bonds=True`.

```python
import torch
from biotite.structure.io.pdbx import CIFFile, get_structure

from tmol import ScoreType, beta2016_score_function, setup_metal_constraints
from tmol.io import pose_stack_from_biotite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
structure = get_structure(
    CIFFile.read("metalloprotein.cif"), model=1, include_bonds=True
)
pose, context = pose_stack_from_biotite(
    structure, device, no_optH=True, return_context=True
)
pose = setup_metal_constraints(pose)

score_function = beta2016_score_function(
    device, param_db=context.parameter_database
)
score_function.set_weight(ScoreType.constraint, 1.0)
score = score_function.render_whole_pose_scoring_module(pose)(pose.coords).sum()
```

The importer creates one connection and one metal virtual proxy for every
explicit donor, supports up to eight connections, removes an attached donor
hydrogen, and retains explicitly coordinated waters. Unrelated crystallographic
waters keep TMol's historical behavior and are discarded. The built-in ion
Lennard-Jones/Lazaridis-Karplus parameters are Rosetta's defaults.

## Geometry model

For every site, `setup_metal_constraints()` adds:

- a proxy--donor harmonic distance with target 0 Å;
- a metal--proxy harmonic distance at the deposited value;
- a harmonic distance between every pair of proxies at the deposited value;
- a metal--donor--donor-parent circular-harmonic angle at the deposited value.

The standard deviations are Rosetta's 0.1 Å and 0.05 rad. The
`distance_multiplier` and `angle_multiplier` arguments scale those energy
families; setting either to zero disables it. Donor-orientation angles are
omitted for deposited waters because an oxygen-only water does not define an
experimental orientation. Its distance geometry remains restrained.

Rosetta exposes these energies as `metalbinding_constraint`; TMol currently
uses its generic `constraint` score type. This is a naming/decomposition
difference, not an energy-form difference, so the constraint weight must be
enabled explicitly.

## Numerical PyRosetta parity

The regression oracle uses PyRosetta 2026.33 and protein-only deposited sites
from 1CA2 and 1CLL. Scores below are unweighted constraint energies after a
single Cartesian x displacement; deposited scores are numerically zero.

| Site | Donors | Constraints | Perturbation | PyRosetta | TMol | Absolute delta |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1CA2 Zn | 3 | 12 | first donor +0.05 Å | 0.252283896 | 0.252285659 | 1.8e-6 |
| 1CA2 Zn | 3 | 12 | Zn +0.05 Å | 0.506329563 | 0.506331503 | 1.9e-6 |
| 1CLL Ca | 6 | 33 | first donor +0.05 Å | 0.484668200 | 0.484665781 | 2.4e-6 |
| 1CLL Ca | 6 | 33 | Ca +0.05 Å | 0.922061032 | 0.922056019 | 5.0e-6 |

Tests also require finite coordinate gradients and exact linear scaling with
the Rosetta distance/angle multipliers on CPU and CUDA.

## Deliberate limits

- Import is explicit-topology-first. TMol does not guess contacts from distance;
  ambiguous proximity is not sufficient evidence for coordination chemistry.
- Only Zn, Mg, and Ca are built in. Redox- and spin-sensitive transition metals,
  hemes, and multinuclear parameterization need a separate chemical model.
- Calling `setup_metal_constraints()` twice adds a second copy, just as adding
  any `ConstraintSet` twice would. Apply it once after import.
- Geometry restraints preserve an observed site during scoring/minimization;
  they do not estimate binding free energy or predict coordination state.

See Rosetta's
[`SetupMetalsMover` documentation](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/SetupMetalsMover)
and
[metalloprotein guide](https://docs.rosettacommons.org/docs/latest/rosetta_basics/non_protein_residues/Metals)
for the corresponding Rosetta workflow.
