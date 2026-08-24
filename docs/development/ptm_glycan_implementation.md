# PTM and glycan implementation report

## Result

TMol can now represent, score with gradients, repack, Cartesian-minimize, and
round-trip PDB-derived peptides containing `SEP`, `TPO`, `PTR`, `MLZ`, `MLY`,
or `M3L`. It also supports explicitly linked N- and O-glycans, including
branched sugar trees. The implementation follows Rosetta's useful separation:
known protein modifications are residue variants; deposited glycans are
chemical residue types connected through explicit inter-residue topology.

This is deliberately not a claim of complete PTM support. New covalent
modifications still need either a built-in variant with scoring parameters or
an externally prepared residue type.

## How it fits into TMol

The implementation has four layers:

1. `VariantType.name3` maps an external PDB component name such as `SEP` back
   to the `SER` I/O equivalence class. Backbone-dependent scoring and packing
   therefore continue to use the canonical amino-acid identity.
2. Chemical patches add PTM atoms, bonds, internal coordinates, torsions,
   aliases, Rosetta atom types, and patch-introduced partial charges without
   duplicating full residue definitions.
3. Generic explicit-bond import creates same-layout connection variants for
   non-polymeric bonds in a Biotite bond table. It excludes only actual
   adjacent polymer links and disulfides already handled by the standard
   importer.
4. Glycosylation specializes that generic layer by virtualizing the leaving
   hydrogen while keeping it graph-connected for kinematics. LK-ball uses a
   term-local filtered bond graph, so virtual atoms neither occlude waters nor
   create water sites.

The last point matters for packing. Removing a leaving hydrogen from the
residue bond graph made scores look correct but produced a disconnected atom
tree. Retaining the graph bond and filtering only the energy term is both
shorter conceptually and compatible with packer kinematics.

## PDB-derived validation set

The compact test fixtures retain deposited coordinates, atom names, residue
identity, intramolecular bonds, and the tested covalent link. Independent
peptide fragments are translated apart and renumbered only to make a small,
deterministic regression input.

| Source | Extracted content |
| --- | --- |
| [3U3Z](https://www.rcsb.org/structure/3U3Z) chain B, residues 1–4 | SEP and PTR peptide |
| [7BA9](https://www.rcsb.org/structure/7BA9) chain B, residues 591–595 | TPO peptide |
| [6L1F](https://www.rcsb.org/structure/6L1F) chain A, residues 140–145 | MLZ peptide |
| [2H6N](https://www.rcsb.org/structure/2H6N) chain C, residues 1–8 | MLY peptide |
| [4IGQ](https://www.rcsb.org/structure/4IGQ) chain B, residues 3–5 | M3L peptide |
| [5VVU](https://www.rcsb.org/structure/5VVU) chain B, residues 393–396 and NAG 401 | SER-OG–NAG-C1 O-glycopeptide |
| [4BYH](https://www.rcsb.org/structure/4BYH) ASN 297 and chain C | ASN-linked, ten-sugar branched glycan |

The downloaded source mmCIF SHA-256 values are recorded below so fixture
provenance can be audited:

```text
3U3Z ef1a1f089bf65d4c58c8aa47f6c02c0e7b562f65d7ef776a55d7b5f470846afe
7BA9 d2604a4faefc43b0b95b4939b9fdbb40a9caf90554759d64658fdf87b61c8d0b
6L1F ca085f98f17038d76729e2c8e0e9e99354849f3bd233f512811b5eeeeddbe921
2H6N fe11fd4cb7def78143a091eab1f404c5e22d40d4fd3557b14eac1ed75d40b987
4IGQ d8f5be3fbd5889bf1309daaa2b1ff18394b65583d491e7489acb9d38fcfb9077
5VVU 4fbc55f434c676b90f0a1cd687212bda500bd36167c356abf1b74ccf7ffb574a
```

CPU end-to-end reference results for the compact inputs were:

| Input | Initial | Repacked | 20-step minimized |
| --- | ---: | ---: | ---: |
| six PTM peptides | 186.946 | 84.482 | 5.995 |
| 5VVU O-GlcNAc peptide | 33.740 | 22.641 | 6.812 |

Absolute energies are regression observations, not scientific acceptance
thresholds. Tests require finite energy/gradients and non-increasing energy
because compiler, device, and floating-point differences can change totals.

Final validation used:

- the complete I/O suite on CPU: 178 passed, 138 CUDA cases skipped;
- the two workflow tests on H200 through both CPU and CUDA parametrizations:
  four passed with Torch 2.13.0, CUDA 13.0;
- a second H200 run using the same JIT cache: four passed in 33.69 seconds of
  pytest time (54 seconds including Slurm startup); and
- the Sphinx HTML build with warnings treated as errors.

The cold H200 pytest run took 736.59 seconds because it built all TMol CUDA
extensions into `TORCH_EXTENSIONS_DIR`; compile time is not workflow runtime.

## Term-level PyRosetta parity

The PTM oracle uses PyRosetta 2026.33 with
`-corrections::beta_nov16 true`. It compares the change on modifying the
central residue of an `ALA-X-ALA` peptide, rather than absolute totals. This
cancels known whole-score-model differences and isolates the PTM patch.
Rosetta's separate inter- and intra-residue xover4 components are summed into
the corresponding combined TMol score type.

| Mapped term | Six-PTM result | Interpretation |
| --- | --- | --- |
| LJ attraction | max absolute delta 0.00245 REU (0.027%) | matched |
| LJ repulsion | max relative delta 0.192% | matched; largest value is the highly clashing M3L ideal conformation |
| LK solvation | max relative delta 2.41% | close, limited by global TMol/Rosetta beta atom-type parameter revisions |
| Rama, omega, Dunbrack, reference | PTM-induced delta exactly zero | variants inherit the canonical residue model, as in Rosetta |
| hydrogen bonds | zero in both models for the ideal oracle peptides | topology agrees; this fixture is not a nonzero hbond oracle |

The audit corrected three concrete issues:

- phosphate terminal oxygens now use Rosetta's `OOC` type, and PTR's ester
  oxygen uses `OH`;
- each methyl carbon now uses a terminal bonded hydrogen as its electrostatic
  count-pair representative, following TMol's existing count-pair convention;
  and
- backbone-table lookup now uses `base_name` and preserves a missing-table
  sentinel. Previously a modified residue missed its table and pandas
  `iloc[-1]` silently selected the final table.

The remaining discrepancies are intentional boundaries of this PR:

- **Electrostatics:** TMol's established protein partial-charge table is not
  the current Rosetta beta_nov16 table. For example, the SEP modification
  delta is -1.110 versus -0.587 REU, and M3L is -0.153 versus -1.559 REU.
  Changing only PTM charges would mix two charge models; changing the global
  table would alter every protein score and requires a separate whole-protein
  migration and benchmark.
- **LK-ball decomposition:** total LK modification deltas are within 2.41%,
  but `lk_ball_iso`/anisotropic components differ because TMol's global
  LK-ball parameterization and water-site implementation are not the current
  Rosetta beta implementation. These components must not be claimed as
  term-identical.
- **Absolute backbone/rotamer energies:** the PTM *delta* is correct, while
  TMol and current Rosetta use different global Rama, omega, and Dunbrack
  tables/implementations. Cartesian-bonded TMol terms also have no direct
  term-for-term entry in the standard Rosetta beta_nov16 score function.

Consequently total beta2016 scores are not interchangeable. Tests enforce the
mapped PTM deltas and exact inheritance behavior instead of baking in a false
total-score equivalence.

Phosphorylation and lysine-methylation atom names, types, heavy-atom geometry,
and patch-introduced charges follow the Rosetta `fa_standard` patches.
Glycosylation follows Rosetta's connection semantics: ASN ND2, SER OG, or THR
OG1 loses its attached hydrogen and gains a connection to the sugar anomeric
atom. TMol additionally preserves the input branch topology explicitly
instead of inferring it from distance.

Rosetta capabilities not yet reproduced are carbohydrate torsion energies,
glycan-tree construction/movers, glycan-specific rotamer sampling, automatic
glycan connection inference, and a general PTM patch library. Glycan blocks
should therefore be held fixed for discrete packing and included in Cartesian
minimization only when the deposited connectivity is trustworthy.

## Metal roadmap

Metals should not reuse the ordinary covalent-bond implementation unchanged.
An attachment atom is allowed one ordinary external covalent bond, while a
metal routinely has coordination number four or six. Coordination also needs
different nonbonded exclusions and geometry restraints.

Rosetta's
[`SetupMetalsMover`](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/SetupMetalsMover)
provides the right behavioral model: recognize
metal and metal-binding atom types, detect contacts using element/Lennard-Jones
radii, add connection variants, remove donor hydrogens where appropriate, and
add input-geometry distance and angle constraints. The constraints are what
keep a site intact during relaxation; connectivity alone is insufficient.
Rosetta's broader
[metalloprotein guide](https://docs.rosettacommons.org/docs/latest/rosetta_basics/non_protein_residues/Metals)
also makes automatic detection an import convenience for known sites, not a
general metal-site prediction algorithm.

Implement this in four reviewable stages:

1. **Explicit import.** Add common ion residue types (Zn, Mg, Ca, Fe, Mn, Cu,
   Co, Ni, Na, and K) with charge/atom parameters. Parse only explicit mmCIF
   `metalc`/coordination records into a separate many-to-one coordination
   graph. Preserve the metal as a block and donor identity; do not apply the
   single-occupancy rule used for covalent attachments.
2. **Scoring and minimization.** Add metal–donor distance and
   donor-base–donor–metal angle constraints centered on the deposited
   geometry, plus explicit exclusions so coordination contacts are not scored
   as ordinary steric clashes. Make the constraint weight opt-in and visible.
3. **Packing.** Add donor-aware variants for HIS ND1/NE2, ASP/GLU carboxylates,
   CYS SG, backbone O, and prepared ligand donors. Remove a donor hydrogen only
   when the chosen coordination state requires it. Freeze the metal and retain
   the current donor rotamer by default; test that repacking and minimization
   preserve coordination number and geometry.
4. **Optional detection.** After explicit topology is robust, add an opt-in
   Rosetta-like proximity detector with element-pair cutoffs and user
   selectors. Report every inferred contact and fail on ambiguous
   over-coordination rather than silently guessing. Explicit mmCIF topology
   must take precedence.

The minimum test matrix should include tetrahedral Zn, octahedral Mg/Ca,
heme Fe with a covalent organic ligand, a binuclear site, alternate HIS donor
atoms, waters as coordinators, and a negative close-contact case. Each case
should check import/export topology, finite gradients, repacking, Cartesian
minimization, donor protonation, nonbonded exclusions, and CPU/CUDA parity.
