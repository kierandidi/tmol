# AtomWorks regression structures

Except for the synthetic amine attachment and separately sourced RCSB 1xvk, 145d and 1aym entries listed in the manifest, these files are copied unchanged from the local `atomworks-dev` checkout at
`a1bda7edfcf325bc140091889b9745220adb5eba`. `provenance.json` records each source
path and SHA-256 digest. AtomWorks is distributed under the BSD 3-Clause license.
The structure files retain their original experimental/generated metadata.

| File | Regression exercised |
|---|---|
| `schiff_base_double_bond.cif` | Retain the declared double bond; after correcting attachment H counts, reject incompatible SINGLE patch parameters. Incomplete partner filtering is covered separately. |
| `unknown_heavy_atom_1a8o.cif` | Distinguish the conflicting author CG / label XYZ identities; retain the author coordinate and reject unknown label atoms or parser deletion. |
| `unresolved_unl.cif` | Retain all 28 unresolved ligand heavy atoms at NaN; explicitly reject unanchored ligand placement. |
| `modified_components_6q9t.cif` | Traverse the whole aromatic acyl cap in the covalently connected 4SO–A1IJ4 pair. The targeted test explicitly selects this pair; the original also contains zinc. |
| `plp_enzyme_7mkv.cif` | Match LLP terminal patches by scope and suffix, and supply finite charge coverage for every LLP variant. |
| `acetylated_peptide_1j8z.cif` | Recognize the BCX backbone separately from its disulfide attachment; retain the peptide connections and score it. |
| `conditional_generation.cif` | Rebuild missing sidechains and alpha hydrogens from valid backbone coordinates; check finite scores and gradients. |

Tests live in `tmol/tests/io/test_atomworks_corpus_regressions.py`,
`test_atomworks_reader.py`, and
`tmol/tests/ligand/test_atomworks_modified_components.py`.
Successful numerical checks do not independently validate the force field.

`macrocycle_1xvk.cif` is the complete RCSB entry used by the wider AtomWorks IO suite. The regression explicitly excludes free Mg and water, verifies QUI cap identity and all 18 covalent links, and scores/minimizes both original and reversed residue orders through both readers. The initial energy must be independent of residue order.

`terminal_nucleotide_145d.cif` is the complete RCSB entry used by that suite. Its first MCY must retain a DNA backbone and 5-prime patch, without proximity-inferred conjugations. The regression checks all 20 phosphodiester links and finite scoring/minimization of the 24 nucleotide blocks.

`conflicting_myristate_1aym.cif.gz` preserves the complete compressed entry. Its `struct_conn` category declares MYR C1 bonded to both GLY N and CA. After explicit free-zinc exclusion, construction must report both partners rather than overwrite the one MYR port. This is an input-conflict regression, not a successful whole-complex minimization.

`repeated_glycans_6mub.cif.gz` is losslessly compressed from the complete authored
AtomWorks fixture. Two MAN–MAN links share a patched type pair but have different
geometry in one generated conformer. Their transferable bond/angle targets must
come from the conformer generator's ideals, not individual strained sites. Both
readers retain every observed non-water residue, every glycan and their source
connections (entirely unresolved protein residues are explicitly excluded), produce identical
records after residue reversal/seed changes, and score/minimize both orders. The
regression disables optional geometric disulfide inference to require the declared
source graph exactly; the corpus exercises the default inference policy.

`generated_amine_attachment.cif` is a synthetic acetylated sugar. Its two declared
components require a sidechain conjugation despite the acyl fragment's inferred
polymer port. Both readers must prepare the linked glycosyl amine with one N–H
instead of the isolated protonated amine's three, conserve Frank's per-residue
charge total, retain the source bond, and score/minimize. This validates topology
and the existing charge convention; coupled local atom typing and parameter-fit
validation remain separate work. The manifest records its complete generation
recipe and stereochemical SMILES.

`af3_cyclic_peptide_7ubd.cif` retains the complete AtomWorks AF3 prediction.
Both readers construct the eight-residue cycle, remove its polymerization leaving
atoms, retain every other observed coordinate, and score/minimize. AtomWorks
continues to reject unknown atom names and preserve retained phosphate oxygens.

`chromophore_3nez.cif.gz` is the complete RCSB entry used by the AtomWorks IO
suite. NRQ supports a C-terminal patch but no N-terminal patch. Canonical ordering
must register that available end without requiring both patches. Both readers
retain all four connected chromophores and score/minimize the constructed pose;
the AtomWorks route also retains unresolved residues in its input array, which
the constructor excludes when their required backbone coordinates are absent.
