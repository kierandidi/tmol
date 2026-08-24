#!/usr/bin/env python3
"""Report Rosetta's carbohydrate connections for a PDB with LINK records."""

import argparse
from contextlib import redirect_stdout
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb", help="PDB containing carbohydrate LINK records")
    parser.add_argument(
        "--score",
        action="store_true",
        help="also evaluate Rosetta's default full-atom score function",
    )
    parser.add_argument(
        "--refine-chain",
        help="run GlycanSampler refinement for glycans in this chain",
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")

    # PyRosetta writes its license banner to stdout; keep stdout valid JSON.
    with redirect_stdout(sys.stderr):
        import pyrosetta

        pyrosetta.init(
            f"-mute all -constant_seed -jran {args.seed} "
            "-corrections::beta_nov16 true "
            "-include_sugars -alternate_3_letter_codes pdb_sugar "
            "-auto_detect_glycan_connections -maintain_links"
        )
    started = time.perf_counter()
    pose = pyrosetta.pose_from_file(args.pdb)
    load_seconds = time.perf_counter() - started

    edges = []
    carbohydrate_residues = []
    for res_index in range(1, pose.total_residue() + 1):
        residue = pose.residue(res_index)
        if residue.is_carbohydrate():
            carbohydrate_residues.append(res_index)
        for conn_index in range(1, residue.n_possible_residue_connections() + 1):
            partner = residue.connected_residue_at_resconn(conn_index)
            if partner <= res_index:
                continue
            partner_residue = pose.residue(partner)
            if not (residue.is_carbohydrate() or partner_residue.is_carbohydrate()):
                continue
            atom_index = residue.residue_connection(conn_index).atomno()
            partner_conn = residue.residue_connection_conn_id(conn_index)
            partner_atom = partner_residue.residue_connection(partner_conn).atomno()
            edges.append(
                {
                    "residue1": res_index,
                    "name1": residue.name(),
                    "atom1": residue.atom_name(atom_index).strip(),
                    "residue2": partner,
                    "name2": partner_residue.name(),
                    "atom2": partner_residue.atom_name(partner_atom).strip(),
                }
            )

    result = {
        "pyrosetta_version": pyrosetta.version().splitlines()[-1],
        "residues": pose.total_residue(),
        "carbohydrate_residues": len(carbohydrate_residues),
        "glycan_edges": edges,
        "glycan_edge_count": len(edges),
        "has_glycan_tree_set": pose.glycan_tree_set() is not None,
        "load_seconds": load_seconds,
    }
    if args.score or args.refine_chain:
        from pyrosetta.rosetta.core.scoring import ScoreType

        started = time.perf_counter()
        score_function = pyrosetta.get_score_function()
        initial_score = float(score_function(pose))
        result["score_seconds"] = time.perf_counter() - started
        initial_sugar_bb = {
            "raw": float(pose.energies().total_energies()[ScoreType.sugar_bb]),
            "weight": float(score_function.get_weight(ScoreType.sugar_bb)),
            "per_residue": [
                {
                    "pose_index": index,
                    "pdb_id": pose.pdb_info().pose2pdb(index).strip(),
                    "raw": float(
                        pose.energies().residue_total_energies(index)[
                            ScoreType.sugar_bb
                        ]
                    ),
                }
                for index in carbohydrate_residues
            ],
        }
        if args.score:
            result["score"] = initial_score
            result["sugar_bb"] = initial_sugar_bb
        if args.refine_chain:
            from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
            from pyrosetta.rosetta.protocols.carbohydrates import GlycanSampler

            sampler = GlycanSampler()
            sampler.set_selector(ChainSelector(args.refine_chain))
            sampler.set_scorefunction(score_function)
            sampler.set_refine(True)
            sampler.set_randomize_first(False)
            sampler.set_use_gaussian_sampling(True)
            sampler.set_population_based_conformer_sampling(True)
            sampler.force_total_rounds(args.rounds)
            sampler.apply(pose)
            final_score = float(score_function(pose))
            final_sugar_bb = float(pose.energies().total_energies()[ScoreType.sugar_bb])
            result["glycan_refinement"] = {
                "chain": args.refine_chain,
                "rounds": args.rounds,
                "seed": args.seed,
                "before": {
                    "score": initial_score,
                    "sugar_bb_raw": initial_sugar_bb["raw"],
                },
                "after": {
                    "score": final_score,
                    "sugar_bb_raw": final_sugar_bb,
                },
                "delta_score": final_score - initial_score,
            }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
