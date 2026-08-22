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
    args = parser.parse_args()

    # PyRosetta writes its license banner to stdout; keep stdout valid JSON.
    with redirect_stdout(sys.stderr):
        import pyrosetta

        pyrosetta.init(
            "-mute all -include_sugars -alternate_3_letter_codes pdb_sugar "
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
    if args.score:
        started = time.perf_counter()
        result["score"] = float(pyrosetta.get_score_function()(pose))
        result["score_seconds"] = time.perf_counter() - started
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
