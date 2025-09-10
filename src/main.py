#!/usr/bin/env python3
"""
Main command-line entry point for the membrane detection pipeline (HARD RSA mode only).

Steps:
    1) Load PDB (Cα table) and DSSP (ACC → RSA).
    2) Merge coordinates + RSA.
    3) Generate directions on a Fibonacci sphere.
    4) Detect best membrane orientation and thickness (hard exposure).
    5) Save JSON (summary) and CSV (per residue).

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Local modules (no relative imports so you can run `python src/main.py`)
from io_dssp import ProteinIO
from detector import MembraneLocator, generate_fibonacci_directions


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Create and return the CLI argument parser for the pipeline (hard RSA only).

    Returns:
        argparse.ArgumentParser: Configured parser with all options.
    """
    parser = argparse.ArgumentParser(
        description="Membrane detection — HARD exposure (thresholded RSA), Q-window + score-based widening."
    )
    # Inputs / IDs
    parser.add_argument(
        "--pdb",
        required=True,
        type=str,
        help="Input PDB file path.",
    )
    parser.add_argument(
        "--dssp",
        required=True,
        type=str,
        help="Input DSSP file path.",
    )
    parser.add_argument(
        "--pdbid",
        type=str,
        default="STRUCT",
        help="Protein ID used in output names.",
    )
    # Algorithm controls
    parser.add_argument(
        "--dirs",
        type=int,
        default=30,
        help="Number of directions to sample on the sphere.",
    )
    parser.add_argument(
        "--win",
        type=float,
        default=16.0,
        help="Window width for Q (Å).",
    )
    parser.add_argument(
        "--bin",
        type=float,
        default=1.0,
        dest="bin_size",
        help="Slice thickness (Å) for the 1D profile.",
    )
    parser.add_argument(
        "--rsa-th",
        type=float,
        default=0.10,
        dest="rsa_threshold",
        help="RSA threshold for exposure (hard mode).",
    )
    parser.add_argument(
        "--rsa-quantile",
        type=float,
        default=None,
        help="If set, overrides RSA threshold by using a global RSA quantile (0–1).",
    )
    parser.add_argument(
        "--widen-step",
        type=float,
        default=1.0,
        help="Step (Å) used when widening slab thickness.",
    )
    parser.add_argument(
        "--max-half-extra",
        type=float,
        default=30.0,
        dest="max_half_extra",
        help="Max extra half-thickness allowed beyond win/2 (Å).",
    )
    parser.add_argument(
        "--min-seg",
        type=int,
        default=16,
        help="Minimal TM segment length (aa) for exported segments.",
    )
    parser.add_argument(
        "--lambda-penalty",
        type=float,
        default=0.8,
        dest="lambda_penalty",
        help="Penalty weight for hydrophiles inside the slab.",
    )
    # Outputs & logs
    parser.add_argument(
        "--out",
        type=str,
        default="outputs",
        help="Output directory (created if missing).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logs.",
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> None:
    """
    Run the end-to-end detection pipeline and write outputs (hard RSA only).

    Args:
        args: Parsed command-line arguments.

    Raises:
        SystemExit: If PDB or DSSP inputs are missing or invalid.
    """
    pdb_path = Path(args.pdb)
    dssp_path = Path(args.dssp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdb_path.exists():
        raise SystemExit(f"[ERROR] Missing PDB: {pdb_path}")
    if not dssp_path.exists():
        raise SystemExit(f"[ERROR] Missing DSSP: {dssp_path}")

    # I/O
    io = ProteinIO(verbose=args.verbose)
    print("[I/O] Reading PDB…", pdb_path)
    df_res = io.load_ca_table(pdb_path)
    if df_res.empty:
        raise SystemExit("[ERROR] No Cα residues found in the PDB.")
    print(f"[I/O] Residues with Cα: {df_res.shape[0]}")

    print("[I/O] Reading DSSP…", dssp_path)
    df_dssp = io.parse_dssp_file(dssp_path)
    print(f"[I/O] DSSP rows: {df_dssp.shape[0]} (ACC/RSA)")

    df = io.merge_coordinates_with_rsa(df_res, df_dssp)
    print(f"[I/O] Merged table: {df.shape}")

    # Directions
    dirs = generate_fibonacci_directions(args.dirs)

    # Detector (HARD only)
    detector = MembraneLocator(
        bin_size=args.bin_size,
        window=args.win,
        rsa_threshold=args.rsa_threshold,
        rsa_quantile=args.rsa_quantile,
        n_dirs=dirs.shape[0],
        widen_step=args.widen_step,
        max_half_increase=args.max_half_extra,
        min_seg_len=args.min_seg,
        lambda_penalty=args.lambda_penalty,
        verbose=args.verbose,
        label="score-only",
    )

    results, df_annot = detector.detect_on_directions(df, dirs)

    # Extra info
    print(f"[INFO] Center of mass (Cα): {results['origin']}")
    print(f"[INFO] Best normal vector: {results['normal']}")

    # Outputs
    pid = args.pdbid.upper()
    out_json = out_dir / f"{pid}_result.json"
    out_csv = out_dir / f"{pid}_annot.csv"

    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "pdb": pid,
                "origin": results["origin"],
                "normal": results["normal"],
                "center": results["center"],
                "Q_max": results["Q_max"],
                "d_in": results["d_in"],
                "d_out": results["d_out"],
                "thickness_A": results["thickness_A"],
                "dirs": int(args.dirs),
                "rsa_mode": "hard",
                "rsa_threshold": args.rsa_threshold,
                "rsa_quantile": args.rsa_quantile,
                "win_A": args.win,
                "bin_A": args.bin_size,
                "widen_step_A": args.widen_step,
                "max_half_extra_A": args.max_half_extra,
                "min_seg_len": args.min_seg,
                "lambda_penalty": args.lambda_penalty,
                "segments": results["segments"],
            },
            fh,
            indent=2,
        )

    df_annot.to_csv(out_csv, index=False)

    print(f"[OK] Wrote outputs → {out_dir}/")
    print(
        "[FINAL] "
        f"Q_max={results['Q_max']:.3f}  thickness={results['thickness_A']:.1f} Å  "
        f"planes=({results['d_in']:.2f},{results['d_out']:.2f})"
    )


if __name__ == "__main__":
    parser = build_arg_parser()
    run_pipeline(parser.parse_args())
