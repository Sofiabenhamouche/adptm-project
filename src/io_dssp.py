"""PDB/DSSP I/O utilities: build a per-residue Cα table, parse DSSP, compute RSA, merge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from Bio.PDB import PDBParser


# 3→1 mapping (tolerant to common variants)
AA3_TO_AA1: Dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # common alternates
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
    "HSD": "H",
    "HSE": "H",
    "HSP": "H",
    "ASX": "B",
    "GLX": "Z",
}

# Absolute ASA (Å²) used for RSA normalization: RSA = ACC / ACC_MAX
ACC_MAX: Dict[str, float] = {
    "A": 129,
    "R": 274,
    "N": 195,
    "D": 193,
    "C": 167,
    "Q": 225,
    "E": 223,
    "G": 104,
    "H": 224,
    "I": 197,
    "L": 201,
    "K": 236,
    "M": 224,
    "F": 240,
    "P": 159,
    "S": 155,
    "T": 172,
    "W": 285,
    "Y": 263,
    "V": 174,
}


def aa3_to_aa1(code3: str) -> str:
    """Convert a 3-letter residue code to 1-letter (fallback 'X')."""
    if not code3:
        return "X"
    return AA3_TO_AA1.get(code3.upper(), "X")


@dataclass
class ProteinIO:
    """I/O helper for PDB (Cα table) and DSSP (ACC→RSA) parsing."""
    verbose: bool = False

    # ---------- PDB ---------- #
    def load_ca_table(self, pdb_path: Path) -> pd.DataFrame:
        """Extract one row per residue that has a Cα atom.

        Returns:
            DataFrame columns:
              chain, resseq, icode, aa3, aa1, x, y, z, idx_in_chain
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("X", str(pdb_path))
        model = structure[0]

        rows: List[dict] = []
        total, no_ca, non_std = 0, 0, 0

        for chain in model:
            idx_in_chain = 0
            chain_id = chain.id
            for res in chain:
                total += 1
                if res.id[0] != " ":
                    non_std += 1
                    continue
                if "CA" not in res:
                    no_ca += 1
                    continue

                resseq = int(res.id[1])
                icode = res.id[2].strip() if isinstance(res.id[2], str) else ""
                aa3 = res.get_resname().upper()
                aa1 = aa3_to_aa1(aa3)
                x, y, z = res["CA"].get_coord()

                rows.append(
                    {
                        "chain": chain_id,
                        "resseq": resseq,
                        "icode": icode,
                        "aa3": aa3,
                        "aa1": aa1,
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "idx_in_chain": idx_in_chain,
                    }
                )
                idx_in_chain += 1

        df = pd.DataFrame(rows)
        if self.verbose:
            print(
                "[PDB] Total residues in model: "
                f"{total} | kept (AA w/ Cα): {df.shape[0]} | no-CA: {no_ca} | non-std: {non_std}"
            )
            if not df.empty:
                per_chain = (
                    df.groupby("chain")["idx_in_chain"].max().reset_index(name="n_ca")
                ).to_dict("records")
                print(f"[PDB] Cα per chain: {per_chain}")
        return df

    # ---------- DSSP ---------- #
    def parse_dssp_file(self, dssp_path: Path) -> pd.DataFrame:
        """Parse DSSP file produced by mkdssp and compute RSA (ACC / ACC_MAX).

        Returns:
            DataFrame columns: chain, resseq, icode, aa1, acc, rsa
        """
        rows: List[dict] = []
        in_table = False
        header_found = False
        skipped = 0
        bad_lines = 0
        not_in_acc = 0

        with open(dssp_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if not in_table:
                    if line.startswith("  #  RESIDUE"):
                        in_table = True
                        header_found = True
                    continue
                if len(line) < 40:
                    bad_lines += 1
                    continue

                try:
                    resseq = int(line[5:10].strip())
                    icode = line[10].strip()
                    chain = line[11].strip() or "A"
                    aa1 = line[13].strip()
                    if not aa1 or aa1 == "!":
                        skipped += 1
                        continue
                    acc = float(line[34:38].strip() or 0.0)
                except Exception:
                    bad_lines += 1
                    continue

                if aa1 not in ACC_MAX:
                    rsa = 0.0
                    not_in_acc += 1
                else:
                    acc_max = ACC_MAX[aa1]
                    rsa = float(acc / acc_max) if acc_max > 0 else 0.0
                rsa = max(0.0, min(1.0, rsa))

                rows.append(
                    {
                        "chain": chain,
                        "resseq": resseq,
                        "icode": icode,
                        "aa1": aa1,
                        "acc": acc,
                        "rsa": rsa,
                    }
                )

        if self.verbose:
            print(
                "[DSSP] header found:",
                header_found,
                "| parsed rows:",
                len(rows),
                "| skipped:",
                skipped,
                "| bad lines:",
                bad_lines,
                "| aa not in ACC_MAX:",
                not_in_acc,
            )
        return pd.DataFrame(rows)

    # ---------- MERGE ---------- #
    def merge_coordinates_with_rsa(
        self, df_res: pd.DataFrame, df_dssp: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge PDB coordinates with DSSP ACC/RSA using (chain, resseq, icode).

        Notes:
            - Keep 'aa1' from PDB only (avoid aa1_x/aa1_y conflicts).
            - Fill missing DSSP with acc=0.0 and rsa=0.0.

        Returns:
            Joined table ready for detection.
        """
        keys = ["chain", "resseq", "icode"]
        right = df_dssp[keys + ["acc", "rsa"]].copy()
        out = pd.merge(df_res, right, on=keys, how="left")
        out["acc"] = out["acc"].fillna(0.0).astype(float)
        out["rsa"] = out["rsa"].fillna(0.0).clip(0.0, 1.0)
        if self.verbose:
            print(
                "[MERGE] residues:",
                df_res.shape[0],
                "| DSSP rows:",
                df_dssp.shape[0],
                "| merged:",
                out.shape[0],
            )
        return out
