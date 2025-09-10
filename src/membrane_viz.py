"""
membrane_viz.py — Membrane visualization in PyMOL
-------------------------------------------------

Displays two planes (CGO) representing the membrane boundaries:
- Inner plane (blue)
- Outer plane (red)
- Outlines (black)
- Normal vector (blue arrow)
- Center point (yellow sphere)

Usage in PyMOL:
---------------
run /mnt/c/Users/sofia/projet_ss/membrane_viz.py
memviz /mnt/c/Users/sofia/projet_ss/1K24.pdb, /mnt/c/Users/sofia/projet_ss/outputs/1K24_result.json

Optional:
memviz <pdb>, <json>, size=200, alpha=0.3
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np


# ---------- Geometry helpers ----------
def _norm(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def _basis_from_normal(n: np.ndarray):
    """Build an orthonormal basis (n, u, v) from a normal vector n."""
    n = _norm(n)
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _norm(np.cross(n, helper))
    v = _norm(np.cross(n, u))
    return n, u, v


def _load(json_path: str | Path):
    """Load membrane geometry parameters from JSON file."""
    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    origin = np.array(d["origin"], float)
    normal = np.array(d["normal"], float)
    d_in = float(d["d_in"])
    d_out = float(d["d_out"])
    center = float(d.get("center", 0.5 * (d_in + d_out)))
    return origin, normal, d_in, d_out, center


# ---------- PyMOL command ----------
def memviz(pdb_file: str, json_file: str, size: float = 160.0, alpha: float = 0.35):
    """
    Visualize the membrane in PyMOL.
    - Loads the protein
    - Adds two CGO planes (in/out) + outlines
    - Draws the normal vector and a central point
    """
    from pymol import cmd
    from pymol.cgo import (
        BEGIN, END, TRIANGLES, LINES, VERTEX, COLOR, ALPHA,
        LINEWIDTH, CONE,
    )

    # Load JSON parameters
    origin, normal, d_in, d_out, center = _load(json_file)
    n, u, v = _basis_from_normal(normal)
    half = float(size) / 2.0

    # Square base perpendicular to normal
    base = np.stack([
        -u * half - v * half,
        +u * half - v * half,
        +u * half + v * half,
        -u * half + v * half,
    ])

    # Plane positions
    cen_in = origin + n * d_in
    cen_out = origin + n * d_out
    quad_in = cen_in + base
    quad_out = cen_out + base

    # Center point and normal tip
    cstar = origin + n * center
    tip = origin + n * 25.0

    # Clean up previous objects
    for obj in (
        "prot", "plane_in", "plane_out",
        "plane_in_outline", "plane_out_outline",
        "normal_vec", "cstar", "membrane",
    ):
        try:
            cmd.delete(obj)
        except Exception:
            pass

    # Protein
    cmd.load(pdb_file, "prot")
    cmd.show("cartoon", "prot")
    cmd.color("cyan", "prot")

    # Utility functions for CGO objects
    def quad(name, Q, rgb):
        a, b, c, d = Q
        r, g, bcol = rgb
        obj = [
            BEGIN, TRIANGLES,
            COLOR, r, g, bcol, ALPHA, alpha,
            VERTEX, *a, VERTEX, *b, VERTEX, *c,
            VERTEX, *a, VERTEX, *c, VERTEX, *d,
            END,
        ]
        cmd.load_cgo(obj, name)

    def outline(name, Q):
        a, b, c, d = Q
        obj = [
            BEGIN, LINES, COLOR, 0.15, 0.15, 0.15, LINEWIDTH, 2.0,
            VERTEX, *a, VERTEX, *b,
            VERTEX, *b, VERTEX, *c,
            VERTEX, *c, VERTEX, *d,
            VERTEX, *d, VERTEX, *a,
            END,
        ]
        cmd.load_cgo(obj, name)

    # Planes + outlines
    quad("plane_in", quad_in, (0.20, 0.60, 1.00))   # blue
    quad("plane_out", quad_out, (1.00, 0.30, 0.30)) # red
    outline("plane_in_outline", quad_in)
    outline("plane_out_outline", quad_out)

    # Normal vector
    nv = [
        BEGIN, LINES, COLOR, 0.10, 0.45, 0.85, LINEWIDTH, 3.0,
        VERTEX, *origin, VERTEX, *tip, END,
        CONE, *tip, *(tip + np.array([0, 0, 5.0])),
        1.6, 0.0, 0.10, 0.45, 0.85, 0.10, 0.45, 0.85, 1.0, 1.0,
    ]
    cmd.load_cgo(nv, "normal_vec")

    # Center point
    cmd.pseudoatom("cstar", pos=cstar.tolist())
    cmd.color("yellow", "cstar")
    cmd.set("sphere_scale", 0.6, "cstar")
    cmd.show("spheres", "cstar")

    # Group objects
    cmd.group(
        "membrane",
        "plane_in plane_out plane_in_outline plane_out_outline normal_vec cstar prot",
    )
    cmd.zoom("membrane", buffer=12)


# Register PyMOL command
from pymol import cmd as _cmd
_cmd.extend("memviz", memviz)
