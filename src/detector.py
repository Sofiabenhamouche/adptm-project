"""
Membrane detection core (HARD exposure only): direction sampling, 1 Å profiling,
Q maximization, and score-based widening (optionally asymmetric).

This module provides:
- `generate_fibonacci_directions(n)`: ~uniform directions on the unit sphere.
- `MembraneLocator`: class implementing the detection pipeline:
    * build hydrophobicity profiles along test directions (HARD RSA exposure),
    * select best center by maximizing windowed average Q,
    * widen the slab by a score balancing hydrophobes vs. hydrophiles,
    * annotate residues as TM if they fall between the final planes,
    * extract contiguous TM segments of minimal length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Set of hydrophobic residues (binary model suitable for TM contexts)
HYDRO = {"A", "I", "L", "M", "F", "V", "W", "Y", "C"}


def generate_fibonacci_directions(n: int) -> np.ndarray:
    """
    Generate ~uniform directions on the unit sphere (golden-angle spiral).

    Args:
        n: Number of directions.

    Returns:
        np.ndarray: Array of shape (n, 3) with unit vectors.
    """
    pts = []
    phi = (1.0 + 5 ** 0.5) / 2.0
    golden = 2.0 * np.pi * (1.0 - 1.0 / phi)
    for i in range(n):
        z = 1.0 - (2 * i + 1) / n
        r = float(np.sqrt(max(0.0, 1.0 - z * z)))
        th = i * golden
        pts.append([r * np.cos(th), r * np.sin(th), z])
    pts = np.asarray(pts, float)
    pts /= (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12)
    return pts


def _center_of_mass(xyz: np.ndarray) -> np.ndarray:
    """
    Compute the center of mass as the simple mean of coordinates.

    Args:
        xyz: Array of shape (N, 3) with positions.

    Returns:
        np.ndarray: Center of mass (3,).
    """
    return np.asarray(xyz, float).mean(axis=0)


def _adaptive_margin(xyz: np.ndarray) -> float:
    """
    Compute a small dynamic margin to extend the profiling range.

    The margin gently expands the min/max(t) window to avoid edge starvation
    on elongated shapes.

    Args:
        xyz: Array of shape (N, 3) with positions.

    Returns:
        float: Margin in Å (clipped to [4, 8]).
    """
    o = xyz.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(xyz - o, axis=1).max()
    return float(np.clip(0.15 * radius, 4.0, 8.0))


@dataclass
class MembraneLocator:
    """
    Detector for membrane placement parameters (HARD RSA exposure).

    Attributes:
        bin_size: Slice thickness (Å) for the 1D profile.
        window: Q(u,c) window width (Å).
        rsa_threshold: RSA threshold for exposure (HARD mode).
        rsa_quantile: If set, overrides rsa_threshold by global RSA quantile.
        n_dirs: Number of directions to scan (informational).
        verbose: Enable verbose logging.
        label: Optional label prefix for prints.
        print_every: Print cadence while scanning directions.
        widen_step: Step (Å) for slab widening.
        max_half_increase: Max additional half-thickness allowed (Å).
        min_seg_len: Minimal TM segment length to keep (aa).
        lambda_penalty: Penalty for hydrophiles inside the slab.
        coverage_ratio: Minimal valid-bin coverage in window for Q.
    """

    # Profiling
    bin_size: float = 1.0
    window: float = 16.0  # Q(u,c) window (Å)
    rsa_threshold: float = 0.10
    rsa_quantile: Optional[float] = None  # if set, overrides rsa_threshold (global quantile)
    n_dirs: int = 30

    # Logging
    verbose: bool = False
    label: str = ""
    print_every: Optional[int] = None

    # Widening (score-only hill-climb)
    widen_step: float = 1.0
    max_half_increase: float = 30.0
    min_seg_len: int = 16
    lambda_penalty: float = 0.8
    coverage_ratio: float = 0.66  # minimal coverage of valid bins in window

    # --------------- public API --------------- #
    def detect(self, df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        """
        Detect best direction and slab using internal direction sampling (hard RSA).

        Args:
            df: Merged table with coordinates and RSA.

        Returns:
            Tuple[Dict, pd.DataFrame]: Results dictionary and annotated table.
        """
        dirs = generate_fibonacci_directions(self.n_dirs)
        return self.detect_on_directions(df, dirs)

    def detect_on_directions(
        self,
        df: pd.DataFrame,
        dirs: np.ndarray,
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Detect on precomputed directions (hard RSA exposure only).

        Args:
            df: Merged table with coordinates and RSA. Must contain columns:
                chain, resseq, icode, aa1, x, y, z, idx_in_chain, acc, rsa
            dirs: Unit directions of shape (n, 3).

        Returns:
            Tuple[Dict, pd.DataFrame]: (results dict, annotated DataFrame).

        Raises:
            ValueError: If required columns are missing.
            RuntimeError: If no valid direction is found.
        """
        required = {
            "chain",
            "resseq",
            "icode",
            "aa1",
            "x",
            "y",
            "z",
            "idx_in_chain",
            "acc",
            "rsa",
        }
        if not required.issubset(df.columns):
            missing = sorted(required - set(df.columns))
            raise ValueError(f"Missing columns: {missing}")

        xyz = df[["x", "y", "z"]].to_numpy(float)
        origin = _center_of_mass(xyz)

        # RSA threshold selection (global quantile override optional)
        if self.rsa_quantile is not None:
            thr = float(np.quantile(df["rsa"].to_numpy(float), self.rsa_quantile))
        else:
            thr = float(self.rsa_threshold)

        n_total = dirs.shape[0]
        tag = f"[{self.label}] " if self.label else ""
        if self.verbose:
            print(
                f"{tag}Scanning {n_total} directions (bin={self.bin_size:.1f} Å, "
                f"win={self.window:.1f} Å, RSA_th={thr:.3f}, widen=score)."
            )
        pe = self.print_every or max(1, n_total // 10)

        # Q(u,c) max → (n*, c*)
        best_Q = -np.inf
        best_n: Optional[np.ndarray] = None
        best_c: Optional[float] = None

        for i, n in enumerate(dirs, start=1):
            centers, hbin, valid = self._profile(n, origin, df, thr)
            if centers.size == 0:
                if self.verbose and (i % pe == 0 or i == n_total):
                    best_txt = f"{best_Q:.3f}" if np.isfinite(best_Q) else "nan"
                    print(f"  dir {i:>4}/{n_total}: (no bins) | best={best_txt}")
                continue

            c_star, Q_star = self._best_center(centers, hbin, valid)
            if Q_star > best_Q:
                best_Q, best_n, best_c = Q_star, n.copy(), float(c_star)

            if self.verbose and (i % pe == 0 or i == n_total):
                qtxt = f"{Q_star:.3f}" if np.isfinite(Q_star) else "nan"
                btxt = f"{best_Q:.3f}" if np.isfinite(best_Q) else "nan"
                print(f"  dir {i:>4}/{n_total}: Q*={qtxt}  (best={btxt})")

        if best_n is None or best_c is None:
            raise RuntimeError("No valid direction found.")

        # Initial planes = c* ± window/2
        d_in = best_c - self.window / 2.0
        d_out = best_c + self.window / 2.0

        if self.verbose:
            print(
                f"{tag}→ best dir: Q_max={best_Q:.3f}, center={best_c:.2f}, "
                f"init_thickness={self.window:.1f} Å."
            )

        # Widening by score only (may become asymmetric if beneficial)
        d_in, d_out = self._widen_by_score(
            df=df,
            origin=origin,
            n=best_n,
            d_in=d_in,
            d_out=d_out,
            rsa_thr=thr,
            step=self.widen_step,
            max_extra=self.max_half_increase,
            lam=self.lambda_penalty,
            tag=tag,
        )

        if self.verbose:
            print(
                f"{tag}→ widened_thickness: {d_out - d_in:.1f} Å "
                f"(planes {d_in:.2f}, {d_out:.2f})"
            )

        out_df = self._annotate(df, origin, best_n, d_in, d_out)
        segs = self._segments(out_df, min_len=self.min_seg_len)

        results = {
            "origin": origin.tolist(),
            "normal": best_n.tolist(),
            "center": best_c,
            "Q_max": float(best_Q),
            "d_in": float(d_in),
            "d_out": float(d_out),
            "thickness_A": float(d_out - d_in),
            "segments": segs,
        }
        return results, out_df

    # --------------- H(z) profile & window max (HARD) --------------- #
    def _profile(
        self,
        n: np.ndarray,
        origin: np.ndarray,
        df: pd.DataFrame,
        rsa_thr: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 1 Å bin hydrophobicity among exposed residues along direction n (HARD mode).

        In HARD mode: exposure = RSA >= rsa_thr (binary).

        Args:
            n: Direction vector (3,).
            origin: Origin for projection (3,).
            df: Input residue table with coordinates and RSA.
            rsa_thr: Threshold for exposure in HARD mode.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                centers: bin centers (Å)
                H:       hydrophobic fraction per bin (np.nan where invalid)
                valid:   boolean mask of valid bins
        """
        n = np.asarray(n, float)
        n /= np.linalg.norm(n) + 1e-12

        r = df[["x", "y", "z"]].to_numpy(float)
        t = (r - origin) @ n

        margin = _adaptive_margin(r)
        t_min = float(np.floor(t.min() - margin))
        t_max = float(np.ceil(t.max() + margin))

        bins = np.arange(t_min, t_max + self.bin_size, self.bin_size, dtype=float)
        if bins.size < 2:
            return np.array([]), np.array([]), np.array([])

        centers = (bins[:-1] + bins[1:]) / 2.0
        rsa = df["rsa"].to_numpy(float)
        aa1 = df["aa1"].astype(str).to_numpy()
        hyd_mask_global = np.isin(aa1, list(HYDRO))

        H = np.full(centers.shape, np.nan, float)
        valid = np.zeros(centers.shape, bool)

        for k in range(centers.size):
            a, b = bins[k], bins[k + 1]
            in_bin = (t >= a) & (t < b)

            # HARD exposure: RSA >= threshold
            expo = in_bin & (rsa >= rsa_thr)
            if not np.any(expo):
                continue

            hyd = expo & hyd_mask_global
            n_tot = int(np.count_nonzero(expo))
            n_hyd = int(np.count_nonzero(hyd))
            if n_tot > 0:
                H[k] = float(n_hyd) / float(n_tot)
                valid[k] = True

        return centers, H, valid

    def _best_center(
        self,
        centers: np.ndarray,
        H: np.ndarray,
        valid: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Find c* maximizing the windowed average Q over H(centers).

        The window size is ~window/bin_size (forced odd length).
        Only windows with sufficient valid-bin coverage (coverage_ratio)
        are considered.

        Args:
            centers: Bin centers (Å).
            H: Hydrophobicity values per bin.
            valid: Valid-bin mask.

        Returns:
            Tuple[float, float]: (best_center_c, Q_value_at_center).
        """
        k = int(round(self.window / max(1e-12, self.bin_size)))
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1

        kernel = np.ones(k, float)
        valid_f = valid.astype(float)
        H0 = np.nan_to_num(H, nan=0.0)

        s = np.convolve(H0, kernel, mode="same")
        c = np.convolve(valid_f, kernel, mode="same")

        min_cov = max(1.0, self.coverage_ratio * k)
        with np.errstate(divide="ignore", invalid="ignore"):
            Q = np.where(c >= min_cov, s / c, -np.inf)

        idx = int(np.argmax(Q))
        return float(centers[idx]), float(Q[idx])

    # --------------- score-based widening (HARD) --------------- #
    def _widen_by_score(
        self,
        df: pd.DataFrame,
        origin: np.ndarray,
        n: np.ndarray,
        d_in: float,
        d_out: float,
        rsa_thr: float,
        step: float = 1.0,
        max_extra: float = 30.0,
        lam: float = 0.8,
        tag: str = "",
    ) -> Tuple[float, float]:
        """
        Hill-climb on slab bounds to maximize (HARD exposure):

            S = Hydrophobic_in / Hydrophobic_tot
                − λ * (Hydrophilic_in / Hydrophilic_tot)

        Exposure is binary with RSA >= rsa_thr.

        Args:
            df: Residue table with coordinates, aa1, and RSA.
            origin: Projection origin (3,).
            n: Best direction vector (3,).
            d_in: Initial inner plane coordinate (Å).
            d_out: Initial outer plane coordinate (Å).
            rsa_thr: RSA threshold for exposure.
            step: Widening step (Å).
            max_extra: Maximum allowed half-thickness increase (Å).
            lam: Penalty weight for hydrophiles inside the slab.
            tag: Optional log prefix.

        Returns:
            Tuple[float, float]: Updated (d_in, d_out) after widening.
        """
        n = np.asarray(n, float)
        n /= np.linalg.norm(n) + 1e-12

        t = (df[["x", "y", "z"]].to_numpy(float) - origin) @ n
        aa1 = df["aa1"].astype(str).to_numpy()
        rsa = df["rsa"].to_numpy(float)
        hyd = np.isin(aa1, list(HYDRO))

        expo = rsa >= rsa_thr  # HARD exposure

        hyd_tot = float(np.count_nonzero(hyd & expo))
        phil_tot = float(np.count_nonzero((~hyd) & expo))

        def score(bounds: Tuple[float, float]) -> float:
            din, dout = bounds
            inside = (t >= din) & (t <= dout) & expo
            hyd_in = float(np.count_nonzero(hyd & inside))
            phil_in = float(np.count_nonzero((~hyd) & inside))

            if hyd_tot <= 0.0 and phil_tot <= 0.0:
                return -np.inf
            a = (hyd_in / hyd_tot) if hyd_tot > 0.0 else 0.0
            b = (phil_in / phil_tot) if phil_tot > 0.0 else 0.0
            return a - lam * b

        half0 = (d_out - d_in) / 2.0
        S = score((d_in, d_out))
        if self.verbose:
            print(f"{tag}  [score-widen] start S={S:.4f}  half={half0:.1f} Å")

        improved = True
        while improved:
            improved = False
            candidates: List[Tuple[str, float, float, float]] = []

            # 1) symmetric widen
            cand1 = (d_in - step, d_out + step)
            S1 = score(cand1)
            candidates.append(("widen", S1, cand1[0], cand1[1]))

            # 2) move inner plane inward
            cand2 = (d_in - step, d_out)
            S2 = score(cand2)
            candidates.append(("move_in", S2, cand2[0], cand2[1]))

            # 3) move outer plane outward
            cand3 = (d_in, d_out + step)
            S3 = score(cand3)
            candidates.append(("move_out", S3, cand3[0], cand3[1]))

            best = max(candidates, key=lambda c: c[1])
            if best[1] > S:
                op, S, new_in, new_out = best
                half_now = (new_out - new_in) / 2.0
                if half_now - half0 > max_extra:
                    break

                if op == "widen":
                    d_in, d_out = new_in, new_out
                    if self.verbose:
                        print(
                            f"{tag}  [score-widen] widen half={half_now:.1f} Å  S={S:.4f}"
                        )
                elif op == "move_in":
                    d_in = new_in
                    if self.verbose:
                        print(f"{tag}  [score-widen] move d_in→ {d_in:.2f}  S={S:.4f}")
                else:
                    d_out = new_out
                    if self.verbose:
                        print(f"{tag}  [score-widen] move d_out→ {d_out:.2f}  S={S:.4f}")

                improved = True

        return d_in, d_out

    # --------------- annotation & segments --------------- #
    def _annotate(
        self,
        df: pd.DataFrame,
        origin: np.ndarray,
        n: np.ndarray,
        d_in: float,
        d_out: float,
    ) -> pd.DataFrame:
        """
        Annotate residues with projection coordinate and in-membrane flag.

        Args:
            df: Input residue table.
            origin: Projection origin (3,).
            n: Best direction vector (3,).
            d_in: Inner plane coordinate (Å).
            d_out: Outer plane coordinate (Å).

        Returns:
            pd.DataFrame: Copy of df with columns 't' and 'in_membrane'.
        """
        n = np.asarray(n, float)
        n /= np.linalg.norm(n) + 1e-12

        r = df[["x", "y", "z"]].to_numpy(float)
        t = (r - origin) @ n
        out = df.copy()
        out["t"] = t
        out["in_membrane"] = (t >= d_in) & (t <= d_out)
        return out

    def _segments(self, df: pd.DataFrame, min_len: int = 16) -> List[Dict]:
        """
        Extract contiguous TM segments per chain using 'in_membrane' flags.

        Args:
            df: Annotated table (must contain 'chain', 'idx_in_chain', 'in_membrane').
            min_len: Minimum length (aa) to report a segment.

        Returns:
            List[Dict]: List of {chain, start, end, length} dictionaries.
        """
        segs: List[Dict] = []
        for ch, sub in df.sort_values(["chain", "idx_in_chain"]).groupby("chain"):
            run = None  # (start_idx, last_idx)
            for _, row in sub.iterrows():
                idx = int(row["idx_in_chain"])
                if bool(row["in_membrane"]):
                    if run is None:
                        run = (idx, idx)
                    else:
                        run = (run[0], idx)
                else:
                    if run is not None:
                        length = run[1] - run[0] + 1
                        if length >= min_len:
                            segs.append(
                                {
                                    "chain": ch,
                                    "start": int(run[0]),
                                    "end": int(run[1]),
                                    "length": int(length),
                                }
                            )
                        run = None
            # flush last run
            if run is not None:
                length = run[1] - run[0] + 1
                if length >= min_len:
                    segs.append(
                        {
                            "chain": ch,
                            "start": int(run[0]),
                            "end": int(run[1]),
                            "length": int(length),
                        }
                    )
        return segs
