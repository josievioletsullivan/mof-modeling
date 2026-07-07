"""LIGA ab-initio structure solution (§6) from a PDF distance list.

Wraps the ``mpbcliga`` solver (diffpy/liga, built from source into the mof env
-- see external/liga). The pipeline this module implements:

    G(r)  --peak pick-->  list of pair distances  --mpbcliga-->  atom positions

LIGA assumes the lattice parameters are known; it places ``natoms`` atoms in
that cell so their inter-atomic distances (including periodic images) reproduce
the target distance list, assigning chemistry by minimizing atomic-radius
overlap. We then compare the solved atoms back to the known CIF.

Everything here is deliberately restricted to the cubic §1 targets, so a single
lattice constant per axis makes Cartesian <-> fractional a plain division.
"""
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.signal import find_peaks
from diffpy.structure import loadStructure

import config


# Radii (A) for the §1 elements: ionic (Shannon, 6-coord) for the compounds,
# metallic/covalent for the elemental solids.  LIGA uses these only to assign
# which element sits where by minimizing atomic-radius overlap -- approximate
# values suffice as long as unlike species are correctly ordered by size.
ATOMIC_RADII = {
    "Cu": 1.28, "Ni": 1.24, "Al": 1.43, "Fe": 1.26, "Si": 1.11,
    "Na": 1.02, "Cl": 1.81, "Mg": 0.72, "O": 1.40, "Cs": 1.67,
    "Zn": 0.74, "S": 1.84, "Ca": 1.00, "F": 1.33,
}


def composition(cif):
    """Return (per_atom_elements, formula_str, radii_str) for a CIF.

    ``formula_str`` is e.g. 'Na4Cl4'; ``radii_str`` is 'Na:1.02, Cl:1.81' built
    from ATOMIC_RADII (None if any element is unknown).
    """
    stru = loadStructure(str(cif))
    elems = [a.element.strip() for a in stru]
    order = list(dict.fromkeys(elems))            # first-seen order, stable
    counts = {e: elems.count(e) for e in order}
    formula = "".join(f"{e}{counts[e]}" for e in order)
    if all(e in ATOMIC_RADII for e in order):
        radii = ", ".join(f"{e}:{ATOMIC_RADII[e]}" for e in order)
    else:
        radii = None
    return elems, formula, radii


def mpbcliga_bin():
    """Absolute path to the solver, PATH first then the mof-env fallback."""
    exe = shutil.which("mpbcliga")
    if exe:
        return exe
    fallback = os.path.expanduser("~/miniforge3/envs/mof/bin/mpbcliga")
    if os.path.exists(fallback):
        return fallback
    raise RuntimeError("mpbcliga not found -- build it in external/liga first")


# --------------------------------------------------------------------------- #
# G(r) -> distance list
# --------------------------------------------------------------------------- #
def load_gr(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1]


def peak_distances(r, g, rmax, rmin=1.0, prominence_frac=0.05, min_sep=0.35):
    """Pair distances = positions of the G(r) coordination-shell maxima.

    Peaks are picked between ``rmin`` and ``rmax``; ``prominence_frac`` is a
    fraction of the strongest peak in that window (rejects FT-truncation
    ripple), and ``min_sep`` (A) keeps neighbouring shells from merging.  The
    centre of each peak is refined by a 3-point parabolic fit for sub-grid
    accuracy.
    """
    win = (r >= rmin) & (r <= rmax)
    rr, gg = r[win], g[win]
    step = rr[1] - rr[0]
    prom = prominence_frac * gg.max()
    idx, _ = find_peaks(gg, prominence=prom, distance=max(1, int(min_sep / step)))
    centres = []
    for i in idx:
        if 0 < i < len(gg) - 1:
            y0, y1, y2 = gg[i - 1], gg[i], gg[i + 1]
            denom = (y0 - 2 * y1 + y2)
            shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            centres.append(rr[i] + shift * step)
        else:
            centres.append(rr[i])
    return np.array(centres)


def true_shell_distances(cif, rmax):
    """Distinct inter-atomic distances of the known structure up to ``rmax``.

    Reference only -- used to report how faithfully peak-picking recovered the
    real shells.  Cubic cell assumed; images span enough cells to reach rmax.
    """
    stru = loadStructure(str(cif))
    a, b, c = stru.lattice.a, stru.lattice.b, stru.lattice.c
    frac = stru.xyz % 1.0
    cart = frac * np.array([a, b, c])
    n = int(np.ceil(rmax / min(a, b, c))) + 1
    shifts = np.array([[i, j, k]
                       for i in range(-n, n + 1)
                       for j in range(-n, n + 1)
                       for k in range(-n, n + 1)]) * np.array([a, b, c])
    dists = []
    for p in cart:
        for q in cart:
            d = np.linalg.norm((q + shifts) - p, axis=1)
            dists.extend(d[(d > 1e-6) & (d <= rmax)])
    uniq = np.sort(np.unique(np.round(dists, 3)))
    # collapse values within 1e-2 A into single shells
    shells = []
    for d in uniq:
        if not shells or d - shells[-1] > 0.02:
            shells.append(d)
    return np.array(shells)


# --------------------------------------------------------------------------- #
# run the solver
# --------------------------------------------------------------------------- #
def solve(distances, latpar, natoms, out_xyz, rmax=None, formula=None,
          radii=None, rngseed=42, maxwalltime=120, tolcost=1e-4, workdir=None):
    """Run mpbcliga on a distance list; return (solved_cart, stdout, found)."""
    workdir = workdir or tempfile.mkdtemp(prefix="liga_")
    os.makedirs(workdir, exist_ok=True)
    distfile = os.path.join(workdir, "target.dst")
    np.savetxt(distfile, np.sort(distances), fmt="%.6f")

    latstr = ",".join(f"{x:g}" for x in latpar)
    args = [
        mpbcliga_bin(),
        f"distfile={distfile}",
        "crystal=true",
        f"latpar={latstr}",
        f"outstru={out_xyz}", "outfmt=rawxyz",
        f"rngseed={rngseed}", f"maxwalltime={maxwalltime}",
        f"tolcost={tolcost}",
    ]
    if formula:
        args.append(f"formula={formula}")
    else:
        args.append(f"natoms={natoms}")
    if radii:
        args.append(f"radii={radii}")
    if rmax:
        args.append(f"rmax={rmax}")

    proc = subprocess.run(args, capture_output=True, text=True, cwd=workdir)
    out = proc.stdout + proc.stderr
    found = "Solution found" in out
    solved = read_xyz(out_xyz) if os.path.exists(out_xyz) else None
    return solved, out, found


def read_xyz(path):
    """rawxyz -> (labels, Nx3 Cartesian array)."""
    labels, coords = [], []
    for line in open(path):
        parts = line.split()
        if len(parts) >= 4:
            labels.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    return labels, np.array(coords)


# --------------------------------------------------------------------------- #
# compare solved vs known structure (cubic, periodic, arbitrary origin)
# --------------------------------------------------------------------------- #
def compare(solved_cart, cif, latpar):
    """Best-origin periodic match of solved atoms to the known CIF.

    LIGA fixes the origin arbitrarily, so we try every translation that lands a
    solved atom on a true atom, wrap into the cell, greedily pair atoms under
    the minimum-image convention, and keep the translation with the smallest
    worst-atom error.  Returns dict with max/rms error in Angstrom and per-atom
    fractional coordinates.
    """
    a, b, c = latpar[0], latpar[1], latpar[2]
    cell = np.array([a, b, c])
    stru = loadStructure(str(cif))
    true_frac = (stru.xyz % 1.0)
    solved_frac = (solved_cart / cell) % 1.0

    def pbc_delta(u, v):
        d = (u - v + 0.5) % 1.0 - 0.5
        return d

    best = None
    for j in range(len(true_frac)):
        t = true_frac[j] - solved_frac[0]
        shifted = (solved_frac + t) % 1.0
        used = set()
        errs = []
        match = []                       # solved atom i -> true atom index
        for p in shifted:
            d = pbc_delta(true_frac, p) * cell
            dist = np.linalg.norm(d, axis=1)
            order = np.argsort(dist)
            for k in order:
                if k not in used:
                    used.add(k)
                    errs.append(dist[k])
                    match.append(int(k))
                    break
        errs = np.array(errs)
        maxerr = errs.max()
        if best is None or maxerr < best["max_err_A"]:
            best = dict(max_err_A=float(maxerr),
                        rms_err_A=float(np.sqrt((errs ** 2).mean())),
                        n_solved=len(solved_frac), n_true=len(true_frac),
                        solved_frac=np.round(shifted, 3),
                        true_frac=np.round(true_frac, 3),
                        match=match)
    return best


def final_cost(stdout):
    """Pull the last reported best-cost (BC) value from LIGA's log, if any."""
    vals = re.findall(r"BC\s+\d+\s+([\d.eE+-]+)", stdout)
    return float(vals[-1]) if vals else None
