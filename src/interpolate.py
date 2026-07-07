"""§3 algorithm: generate a synthetic intermediate structure by linearly
morphing an `initial` structure toward a `final` one.

Both the lattice parameters (a, b, c, alpha, beta, gamma) and the fractional
coordinates blend with a fraction t, using the convention

    intermediate = t * initial + (1 - t) * final          (t = weight on initial)

so t=1 -> initial, t=0 -> final, t=0.5 -> midpoint. This is the SAME convention
as the sanity check  t*G_initial + (1-t)*G_final ~ G_intermediate, which lets
the two be compared weight-for-weight.

Requirements / choices:
  * Equal atom counts. Interpolating coordinates needs a 1:1 atom
    correspondence; here atoms are mapped by index (row k <-> row k).
  * The final atom is taken in its minimum-image position relative to the
    initial atom, so each atom travels the shortest path and no atom sweeps
    across the cell (0.02 and 0.98 are treated as 0.04 apart, not 0.96).
  * The intermediate keeps the INITIAL structure's chemistry (per-atom
    species). The morph is therefore purely geometric and always yields a
    single, well-defined structure even when the two endpoints differ in
    composition (e.g. Si -> NaCl stays pure Si with a rocksalt-like geometry).
"""
import copy
import numpy as np


def _latpar(lattice):
    return np.array([lattice.a, lattice.b, lattice.c,
                     lattice.alpha, lattice.beta, lattice.gamma], float)


def interpolate_structures(initial, final, t=0.5):
    """Return a new diffpy Structure blended between `initial` and `final`.

    t is the weight on `initial` (t in (0,1) for a true intermediate).
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be in [0, 1], got {t}")
    if len(initial) != len(final):
        raise ValueError(
            f"atom-count mismatch: {len(initial)} vs {len(final)} -- "
            "interpolation needs a 1:1 atom correspondence (equal atom counts)"
        )

    inter = copy.deepcopy(initial)          # inherits the initial's species

    # --- lattice: linear blend of the six cell parameters ---
    a, b, c, al, be, ga = t * _latpar(initial.lattice) \
        + (1 - t) * _latpar(final.lattice)
    inter.lattice.setLatPar(a, b, c, al, be, ga)

    # --- fractional coords: shortest-path (minimum-image) blend, index-mapped ---
    xi = np.asarray(initial.xyz, float)
    xf = np.asarray(final.xyz, float)
    delta = xf - xi
    delta -= np.round(delta)                # nearest periodic image of each final atom
    inter.xyz = xi + (1 - t) * delta        # = t*xi + (1-t)*xf_image
    return inter
