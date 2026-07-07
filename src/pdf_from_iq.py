"""§4 (alternate route) — PDF from I(Q) by Fourier transform.

The §2 route goes structure -> G(r) directly. This route mirrors what an
*experiment* does: measured intensity I(Q) -> normalize to S(Q) -> reduced
structure function F(Q) = Q[S(Q) - 1] -> sine transform to G(r):

    G(r) = (2/pi) * integral_{Qmin}^{Qmax}  F(Q) * sin(Q*r) dQ .

Engine note: the reference engine is PDFgetX3 (`diffpy.pdfgetx`), which is a
separately *licensed* package and is not installed here. This module is a
faithful, dependency-free reimplementation of its ad-hoc-correction transform
(Juhas et al., J. Appl. Cryst. 46 (2013) 560): the same Qmin/Qmax cropping,
the same rpoly polynomial background removal, and the same composition-based
normalization. `iq_to_gr` accepts any (Q, I) array, so a real PDFgetX3 output
or a raw .chi/.iq file can be dropped in unchanged.

Because the project has no measured data, `structure_to_iq` forward-models a
measured-like I(Q) from a known structure (Uiso = 0.07, consistent with §2).
Feeding that through `iq_to_gr` and comparing to the §2 PDF is a closed-loop
validation: recover the G(r) you started from.
"""
from pathlib import Path

import numpy as np
from diffpy.structure import loadStructure
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal.scatteringfactortable import SFTXray

import config

_SFT = SFTXray()


def composition_of(stru):
    """Atom-fraction composition {element: fraction} from a diffpy Structure."""
    els = list(stru.element)
    uniq, counts = np.unique(els, return_counts=True)
    return {e: c / counts.sum() for e, c in zip(uniq, counts)}


def average_scattering(composition, q):
    """Composition-weighted X-ray form factors on grid q:
    returns <f(Q)>, <f(Q)^2>  (the Faber-Ziman averages used to normalize I(Q))."""
    f_avg = np.zeros_like(q, float)
    f2_avg = np.zeros_like(q, float)
    for el, frac in composition.items():
        fq = np.array([_SFT.lookup(el, qi) for qi in q])
        f_avg += frac * fq
        f2_avg += frac * fq ** 2
    return f_avg, f2_avg


def structure_to_iq(cif_or_stru, uiso=config.UISO, qmax=config.QMAX, qstep=0.02,
                    add_background=True, seed_amp=1.0):
    """Forward-model a measured-like coherent intensity I(Q) from a structure.

    Faber-Ziman:  I(Q) = <f^2> + <f>^2 * (S(Q) - 1),
    with S(Q)-1 = F(Q)/Q taken from the *same bulk-crystal engine as §2*
    (diffpy PDFCalculator exposes F(Q) via .fq), Uiso baked in for consistency.
    Using this engine -- rather than the finite-cluster Debye equation -- means
    the forward I(Q) describes the same infinite crystal §2 models, so a correct
    transform recovers the §2 G(r). Optionally adds the two things a real pattern
    carries and that the transform must handle: a low-Q small-angle upturn
    (macroscopic/SAS junk) and a smooth instrumental background.
    Returns (Q, I, composition)."""
    stru = loadStructure(str(cif_or_stru)) \
        if isinstance(cif_or_stru, (str, bytes, Path)) else cif_or_stru
    stru.Uisoequiv = uiso
    # bulk F(Q) = Q[S(Q)-1] over the full Q range (qmin=0, no qdamp: applied on inversion)
    pc = PDFCalculator(qmin=0.0, qmax=qmax, qdamp=0.0,
                       rmin=0.0, rmax=config.RMAX, rstep=config.RSTEP)
    pc(stru)
    q = np.arange(0.0, qmax + 0.5 * qstep, qstep)          # smooth uniform grid
    fq = np.interp(q, np.asarray(pc.qgrid, float), np.asarray(pc.fq, float))

    comp = composition_of(stru)
    f_avg, f2_avg = average_scattering(comp, q)

    sq_minus_1 = np.zeros_like(q)
    nz = q > 0
    sq_minus_1[nz] = fq[nz] / q[nz]
    iq = f2_avg + f_avg ** 2 * sq_minus_1          # coherent intensity per atom

    if add_background:
        # low-Q upturn: large length-scale scattering with no local structure
        iq += seed_amp * 8.0 * f2_avg[0] * np.exp(-(q / 0.6) ** 2)
        # smooth slowly-varying instrumental background (removed by rpoly)
        iq += seed_amp * f2_avg[0] * (0.15 + 0.02 * q - 0.0008 * q ** 2)
    return q, iq, comp


def _rpoly_degree(qmax, rpoly):
    """Number of polynomial terms rpoly implies: a degree-n polynomial in Q
    only affects G(r) below ~ n*pi/qmax, so features below rpoly are captured by
    n ~ qmax*rpoly/pi terms (Juhas 2013). rpoly sits just below the first peak."""
    return max(1, int(round(qmax * rpoly / np.pi)))


def sine_transform(q, fq, r):
    """G(r) = (2/pi) * integral F(Q) sin(Qr) dQ, trapezoid over the q grid."""
    integrand = fq[None, :] * np.sin(np.outer(r, q))     # (nr, nq)
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy>=2 vs <2
    return (2.0 / np.pi) * _trap(integrand, q, axis=1)


def iq_to_gr(q, iq, composition, r=None, qmin=config.QMIN, qmax=config.QMAX,
             rpoly=config.RPOLY, qdamp=config.QDAMP, return_fq=False):
    """Measured I(Q) -> G(r), the PDFgetX3-style ad-hoc-correction transform.

    Steps: normalize to S(Q) with the composition form factors, form F(Q),
    remove the rpoly-controlled polynomial background, crop to [Qmin, Qmax],
    sine-transform, and apply the qdamp real-space resolution envelope.
    `composition` may be a diffpy Structure or an {element: fraction} dict.
    """
    if r is None:
        r = config.R
    if not isinstance(composition, dict):
        composition = composition_of(composition)
    q = np.asarray(q, float)
    iq = np.asarray(iq, float)

    # --- normalize intensity -> reduced structure function F(Q) ---
    f_avg, f2_avg = average_scattering(composition, q)
    sq = np.ones_like(q)
    nz = f_avg > 0
    sq[nz] = (iq[nz] - (f2_avg[nz] - f_avg[nz] ** 2)) / f_avg[nz] ** 2   # Faber-Ziman
    fq = q * (sq - 1.0)

    # --- rpoly: subtract the best-fit low-order polynomial (slow background) ---
    deg = _rpoly_degree(qmax, rpoly)
    win = (q >= qmin) & (q <= qmax)
    # Polynomial.fit rescales Q to [-1, 1] internally -> well-conditioned at high deg
    bg = np.polynomial.Polynomial.fit(q[win], fq[win], deg)
    fq_corr = fq - bg(q)

    # --- crop reciprocal space, then sine transform ---
    fq_crop = np.where(win, fq_corr, 0.0)
    g = sine_transform(q, fq_crop, r)

    # --- qmin: throw out small Q (done by the crop); qdamp: resolution envelope ---
    g = g * np.exp(-(r * qdamp) ** 2 / 2.0)
    if return_fq:
        return r, g, q, fq_corr
    return r, g
