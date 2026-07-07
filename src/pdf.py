"""Forward PDF generation: structure -> G(r) on the canonical config.R grid."""
import numpy as np
from diffpy.structure import loadStructure
from diffpy.srreal.pdfcalculator import PDFCalculator

import config


def pdf_from_structure(stru, uiso=config.UISO):
    """Core: a diffpy Structure -> G(r) resampled onto config.R.
    Every PDF in the project (endpoints §2, intermediates §3) goes through
    here, so they share an identical r-grid and normalization by construction."""
    stru.Uisoequiv = uiso  # isotropic ADP applied to all atoms
    pc = PDFCalculator(
        rmin=config.RMIN, rmax=config.RMAX, rstep=config.RSTEP,
        qmin=config.QMIN, qmax=config.QMAX, qdamp=config.QDAMP,
    )
    r, g = pc(stru)
    return np.interp(config.R, r, g)  # force onto the canonical grid


def compute_pdf(cif_path, uiso=config.UISO):
    """Compute G(r) for one CIF, resampled onto config.R so every
    output vector is row-aligned for PCA / NMF / correlation."""
    stru = loadStructure(str(cif_path))
    return pdf_from_structure(stru, uiso)


def save_gr(path, g):
    """Write a two-column .gr (r, G) on config.R."""
    np.savetxt(path, np.column_stack([config.R, g]),
               header="r(A)  G(r)", fmt="%.6f")
