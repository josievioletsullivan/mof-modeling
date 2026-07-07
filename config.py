"""Single source of truth: paths + PDF physics params.
Every module imports the grid and params from here so all PDFs are
comparable (identical r-grid) for PCA / NMF / correlation / Hungarian.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STRUCTURES = DATA / "structures"
PDFS = DATA / "pdfs"
INTERMEDIATES = DATA / "intermediates"   # §3 morphed structures + their PDFs
IQ = DATA / "iq"                         # §4 I(Q) + PDFs via Fourier transform
MIXTURES = DATA / "mixtures"
RESULTS = DATA / "results"
for _d in (STRUCTURES, PDFS, INTERMEDIATES, IQ, MIXTURES, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

# real-space grid (Angstrom) -- the canonical grid; never rebuild ad hoc
RMIN, RMAX, RSTEP = 0.0, 30.0, 0.01
R = np.arange(RMIN, RMAX + 0.5 * RSTEP, RSTEP)

# scattering / calculator params
QMIN = 0.5      # A^-1; drop low-Q (macroscopic / SAS regime)
QMAX = 35.0     # A^-1; FT truncation -> real-space resolution
QDAMP = 0.04    # A^-1; instrument resolution damping
UISO = 0.07     # A^2; isotropic ADP applied to every atom (deliberately broad)

RPOLY = 0.9     # PDFgetX3 branch only; tune just below first PDF peak

TARGETS = ["Ni", "Cu", "Al", "Fe", "Si", "NaCl", "MgO", "CsCl", "ZnS", "CaF2"]
