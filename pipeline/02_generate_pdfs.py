"""Stage 02: CIFs in data/structures/ -> G(r) on config.R.

Writes one .gr per structure, plus stacked arrays for the decomposition
stage: G.npy (n_structures, n_r), r.npy, labels.json.
"""
import sys
import pathlib
import json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import config
from src.pdf import compute_pdf, save_gr

cifs = sorted(config.STRUCTURES.glob("*.cif"))
if not cifs:
    raise SystemExit("no CIFs in data/structures/ -- run pipeline/01 first")

labels, rows = [], []
for cif in cifs:
    label = cif.stem                       # e.g. NaCl_mp-22862
    g = compute_pdf(cif)
    save_gr(config.PDFS / f"{label}.gr", g)
    labels.append(label)
    rows.append(g)
    print(f"{label:22s} points={g.size}")

G = np.vstack(rows)                         # (n_structures, n_r)
np.save(config.PDFS / "G.npy", G)
np.save(config.PDFS / "r.npy", config.R)
(config.PDFS / "labels.json").write_text(json.dumps(labels, indent=2))
print(f"\nstacked G shape={G.shape} -> {config.PDFS / 'G.npy'}")
