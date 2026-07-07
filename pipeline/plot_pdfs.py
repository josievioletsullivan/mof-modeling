"""Quick-look: plot all G(r) in data/pdfs/ as an offset waterfall."""
import sys
import pathlib
import json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import config

r = np.load(config.PDFS / "r.npy")
G = np.load(config.PDFS / "G.npy")
labels = json.loads((config.PDFS / "labels.json").read_text())

off = 1.15 * np.nanmax(np.abs(G))          # vertical spacing between curves
fig, ax = plt.subplots(figsize=(8, 10))
for i, (g, lab) in enumerate(zip(G, labels)):
    ax.plot(r, g + i * off, lw=0.8)
    ax.text(r[-1], i * off, " " + lab, va="center", fontsize=8)

ax.set_xlabel("r (A)")
ax.set_ylabel("G(r)  (offset per structure)")
ax.set_xlim(r[0], r[-1] * 1.18)
ax.set_yticks([])
fig.tight_layout()

out = config.PDFS / "pdfs_overview.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
plt.show()
