"""Fetch simple, high-symmetry ground-state structures from Materials Project."""
import os
from pathlib import Path

from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def fetch_structures(targets, outdir, api_key=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    api_key = api_key or os.environ.get("MP_API_KEY")
    saved = []
    with MPRester(api_key) as mpr:
        for formula in targets:
            docs = mpr.materials.summary.search(
                formula=formula,
                fields=["material_id", "structure", "formula_pretty",
                        "symmetry", "nsites", "energy_above_hull"],
            )
            if not docs:
                print(f"{formula:7s} (no match)")
                continue
            doc = min(docs, key=lambda d: d.energy_above_hull)
            conv = SpacegroupAnalyzer(doc.structure).get_conventional_standard_structure()
            path = outdir / f"{formula}_{doc.material_id}.cif"
            conv.to(filename=str(path), fmt="cif")
            print(f"{formula:7s} {str(doc.material_id):13s} "
                  f"{doc.symmetry.symbol:10s} nsites={len(conv)}")
            saved.append(path)
    return saved
