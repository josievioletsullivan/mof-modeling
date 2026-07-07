"""Stage 03: initial / final / intermediate states.

For every ordered pair of endpoint structures that share an atom count (the
requirement for a 1:1 atom correspondence), morph the initial toward the final
to build an intermediate, then compute the intermediate's PDF the *same way as
§2* (src.pdf.pdf_from_structure -> config.R).

Writes per intermediate:
  data/intermediates/<init>__<final>__t<NN>.cif   the morphed structure
  data/intermediates/<init>__<final>__t<NN>.gr    its G(r) on config.R
plus stacked arrays (G_inter.npy, r.npy, labels.json) and a sanity summary
(sanity.json) + overview figure (intermediates_overview.png).

Sanity check (§3). Two things get confirmed, and they are deliberately
distinct because correlation alone conflates a real bug with real physics:

  * GRID + SCALE (hard invariants). All curves live on config.R by
    construction (asserted once, globally). "Same normalization" is checked
    with a shift-robust amplitude ratio: rms(G_inter) / sqrt(rms_i * rms_f)
    should sit near 1 (band [0.25, 4]). This is what actually catches a
    grid/scale bug -- and it is blind to where the peaks land.

  * BLEND RESEMBLANCE (informational). corr(G_inter, t*G_i + (1-t)*G_f).
    PDF is NOT linear in atomic position, so this is high ONLY when the two
    endpoints' peaks still overlap (small lattice mismatch, e.g. Cu<->Ni ~0.9)
    and falls -- even goes negative -- once peaks shift by more than a peak
    width. That drop is PHYSICS, not a bug: a summed blend keeps both peak
    sets while the intermediate has a single peak between them. It falls
    further still for *reconstructive* morphs (coord_rmsd large, e.g. diamond
    Si <-> rocksalt), where the coordination itself reorganizes. So a low corr
    is only flagged when the grid/scale invariant ALSO fails.
"""
import sys
import pathlib
import json
from itertools import permutations

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from diffpy.structure import loadStructure

import config
from src.pdf import pdf_from_structure
from src.interpolate import interpolate_structures

T = 0.5  # weight on the initial endpoint; t in (0,1). 0.5 = geometric midpoint

# ---- endpoints: structures + their §2 PDFs (same r-grid / normalization) ----
labels = json.loads((config.PDFS / "labels.json").read_text())
G = np.load(config.PDFS / "G.npy")                 # (n_endpoints, n_r) from §2
r = np.load(config.PDFS / "r.npy")
assert np.allclose(r, config.R), "endpoint r-grid != config.R -- rerun §2"
Gend = {lab: G[i] for i, lab in enumerate(labels)}
stru = {lab: loadStructure(str(config.STRUCTURES / f"{lab}.cif")) for lab in labels}
natoms = {lab: len(stru[lab]) for lab in labels}

# ---- enumerate all valid (initial, final) permutations: equal atom count ----
groups = {}
for lab in labels:
    groups.setdefault(natoms[lab], []).append(lab)

pairs = [(i, f) for members in groups.values()
         for (i, f) in permutations(members, 2)]
skipped = [labs[0] for labs in groups.values() if len(labs) == 1]

print(f"{len(labels)} endpoints -> atom-count groups: "
      + ", ".join(f"{n}:[{','.join(m)}]" for n, m in sorted(groups.items())))
if skipped:
    print(f"no partner (skipped, cannot map atoms 1:1): {', '.join(skipped)}")
print(f"generating {len(pairs)} intermediates at t={T} "
      f"(weight on initial; t=1->initial, t=0->final)\n")

# ---- morph -> PDF -> sanity, per pair ----
_fit = (config.R >= 1.0) & (config.R <= 10.0)      # window for amplitude/scale


def _rms(g):
    return float(np.sqrt(np.mean(g[_fit] ** 2)))


def _coord_rmsd(a, b):
    """Min-image fractional RMSD between index-mapped atom sets.
    ~0 => lattice-only morph (coords identical); large => reconstructive."""
    d = np.asarray(a.xyz, float) - np.asarray(b.xyz, float)
    d -= np.round(d)
    return float(np.sqrt(np.mean(d ** 2)))


SCALE_BAND = (0.25, 4.0)   # rms ratio outside this => genuine grid/scale bug
RECON_THRESH = 0.05        # coord_rmsd above this => reconstructive morph

tag = f"t{int(round(T * 100)):02d}"
rows, meta, summary = [], [], []
for i_lab, f_lab in pairs:
    name = f"{i_lab}__{f_lab}__{tag}"
    inter = interpolate_structures(stru[i_lab], stru[f_lab], t=T)
    inter.write(str(config.INTERMEDIATES / f"{name}.cif"), "cif")

    g_inter = pdf_from_structure(inter)            # §2 machinery -> config.R
    np.savetxt(config.INTERMEDIATES / f"{name}.gr",
               np.column_stack([config.R, g_inter]),
               header="r(A)  G(r)", fmt="%.6f")

    # --- hard invariant: consistent normalization (shift-robust amplitude) ---
    scale_ratio = float(_rms(g_inter) / np.sqrt(_rms(Gend[i_lab]) * _rms(Gend[f_lab])))
    scale_ok = bool(SCALE_BAND[0] <= scale_ratio <= SCALE_BAND[1])

    # --- informational: resemblance to the weight-matched linear blend ---
    blend = T * Gend[i_lab] + (1 - T) * Gend[f_lab]
    corr = float(np.corrcoef(g_inter, blend)[0, 1])
    coord_rmsd = _coord_rmsd(stru[i_lab], stru[f_lab])
    regime = "reconstructive" if coord_rmsd > RECON_THRESH else "lattice"
    # only a failed grid/scale invariant is a bug; a low corr on its own is physics
    flag = "" if scale_ok else "  <-- SCALE OUT OF BAND: grid/scale bug"
    print(f"{name:48s} atoms={natoms[i_lab]:2d}  scale={scale_ratio:.2f}  "
          f"corr={corr:+.3f}  {regime:14s}{flag}")

    rows.append(g_inter)
    meta.append(name)
    summary.append(dict(initial=i_lab, final=f_lab, t=T, natoms=natoms[i_lab],
                        species=sorted(set(inter.element)),
                        morph=regime, coord_rmsd=coord_rmsd,
                        scale_ratio=scale_ratio, scale_ok=scale_ok,
                        corr_vs_blend=corr))

# ---- stacked outputs for the decomposition stage ----
Gi = np.vstack(rows)
np.save(config.INTERMEDIATES / "G_inter.npy", Gi)
np.save(config.INTERMEDIATES / "r.npy", config.R)
(config.INTERMEDIATES / "labels.json").write_text(json.dumps(meta, indent=2))
(config.INTERMEDIATES / "sanity.json").write_text(json.dumps(summary, indent=2))

corrs = np.array([s["corr_vs_blend"] for s in summary])
bugs = [s for s in summary if not s["scale_ok"]]
lat = [s["corr_vs_blend"] for s in summary if s["morph"] == "lattice"]
print(f"\nstacked G_inter shape={Gi.shape} -> {config.INTERMEDIATES / 'G_inter.npy'}")
print(f"GRID:  all {len(summary)} intermediates on config.R (asserted).")
print(f"SCALE: {len(summary) - len(bugs)}/{len(summary)} within band {SCALE_BAND} "
      f"-> normalization consistent."
      + ("" if not bugs else "  BUGS: " + ", ".join(s['initial'] + '->' + s['final'] for s in bugs)))
print(f"BLEND corr (informational): all={corrs.min():+.2f}..{corrs.max():+.2f}; "
      f"lattice-only morphs {min(lat):+.2f}..{max(lat):+.2f} "
      f"(high when endpoints similar, e.g. Cu<->Ni). Low corr = peak-shift "
      f"physics, not a bug -- grid & scale above are the real check.")

# ---- overview figure: for each pair, initial / final / intermediate / blend ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rmax_plot = 10.0
    m = config.R <= rmax_plot
    ncol = 3
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.4 * nrow),
                             squeeze=False)
    for ax, (i_lab, f_lab), g_inter, s in zip(
            axes.ravel(), pairs, rows, summary):
        blend = T * Gend[i_lab] + (1 - T) * Gend[f_lab]
        ax.plot(config.R[m], Gend[i_lab][m], lw=0.7, alpha=0.6, label="initial")
        ax.plot(config.R[m], Gend[f_lab][m], lw=0.7, alpha=0.6, label="final")
        ax.plot(config.R[m], g_inter[m], lw=1.3, color="k", label="intermediate")
        ax.plot(config.R[m], blend[m], lw=0.8, ls="--", color="r",
                label="t·Gi+(1-t)·Gf")
        ax.set_title(f"{i_lab.split('_')[0]}→{f_lab.split('_')[0]}  "
                     f"corr={s['corr_vs_blend']:.2f} · {s['morph'][:3]} · "
                     f"×{s['scale_ratio']:.2f}", fontsize=8)
        ax.set_xlim(0, rmax_plot)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6, loc="upper right")
    fig.supxlabel("r (Å)")
    fig.supylabel("G(r)")
    fig.suptitle(f"§3 initial / final / intermediate (t={T})", fontsize=11)
    fig.tight_layout()
    out = config.INTERMEDIATES / "intermediates_overview.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
except Exception as e:  # plotting is optional; never block the data outputs
    print(f"(skipped overview figure: {e})")
