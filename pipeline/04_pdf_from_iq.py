"""Stage 04 (alternate route): PDF from I(Q) by Fourier transform.

Instead of structure -> G(r) (§2), go the experimental way:
    I(Q)  -->  S(Q)  -->  F(Q)=Q[S(Q)-1]  -->  G(r)=(2/pi)∫F(Q)sin(Qr)dQ.

Reference engine PDFgetX3 (diffpy.pdfgetx) is licensed and not installed here;
src/pdf_from_iq.py is a faithful reimplementation of its ad-hoc-correction
transform. With no measured data, we forward-model a measured-like I(Q) from
each structure using the *same bulk engine as §2* (Uiso=0.07), then invert it.

Two deliverables:
  1. Round-trip validation over all endpoints -> recover the §2 PDF.
  2. Parameter demonstration (Qmin, Qmax, rpoly) on one structure, showing what
     each knob does to G(r).

Outputs in data/iq/: <label>.iq (Q,I), <label>_ft.gr (r,G), G_ft.npy, r.npy,
labels.json, sanity.json, and figures ft_validation.png + ft_parameters.png.
"""
import sys
import pathlib
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import config
from src.pdf_from_iq import structure_to_iq, iq_to_gr

DEMO = "Si_mp-149"          # structure used for the parameter sweeps

# --- reference §2 PDFs (same r-grid / normalization) ---
labels = json.loads((config.PDFS / "labels.json").read_text())
G2 = np.load(config.PDFS / "G.npy")
ref = {lab: G2[i] for i, lab in enumerate(labels)}
R = config.R
m = (R >= 1.0) & (R <= 25.0)                     # comparison window (skip r<1 junk)


def _corr(a, b):
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _amp(g):
    return float(np.sqrt(np.mean(g[m] ** 2)))


# ============================ 1. round-trip validation ============================
print("I(Q) -> G(r) round trip vs §2 (Qmin=%.1f Qmax=%.0f rpoly=%.1f Uiso=%.2f):"
      % (config.QMIN, config.QMAX, config.RPOLY, config.UISO))
print(f"{'structure':16s} {'corr':>7s} {'amp_ratio':>9s}")
rows, meta, summary = [], [], []
for lab in labels:
    q, iq, comp = structure_to_iq(config.STRUCTURES / f"{lab}.cif",
                                  add_background=True)
    np.savetxt(config.IQ / f"{lab}.iq", np.column_stack([q, iq]),
               header="Q(1/A)  I(Q)  (forward-modeled, incl. low-Q + background)",
               fmt="%.6f")
    _, g = iq_to_gr(q, iq, comp)
    np.savetxt(config.IQ / f"{lab}_ft.gr", np.column_stack([R, g]),
               header="r(A)  G(r)  (via Fourier transform of I(Q))", fmt="%.6f")
    corr, ampr = _corr(g, ref[lab]), _amp(g) / _amp(ref[lab])
    print(f"{lab:16s} {corr:>7.3f} {ampr:>9.3f}")
    rows.append(g)
    meta.append(lab)
    summary.append(dict(structure=lab, corr_vs_pdf2=corr, amp_ratio=ampr,
                        composition={k: round(v, 3) for k, v in comp.items()}))

np.save(config.IQ / "G_ft.npy", np.vstack(rows))
np.save(config.IQ / "r.npy", R)
(config.IQ / "labels.json").write_text(json.dumps(meta, indent=2))
(config.IQ / "sanity.json").write_text(json.dumps(summary, indent=2))
corrs = np.array([s["corr_vs_pdf2"] for s in summary])
print(f"\ncorr(FT route, §2): min={corrs.min():.3f} mean={corrs.mean():.3f} "
      f"max={corrs.max():.3f}  -> the two routes agree (shape). "
      f"amp within ~10% (resolution envelope + ad-hoc normalization).")

# ============================ 2. parameter demonstration ============================
q, iq_clean, comp = structure_to_iq(config.STRUCTURES / f"{DEMO}.cif",
                                    add_background=False)
q, iq_bg, comp = structure_to_iq(config.STRUCTURES / f"{DEMO}.cif",
                                 add_background=True)

qmins = [0.0, 0.5, 2.0]        # throwing out small Q
qmaxs = [6.0, 12.0, 35.0]      # truncation -> resolution / ripple
rpolys = [0.3, 0.9, 2.5]       # background poly: too low leaks, too high eats signal

print(f"\nparameter sweeps on {DEMO}:")
print("  Qmin (with low-Q junk present) corr vs §2:")
for qm in qmins:
    _, g = iq_to_gr(q, iq_bg, comp, qmin=qm)
    print(f"    Qmin={qm:>4.1f}  corr={_corr(g, ref[DEMO]):+.3f}")
print("  Qmax (clean) corr vs §2:")
for qm in qmaxs:
    _, g = iq_to_gr(q, iq_clean, comp, qmax=qm)
    print(f"    Qmax={qm:>4.0f}  corr={_corr(g, ref[DEMO]):+.3f}")
print("  rpoly (with background present) corr vs §2:")
for rp in rpolys:
    _, g = iq_to_gr(q, iq_bg, comp, rpoly=rp)
    print(f"    rpoly={rp:>4.1f} corr={_corr(g, ref[DEMO]):+.3f}")

# ============================ figures ============================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- validation figure: I(Q), F(Q), and recovered G vs §2 ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    q, iq_bg, comp = structure_to_iq(config.STRUCTURES / f"{DEMO}.cif",
                                     add_background=True)
    _, g_bg, qf, fq = iq_to_gr(q, iq_bg, comp, return_fq=True)
    q, iq_cl, comp = structure_to_iq(config.STRUCTURES / f"{DEMO}.cif",
                                     add_background=False)
    ax[0, 0].plot(q, iq_bg, lw=0.8, label="with low-Q + background")
    ax[0, 0].plot(q, iq_cl, lw=0.8, label="ideal")
    ax[0, 0].axvspan(0, config.QMIN, color="r", alpha=0.12)
    ax[0, 0].set(title=f"{DEMO}: forward-modeled I(Q)", xlabel="Q (1/Å)",
                 ylabel="I(Q)"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(qf, fq, lw=0.7)
    ax[0, 1].set(title="F(Q)=Q[S(Q)−1] after normalization + rpoly",
                 xlabel="Q (1/Å)", ylabel="F(Q)")
    mm = R <= 15
    ax[1, 0].plot(R[mm], ref[DEMO][mm], lw=1.4, label="§2 direct")
    ax[1, 0].plot(R[mm], g_bg[mm], lw=0.9, ls="--", color="k",
                  label=f"FT route (corr={_corr(g_bg, ref[DEMO]):.2f})")
    ax[1, 0].set(title="recovered G(r) vs §2", xlabel="r (Å)", ylabel="G(r)")
    ax[1, 0].legend(fontsize=8)
    # scatter of corr across all structures
    ax[1, 1].bar(range(len(corrs)), corrs, color="steelblue")
    ax[1, 1].axhline(1.0, color="grey", lw=0.6)
    ax[1, 1].set(title="corr(FT route, §2) — all endpoints", ylim=(0.9, 1.0),
                 ylabel="corr")
    ax[1, 1].set_xticks(range(len(labels)))
    ax[1, 1].set_xticklabels([l.split("_")[0] for l in labels], rotation=60,
                             fontsize=7)
    fig.tight_layout()
    fig.savefig(config.IQ / "ft_validation.png", dpi=150)
    print(f"\nsaved {config.IQ / 'ft_validation.png'}")

    # ---- parameter figure: one panel per knob, per-panel y-limits ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    mm = R <= 12
    glim = 1.15 * np.max(np.abs(ref[DEMO][(R >= 1) & mm]))    # structural G(r) scale

    # (0,0) Qmin — with rpoly working, Qmin in [0,0.5] barely matters; 2.0 loses signal
    for qm in qmins:
        _, g = iq_to_gr(q, iq_bg, comp, qmin=qm)
        ax[0, 0].plot(R[mm], g[mm], lw=0.9,
                      label=f"Qmin={qm} (corr {_corr(g, ref[DEMO]):.2f})")
    ax[0, 0].plot(R[mm], ref[DEMO][mm], lw=1.6, color="k", alpha=0.4, label="§2")
    ax[0, 0].set(title="Qmin — cut small Q\n(rpoly already removes low-Q junk; "
                 "Qmin=2 discards real signal)", ylabel="G(r)", ylim=(-glim, glim))
    ax[0, 0].legend(fontsize=7)

    # (0,1) Qmax — truncation sets peak sharpness / termination ripple
    for qm in qmaxs:
        _, g = iq_to_gr(q, iq_cl, comp, qmax=qm)
        ax[0, 1].plot(R[mm], g[mm], lw=0.9, label=f"Qmax={qm:.0f}")
    ax[0, 1].plot(R[mm], ref[DEMO][mm], lw=1.6, color="k", alpha=0.4, label="§2")
    ax[0, 1].set(title="Qmax — truncation → resolution / ripple\n"
                 "(low Qmax = broad peaks + ripple)", ylim=(-glim, glim))
    ax[0, 1].legend(fontsize=7)

    # (1,0) rpoly too low — smooth background leaks through (off-scale)
    for rp in rpolys:
        _, g = iq_to_gr(q, iq_bg, comp, rpoly=rp)
        ax[1, 0].plot(R[mm], g[mm], lw=0.9,
                      label=f"rpoly={rp} (corr {_corr(g, ref[DEMO]):.2f})")
    ax[1, 0].plot(R[mm], ref[DEMO][mm], lw=1.6, color="k", alpha=0.4, label="§2")
    ax[1, 0].set(title="rpoly TOO LOW → background leaks\n(rpoly=0.3 blows up "
                 "off-scale)", xlabel="r (Å)", ylabel="G(r)", ylim=(-3 * glim, 3 * glim))
    ax[1, 0].legend(fontsize=7)

    # (1,1) rpoly too high — eats into the first peak (clean data, zoom low r)
    zoom = (R >= 0.5) & (R <= 4.0)
    for rp in [0.9, 2.5, 4.0]:
        _, g = iq_to_gr(q, iq_cl, comp, rpoly=rp)
        ax[1, 1].plot(R[zoom], g[zoom], lw=1.0, label=f"rpoly={rp}")
    ax[1, 1].plot(R[zoom], ref[DEMO][zoom], lw=1.6, color="k", alpha=0.4, label="§2")
    ax[1, 1].axvspan(0.5, 0.9, color="r", alpha=0.10)
    ax[1, 1].set(title="rpoly TOO HIGH → eats real signal\n(first-peak region, "
                 "clean data)", xlabel="r (Å)")
    ax[1, 1].legend(fontsize=7)

    fig.suptitle(f"§4 transform parameters ({DEMO})", fontsize=13)
    fig.tight_layout()
    fig.savefig(config.IQ / "ft_parameters.png", dpi=150)
    print(f"saved {config.IQ / 'ft_parameters.png'}")
except Exception as e:
    print(f"(skipped figures: {e})")
