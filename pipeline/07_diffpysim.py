"""Stage 07: stretched-NMF on data from the PAPER'S SIMULATOR (diffpysim).

github.com/yevgenyr/diffpysim (Gu et al., stretched-NMF paper). Unlike stage 06
(which stretches PDFs by interpolation), diffpysim *physically* re-simulates each
component's PDF at a genuinely expanded lattice via diffpy.srfit's PDFGenerator,
so peaks shift as a real lattice expansion. We then separate the mixtures.

Pipeline (all in env `mof`: diffpysim + sklearn + numpy/scipy):
  1. diffpysim -> physical stretched PDF series for K phases (ground-truth
     components A0, per-component stretch S, on grid r starting at 0).
  2. build M mixtures: sample m = Σ_k C[m,k]·(phase k at strain step m), random
     weights C. Stretch is monotonic across samples (a T-series).
  3. PCA (SVD): physical stretch inflates the rank (K phases -> many components).
  4. stretched-NMF (src/snmf.py, grid-aware so the stretch is about r=0 =
     physical) vs plain NMF -> recover components + stretch + weights.

Key settings learned: broad peaks (Uiso = §2's 0.07; sharp peaks make the stretch
search a needle-in-a-haystack), grid from r=0 (so the r-scaling stretch matches a
physical expansion), NMF-seeded init.
Outputs in data/results/.
"""
import sys
import pathlib
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment

import config
from src.diffpysim_run import run_diffpysim
from src.mixtures import dirichlet_coeffs, lift
from src.snmf import stretched_nmf

SEED = 3
M = 24                       # samples (a monotonic strain / T series)
DOWN = 4                     # r-grid downsample for tractability
COMPOUNDS = {
    "Ni":   dict(cifname="Ni_mp-23.cif",     weight=0.34, max_strain=(0.06,) * 3),
    "NaCl": dict(cifname="NaCl_mp-22862.cif", weight=0.33, max_strain=(0.10,) * 3),
    "Si":   dict(cifname="Si_mp-149.cif",     weight=0.33, max_strain=(0.04,) * 3),
}

# ---------- 1. paper's simulator: physical stretched PDF series ----------
print("running diffpysim (physical lattice expansion, Uiso=%.2f)..." % config.UISO)
sim = run_diffpysim(str(config.STRUCTURES.parent / "diffpysim" / "cif_bank"),
                    COMPOUNDS, steps=M, qmin=config.QMIN, qmax=30.0,
                    uiso=config.UISO, rmax=30.0)
names = sim["names"]
K = len(names)
r = sim["r"]
sel = (r >= 0.0) & (r <= 25.0)                 # grid MUST start at 0 (physical stretch)
rr = r[sel][::DOWN]
comps = sim["components"][:, :, sel][:, :, ::DOWN]     # (K, M, n)
A0 = comps[:, 0, :]                                    # unstrained pure signals
S_true = sim["S_true"]                                 # (K, M) physical stretch
print(f"  {K} phases {names}, stretch to {np.round(S_true[:, -1], 3)}, grid {rr.size} pts")

# ---------- 2. build mixtures: random weights, physical stretch per sample ----------
rng = np.random.default_rng(SEED)
C = dirichlet_coeffs(M, K, alpha=0.7, rng=SEED)        # ground-truth weights
X = np.array([sum(C[m, k] * comps[k, m] for k in range(K)) for m in range(M)])
Xu = np.array([sum(C[m, k] * A0[k] for k in range(K)) for m in range(M)])  # unstretched
np.save(config.MIXTURES / "diffpysim_X.npy", X)
np.save(config.MIXTURES / "diffpysim_C_true.npy", C)
np.save(config.MIXTURES / "diffpysim_S_true.npy", S_true)
np.save(config.MIXTURES / "diffpysim_A0.npy", A0)      # raw pure components (K, n) for §7 match-quality


def svd_rank(A, thr=0.99):
    Ac = A - A.mean(0)
    sv = np.linalg.svd(Ac, compute_uv=False)
    ev = sv ** 2 / np.sum(sv ** 2)
    return int(np.searchsorted(np.cumsum(ev), thr) + 1), ev


# ---------- 3. PCA rank ----------
ru, evu = svd_rank(Xu)
rs, evs = svd_rank(X)
print(f"\nPCA rank (99%): unstretched {ru} (≈K−1), physically stretched {rs} "
      f"(real lattice expansion inflates the rank)")

# ---------- 4. stretched-NMF vs plain NMF ----------
shift = float(np.abs(X.min()))
Xl = lift(X.T)                                          # (n, M) nonnegative
At = A0.T + shift

best = None                                             # plain NMF (seed init)
for s in range(6):
    m = NMF(K, init="nndsvdar", max_iter=4000, random_state=s)
    m.fit(Xl.T)
    if best is None or m.reconstruction_err_ < best.reconstruction_err_:
        best = m
A_nmf = best.components_.T

snmf = stretched_nmf(Xl, K, grid=rr, n_iter=150, n_restarts=8, seed=0,
                     s_bounds=(0.9, 1.2), smooth=0.5, nonneg=True, A_init=A_nmf)


def match(A):
    cc = np.array([[abs(np.corrcoef(A[:, i], At[:, j])[0, 1]) for j in range(K)]
                   for i in range(K)])
    ri, cj = linear_sum_assignment(-cc)
    return cc, ri, cj


cc_s, ri, cj = match(snmf["A"])
cc_p, riP, cjP = match(A_nmf)
Sr = snmf["S"][ri]
stretch_corr = [float(np.corrcoef(Sr[a], S_true[cj[a]])[0, 1]) for a in range(K)]
Wr = snmf["W"][ri]
weight_corr = [float(np.corrcoef(Wr[a], C[:, cj[a]])[0, 1]) for a in range(K)]

print(f"\nresidual: SNMF {snmf['obj'][-1]:.1f}  plain NMF (no stretch) {'-'}")
print(f"{'phase':6s} {'SNMF comp':>10s} {'NMF comp':>9s} {'stretch':>8s} {'weight':>7s}")
for a in range(K):
    print(f"{names[cj[a]]:6s} {cc_s[ri[a], cj[a]]:>10.3f} "
          f"{cc_p[riP[a], cjP[a]]:>9.3f} {stretch_corr[a]:>8.3f} {weight_corr[a]:>7.3f}")
print(f"mean component corr: SNMF {cc_s[ri, cj].mean():.3f}  plain NMF {cc_p[riP, cjP].mean():.3f}")

summary = dict(source="diffpysim (physical lattice expansion)", phases=names,
               n_samples=M, uiso=config.UISO,
               pca_rank_unstretched=ru, pca_rank_stretched=rs,
               snmf_component_corr=dict(zip([names[j] for j in cj],
                                            [round(float(cc_s[ri[a], cj[a]]), 4) for a in range(K)])),
               plain_nmf_component_corr_mean=round(float(cc_p[riP, cjP].mean()), 4),
               stretch_corr=dict(zip([names[j] for j in cj], [round(x, 4) for x in stretch_corr])),
               weight_corr=dict(zip([names[j] for j in cj], [round(x, 4) for x in weight_corr])))
(config.RESULTS / "diffpysim_snmf_summary.json").write_text(json.dumps(summary, indent=2))
np.save(config.RESULTS / "diffpysim_components.npy", snmf["A"])   # recovered (n, K) for §7 match-quality

# ---------- figure ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    # physical stretched series (one phase) to show real peak shift
    kshow = int(np.argmax(S_true[:, -1]))
    off = 0.6 * np.max(np.abs(comps[kshow]))
    for i in range(0, M, max(1, M // 8)):
        ax[0, 0].plot(rr, comps[kshow, i] + (i / M) * off * 3, lw=0.7)
    ax[0, 0].set(title=f"diffpysim physical stretch: {names[kshow]} PDF vs strain step",
                 xlabel="r (Å)", ylabel="G(r) (offset by step)", xlim=(1, 12))
    # PCA scree
    x = np.arange(1, 9)
    ax[0, 1].bar(x - 0.2, evu[:8], width=0.4, label=f"unstretched (rank {ru})")
    ax[0, 1].bar(x + 0.2, evs[:8], width=0.4, label=f"stretched (rank {rs})")
    ax[0, 1].axvline(K + 0.5, color="grey", ls="--", lw=1, label=f"K={K}")
    ax[0, 1].set(title="PCA scree — physical stretch inflates rank",
                 xlabel="component", ylabel="variance ratio")
    ax[0, 1].legend(fontsize=8)
    # recovered components vs true
    offc = 1.1 * np.max(np.abs(At))
    for a in range(K):
        rec = snmf["A"][:, ri[a]]
        sc = float(rec @ At[:, cj[a]]) / float(rec @ rec + 1e-12)
        ax[1, 0].plot(rr, At[:, cj[a]] + a * offc, lw=1.3, color=f"C{a}")
        ax[1, 0].plot(rr, rec * sc + a * offc, lw=0.9, ls="--", color="k")
        ax[1, 0].text(rr[-1], a * offc, f" {names[cj[a]]} ({cc_s[ri[a], cj[a]]:.2f})",
                      fontsize=8, va="center")
    ax[1, 0].set(title="SNMF components (dashed) vs true (solid)", xlabel="r (Å)",
                 ylabel="G+shift (offset)")
    # stretch trajectories
    for a in range(K):
        ax[1, 1].plot(S_true[cj[a]], color=f"C{a}", lw=1.4, label=f"{names[cj[a]]} true")
        g = np.mean(S_true[cj[a]]) / np.mean(Sr[a])
        ax[1, 1].plot(Sr[a] * g, color=f"C{a}", lw=0.9, ls="--", marker="o", ms=3)
    ax[1, 1].set(title=f"stretch: true (solid) vs SNMF (dashed)\ncorr {np.round(stretch_corr, 3)}",
                 xlabel="sample (strain step)", ylabel="stretch (gauge-aligned)")
    ax[1, 1].legend(fontsize=8)
    fig.suptitle("§5 diffpysim — stretched-NMF on the paper's physical simulator", fontsize=12)
    fig.tight_layout()
    out = config.RESULTS / "diffpysim_snmf_overview.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")
except Exception as e:
    print(f"(skipped figure: {e})")
