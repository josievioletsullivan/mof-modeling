"""Stage 06: stretched-NMF — separation when mixtures also carry x-axis stretch.

Builds mixtures of K pure PDFs where each component appears at a per-sample
uniform x-axis stretch (mimicking lattice expansion across a T/P series):

    X[:, m] = Σ_k C[m,k] · stretch(G_k, S[m,k])

and shows the payoff of stretched-NMF over plain PCA/NMF:

  * PCA (SVD): stretch smears each phase across many singular directions, so the
    apparent rank explodes (K phases -> far more than K components). Plain PCA/NMF
    cannot absorb a moving x-axis.
  * plain NMF (= this SNMF with stretch frozen at 1): fits the stretched data
    badly and recovers blurred components.
  * stretched-NMF (src/snmf.py, built on diffpy.snmf's stretch operator): with
    K components it fits well AND recovers both the pure components and the
    per-sample stretch trajectories S.

Runs in env `pdfsep` (has diffpy.snmf; no sklearn -> PCA via SVD, NMF via the
stretch-frozen solver). Outputs in data/results/.
"""
import sys
import pathlib
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import linear_sum_assignment

import config
from src.mixtures import (dirichlet_coeffs, stretch_factors, stretched_mixtures,
                          linear_mixtures, lift)
from src.snmf import stretched_nmf

SEED = 0
PICK = ["Ni_mp-23", "NaCl_mp-22862", "Si_mp-149"]   # 3 structurally distinct phases
M = 24               # samples (ordered = a T/P series)
MAX_EXPANSION = 0.10  # up to 10% lattice expansion
RESTARTS = 14

labels = json.loads((config.PDFS / "labels.json").read_text())
G = np.load(config.PDFS / "G.npy")
r = np.load(config.PDFS / "r.npy")
idx = [labels.index(p) for p in PICK]
K = len(PICK)

# crop + downsample the r-grid so the SNMF inner loop is tractable
sel = (r >= 1.0) & (r <= 25.0)
rr = r[sel][::4]
A_true = G[idx][:, sel][:, ::4]                      # (K, n)
print(f"{K} phases {PICK}  |  grid {rr.size} pts  |  {M} samples")

# --- ground truth: weights C and stretch trajectories S ---
C = dirichlet_coeffs(M, K, alpha=0.7, rng=SEED + 1)
S = stretch_factors(M, K, max_expansion=MAX_EXPANSION, rng=SEED + 2)
Xs = stretched_mixtures(A_true, C, S)                # stretched mixtures (M, n)
Xu = linear_mixtures(A_true, C)                      # unstretched counterpart
np.save(config.MIXTURES / "snmf_C_true.npy", C)
np.save(config.MIXTURES / "snmf_S_true.npy", S)
np.save(config.MIXTURES / "snmf_X_stretched.npy", Xs)
np.save(config.MIXTURES / "snmf_A_true.npy", A_true)   # raw pure components (K, n) for §7 match-quality


def svd_rank(X, thr=0.99):
    Xc = X - X.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    ev = sv ** 2 / np.sum(sv ** 2)
    return int(np.searchsorted(np.cumsum(ev), thr) + 1), ev


ru, evu = svd_rank(Xu)
rs, evs = svd_rank(Xs)
print(f"\nPCA rank (99% var): unstretched = {ru} (≈K−1), stretched = {rs} "
      f"(stretch inflates rank -> PCA cannot absorb it)")

# --- factorize the stretched data: stretch ON (SNMF) vs OFF (plain NMF) ---
Xl = lift(Xs.T)                                      # (n, M) nonnegative
shift = float(np.abs(Xs.min()))
At = A_true.T + shift                                # true components, same lift

snmf = stretched_nmf(Xl, K, n_iter=150, n_restarts=RESTARTS, seed=0,
                     s_bounds=(0.85, 1.25), smooth=0.5)
plain = stretched_nmf(Xl, K, n_iter=150, n_restarts=RESTARTS, seed=0,
                      s_bounds=(1.0, 1.0), smooth=0.0)   # stretch frozen = NMF


def match(A):
    corr = np.array([[abs(np.corrcoef(A[:, i], At[:, j])[0, 1]) for j in range(K)]
                     for i in range(K)])
    ri, cj = linear_sum_assignment(-corr)
    return corr[ri, cj], ri, cj


mc_s, ri, cj = match(snmf["A"])
mc_p, _, _ = match(plain["A"])
Sr = snmf["S"][ri]                                   # recovered stretch, phase-ordered
S_true = S.T[cj]
stretch_corr = [float(np.corrcoef(Sr[a], S_true[a])[0, 1]) for a in range(K)]

print(f"\nfit residual:   SNMF = {snmf['obj'][-1]:.2e}   plain NMF = {plain['obj'][-1]:.2e}"
      f"   ({plain['obj'][-1]/snmf['obj'][-1]:.0f}× worse without stretch)")
print(f"component corr: SNMF mean = {mc_s.mean():.3f}   plain mean = {mc_p.mean():.3f}")
for a in range(K):
    print(f"  {PICK[cj[a]]:16s} SNMF={mc_s[a]:.3f}  plain={mc_p[a]:.3f}  "
          f"stretch_corr={stretch_corr[a]:.3f}")

summary = dict(phases=PICK, n_samples=M, max_expansion=MAX_EXPANSION,
               pca_rank_unstretched=ru, pca_rank_stretched=rs,
               snmf_residual=float(snmf["obj"][-1]),
               plain_residual=float(plain["obj"][-1]),
               snmf_component_corr=dict(zip([PICK[j] for j in cj], [round(float(x), 4) for x in mc_s])),
               plain_component_corr_mean=round(float(mc_p.mean()), 4),
               stretch_corr=dict(zip([PICK[j] for j in cj], [round(x, 4) for x in stretch_corr])))
(config.RESULTS / "snmf_summary.json").write_text(json.dumps(summary, indent=2))
np.save(config.RESULTS / "snmf_components.npy", snmf["A"])
np.save(config.RESULTS / "snmf_stretch.npy", snmf["S"])

# ============================ figures ============================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    # (0,0) PCA scree: unstretched vs stretched
    x = np.arange(1, 9)
    ax[0, 0].bar(x - 0.2, evu[:8], width=0.4, label=f"unstretched (rank {ru})")
    ax[0, 0].bar(x + 0.2, evs[:8], width=0.4, label=f"stretched (rank {rs})")
    ax[0, 0].axvline(K + 0.5, color="grey", ls="--", lw=1, label=f"K = {K} phases")
    ax[0, 0].set(title="PCA scree — stretch inflates the rank",
                 xlabel="singular component", ylabel="variance ratio")
    ax[0, 0].legend(fontsize=8)
    # (0,1) recovered components (SNMF) vs true. SNMF components are unit-norm,
    # so rescale each to the true amplitude (LSQ scalar) for the overlay.
    off = 1.1 * np.max(np.abs(At))
    for a in range(K):
        rec = snmf["A"][:, ri[a]]
        true = At[:, cj[a]]
        scale = float(rec @ true) / float(rec @ rec + 1e-12)
        ax[0, 1].plot(rr, true + a * off, lw=1.3, color=f"C{a}")
        ax[0, 1].plot(rr, rec * scale + a * off, lw=0.9, ls="--", color="k")
        ax[0, 1].text(rr[-1], a * off, f" {PICK[cj[a]].split('_')[0]} "
                      f"({mc_s[a]:.2f})", fontsize=8, va="center")
    ax[0, 1].set(title="SNMF components (dashed) vs true (solid)", xlabel="r (Å)",
                 ylabel="G+shift (offset)")
    # (1,0) stretch trajectory recovery
    for a in range(K):
        ax[1, 0].plot(S_true[a], color=f"C{a}", lw=1.4,
                      label=f"{PICK[cj[a]].split('_')[0]} true")
        # align gauge: recovered stretch may differ by a constant scale
        g = np.mean(S_true[a]) / np.mean(Sr[a])
        ax[1, 0].plot(Sr[a] * g, color=f"C{a}", lw=0.9, ls="--", marker="o", ms=3)
    ax[1, 0].set(title=f"stretch trajectories: true (solid) vs SNMF (dashed)\n"
                 f"corr = {np.round(stretch_corr, 3)}", xlabel="sample index (T/P series)",
                 ylabel="stretch factor (gauge-aligned)")
    ax[1, 0].legend(fontsize=8)
    # (1,1) SNMF vs plain NMF: component corr + residual
    xb = np.arange(K)
    ax[1, 1].bar(xb - 0.2, sorted(mc_s, reverse=True), width=0.4, label="SNMF")
    ax[1, 1].bar(xb + 0.2, sorted(mc_p, reverse=True), width=0.4, label="plain NMF")
    ax[1, 1].set(title=f"component recovery on stretched data\n"
                 f"residual: SNMF {snmf['obj'][-1]:.1f} vs plain {plain['obj'][-1]:.0f}",
                 xlabel="component (sorted)", ylabel="corr to true", ylim=(0, 1))
    ax[1, 1].legend(fontsize=8)
    fig.suptitle("§5.4 stretched-NMF — recovering components + stretch that PCA/NMF cannot",
                 fontsize=12)
    fig.tight_layout()
    out = config.RESULTS / "snmf_overview.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")
except Exception as e:
    print(f"(skipped figures: {e})")
