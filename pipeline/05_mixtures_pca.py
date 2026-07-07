"""Stage 05: linear combinations + signal separation (no stretch).

Build mixtures G_mix = C @ G from the endpoint PDFs with KNOWN nonnegative
coefficients C (the ground truth), then try to recover the underlying signals:

  * PCA (sklearn): how many components carry the variance? The mixtures live on
    a simplex (rows of C sum to 1), so after centering the variance sits in
    (n_phases - 1) directions. PCA components are orthogonal *mixtures*, not
    pure phases -- good for counting rank, not for reading off phases.
  * NMF (sklearn): nonnegative, so its components look like actual pure phases.
    We Hungarian-match them back to the input PDFs and also check the recovered
    weights against the ground-truth C.

Runs in env `mof` (needs sklearn). Outputs in data/mixtures/ + data/results/.
"""
import sys
import pathlib
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA, NMF

import config
from src.mixtures import dirichlet_coeffs, linear_mixtures, lift

SEED = 0
N_MIX = 150         # number of mixture samples
ALPHA = 0.4         # Dirichlet concentration (lower = sparser, more near-pure)

# --- pure components = the §2 endpoint PDFs ---
labels = json.loads((config.PDFS / "labels.json").read_text())
G = np.load(config.PDFS / "G.npy")            # (n_phases, n_r)
r = np.load(config.PDFS / "r.npy")
n_phases = len(labels)
print(f"{n_phases} pure phases, {G.shape[1]} r-points")

# --- build mixtures with known coefficients (ground truth) ---
C = dirichlet_coeffs(N_MIX, n_phases, alpha=ALPHA, rng=SEED)   # (N_MIX, n_phases)
X = linear_mixtures(G, C)                                      # (N_MIX, n_r)
np.save(config.MIXTURES / "C_true.npy", C)
np.save(config.MIXTURES / "X_mix.npy", X)
np.save(config.MIXTURES / "r.npy", r)
(config.MIXTURES / "labels.json").write_text(json.dumps(labels, indent=2))
print(f"built {N_MIX} mixtures -> {config.MIXTURES/'X_mix.npy'}  (C_true saved)")

# ============================ PCA ============================
pca = PCA(n_components=min(N_MIX, n_phases + 3)).fit(X)   # rows=mixtures, cols=r
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
n_99 = int(np.searchsorted(cum, 0.99) + 1)
print(f"\nPCA: {n_99} components explain 99% of variance "
      f"(expected ~{n_phases - 1} from the sum-to-1 simplex). "
      f"first {n_phases} EVR: {np.round(evr[:n_phases], 4)}")

# ============================ NMF ============================
# NMF needs nonnegative data; lift the whole stack by the same constant.
shift = float(np.abs(np.min(X)))
Xl = X + shift
# blind NMF is seed-sensitive; keep the fit with the lowest reconstruction error
best_nmf, Wn = None, None
for s in range(8):
    m = NMF(n_components=n_phases, init="nndsvdar", max_iter=6000, random_state=s)
    w = m.fit_transform(Xl)
    if best_nmf is None or m.reconstruction_err_ < best_nmf.reconstruction_err_:
        best_nmf, Wn = m, w
nmf = best_nmf
Hn = nmf.components_              # (n_phases, n_r) recovered components

# lift the pure phases the same way for a fair component comparison
Gl = G + shift
corr = np.array([[abs(np.corrcoef(Hn[i], Gl[j])[0, 1]) for j in range(n_phases)]
                 for i in range(n_phases)])
ri, cj = linear_sum_assignment(-corr)             # match NMF comp -> true phase
match_corr = corr[ri, cj]
order = np.argsort(cj)
print(f"\nNMF: recovered-component vs true-phase correlation (Hungarian-matched):")
for i in order:
    print(f"  NMF#{ri[i]} -> {labels[cj[i]]:16s} corr={match_corr[i]:.3f}")

# coefficient recovery: reorder NMF weights to phase order, rescale per column
Wn_ord = Wn[:, ri[np.argsort(cj)]]                # columns in true-phase order
# NMF has per-component scale freedom; align each column by least-squares scalar
scales = np.array([np.dot(Wn_ord[:, j], C[:, j]) / (np.dot(Wn_ord[:, j], Wn_ord[:, j]) + 1e-12)
                   for j in range(n_phases)])
Wn_scaled = Wn_ord * scales
coef_r = [float(np.corrcoef(Wn_scaled[:, j], C[:, j])[0, 1]) for j in range(n_phases)]
print(f"coefficient recovery corr per phase: min={min(coef_r):.3f} "
      f"mean={np.mean(coef_r):.3f}")

summary = dict(n_phases=n_phases, n_mixtures=N_MIX, alpha=ALPHA,
               pca_components_for_99pct=n_99,
               pca_explained_variance_ratio=[round(float(x), 5) for x in evr],
               nmf_component_corr={labels[cj[i]]: round(float(match_corr[i]), 4)
                                   for i in range(n_phases)},
               nmf_coefficient_corr=dict(zip(labels, [round(x, 4) for x in coef_r])))
(config.RESULTS / "pca_nmf_summary.json").write_text(json.dumps(summary, indent=2))
np.save(config.RESULTS / "nmf_components.npy", Hn)
np.save(config.RESULTS / "nmf_weights.npy", Wn)

# ============================ figures ============================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    # scree
    ax[0, 0].bar(np.arange(1, len(evr) + 1), evr, color="steelblue")
    ax[0, 0].axvline(n_phases - 1 + 0.5, color="r", ls="--", lw=1,
                     label=f"n_phases−1 = {n_phases-1}")
    ax[0, 0].set(title="PCA scree — variance per component", xlabel="component",
                 ylabel="explained variance ratio"); ax[0, 0].legend(fontsize=8)
    # cumulative
    ax[0, 1].plot(np.arange(1, len(cum) + 1), cum, "o-", ms=4)
    ax[0, 1].axhline(0.99, color="grey", lw=0.7)
    ax[0, 1].set(title="PCA cumulative variance", xlabel="components",
                 ylabel="cumulative EVR", ylim=(0, 1.02))
    # NMF component vs true (best match). NMF components carry an arbitrary
    # scale, so rescale to the true amplitude (LSQ scalar) for the overlay.
    mm = r <= 15
    best = order[int(np.argmax(match_corr[order]))]
    rec = Hn[ri[best]]
    true = Gl[cj[best]]
    sc = float(rec @ true) / float(rec @ rec + 1e-12)
    ax[1, 0].plot(r[mm], true[mm], lw=1.4, label=f"true {labels[cj[best]].split('_')[0]}")
    ax[1, 0].plot(r[mm], (rec * sc)[mm], lw=0.9, ls="--", color="k",
                  label=f"NMF (corr {match_corr[best]:.2f})")
    ax[1, 0].set(title="NMF component — best match", xlabel="r (Å)", ylabel="G+shift")
    ax[1, 0].legend(fontsize=8)
    # coefficient recovery scatter (all phases pooled)
    ax[1, 1].scatter(C.ravel(), Wn_scaled.ravel(), s=6, alpha=0.4)
    lim = [0, C.max() * 1.05]
    ax[1, 1].plot(lim, lim, "r-", lw=0.8)
    ax[1, 1].set(title=f"coefficient recovery (mean corr {np.mean(coef_r):.3f})",
                 xlabel="true cᵢ", ylabel="NMF weight (scaled)", xlim=lim, ylim=lim)
    fig.suptitle("§5 mixtures → PCA + NMF (no stretch)", fontsize=12)
    fig.tight_layout()
    out = config.RESULTS / "pca_nmf_overview.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")
except Exception as e:
    print(f"(skipped figures: {e})")
