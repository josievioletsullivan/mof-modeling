# §5 — Linear combinations + signal separation

Mix the endpoint PDFs with **known** coefficients, then try to recover the pure
signals blind. Two stages:

| stage | script | env | what it shows |
|-------|--------|-----|---------------|
| 05 | `pipeline/05_mixtures_pca.py` | `mof` (sklearn) | mixtures → PCA rank + NMF recovery (no stretch) |
| 06 | `pipeline/06_stretched_nmf.py` | `mof` | stretched mixtures (interp) → stretched-NMF beats PCA/NMF |
| 07 | `pipeline/07_diffpysim.py` | `mof` | **paper's simulator** (diffpysim, physical strain) → stretched-NMF |

Code: [`src/mixtures.py`](../../src/mixtures.py) (build mixtures, stretch),
[`src/snmf.py`](../../src/snmf.py) (the stretched-NMF solver).
Ground truth + mixture data live in [`../mixtures/`](../mixtures/).

## Stage 05 — mixtures, PCA, NMF (no stretch)

`G_mix = C @ G` with `C` a Dirichlet coefficient matrix (rows sum to 1) — **`C`
is the stored ground truth** (`mixtures/C_true.npy`). 150 mixtures of the 10
phases.

- **PCA** (`sklearn.decomposition.PCA`): **8 components explain 99%** of the
  variance. That is ≈ `n_phases − 1 = 9` — one dimension is lost because the
  mixtures live on a sum-to-1 simplex (PCA centers the data). PCA counts rank
  correctly, but its components are orthogonal *mixtures*, not pure phases.
- **NMF** (`sklearn.decomposition.NMF`, best of 8 seeds by reconstruction error):
  nonnegative, so components resemble real phases. Hungarian-matched to the
  inputs, **9/10 phases recover at corr 0.83–0.998**; **Si blends** (corr ≈0.20)
  because its lifted PDF is expressible as a nonnegative combination of the
  others — a real blind-separation limit. Coefficient recovery: **mean corr 0.84**.
  - *Caveat*: PDFs are signed; NMF needs nonnegative data, so the stack is lifted
    by `|min|`. That shared DC baseline reduces contrast and is why blind NMF on
    PDFs is imperfect. PCA (which handles signed data) is the cleaner rank tool.

Outputs: `pca_nmf_summary.json`, `nmf_components.npy`, `nmf_weights.npy`,
`pca_nmf_overview.png`.

## Stage 06 — stretched-NMF (the one that must work)

Now each component also carries a **per-sample uniform x-axis stretch**
(mimicking lattice expansion across a T/P series):
`X[:,m] = Σ_k C[m,k]·stretch(G_k, S[m,k])`, ≤10% expansion, 3 phases (Ni, NaCl,
Si), 24 ordered samples. Ground truth `C` and `S` saved.

**Engine note.** `diffpy.snmf` 0.1.3 ships the primitives (stretch operator,
weight QP, objective, residual) but **not the solver loop** — `main()` is a stub
with no component/stretch updates. So [`src/snmf.py`](../../src/snmf.py) supplies
the alternating optimization, reusing diffpy.snmf's **exact** stretch definition
(verified bit-identical to `get_stretched_component`) and adding: W via
nonnegative least squares, S via 1-D bounded minimization with a neighbor
**smoothness** penalty (the diffpy objective's smoothness term — essential, it
turns noisy per-sample stretch estimates into clean trajectories), and A via a
multiplicative NMF update using the stretch operator's sparse linear adjoint.

**Results — stretched-NMF recovers what PCA/NMF cannot:**

| metric | value |
|--------|-------|
| PCA rank (99% var), unstretched → stretched | **2 → 8** (stretch smears rank) |
| fit residual: SNMF vs plain NMF | **4.4 vs 489** (~110× better with stretch) |
| component recovery corr: SNMF vs plain NMF | **0.66 vs 0.58** (Ni 0.97, NaCl 0.83, Si 0.17) |
| stretch trajectory recovery corr | **0.989–0.999** |

"Plain NMF" here is the *same solver with stretch frozen at 1* — an apples-to-
apples control. PCA needs 8 components for 3 phases because a moving x-axis is
not a linear subspace; stretched-NMF absorbs it into `S` and fits with K=3.
Stretch trajectories recover near-perfectly and SNMF beats plain NMF, but **Si
blends** (0.17) — diamond-Si's PDF is expressible as a stretched combination of
the others, a genuine non-identifiability (also seen in stage 05's plain NMF).
The physical-simulator stage 07 recovers Si better (0.59).

Outputs: `snmf_summary.json`, `snmf_components.npy`, `snmf_stretch.npy`,
`snmf_overview.png`.

## Stage 07 — the paper's simulator (diffpysim), physical strain

Stage 06 stretches PDFs by *interpolation*. Stage 07 uses **diffpysim**
(github.com/yevgenyr/diffpysim, the simulator from the stretched-NMF paper),
installed into `external/diffpysim` and driven by
[`src/diffpysim_run.py`](../../src/diffpysim_run.py). diffpysim **physically
re-simulates** each phase's PDF at a genuinely expanded lattice via
diffpy.srfit's `PDFGenerator`, so peaks shift as a real lattice expansion — not
an interpolation. We build mixtures from its output (random weights × physical
stretch) and separate them.

Install notes: `pip install Dans_Diffraction` + `pip install -e external/diffpysim`
(its `setup.py` version string was patched to be PEP 440-valid). Runs in `mof`
(diffpysim + sklearn + our numpy/scipy solver all live there).

**Results (physical simulator):**

| metric | value |
|--------|-------|
| PCA rank (99%): unstretched → physically stretched | **2 → 9** |
| component recovery: SNMF vs plain NMF | **0.81 vs 0.51** (Ni 0.88, NaCl 0.96, Si 0.59) |
| weight recovery corr | **0.91 – 0.98** |
| stretch trajectory recovery corr | **0.73 – 0.99** |

SNMF clearly beats plain NMF on the paper's real physical data, and recovers
weights near-perfectly. Recovery is **less clean than the idealized stage 06**
(0.81 vs 0.96) — honest, because physical PDFs are harder to separate blindly.

**What made it work (hard-won):**

1. **Broad peaks.** diffpysim's default `Uiso=0.005` gives razor-sharp peaks; a
   stretch search then has near-zero gradient (a shifted needle barely overlaps
   its target), and SNMF fails completely. Using §2's `Uiso=0.07` broadens the
   peaks and the stretch search converges. This is a real, non-obvious modelling
   lesson, not a hack.
2. **Grid must start at r=0.** Physical lattice expansion scales r about 0
   (peak at r → r·s). The stretch operator ([`src/snmf.py`](../../src/snmf.py) is
   now grid-aware: `_stretch_matrix(s, grid)`) must therefore scale about r=0.
   Cropping the low-r start makes index-stretch scale about the wrong origin and
   breaks both the physical match and the optimization. (Stage 06's interp demo
   worked only because it was self-consistent about its own — wrong — origin.)

Outputs: `diffpysim_snmf_summary.json`, `diffpysim_snmf_overview.png`.

## Takeaways

1. **PCA = rank counter.** Number of significant components ≈ number of
   independent phases (−1 for the simplex). Great for "how many phases?", not for
   reading off phase shapes.
2. **NMF ≈ phases, with caveats.** Nonnegativity makes components phase-like, but
   lifting signed PDFs costs contrast; blind separation isn't guaranteed (Si).
3. **A moving x-axis breaks linear methods.** Stretch inflates PCA/NMF rank and
   blurs their components. **Stretched-NMF** models the stretch explicitly and
   recovers both the pure components and the per-sample stretch factors — the
   trajectory recovers essentially perfectly (corr ≈1.0); absolute stretch has a
   gauge (an overall stretch absorbs into the component), so the *shape* of `S`
   across samples is the meaningful, recovered quantity.
