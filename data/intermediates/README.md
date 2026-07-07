# §3 — Initial / final / intermediate states

Generate a synthetic **intermediate** structure by linearly morphing an
`initial` endpoint toward a `final` one, then compute its PDF the *same way as
§2* so all three curves are directly comparable.

Run: `python pipeline/03_interpolate.py` (env `mof`).
Code: [`src/interpolate.py`](../../src/interpolate.py) (the morph) +
[`src/pdf.py`](../../src/pdf.py) `pdf_from_structure` (shared §2/§3 PDF core).

## The algorithm

```
intermediate = t · initial + (1 − t) · final        (t = weight on initial)
```

- **Lattice**: the six cell parameters (a, b, c, α, β, γ) blend linearly.
- **Fractional coords**: index-mapped, blended along the **minimum-image**
  shortest path (0.02 and 0.98 are 0.04 apart, not 0.96).
- **Chemistry**: the intermediate keeps the **initial's** per-atom species, so
  the morph is purely geometric and always yields one well-defined structure —
  even across compositions (Si→NaCl stays pure Si with rocksalt-like geometry).
- **t convention**: `t=1`→initial, `t=0`→final, `t=0.5`→midpoint. Chosen to
  match the sanity formula `t·G_i + (1−t)·G_f` weight-for-weight. Default `t=0.5`.

**Requirement:** equal atom counts (needed for a 1:1 atom correspondence). Pairs
are enumerated within atom-count groups, so **all 14 valid permutations** run:
4-atom FCC {Al, Cu, Ni}, 8-atom {MgO, NaCl, Si}, 2-atom {CsCl, Fe}. CaF₂ (12)
and ZnS (56) have no partner and are skipped (reported, not silent).

## Outputs (this folder)

| file | contents |
|------|----------|
| `<init>__<final>__t50.cif` | the morphed structure (t=0.5) |
| `<init>__<final>__t50.gr`  | its G(r) on `config.R` (r, G columns) |
| `G_inter.npy` | stacked (14, 3001) intermediates for the decomposition stage |
| `r.npy` | the shared grid (== `config.R`) |
| `sanity.json` | per-pair diagnostics (below) |
| `intermediates_overview.png` | initial / final / intermediate / blend, per pair |

## Sanity check — and a correction to the naive version

All curves share `config.R` by construction, so "same grid/normalization" is a
**hard invariant**, verified two ways:

1. **Grid**: every intermediate `.gr` is asserted equal to `config.R` (3001 pts).
2. **Scale**: shift-robust amplitude ratio `rms(G_inter)/√(rms_i·rms_f)` sits in
   `[0.25, 4]` for all 14. **This is the real bug-detector** — it is blind to
   where peaks land, so it catches a grid/scale error without being fooled by
   physics. All 14 pass.

The tempting check — `corr(G_inter, t·G_i+(1−t)·G_f)` — is reported but only
**informational**, because a low value is usually *physics, not a bug*:

- PDF is **not linear** in atomic position. When two endpoints' lattices differ
  by more than a peak width, the summed blend keeps **both** peak sets while the
  intermediate has a **single** peak between them → they anti-correlate *by
  construction*.
- So correlation is high **only** when endpoints are already similar:
  **Cu↔Ni (Δa≈3%) → corr ≈ +0.92** (intermediate lands right on the blend), the
  textbook "resembles" case. Al↔Cu (Δa≈12%) goes slightly negative. This drop is
  expected, not a defect.

### Two morph regimes (see `sanity.json` → `morph`)

- **lattice** (`coord_rmsd ≈ 0`): endpoints share fractional coords, so the morph
  is pure isotropic lattice scaling. FCC, rocksalt, CsCl/Fe. Scale ≈ 1.0.
- **reconstructive** (`coord_rmsd` large): coordination itself reorganizes
  (diamond Si ↔ rocksalt). Lower amplitude (~0.57×) but still on-grid, in-band —
  no bug, just a bigger structural change.

**Takeaway:** confirm grid + amplitude (shift-robust) as the invariants; treat
blend-correlation as a *physics readout* (tracks lattice mismatch), never as a
standalone bug flag. The original "large divergence ⇒ grid/scale bug" heuristic
only holds for small morphs — Cu↔Ni is the pair that demonstrates it cleanly.
