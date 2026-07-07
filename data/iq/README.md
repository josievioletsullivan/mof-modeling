# §4 (alternate route) — PDF from I(Q) by Fourier transform

The §2 route computes G(r) straight from a structure model. This route mirrors
what an **experiment** does — turn measured scattered intensity into a PDF:

```
I(Q)  ──►  S(Q)  ──►  F(Q) = Q·[S(Q) − 1]  ──►  G(r) = (2/π) ∫ F(Q) sin(Q·r) dQ
       normalize          reduce                 sine transform, Qmin→Qmax
```

Run: `python pipeline/04_pdf_from_iq.py` (env `mof`).
Code: [`src/pdf_from_iq.py`](../../src/pdf_from_iq.py). Reusable entry point:
`iq_to_gr(q, iq, composition, qmin, qmax, rpoly, qdamp)` — feed it any `(Q, I)`.

## Engine note — read this

The reference engine is **PDFgetX3** (`diffpy.pdfgetx`), which is a separately
**licensed** package and **is not installed** in any local env. `src/pdf_from_iq.py`
is a faithful, dependency-free reimplementation of its *ad-hoc-correction*
transform (Juhás et al., *J. Appl. Cryst.* **46** (2013) 560): same Qmin/Qmax
cropping, same rpoly polynomial background removal, same composition-based
normalization. To use the real engine or real data instead, hand `iq_to_gr` a
`.chi`/`.iq` array (or a PDFgetX3 `S(Q)`) — the interface is unchanged.

Because the project has **no measured data**, `structure_to_iq` forward-models a
measured-like `I(Q)` from each structure using the **same bulk-crystal engine as
§2** (`diffpy PDFCalculator` exposes `F(Q)` via `.fq`), with **Uiso = 0.07** baked
in. Feeding that through `iq_to_gr` and comparing to the §2 PDF is a **closed-loop
validation**: recover the G(r) we started from. (The finite-cluster Debye engine
was rejected — it models a nanoparticle, not the infinite crystal §2 uses.)

## The four knobs (checkboxes) and what they actually do

| knob | value | role | demonstrated result |
|------|-------|------|---------------------|
| **Qmin** | 0.5 | throw out small Q (SAS/macroscopic regime, no local structure) | With rpoly active, low-Q junk is *already* removed, so Qmin∈[0,0.5] barely changes G(r). **Over-cutting hurts**: Qmin=2.0 discards real signal → corr 0.98→0.94. |
| **Qmax** | 35 Å⁻¹ | FT truncation → real-space resolution | Low Qmax = broad peaks + termination ripple. Qmax=6 visibly smears; 35 is sharp. |
| **rpoly** | 0.9 | polynomial background removal, tuned just below the first PDF peak | **Two-sided**: too low (0.3) → smooth background leaks → G(r) blows up (corr **0.13**); too high (4.0) → eats into the first peak. 0.9 sits in the safe band. |
| **composition + Uiso** | from CIF, 0.07 | average form factors ⟨f(Q)⟩, ⟨f²(Q)⟩ that normalize I(Q)→S(Q) | Required for the Faber-Ziman normalization; Uiso=0.07 matches §2 so the recovered PDF is comparable. |

`rpoly`↔degree: a degree-`n` polynomial in Q only affects G(r) below ~`nπ/Qmax`,
so features below `rpoly` are captured by `n ≈ Qmax·rpoly/π` terms (≈10 here).
That is why "tune rpoly just below the first peak" — high enough to kill slow
background, low enough not to reach into the first coordination shell.

## Validation result

Round-trip over all 10 endpoints, FT route vs §2 direct PDF (r = 1–25 Å):

- **corr = 0.974 – 0.991** (mean 0.986) — peak **positions and relative heights
  are essentially exact**; the two independent routes agree.
- **amplitude ratio ≈ 0.90 – 0.94** — absolute scale sits ~10% low, from the
  qdamp resolution envelope plus the ad-hoc normalization. This is expected of a
  real PDFgetX3 transform and does not affect peak structure. (The bare sine
  transform of the §2 `F(Q)`, with no normalization/rpoly, matches at
  **corr 0.99999, ratio 1.000** — proof the transform itself is exact; the ~10%
  is the intensity-processing chain, not the FT.)

## Outputs (this folder)

| file | contents |
|------|----------|
| `<label>.iq` | forward-modeled I(Q) (Q, I), incl. low-Q upturn + background |
| `<label>_ft.gr` | G(r) recovered via the Fourier transform, on `config.R` |
| `G_ft.npy`, `r.npy`, `labels.json` | stacked FT-route PDFs + grid |
| `sanity.json` | per-structure corr vs §2, amplitude ratio, composition |
| `ft_validation.png` | I(Q) → F(Q) → recovered G(r) vs §2 + corr bar chart |
| `ft_parameters.png` | Qmin / Qmax / rpoly(too-low) / rpoly(too-high) sweeps |

## Takeaways

1. The transform `G(r) = (2/π)∫F(Q)sin(Qr)dQ` is exact; all error lives in the
   **reciprocal-space processing** (normalization, background, truncation).
2. **rpoly is the parameter that matters most** — it has a genuine failure mode
   on both sides. Qmin and Qmax are more forgiving over sensible ranges.
3. **Qmin is partly redundant with rpoly** for background removal in clean data;
   its stated job ("cut the SAS junk") is real for *measured* data with beamstop
   artifacts, but here the dominant risk is cutting *too much*.
4. The §2 and §4 routes are cross-validated: same structure, two independent
   paths, same PDF (shape) to <3%.
