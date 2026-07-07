"""§5 — build linear combinations of PDFs (with known coefficients) and, for
the stretched-NMF demo, mixtures that also carry a per-sample uniform x-axis
stretch (mimicking lattice expansion between states).

Two mixing models:
  plain     : X[m] = Σ_k C[m,k] · G_k
  stretched : X[m] = Σ_k C[m,k] · stretch(G_k, S[m,k])

`stretch` matches diffpy.snmf exactly: the signal is re-interpolated on a
normalized integer grid, x(r) -> x(r/s), with zero fill past the domain
(diffpy.snmf.subroutines.get_stretched_component). s>1 dilates (peaks move out,
= lattice expansion); s<1 contracts.
"""
import numpy as np


def stretch(signal, s):
    """x(r) -> x(r/s) on the normalized integer grid; zero-filled past the edge.
    Identical to diffpy.snmf's get_stretched_component (validated in tests)."""
    signal = np.asarray(signal, float)
    n = len(signal)
    idx = np.arange(n)
    return np.interp(idx / s, idx, signal, left=0.0, right=0.0)


def dirichlet_coeffs(n_samples, n_components, alpha=0.6, rng=None):
    """Nonnegative mixing coefficients that sum to 1 per sample (a simplex).
    Lower alpha -> sparser (more single-phase-dominant) mixtures."""
    rng = np.random.default_rng(rng)
    return rng.dirichlet(np.full(n_components, alpha), size=n_samples)


def linear_mixtures(components, C):
    """Plain linear combinations: X = C @ components. Rows = samples."""
    return np.asarray(C) @ np.asarray(components)


def stretch_factors(n_samples, n_components, max_expansion=0.25, rng=None,
                    smooth=True):
    """Per-sample, per-component stretch factors S[m,k].

    If smooth, each component's stretch ramps monotonically across the sample
    index (a temperature/pressure series: component k expands from 1 to
    1+max_expansion*f_k). Otherwise random. Always >= 1 (expansion)."""
    rng = np.random.default_rng(rng)
    if smooth:
        t = np.linspace(0.0, 1.0, n_samples)[:, None]        # (n_samples, 1)
        amp = rng.uniform(0.4, 1.0, size=(1, n_components)) * max_expansion
        return 1.0 + t * amp                                  # (n_samples, n_comp)
    return 1.0 + rng.uniform(0, max_expansion, size=(n_samples, n_components))


def stretched_mixtures(components, C, S):
    """X[m] = Σ_k C[m,k] · stretch(components[k], S[m,k]). Rows = samples."""
    components = np.asarray(components, float)
    C = np.asarray(C, float)
    S = np.asarray(S, float)
    n_samples, n_components = C.shape
    n_r = components.shape[1]
    X = np.zeros((n_samples, n_r))
    for m in range(n_samples):
        for k in range(n_components):
            X[m] += C[m, k] * stretch(components[k], S[m, k])
    return X


def lift(X, factor=1.0):
    """Shift data up so it is nonnegative (required for NMF/SNMF).
    lifted = X + |min(X)| * factor. PDFs oscillate about 0, so this is needed;
    matches diffpy.snmf.subroutines.lift_data."""
    X = np.asarray(X, float)
    return X + np.abs(np.min(X)) * factor
