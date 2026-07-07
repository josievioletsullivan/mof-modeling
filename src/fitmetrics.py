"""Fit- and match-quality metrics (§7).

Two layers:

1. Pairwise quality of an obs vs calc profile -- weighted profile residual Rw,
   chi-square, and Pearson correlation.
2. Set-vs-set matching -- build the NxN similarity matrix between two sets of
   profiles and solve the optimal assignment (Hungarian / linear_sum_assignment)
   instead of scoring all N! permutations.  The product (or mean) of the matched
   correlations is a single-number score for "did the method recover the
   components".

Scale/offset gauge
------------------
Blind decompositions (NMF/SNMF) return components with an arbitrary per-component
scale, and our NMF pipeline also lifts data by a global constant, so a recovered
component may differ from the truth by ``a*calc + b``.  Pearson r is invariant to
that affine map, so matching is unaffected.  For Rw and chi-square -- which are
NOT invariant -- ``calc`` is first affine-aligned to ``obs`` by least squares
(remove exactly the gauge freedom, nothing more).  Feed the RAW (un-lifted) true
profile as ``obs`` so its baseline sits near zero and Sum(w*obs^2) is physical.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- #
# pairwise metrics
# --------------------------------------------------------------------------- #
def pearson(a, b):
    """Pearson correlation coefficient (scale/offset invariant)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def align_affine(obs, calc, w=None):
    """Least-squares (a, b) minimizing Sum w*(obs - (a*calc + b))^2."""
    obs = np.asarray(obs, float)
    calc = np.asarray(calc, float)
    w = np.ones_like(obs) if w is None else np.asarray(w, float)
    Sw = w.sum()
    Sc = np.sum(w * calc)
    Scc = np.sum(w * calc * calc)
    So = np.sum(w * obs)
    Sco = np.sum(w * calc * obs)
    A = np.array([[Scc, Sc], [Sc, Sw]])
    rhs = np.array([Sco, So])
    a, b = np.linalg.solve(A, rhs)
    return float(a), float(b)


def rw(obs, calc, w=None, align=True):
    """Weighted profile residual  Rw = sqrt( Sum w (obs-calc)^2 / Sum w obs^2 )."""
    obs = np.asarray(obs, float)
    calc = np.asarray(calc, float)
    w = np.ones_like(obs) if w is None else np.asarray(w, float)
    if align:
        a, b = align_affine(obs, calc, w)
        calc = a * calc + b
    num = np.sum(w * (obs - calc) ** 2)
    den = np.sum(w * obs ** 2)
    return float(np.sqrt(num / den))


def chi_square(obs, calc, var=None, align=True, n_params=2):
    """Chi-square and reduced chi-square.

    ``var`` is the per-point variance sigma_i^2; the weights are w_i = 1/var_i,
    matching the Rw convention.  These are noise-free computed PDFs, so there are
    no measured uncertainties -- with ``var=None`` a uniform sigma^2 = 1 is used,
    which makes chi-square the (aligned) sum of squared residuals.  Pass a real
    variance array for measured data.  Reduced chi-square divides by
    (N - n_params); with affine alignment n_params defaults to 2.
    """
    obs = np.asarray(obs, float)
    calc = np.asarray(calc, float)
    var = np.ones_like(obs) if var is None else np.asarray(var, float)
    w = 1.0 / var
    if align:
        a, b = align_affine(obs, calc, w)
        calc = a * calc + b
    chi2 = float(np.sum((obs - calc) ** 2 / var))
    dof = max(1, obs.size - n_params)
    return chi2, chi2 / dof


# --------------------------------------------------------------------------- #
# set-vs-set matching
# --------------------------------------------------------------------------- #
def similarity_matrix(A, B, metric="pearson", w=None):
    """N_A x N_B matrix of pairwise similarity between rows of A and rows of B.

    metric='pearson' -> |Pearson r| (in [0,1]); metric='one_minus_rw' -> 1 - Rw
    (higher = better, may go negative for a poor pair).
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    M = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if metric == "pearson":
                M[i, j] = abs(pearson(a, b))
            elif metric == "one_minus_rw":
                M[i, j] = 1.0 - rw(a, b, w=w)
            else:
                raise ValueError(metric)
    return M


def match(A, B, w=None, var=None):
    """Optimal one-to-one assignment of rows of A (obs) to rows of B (calc).

    Cost = 1 - |Pearson r| solved with linear_sum_assignment (Hungarian).
    Returns a dict with the pairing and, for each matched pair, the signed
    Pearson r, Rw and reduced chi-square, plus single-number read-outs:
    product and mean of the matched correlations, and the mean Rw.
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    sim = similarity_matrix(A, B, metric="pearson")
    row, col = linear_sum_assignment(1.0 - sim)

    pairs = []
    for i, j in zip(row, col):
        r = pearson(A[i], B[j])
        pr = dict(a_index=int(i), b_index=int(j),
                  pearson=r,
                  rw=rw(A[i], B[j], w=w),
                  chi2_reduced=chi_square(A[i], B[j], var=var)[1])
        pairs.append(pr)

    matched_r = np.array([abs(p["pearson"]) for p in pairs])
    matched_rw = np.array([p["rw"] for p in pairs])
    return dict(
        similarity=sim,
        row=row, col=col,
        pairs=pairs,
        product_r=float(np.prod(matched_r)),
        mean_r=float(matched_r.mean()),
        min_r=float(matched_r.min()),
        mean_rw=float(matched_rw.mean()),
        max_rw=float(matched_rw.max()),
    )
