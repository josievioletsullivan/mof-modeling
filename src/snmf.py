"""Stretched-NMF solver.

Model (per the stretched-NMF paper, Gu/Thatcher/Billinge): a set of M signals
is explained by K nonnegative components, each of which may appear at a
per-signal uniform x-axis stretch:

    X[:, m] ≈ Σ_k  W[k, m] · T(S[k, m]) · A[:, k]        A, W ≥ 0,  S > 0

where T(s) is the same normalized-integer-grid stretch as diffpy.snmf
(x(r) -> x(r/s), zero-filled). diffpy.snmf 0.1.3 ships the primitives for this
(stretch operator, weight QP, objective, residual) but NOT the alternating
solver loop -- `main()` is a stub with no component/stretch updates. This module
supplies that loop, reusing diffpy.snmf's stretch definition (validated identical
in tests) and adding the missing updates:

  * W : per signal, nonnegative least squares in the stretched basis (scipy nnls)
  * S : per (k, m), 1-D bounded minimization of the residual (scipy)
  * A : multiplicative NMF update using T(s) and its exact linear adjoint T(s)^T
        (a sparse interpolation matrix), which the multiplicative rule requires.

Data must be nonnegative (lift PDFs first; see mixtures.lift).
"""
import numpy as np
from scipy.optimize import nnls, minimize_scalar
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator


def _stretch_matrix(s, grid):
    """Sparse (n x n) linear operator T(s): row i samples the input at grid[i]/s
    by linear interpolation (zero past the domain), i.e. x(r) -> x(r/s) scaled
    about r=0 -- matching a physical lattice expansion. `grid` is the r-vector.
    T(s).T is its exact adjoint (needed by the multiplicative A-update).

    When `grid` is a uniform grid starting at 0 this reduces exactly to
    diffpy.snmf's index-based get_stretched_component."""
    grid = np.asarray(grid, float)
    n = len(grid)
    p = grid / s                                  # source r-position for each row
    j = np.clip(np.searchsorted(grid, p) - 1, 0, n - 2)   # left node index
    frac = (p - grid[j]) / (grid[j + 1] - grid[j])
    inside = (p >= grid[0]) & (p <= grid[-1])
    idx = np.arange(n)
    rows, cols, vals = [], [], []
    for node, w in ((j, 1.0 - frac), (j + 1, frac)):
        ok = inside & (w != 0)
        rows.append(idx[ok]); cols.append(node[ok]); vals.append(w[ok])
    return sparse.csr_matrix((np.concatenate(vals),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, n))


def _reconstruct(A, W, S, mats):
    n, K = A.shape
    M = W.shape[1]
    Xhat = np.zeros((n, M))
    for m in range(M):
        for k in range(K):
            Xhat[:, m] += W[k, m] * (mats[(k, m)] @ A[:, k])
    return Xhat


def objective(A, W, S, X, mats):
    R = _reconstruct(A, W, S, mats) - X
    return 0.5 * float(np.sum(R * R))


def stretched_nmf(X, n_components, grid=None, n_iter=120, s_bounds=(0.8, 1.6),
                  tol=1e-6, seed=0, n_restarts=1, smooth=0.5, A_init=None,
                  nonneg=True, verbose=False):
    """Factorize X (n_r x M, nonnegative) into A (n_r x K), W (K x M), S (K x M).

    `grid` is the r-vector the rows live on; the stretch is applied about r=0 so
    it matches a physical lattice expansion (essential when r does not start at
    0). If None, the index grid is used (equivalent for r starting at 0).

    `smooth` weights a second-difference penalty on each component's stretch
    trajectory across samples (the diffpy.snmf objective's smoothness term),
    which suppresses noisy per-sample stretch estimates -- assumes samples are
    ordered (e.g. a T/P series). Set 0 to disable.

    SNMF is non-convex; with n_restarts>1 the run with the lowest final objective
    is returned. Returns dict with A, W, S, obj (history), n_iter_run.
    """
    if n_restarts > 1:
        best = None
        for r in range(n_restarts):
            out = stretched_nmf(X, n_components, grid, n_iter, s_bounds, tol,
                                seed=seed + r, n_restarts=1, smooth=smooth,
                                A_init=A_init, nonneg=nonneg, verbose=verbose)
            if best is None or out["obj"][-1] < best["obj"][-1]:
                best = out
        return best

    X = np.asarray(X, float)
    n, M = X.shape
    K = n_components
    grid = np.arange(n, dtype=float) if grid is None else np.asarray(grid, float)
    rng = np.random.default_rng(seed)

    # init: from a provided basis (e.g. plain NMF components) + small jitter so
    # restarts differ, else seeded from actual data columns (standard NMF init).
    if A_init is not None:
        A = np.asarray(A_init, float) * (1.0 + 0.02 * rng.standard_normal((n, K)))
    else:
        cols = rng.choice(M, size=K, replace=(M < K))
        if nonneg:
            A = X[:, cols] + 0.01 * X.mean() * rng.random((n, K))
        else:
            A = X[:, cols] + 0.01 * np.abs(X).mean() * rng.standard_normal((n, K))
    if nonneg:
        A = np.clip(A, 1e-6, None)
    S = np.ones((K, M))
    W = np.full((K, M), 1.0 / K)
    mats = {(k, m): _stretch_matrix(S[k, m], grid) for k in range(K) for m in range(M)}

    obj_hist = [objective(A, W, S, X, mats)]
    for it in range(n_iter):
        # --- W: nonnegative least squares per signal in the stretched basis ---
        for m in range(M):
            B = np.column_stack([mats[(k, m)] @ A[:, k] for k in range(K)])
            W[:, m], _ = nnls(B, X[:, m])

        # --- S: 1-D bounded minimize residual per (k, m), + neighbor smoothness ---
        # smoothness scale ~ data curvature, so the penalty is commensurate with
        # the residual term regardless of overall intensity
        lam = smooth * float(np.mean(X ** 2)) * n
        for m in range(M):
            base = np.column_stack([mats[(k, m)] @ A[:, k] for k in range(K)])
            recon = base @ W[:, m]
            for k in range(K):
                partial = recon - W[k, m] * base[:, k]      # residual w/o comp k
                target = X[:, m] - partial
                # reference = mean of current neighbors along the sample axis
                nb = [S[k, j] for j in (m - 1, m + 1) if 0 <= j < M]
                s_ref = float(np.mean(nb)) if nb else None

                def resid(s, k=k, m=m, target=target, s_ref=s_ref):
                    v = _stretch_matrix(s, grid) @ A[:, k]
                    d = W[k, m] * v - target
                    val = float(d @ d)
                    if s_ref is not None:
                        val += lam * (s - s_ref) ** 2
                    return val

                r = minimize_scalar(resid, bounds=s_bounds, method="bounded")
                S[k, m] = r.x
                mats[(k, m)] = _stretch_matrix(S[k, m], grid)

        if nonneg:
            # --- A: multiplicative NMF update with linear stretch operators ---
            num = np.zeros((n, K))
            den = np.zeros((n, K))
            Xhat = _reconstruct(A, W, S, mats)
            for m in range(M):
                for k in range(K):
                    Tt = mats[(k, m)].T
                    num[:, k] += W[k, m] * (Tt @ X[:, m])
                    den[:, k] += W[k, m] * (Tt @ Xhat[:, m])
            A *= num / (den + 1e-12)
            A = np.clip(A, 0, None)
        else:
            # --- A: block-coordinate least squares (signed components; correct
            # for PDFs, which are not nonnegative). Solve the exact normal
            # equations per component with CG -- monotonically decreases the fit
            # and keeps the true solution a fixed point. ---
            for k in range(K):
                Tk = [mats[(k, m)] for m in range(M)]
                wk = W[k, :]
                resid_k = X - _reconstruct(A, W, S, mats) \
                    + np.column_stack([wk[m] * (Tk[m] @ A[:, k]) for m in range(M)])

                def matvec(a, Tk=Tk, wk=wk):
                    out = np.zeros(n)
                    for m in range(M):
                        out += wk[m] ** 2 * (Tk[m].T @ (Tk[m] @ a))
                    return out

                b = np.zeros(n)
                for m in range(M):
                    b += wk[m] * (Tk[m].T @ resid_k[:, m])
                A[:, k], _ = cg(LinearOperator((n, n), matvec=matvec), b,
                                x0=A[:, k], rtol=1e-4, maxiter=60)

        # normalize components (fix scale ambiguity), absorb into W
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1.0
        A /= norms
        W *= norms[:, None]

        obj_hist.append(objective(A, W, S, X, mats))
        if verbose and (it % 10 == 0 or it == n_iter - 1):
            print(f"  iter {it:3d}  obj={obj_hist[-1]:.4e}")
        if abs(obj_hist[-2] - obj_hist[-1]) <= tol * obj_hist[0]:
            break

    return dict(A=A, W=W, S=S, obj=np.array(obj_hist), n_iter_run=it + 1)
