#!/usr/bin/env python3
"""
kde_density.py  (src/)

Kernel Density Estimation (KDE) anomaly scoring in the embedding space. Two uses:

  single-sample (default): fit a KDE on all points, score each point by its negative
    log-density -- points in low-density (rare) regions get high anomaly scores. This is
    an independent, A/B-agnostic detector, usable as a filter alongside LOF / Isolation
    Forest for non-circular validation.

  two-sample (--reference B): fit one KDE on A and one on B and score each A-point by the
    log density-ratio log p_A(x) - log p_B(x). High = dense in A but rare in B -- the same
    question the KL method asks, via KDE instead of k-NN.

IMPORTANT: KDE degrades badly in high dimensions (the embedding is ~768-D). Always run on
PCA-projected embeddings (e.g. --pca-dim 16..64), never on the raw 768-D vectors, or the
density becomes near-uniform and the scores are meaningless. The functions here take
already-projected X; the CLI projects for you.

Bandwidth: the single most important parameter. Chosen by Scott's / Silverman's rule by
default (fast, reasonable), or by cross-validation with --bandwidth cv (slower, better).

As a library:
    from kde_density import kde_scores, kde_ratio_scores
    s = kde_scores(XA)                       # neg log-density, higher = more anomalous
    r = kde_ratio_scores(XA, XB)             # log p_A - p_B, higher = specific to A

CLI (standalone, on a saved embeddings .npz with an 'emb' or 'XA' array):
    python src/kde_density.py --emb results/.../embeddings_proj.npz --pca-dim 32
"""
import argparse

import numpy as np


def rips_death_diameters(X):
    """The finite death diameters of 0-dim Rips persistence = the edge weights of the
    Euclidean minimum spanning tree (0-dim persistent homology is single-linkage
    clustering). Returns them sorted ascending (n-1 values for n points)."""
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import minimum_spanning_tree
    n = len(X)
    if n < 3:
        return np.array([0.0])
    D = squareform(pdist(X))
    mst = minimum_spanning_tree(D).toarray()
    w = mst[mst > 0]
    return np.sort(w)


def rips_bandwidth(X, gamma=0.98):
    """lookout v2 bandwidth: the gamma-quantile of the Rips death diameters. Proven to give
    a consistent KDE under mild conditions, and robust to outliers (an upper quantile, not
    the largest gap)."""
    d = rips_death_diameters(X)
    bw = float(np.quantile(d, gamma))
    return bw if bw > 0 else float(np.quantile(d[d > 0], gamma)) if (d > 0).any() else 1e-6


def _bandwidth(X, rule="scott"):
    """Scott's or Silverman's rule-of-thumb bandwidth for a d-dim sample of size n."""
    n, d = X.shape
    std = np.std(X, axis=0).mean() + 1e-12
    if rule == "silverman":
        factor = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))
    else:  # scott
        factor = n ** (-1.0 / (d + 4))
    return float(std * factor)


def robust_standardize(X):
    """Robust multivariate standardization (lookout v2): center on the median and whiten
    with a robust covariance estimate, so the data has ~zero correlations and unit scale.
    Falls back to median/MAD if the robust covariance is singular. More resistant to the
    very outliers we are hunting than mean/std standardization."""
    import warnings
    import numpy.linalg as la
    X = np.asarray(X, dtype=np.float64)
    med = np.median(X, axis=0)
    Xc = X - med
    n, d = Xc.shape
    # MinCovDet is unstable when d is large relative to n (determinant may increase,
    # near-singular covariance). Use a high support fraction and a diagonal ridge, and
    # fall back to the robust median/MAD whitening if it still fails.
    try:
        from sklearn.covariance import MinCovDet
        # support_fraction >= (n+d+1)/(2n); use max(0.9, that) so the estimate is stable
        sf = max(0.9, min(0.99, (n + d + 1) / (2 * n)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")               # silence the "determinant increased" note
            cov = MinCovDet(random_state=0, support_fraction=sf).fit(Xc).covariance_
        ridge = 1e-6 * np.trace(cov) / max(d, 1)          # scale-aware ridge
        U = la.cholesky(la.inv(cov + ridge * np.eye(d))).T
        Z = Xc @ U.T
        if np.all(np.isfinite(Z)):
            return Z
    except Exception:
        pass
    # fallback: per-dimension robust scaling (median / MAD) -- always stable
    mad = np.median(np.abs(Xc), axis=0) * 1.4826 + 1e-12
    return Xc / mad


def _fit_kde(X, bandwidth="scott", cv_grid=None, rips_gamma=0.98):
    """Fit sklearn KernelDensity. bandwidth: 'scott'|'silverman'|'cv'|'rips'|float."""
    from sklearn.neighbors import KernelDensity
    if bandwidth == "rips":
        bw = rips_bandwidth(X, rips_gamma)
        return KernelDensity(kernel="gaussian", bandwidth=bw).fit(X), bw
    if bandwidth == "cv":
        from sklearn.model_selection import GridSearchCV
        base = _bandwidth(X, "scott")
        grid = cv_grid if cv_grid is not None else np.logspace(np.log10(base / 5),
                                                               np.log10(base * 5), 12)
        gs = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": grid}, cv=3)
        gs.fit(X)
        return gs.best_estimator_, float(gs.best_params_["bandwidth"])
    bw = _bandwidth(X, bandwidth) if isinstance(bandwidth, str) else float(bandwidth)
    return KernelDensity(kernel="gaussian", bandwidth=bw).fit(X), bw


def kde_scores(X, bandwidth="cv", leave_one_out=False, rips_gamma=0.98):
    """Single-sample anomaly score = negative log-density. Higher = rarer = more anomalous.
    bandwidth defaults to 'cv' (cross-validated) -- rule-of-thumb bandwidths (scott/
    silverman) tend to over-smooth and miss grouped outliers. With leave_one_out=True each
    point is scored against the density of the OTHER points, so tight outlier groups cannot
    hide inside their own density (slower, O(n^2); worth it for small/medium n).
    Returns (scores, bandwidth)."""
    X = np.asarray(X, dtype=np.float64)
    kde, bw = _fit_kde(X, bandwidth, rips_gamma=rips_gamma)
    if not leave_one_out:
        return -kde.score_samples(X), bw
    # leave-one-out: density of each point under a KDE of all the others
    from sklearn.neighbors import KernelDensity
    n = len(X)
    logp = np.empty(n)
    # vectorised LOO via the kernel matrix would need O(n^2) memory; do it in blocks
    for i in range(n):
        other = KernelDensity(kernel="gaussian", bandwidth=bw).fit(np.delete(X, i, axis=0))
        logp[i] = other.score_samples(X[i:i + 1])[0]
    return -logp, bw


def kde_ratio_scores(XA, XB, bandwidth="scott"):
    """Two-sample score for each A-point = log p_A(x) - log p_B(x). Higher = dense in A but
    rare in B. Returns (scores, bw_A, bw_B)."""
    XA = np.asarray(XA, dtype=np.float64); XB = np.asarray(XB, dtype=np.float64)
    kdeA, bwA = _fit_kde(XA, bandwidth)
    kdeB, bwB = _fit_kde(XB, bandwidth)
    return kdeA.score_samples(XA) - kdeB.score_samples(XA), bwA, bwB


def _project(emb, dim, robust=False):
    """PCA to `dim`, then standardize the components. Standardizing AFTER PCA is essential
    for KDE: raw PCA components have very different variances (PC1 huge, later ones tiny),
    so a single global bandwidth would be dominated by PC1 and smear everything together
    (symptom: a huge bandwidth and near-identical scores). Whitening the components gives
    them comparable scale so the density is meaningful."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    E = np.asarray(emb, dtype=np.float64)
    X = robust_standardize(E) if robust else StandardScaler().fit_transform(E)
    d = min(dim, X.shape[1], X.shape[0])
    Xp = PCA(n_components=d, random_state=0).fit_transform(X)
    # whiten the PCA components (unit variance each) so KDE's single bandwidth is balanced
    return (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True,
                    help=".npz with an 'emb'/'XA' array of ALREADY-COMPUTED embeddings. This "
                         "script only SCORES saved embeddings; to embed flows with Chronos-2 and "
                         "get results + plots, run kde_anomaly.py --csv <features> instead.")
    ap.add_argument("--reference", default=None, help=".npz with a 'XB'/'emb' array -> two-sample ratio")
    ap.add_argument("--pca-dim", type=int, default=32, help="project to this many dims (KDE needs low-D)")
    ap.add_argument("--bandwidth", default="cv", help="cv (default) | scott | silverman | a float")
    ap.add_argument("--leave-one-out", action="store_true",
                    help="score each point against the density of the OTHERS (slower, O(n^2), "
                         "but robust to tight outlier groups hiding in their own density)")
    ap.add_argument("--out", default=None, help="CSV to write (scores, and cid if present)")
    ap.add_argument("--no-project", action="store_true", help="use the embeddings as-is (already low-D)")
    args = ap.parse_args()

    def _load(path):
        z = np.load(path, allow_pickle=True)
        for k in ("emb", "XA", "X", "embeddings"):
            if k in z:
                return z[k], (z["cid"] if "cid" in z else None)
        raise SystemExit(f"{path}: no 'emb'/'XA'/'X' array found (keys: {list(z.keys())})")

    emb, cid = _load(args.emb)
    X = emb if args.no_project else _project(emb, args.pca_dim)
    print(f"[kde] {X.shape[0]:,} points, {X.shape[1]}-D ({'raw' if args.no_project else 'PCA'})")

    import pandas as pd
    if args.reference:
        embB, _ = _load(args.reference)
        XB = embB if args.no_project else _project(embB, args.pca_dim)
        s, bwA, bwB = kde_ratio_scores(X, XB, args.bandwidth)
        print(f"[kde] two-sample: bw_A={bwA:.4g}, bw_B={bwB:.4g}; score = log p_A - log p_B")
        col = "log_density_ratio"
    else:
        s, bw = kde_scores(X, args.bandwidth, leave_one_out=args.leave_one_out)
        print(f"[kde] single-sample: bandwidth={bw:.4g}"
              + (" (leave-one-out)" if args.leave_one_out else "") + "; score = -log density")
        col = "neg_log_density"
    df = pd.DataFrame({col: s})
    if cid is not None:
        df.insert(0, "cid", [str(c) for c in cid])
    df = df.sort_values(col, ascending=False)
    q = np.quantile(s, [0.5, 0.9, 0.99])
    print(f"[kde] {col}: median {q[0]:.3f}, p90 {q[1]:.3f}, p99 {q[2]:.3f}, max {s.max():.3f}")
    print("[kde] top 10 most anomalous:")
    print(df.head(10).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"[kde] wrote {args.out}")


if __name__ == "__main__":
    main()


def gpd_anomaly_prob(scores, beta=0.90, cap_xi_nonpositive=True):
    """lookout v2 thresholding. Fit a Generalized Pareto Distribution to the upper tail
    (largest 1-beta) of the surprisals `scores` (= -log density), then return, per point,
    the tail probability p_i = (1-beta) * P(S > s_i). Small p_i = anomalous. Unlike a fixed
    top-fraction, this can flag ZERO points when the data is clean (no heavy upper tail).
    The GPD shape xi is constrained <= 0 (the surprisals are bounded, so the limiting
    distribution is Weibull-type)."""
    from scipy.stats import genpareto
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    u = np.quantile(s, beta)                      # threshold: top (1-beta) tail
    tail = s[s > u] - u
    if len(tail) < 5:
        return np.ones(n)                         # no usable tail -> nothing anomalous
    if cap_xi_nonpositive:
        # fit with xi <= 0 by searching; genpareto.fit gives c=xi
        c, loc, scale = genpareto.fit(tail, floc=0)
        if c > 0:                                 # constrain to non-positive shape
            c = 0.0
            scale = tail.mean()
    else:
        c, loc, scale = genpareto.fit(tail, floc=0)
    p = np.ones(n)
    over = s > u
    p[over] = (1 - beta) * genpareto.sf(s[over] - u, c, loc=0, scale=scale)
    return p