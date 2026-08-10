"""
spike_sorting.py

Unsupervised discovery of recurring event-waveform *families* across a
fungal MEA recording -- a "spike sorting" pipeline adapted from classical
extracellular neurophysiology (Rey et al. 2015; Quian Quiroga et al. 2004),
where the unit of analysis is the *compound event* rather than a single
action potential.

Pipeline (per channel, then merged across channels):

  1. STANDARDIZATION. Events have variable window length W_i; to compare
     shapes they are first resampled to a common length L and normalized to
     unit energy (scale-invariance):
         x_i[k] = w_i((k+1/2) * W_i / L),        k = 0..L-1    (linear interp)
         x~_i   = (x_i - mean(x_i)) / ||x_i - mean(x_i)||_2
     Raw amplitude is preserved separately as a descriptor (families may be
     the same shape at different magnitudes).

  2. FEATURES. Two interchangeable representations (chosen via --method):
       (a) PCA:  z_i = U_K^T x~_i,  K from 95% cumulative variance of the
           covariance Sigma = (1/M) sum_i x~_i x~_i^T.  [PCA scores]
       (b) WAVELET: discrete wavelet coefficients c_i of x~_i (mother
           wavelet sym4), keeping the N_feat coefficients whose distributions
           deviate most from Gaussian (Anderson-Darling normality statistic)
           -- multimodal coefficients separate shapes (Quian Quiroga 2004).

  3. CLUSTERING.
       (a) pca-gmm: Gaussian Mixture Model on z, model order K selected by the
           average silhouette coefficient (default; BIC available via
           --criterion bic). BIC is monotone decreasing on these data -- it
           never identifies a minimum within the tested range -- whereas the
           silhouette is bounded in [-1, 1] and peaks at the natural number of
           compact families. Events are then (re)assigned to the nearest
           template and labelled irregular (-1) if the normalized template
           distance exceeds 3 sigma of that family, where
               sigma_j = sqrt( (1/M_j) sum_{i in C_j} ||x~_i - tau_j||_2^2 )
           is the RMS member-to-template distance (the scale-matched analogue
           of the Wave_clus sigma_T). Chebyshev's inequality guarantees at
           least 88.9% of members lie within 3 sigma regardless of shape.
       (b) wavelet-dbscan: DBSCAN on the selected wavelet features with eps
           set at the kneedle (elbow) of the k-distance curve (Satopaa et
           al. 2011; quantile fallback when no elbow exists); points not
           density-reachable (outliers) are *by construction* the irregular
           class (label -1).

  4. TEMPLATES. For each family j, the template is the mean standardized
     waveform tau_j = (1/M_j) sum_{i in C_j} x~_i, a canonical shape that can
     be compared across channels and time.

  5. CROSS-CHANNEL MERGE. Families from different channels whose templates
      are near-identical are merged into a *global family* using
          |<tau_a, tau_b>| >= r_thr        (normalized inner product),
      implemented as connected components (union-find) on the template
      correlation graph. This answers: does the same event shape recur across
      channels (and time)? The default r_thr = 0.85 is justified in
      validate_thresholds() -- it exceeds the maximum correlation ever
      observed between two families that the per-channel model separates
      (self-consistency), while admitting only the extreme right tail of the
      cross-channel template correlation distribution (the true repeats).

  6. OUTPUT. families.npz (labels, global mapping, templates, descriptors),
     a console report, a gallery HTML (template + (time, channel) occurrence
     scatter per global family), and a threshold-validation report
     (threshold_report.txt + thresholds.png) that documents the empirical
     basis for every threshold, for reproducibility in the paper.

Run (from fungi-signaling/):
    python spike_sorting.py -o outputs/<ts>/waveforms/waveforms.npz
    python spike_sorting.py -o outputs/<ts>/waveforms/waveforms.npz --method wavelet
    python spike_sorting.py -o outputs/<ts>/waveforms/waveforms.npz --method both
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pywt
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from raw_analysis import SAMPLE_RATE_HZ, load_waveforms

# ---------------------------------------------------------------------------
# default hyper-parameters
# ---------------------------------------------------------------------------
DEFAULT_L: int = 128                 # common resampled length per event
PCA_ENERGY: float = 0.95             # fraction of variance kept for PCA features
DEFAULT_MAX_COMPONENTS: int = 8      # max GMM components to try
DEFAULT_WAVELET: str = "sym4"        # mother wavelet for --method wavelet
DEFAULT_LEVEL: int = 4               # DWT decomposition level (needs L % 2**level == 0)
DEFAULT_N_FEATURES: int = 10         # wavelet coefficients selected for clustering
DEFAULT_MIN_SAMPLES: int = 5         # DBSCAN min_samples
DEFAULT_EPS: Optional[float] = None  # DBSCAN eps; None -> k-distance heuristic
DEFAULT_MERGE_THRESHOLD: float = 0.85  # |cos(tau_a, tau_b)| above which per-channel families merge
TEMPLATE_GATE_MULT: float = 3.0      # irregular bucket: distance > 3 sigma_tau
RANDOM_SEED: int = 0


def standardize_waveforms(waveforms: List[np.ndarray], L: int = DEFAULT_L
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Resample variable-length windows to length L and L2-normalize.

    Formal description
    ------------------
    Event i has W_i samples. The resampled signal is the piecewise-linear
    interpolation of w_i evaluated on a uniform grid of L points:

        x_i[k] = (1 - f) * w_i[floor(t_k)] + f * w_i[floor(t_k)+1],
        t_k    = (k + 1/2) * W_i / L,      k = 0..L-1,

    then centered and normalized to unit energy (shape, not magnitude):

        x~_i = (x_i - mean(x_i)) / ||x_i - mean(x_i)||_2.

    Returns (X, norms): X is (M, L) standardized rows; norms[i] = ||x_i - mean||
    is the scale factor removed (a magnitude descriptor, in uV).
    """
    M = len(waveforms)
    if M == 0:
        return np.zeros((0, L)), np.zeros(0)
    X = np.zeros((M, L))
    norms = np.zeros(M)
    for i, w in enumerate(waveforms):
        w = np.asarray(w, dtype=float)
        n = len(w)
        t = (np.arange(L) + 0.5) * n / L
        idx = np.clip(np.floor(t).astype(int), 0, n - 1)
        nxt = np.clip(idx + 1, 0, n - 1)
        frac = t - np.floor(t)
        x = w[idx] * (1.0 - frac) + w[nxt] * frac
        x = x - float(x.mean())
        norms[i] = float(np.linalg.norm(x))
        if norms[i] > 0:
            x = x / norms[i]
        X[i] = x
    return X, norms


def pca_features(X: np.ndarray, energy: float = PCA_ENERGY) -> Tuple[np.ndarray, int]:
    """Project standardized waveforms onto the top-K principal components.

    Formal description
    ------------------
    Sigma = (1/M) X^T X  (L x L). Its eigen-decomposition Sigma = U Lambda U^T
    orders the variance directions by eigenvalue. K is the smallest index with

        sum_{j<=K} lambda_j / sum_j lambda_j >= energy,

    and the scores are z_i = U_K^T x~_i  (M x K), the low-dimensional shape
    coordinates used for clustering.
    """
    M = X.shape[0]
    pca = PCA(n_components=min(M, X.shape[1]), random_state=RANDOM_SEED)
    Z_full = pca.fit_transform(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    K = int(np.argmax(cum >= energy)) + 1
    K = int(np.clip(K, 2, M - 1 if M > 1 else 1))
    return Z_full[:, :K], K


def gmm_best_model(Z: np.ndarray, max_components: int = DEFAULT_MAX_COMPONENTS,
                   criterion: str = "silhouette", min_members: int = 2,
                   ) -> Tuple[GaussianMixture, int, np.ndarray]:
    """Fit GMMs on scores z and select the model order.

    Formal description
    ------------------
    p(z) = sum_{j=1..K} pi_j N(z | mu_j, Sigma_j)  fitted by expectation
    maximization for K = 1..max_components. Model order is selected by one of

      * silhouette (default): for each valid K (every cluster >= min_members
        events) compute the mean silhouette coefficient

            s(i) = (b(i) - a(i)) / max(a(i), b(i)),

        with a(i) the mean intra-cluster distance and b(i) the mean distance
        to the nearest other cluster. K maximizes the average silhouette.
        This is the standard compactness/separability criterion; it resists
        the over-splitting that BIC exhibits here, because fragmenting a
        compact cluster lowers its separation.

      * bic:  BIC(K) = dof(K) * ln M - 2 * ln L_hat(K),  K minimizes it.

    Returns (model, K, scores).
    """
    from sklearn.metrics import silhouette_score
    M = Z.shape[0]
    kmax = int(min(max_components, max(1, M // min_members)))
    scores = np.full(kmax, np.nan)
    best, best_key = None, None
    for k in range(1, kmax + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="diag",
                              n_init=4, random_state=RANDOM_SEED, tol=1e-4)
        gmm.fit(Z)
        labels = gmm.predict(Z)
        sizes = np.bincount(labels, minlength=k)
        if not (sizes >= min_members).all():
            continue
        if criterion == "silhouette" and k > 1 and sizes.min() >= 2:
            key = float(silhouette_score(Z, labels))
            better = best_key is None or key > best_key
        elif criterion == "bic":
            key = float(gmm.bic(Z))
            better = best_key is None or key < best_key
        else:
            continue
        scores[k - 1] = key
        if better:
            best, best_key = (gmm, k), key
    if best is None:
        # degenerate fallback: single family
        best = (GaussianMixture(n_components=1, covariance_type="diag",
                                random_state=RANDOM_SEED).fit(Z), 1)
    return best[0], best[1], scores


def _template_sigmas(X: np.ndarray, labels: np.ndarray,
                     templates: np.ndarray) -> np.ndarray:
    """Per-template RMS member distance (see template_assign for rationale).

    sigma_j = sqrt( (1/M_j) sum_{i in C_j} ||x~_i - tau_j||_2^2 )

    with a global fallback scale (1.4826 * robust typical nearest-template
    distance) for families with fewer than 2 members. Returns an array of
    length len(templates) (empty array if there are no templates).
    """
    if len(templates) == 0:
        return np.zeros(0)
    D = np.linalg.norm(X[:, None, :] - templates[None, :, :], axis=2)  # (M, F)
    sigmas = np.zeros(len(templates))
    for j in range(len(templates)):
        members = labels == j
        if members.sum() >= 2:
            sigmas[j] = float(np.sqrt(np.mean(D[members, j] ** 2)))
    l = np.argmin(D, axis=1)
    dmin = D[np.arange(len(D)), l]
    fallback = float(1.4826 * np.median(dmin)) if len(dmin) else 1.0
    if fallback <= 0:
        fallback = float(np.mean(dmin)) if len(dmin) else 1.0
    sigmas[sigmas <= 0] = fallback
    return sigmas


def template_assign(X: np.ndarray, templates: np.ndarray,
                    gate_mult: float = TEMPLATE_GATE_MULT) -> Tuple[np.ndarray, np.ndarray]:
    """Assign events to the nearest template with a 3-sigma irregular gate.

    Formal description
    ------------------
    All distances are in the unit-norm shape space (||x~|| = 1), so the
    natural spread of family j is the RMS distance of its members to the
    template,

        sigma_j = sqrt( (1/M_j) sum_{i in C_j} ||x~_i - tau_j||_2^2 ).

    (This is the scale-invariant analogue of the Wave_clus sigma_T, which is
    defined on raw waveforms; the per-sample-variance form does not transfer
    to normalized vectors because it shrinks by sqrt(L) relative to the L2
    distance.) Families with fewer than 2 members take a global fallback
    scale = 1.4826 * median_i min_j d_ij (robust typical nearest-template
    distance). Event i is assigned to

        l_i = argmin_j ||x~_i - tau_j||_2   if d_ij <= 3 * sigma_{l_i},
        l_i = -1 (irregular)               otherwise.

    Returns (labels, sigma_per_template).
    """
    D = np.linalg.norm(X[:, None, :] - templates[None, :, :], axis=2)  # (M, F)
    l = np.argmin(D, axis=1)
    dmin = D[np.arange(len(D)), l]
    sigmas = _template_sigmas(X, l, templates)
    gate = gate_mult * sigmas[l]
    labels = np.where(dmin <= gate, l, -1)
    return labels, sigmas


def cluster_pca_gmm(X: np.ndarray, energy: float = PCA_ENERGY,
                    max_components: int = DEFAULT_MAX_COMPONENTS,
                    criterion: str = "silhouette",
                    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Method (a): PCA features -> GMM/silhouette -> 3-sigma template assignment.

    See pca_features / gmm_best_model / template_assign. Returns (labels,
    templates, info) with info = {K_features, K_clusters, scores, sigmas}.
    """
    Z, Kf = pca_features(X, energy)
    gmm, Kc, scores = gmm_best_model(Z, max_components, criterion)
    init_labels = gmm.predict(Z)
    templates = np.vstack([
        X[init_labels == j].mean(axis=0) if (init_labels == j).sum() > 0
        else X.mean(axis=0) for j in range(Kc)])
    labels, sigmas = template_assign(X, templates)
    # drop families that ended up empty after the gate
    keep = np.array([(labels == j).sum() > 0 for j in range(Kc)])
    labels_map = np.zeros(Kc, dtype=int) - 1
    new_id, c = 0, 0
    for j in range(Kc):
        if keep[j]:
            labels_map[j] = new_id
            new_id += 1
    labels = np.where(labels >= 0, labels_map[labels], -1)
    templates = templates[keep]
    sigmas = sigmas[keep]
    return labels, templates, {"K_features": int(Kf), "n_clusters": int(new_id),
                               "scores": scores, "sigmas": sigmas}


def wavelet_features(X: np.ndarray, wavelet: str = DEFAULT_WAVELET,
                     level: int = DEFAULT_LEVEL) -> np.ndarray:
    """Discrete wavelet transform coefficients of each standardized event.

    Formal description
    ------------------
    x~_i is decomposed in a multiresolution analysis (Mallat):
        x~_i = a_J + sum_{m=1..J} d_m,
    where d_m are detail coefficient vectors at scales m (mother wavelet
    dilated by 2^m, localized in time) and a_J the coarsest approximation.
    The full coefficient vector c_i = [d_J ... d_1 a_J] has length L (the
    transform is orthonormal and non-redundant) and localizes shape
    differences in time-scale space. Returns (M, L) coefficient matrix.
    """
    coeffs = [pywt.wavedec(X[i], wavelet, level=level) for i in range(X.shape[0])]
    return np.vstack([np.concatenate([c.ravel() for c in ci]) for ci in coeffs])


def select_non_gaussian_features(F: np.ndarray,
                                 n_feat: int = DEFAULT_N_FEATURES) -> np.ndarray:
    """Keep the coefficients whose distributions deviate most from Gaussian.

    Formal description
    ------------------
    A coefficient that separates waveform families has a multimodal (non-
    Gaussian) distribution across events (Quian Quiroga 2004). Each column of
    F is scored with the Anderson-Darling normality statistic A (location/
    scale-free; large A = far from normal) and the N_feat highest-scoring
    columns are retained. Returns the selected feature matrix, standardized
    to zero mean / unit variance per column.
    """
    scores = np.zeros(F.shape[1])
    for j in range(F.shape[1]):
        x = F[:, j]
        x = x[np.isfinite(x)]
        if x.size >= 8 and x.std() > 0:
            scores[j] = float(stats.anderson(x, dist="norm",
                                             method="interpolate").statistic)
    sel = np.argsort(scores)[::-1][:n_feat]
    Z = F[:, sel]
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    return (Z - Z.mean(axis=0)) / sd


def estimate_eps(Z: np.ndarray, min_samples: int = DEFAULT_MIN_SAMPLES,
                 quantile: float = 0.9) -> float:
    """Heuristic DBSCAN eps from the k-distance distribution.

    Let kd = sorted min_samples-nearest-neighbour distances (ascending).
    The knee of this curve separates the flat core region (dense clusters)
    from the steep tail (noise); eps is taken at the knee. The knee is the
    point of maximum perpendicular distance from the chord joining the two
    curve endpoints (Satopaa et al., "Finding a 'kneedle' in a haystack").
    Falls back to the 'quantile' quantile when no well-defined knee exists.
    """
    nn = NearestNeighbors(n_neighbors=min(min_samples, Z.shape[0]))
    nn.fit(Z)
    dists, _ = nn.kneighbors(Z)
    kd = np.sort(dists[:, -1])
    n = len(kd)
    if n < 4:
        return float(kd[-1])
    x = np.arange(n, dtype=float)
    norm = float(np.hypot(x[-1] - x[0], kd[-1] - kd[0]))
    if norm == 0:
        return float(kd[-1])
    # perpendicular distance of each point to the chord (Satopaa et al. 2011)
    d = np.abs((kd[-1] - kd[0]) * (x - x[0]) - (x[-1] - x[0]) * (kd - kd[0])) / norm
    knee = int(np.argmax(d))
    # require the knee to be a genuine elbow: kd must accelerate there
    if knee >= n - 2 or knee < 1:
        return float(kd[int(quantile * (n - 1))])
    slope_before = (kd[knee] - kd[0]) / max(knee, 1)
    slope_after = (kd[-1] - kd[knee]) / max(n - knee, 1)
    if slope_after <= slope_before:
        return float(kd[int(quantile * (n - 1))])
    return float(kd[knee])


def cluster_wavelet_dbscan(X: np.ndarray, wavelet: str = DEFAULT_WAVELET,
                           level: int = DEFAULT_LEVEL,
                           n_feat: int = DEFAULT_N_FEATURES,
                           min_samples: int = DEFAULT_MIN_SAMPLES,
                           eps: Optional[float] = DEFAULT_EPS,
                           ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Method (b): wavelet features -> DBSCAN (outliers = irregular).

    See wavelet_features / select_non_gaussian_features. DBSCAN labels
    density-connected points; points reachable from no core point are
    labelled -1 (noise), giving the irregular class by construction.
    Templates are the mean standardized waveform per non-noise cluster.
    Returns (labels, templates, info).
    """
    F = wavelet_features(X, wavelet, level)
    Z = select_non_gaussian_features(F, n_feat)
    eps = float(estimate_eps(Z, min_samples)) if eps is None else float(eps)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Z)
    labels = db.labels_
    F_clus = int(labels.max()) + 1 if labels.max() >= 0 else 0
    templates = np.vstack([
        X[labels == j].mean(axis=0) for j in range(F_clus)]) if F_clus > 0 \
        else np.zeros((0, X.shape[1]))
    sigmas = _template_sigmas(X, labels, templates)
    return labels, templates, {"eps": eps, "min_samples": min_samples,
                               "n_features": int(n_feat),
                               "n_clusters": F_clus,
                               "n_irregular": int((labels < 0).sum()),
                               "sigmas": sigmas}


def merge_templates_global(families: List[Dict[str, Any]],
                           threshold: float = DEFAULT_MERGE_THRESHOLD
                           ) -> Tuple[List[List[int]], Dict[int, int]]:
    """Merge per-channel families whose templates correlate above threshold.

    Formal description
    ------------------
    Each per-channel family a is a node with template tau_a (mean waveform).
    Two nodes are connected when their absolute template overlap
        c_ab = |<tau_a, tau_b>|  (raw dot product)
    is >= r_thr; connected components (union-find) define the global
    families. NOTE: tau_a are means of unit-norm events, so ||tau_a|| <= 1
    and varies with event amplitude; the raw overlap therefore scores shape
    AND magnitude together (a family of small events can only merge with
    another small-event family even at identical shape). The 0.85 default was
    calibrated empirically against this metric (see threshold report: within-
    channel max ~0.81, cross-channel p99 ~0.89). Absolute value makes
    mirrored (sign-flipped) shapes map to the same family.

    Returns (components, node_to_global) where node index = (ch, fam) pairs
    enumerated in order of `families`; components are lists of node indices.
    """
    n = len(families)
    T = np.vstack([f["template"] for f in families])          # (n, L)
    corr = np.abs(T @ T.T)                                     # magnitude-weighted overlap
    np.fill_diagonal(corr, 0.0)

    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a in range(n):
        for b in range(a + 1, n):
            if corr[a, b] >= threshold:
                union(a, b)
    comps: Dict[int, List[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    node_to_global = {node: gid for gid, nodes in enumerate(comps.values())
                      for node in nodes}
    return sorted(comps.values(), key=len, reverse=True), node_to_global


def _event_index(results: Dict[int, Dict[str, Any]]) -> List[int]:
    """Channels that actually have extracted events, sorted."""
    return [ch for ch in sorted(results.keys()) if results[ch]["n_extracted"] > 0]


def run_sorting(npz_path: str, method: str, L: int = DEFAULT_L,
                merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
                max_components: int = DEFAULT_MAX_COMPONENTS,
                criterion: str = "silhouette",
                wavelet: str = DEFAULT_WAVELET, level: int = DEFAULT_LEVEL,
                n_feat: int = DEFAULT_N_FEATURES,
                min_samples: int = DEFAULT_MIN_SAMPLES,
                eps: Optional[float] = DEFAULT_EPS,
                ) -> Dict[str, Any]:
    """Execute the full per-channel -> cross-channel sorting pipeline.

    Returns a result dict holding, for every channel: the cluster labels
    (aligned with the channel's event arrays), templates, descriptors, and
    the event->global-family mapping; plus global family summaries.
    """
    results = load_waveforms(npz_path)
    channels = _event_index(results)

    families: List[Dict[str, Any]] = []        # one entry per (ch, family) node
    channel_out: Dict[int, Dict[str, Any]] = {}
    global_event_ids: Dict[int, np.ndarray] = {}   # ch -> global family id per event

    for ch in channels:
        d = results[ch]
        X, norms = standardize_waveforms(d["waveforms"], L)
        M = X.shape[0]
        if M < 3:
            # too few events to cluster reliably; all treated as irregular
            labels = np.full(M, -1, dtype=int)
            templates = np.zeros((0, L))
            info = {"n_clusters": 0, "sigmas": np.zeros(0)}
        elif method == "pca":
            labels, templates, info = cluster_pca_gmm(
                X, max_components=max_components, criterion=criterion)
        elif method == "wavelet":
            labels, templates, info = cluster_wavelet_dbscan(
                X, wavelet=wavelet, level=level, n_feat=n_feat,
                min_samples=min_samples, eps=eps)
        else:
            raise ValueError(f"unknown method: {method}")

        # per-family descriptors
        fam_desc = []
        for j in range(len(templates)):
            m = labels == j
            fam_desc.append({
                "n_members": int(m.sum()),
                "n_osc_median": float(np.median(d["n_oscillations"][m])),
                "n_osc_range": [int(d["n_oscillations"][m].min()),
                                int(d["n_oscillations"][m].max())],
                "amp_median_uV": float(np.median(np.abs(d["amplitudes"][m]))),
                "amp_iqr_uV": float(np.percentile(np.abs(d["amplitudes"][m]), 75)
                                    - np.percentile(np.abs(d["amplitudes"][m]), 25)),
                "time_span_s": [float(d["spike_times"][m].min()),
                                float(d["spike_times"][m].max())],
            })
            families.append({"ch": ch, "family": j, "template": templates[j],
                             "sigma": float(info["sigmas"][j]),
                             "n_members": int(m.sum())})

        channel_out[ch] = {
            "labels": labels, "templates": templates,
            "norms": norms, "info": info, "families": fam_desc, "X": X,
        }
        print(f"  ch{ch}: {M} events -> {info['n_clusters']} families "
              f"({int((labels < 0).sum())} irregular), {method}")

    # cross-channel merge
    components, node_to_global = merge_templates_global(families, merge_threshold)
    for i, f in enumerate(families):
        f["global"] = node_to_global[i]

    for ch in channels:
        labels = channel_out[ch]["labels"]
        gid = np.full(len(labels), -1, dtype=int)
        for j in range(len(channel_out[ch]["templates"])):
            node = next(i for i, f in enumerate(families)
                        if f["ch"] == ch and f["family"] == j)
            gid[labels == j] = node_to_global[node]
        global_event_ids[ch] = gid

    # global family summaries
    global_summary = []
    for cid, nodes in enumerate(components):
        members = []
        for i in nodes:
            f = families[i]
            members.append(f)
        global_summary.append({
            "global_id": cid,
            "n_members": int(sum(m["n_members"] for m in members)),
            "channels": sorted({m["ch"] for m in members}),
            "n_channels": len({m["ch"] for m in members}),
            "member_families": len(members),
        })

    return {
        "method": method, "L": L, "merge_threshold": merge_threshold,
        "criterion": criterion,
        "channels": channels, "channel_out": channel_out,
        "families": families, "components": components,
        "global_summary": global_summary, "global_event_ids": global_event_ids,
        "results": results,
    }


def save_families(out_path: Path, run: Dict[str, Any], npz_path: Path) -> None:
    """Persist the sorting output to a families.npz archive.

    One row per channel (object arrays) mirroring raw_analysis's layout:
    labels (per-channel cluster ids, -1 irregular), global_event_ids,
    templates (normalized), plus scalar metadata (method, params) and the
    source waveforms path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    channels = run["channels"]
    labels = np.empty(len(channels), dtype=object)
    glob_ids = np.empty(len(channels), dtype=object)
    templates = np.empty(len(channels), dtype=object)
    for i, ch in enumerate(channels):
        co = run["channel_out"][ch]
        labels[i] = co["labels"].astype(int)
        glob_ids[i] = run["global_event_ids"][ch].astype(int)
        templates[i] = co["templates"]
    payload = {
        "channels": np.asarray(channels, dtype=int),
        "labels": labels, "global_event_ids": glob_ids, "templates": templates,
        "method": run["method"], "L": run["L"],
        "criterion": run.get("criterion", ""),
        "merge_threshold": run["merge_threshold"],
        "source_waveforms": str(npz_path),
        "sample_rate": SAMPLE_RATE_HZ,
    }
    np.savez_compressed(out_path, **payload)
    print(f"Saved families to: {out_path}")


def gen_home_html(run_dir: Path) -> Path:
    """Build html/index.html linking every visualization in a run directory.

    Scans <run_dir> for the raw_analysis and spike_sorting outputs that
    actually exist (waveform grid, full traces, per-channel interactive views,
    family galleries, detail pages, threshold reports, meta) and writes a
    single index page. Returns the index path.
    """
    html_dir = run_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    fam_dir = run_dir / "families"
    index = html_dir / "index.html"

    rel = lambda p: p.relative_to(run_dir).as_posix()

    def link(title: str, path: Path, note: str = "") -> str:
        return (f'<li><a href="../{rel(path)}">{title}</a>'
                + (f" <span style='color:#888'>&mdash; {note}</span>" if note else "")
                + "</li>")

    links: List[str] = []
    links.append(link("Event waveform grid", html_dir / "waveforms_grid.html"))
    links.append(link("Full per-channel traces (with events)",
                      html_dir / "all_ch_spikes.html"))
    inter_dir = html_dir / "interactive_ch_views"
    inter_files = sorted(inter_dir.glob("channel_*_interactive.html")) \
        if inter_dir.exists() else []
    if inter_files:
        links.append(link(f"Per-channel interactive views ({len(inter_files)})",
                          inter_files[0],
                          "each page zooms into any channel"))

    for meth, sfx in (("pca", ""), ("wavelet", "_wavelet")):
        gal = fam_dir / f"gallery{sfx}.html"
        if gal.exists():
            links.append(link(f"Family gallery ({meth})", gal))
        for rep, repname in ((f"threshold_report{sfx}.txt", "threshold report"),
                             (f"thresholds{sfx}.png", "threshold figure")):
            p = fam_dir / rep
            if p.exists():
                links.append(link(f"{repname} ({meth})", p))
    agg = fam_dir / "agreement.txt"
    if agg.exists():
        links.append(link("Method agreement (pca vs wavelet)", agg))

    n_detail = len(list(fam_dir.glob("family*.html"))) if fam_dir.exists() else 0
    if n_detail:
        links.append(link(f"Family drill-down pages ({n_detail})",
                          fam_dir / "family_0.html",
                          "every member waveform with ch + time"))

    meta = run_dir / "run_meta.json"
    if meta.exists():
        links.append(link("Run metadata (parameters, per-channel summary)", meta))

    body = "\n".join(f"<ul>{l}</ul>" for l in links)
    html = [
        "<!DOCTYPE html><html><head><title>MEA analysis index</title><style>",
        "body{font-family:Arial,sans-serif;margin:30px;background:#f5f5f5;}",
        ".card{background:white;padding:20px;border-radius:8px;max-width:760px;",
        "box-shadow:0 2px 4px rgba(0,0,0,0.1);}",
        "a{color:#1f77b4;text-decoration:none;}a:hover{text-decoration:underline;}",
        "li{margin:6px 0;}</style></head><body>",
        f'<div class="card"><h1>MEA analysis index</h1>'
        f"<p>Run: <code>{run_dir.name}</code></p>{body}</div></body></html>",
    ]
    index.write_text("\n".join(html))
    print(f"Saved home page to: {index}")
    return index


def _waveform_tile(w: np.ndarray, ax) -> None:
    """Plot one raw event waveform as a compact tile (no axes decorations)."""
    ax.plot(w, linewidth=0.6, color="#1f77b4")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def gen_family_detail_html(run: Dict[str, Any], out_dir: Path,
                           cid: int, tiles_per_fig: int = 96,
                           cols: int = 12) -> None:
    """Clickable drill-down page for ONE global family: every member waveform.

    Renders all member events of global family `cid` as a grid of small
    waveform tiles, each labelled with its channel and event time (so a shape
    can be traced back to ch + t). Very large families are chunked into
    multiple <img> grids on a single page. Returns nothing; writes
    family_<cid>.html into out_dir.
    """
    nodes = run["components"][cid]
    gsum = run["global_summary"][cid]
    results = run["results"]

    rows = []  # (ch, t, waveform)
    for i in nodes:
        f = run["families"][i]
        d = results[f["ch"]]
        lbl = run["channel_out"][f["ch"]]["labels"]
        m = np.flatnonzero(lbl == f["family"])
        for mi in m:
            rows.append((f["ch"], float(d["spike_times"][mi]),
                         np.asarray(d["waveforms"][mi], dtype=float)))

    n = len(rows)
    rows_sorted = sorted(rows, key=lambda r: (r[0], r[1]))  # ch, then time

    img_parts = []
    for start in range(0, n, tiles_per_fig):
        chunk = rows_sorted[start:start + tiles_per_fig]
        nrow = int(np.ceil(len(chunk) / cols))
        fig, axes = plt.subplots(nrow, cols, figsize=(cols * 1.05, nrow * 0.85))
        axes = np.atleast_2d(axes)
        for k, (ch, t, w) in enumerate(chunk):
            r, c = divmod(k, cols)
            _waveform_tile(w, axes[r, c])
            axes[r, c].set_title(f"ch{ch} t={t:.2f}s", fontsize=5, pad=1)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        fig.tight_layout(pad=0.2)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        img_parts.append(f'<img src="data:image/png;base64,'
                         f'{base64.b64encode(buf.getvalue()).decode()}" '
                         f'style="display:block;margin:6px auto;max-width:100%;">')

    chs_str = ", ".join(str(c) for c in gsum["channels"][:20])
    if len(gsum["channels"]) > 20:
        chs_str += ", ..."
    sfx = suffix_of(run)
    html = [
        "<!DOCTYPE html><html><head><title>Family "
        f"{cid} detail</title><style>",
        "body{font-family:Arial,sans-serif;margin:20px;background:#f5f5f5;}",
        ".hd{background:#e8e8e8;padding:12px;border-radius:8px;}",
        "a{color:#1f77b4;}</style></head><body>",
        f'<div class="hd"><a href="gallery{sfx}.html">'
        f'&larr; back to gallery</a><br>',
        f"<h2>Family {cid} &mdash; {gsum['n_members']} events on "
        f"{gsum['n_channels']} channels (ch {chs_str})</h2>",
        f"<p>{n} waveforms; sorted by channel then time.</p></div>",
        *img_parts,
        "</body></html>",
    ]
    (out_dir / f"family{sfx}_{cid}.html").write_text("\n".join(html))


def suffix_of(run: Dict[str, Any]) -> str:
    """Filename suffix used for this run's gallery ('' for pca, '_wavelet')."""
    return "" if run["method"] == "pca" else f"_{run['method']}"


def gen_gallery_html(run: Dict[str, Any], out_path: Path) -> None:
    """Render one section per global family: template + (time, channel) map.

    For each global family: left panel shows the normalized template (mean
    standardized waveform, resampled index axis); right panel plots every
    member event's (time, channel) location -- directly visualizing whether
    a morphology family recurs across time and across electrodes.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = run["results"]
    parts = [
        "<!DOCTYPE html><html><head><title>Event morphology families</title><style>",
        "body{font-family:Arial,sans-serif;margin:20px;background:#f5f5f5;}",
        ".fam{margin-bottom:40px;background:white;padding:20px;border-radius:8px;",
        "box-shadow:0 2px 4px rgba(0,0,0,0.1);}",
        ".h{background:#e8e8e8;padding:12px;margin:-20px -20px 15px -20px;",
        "border-radius:8px 8px 0 0;}",
        "img{max-width:100%;}</style></head><body>",
        f"<h1>Event morphology families ({run['method']})</h1>",
    ]

    # irregular totals
    total_irr = sum(int((run["channel_out"][ch]["labels"] < 0).sum())
                    for ch in run["channels"])
    parts.append(f"<p><strong>{len(run['components'])} global families</strong>; "
                 f"{total_irr} irregular events unassigned.</p>")

    rng = np.random.default_rng(0)
    colors = {}
    for cid, nodes in enumerate(run["components"]):
        gsum = run["global_summary"][cid]
        # assemble member (time, channel) pairs
        ts, chs, oscs = [], [], []
        for i in nodes:
            f = run["families"][i]
            d = results[f["ch"]]
            lbl = run["channel_out"][f["ch"]]["labels"]
            m = lbl == f["family"]
            ts.extend(d["spike_times"][m].tolist())
            chs.extend([f["ch"]] * int(m.sum()))
            oscs.extend(d["n_oscillations"][m].tolist())
            color = None
        ts = np.asarray(ts)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6),
                                       gridspec_kw={"width_ratios": [1, 1.4]})
        templ = run["families"][nodes[0]]["template"]
        ax1.plot(templ, color="#1f77b4", linewidth=1.2)
        ax1.set_title(f"template (family {cid})", fontsize=8)
        ax1.set_ylabel("normalized uV", fontsize=7)
        ax1.set_xlabel("resampled index", fontsize=7)
        ax1.tick_params(labelsize=6)
        ax2.scatter(ts, chs, s=8, c="#1f77b4", alpha=0.7)
        ax2.set_title(f"{gsum['n_members']} events on {gsum['n_channels']} "
                      f"channels", fontsize=8)
        ax2.set_xlabel("time (s)", fontsize=7)
        ax2.set_ylabel("channel", fontsize=7)
        ax2.tick_params(labelsize=6)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        img = base64.b64encode(buf.getvalue()).decode("utf-8")

        chs_str = ", ".join(str(c) for c in gsum["channels"][:12])
        if len(gsum["channels"]) > 12:
            chs_str += ", ..."
        parts.append(
            f'<div class="fam"><div class="h"><strong>Family {cid}</strong> '
            f'&mdash; {gsum["n_members"]} events, {gsum["n_channels"]} channels '
            f'(ch {chs_str}) &mdash; '
            f'<a href="family{suffix_of(run)}_{cid}.html">drill-down: all '
            f'waveforms</a></div>'
            f'<img src="data:image/png;base64,{img}"></div>')

    parts.append("</body></html>")
    out_path.write_text("\n".join(parts))
    print(f"Saved gallery to: {out_path}")


def report(run: Dict[str, Any]) -> None:
    """Human-readable console summary of the sorting result."""
    print("\n" + "=" * 64)
    print(f"Sorting summary  [method={run['method']}, L={run['L']}, "
          f"criterion={run.get('criterion', '-')}, "
          f"merge_threshold={run['merge_threshold']}]")
    print("=" * 64)
    for g in run["global_summary"]:
        print(f"  family {g['global_id']:>3}: {g['n_members']:>4} events "
              f"over {g['n_channels']:>2} channels "
              f"({g['member_families']} per-channel families)")
    total = sum(g["n_members"] for g in run["global_summary"])
    irr = sum(int((run["channel_out"][ch]["labels"] < 0).sum())
              for ch in run["channels"])
    print(f"  total assigned: {total}, irregular: {irr}")
    print("=" * 64)


def validate_thresholds(run: Dict[str, Any], out_dir: Path,
                        merge_threshold: float,
                        suffix: str = "") -> Dict[str, Any]:
    """Empirically justify the merge threshold and irregular gate.

    This is the reproducibility/paper check. Using the per-channel family
    templates tau (unit norm), every pair (a, b) is scored c = |<tau_a,
    tau_b>| and split into within-channel pairs (two families the per-channel
    model kept *separate* on the same channel) and cross-channel pairs (the
    candidate merge population). The report records:

      * the distribution of within-channel correlations -- an upper bound on
        how similar two families can be while still being treated as distinct
        by the per-channel clustering (self-consistency of the merge rule);
      * the quantiles of the cross-channel distribution, showing that
        r_thr admits only the far right tail (the recurring shapes);
      * the resulting number of merged (multi-channel) global families;
      * the irregular-fraction vs gate-multiplier curve, confirming that 3
        sigma sits in a stable plateau (raising the gate barely changes the
        irregular class).

    Writes threshold_report.txt and thresholds.png into out_dir and returns
    the recorded statistics. When suffix is given (e.g. "_pca") the files are
    threshold_report<suffix>.txt / thresholds<suffix>.png.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"_report{suffix}" if suffix else "_report"
    imgprefix = f"{suffix}" if suffix else ""
    fam = run["families"]
    T = np.vstack([f["template"] for f in fam])
    C = np.abs(T @ T.T)  # magnitude-weighted overlap (matches merge_templates_global)
    np.fill_diagonal(C, 0)
    n = len(fam)
    within, cross = [], []
    for a in range(n):
        for b in range(a + 1, n):
            (within if fam[a]["ch"] == fam[b]["ch"] else cross).append(C[a, b])
    within, cross = np.asarray(within), np.asarray(cross)
    cross_sorted = np.sort(cross)

    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")
    stats_out: Dict[str, Any] = {
        "n_per_channel_families": n,
        "within_channel_corr": {"p50": q(within, 50), "p90": q(within, 90),
                                "p99": q(within, 99), "max": q(within, 100)},
        "cross_channel_corr": {"p50": q(cross, 50), "p90": q(cross, 90),
                               "p95": q(cross, 95), "p99": q(cross, 99),
                               "max": q(cross, 100)},
        "merge_threshold": merge_threshold,
        "n_cross_pairs_merged": int((cross >= merge_threshold).sum()),
    }
    self_consistent = stats_out["within_channel_corr"]["max"] < merge_threshold
    stats_out["self_consistent"] = bool(self_consistent)

    n_multi = sum(1 for g in run["global_summary"] if g["n_channels"] > 1)
    stats_out["n_multi_channel_families"] = int(n_multi)

    # irregular fraction vs gate multiplier (works for pca AND wavelet: both
    # store per-channel X and templates)
    if run["method"] in ("pca", "wavelet", "both"):
        irr_curve = {}
        for mult in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            tot_irr = 0
            for ch in run["channels"]:
                co = run["channel_out"][ch]
                if len(co["templates"]) == 0:
                    tot_irr += int((co["labels"] < 0).sum())
                    continue
                D = np.linalg.norm(co["X"][:, None, :]
                                   - co["templates"][None, :, :], axis=2)
                l = np.argmin(D, axis=1)
                dmin = D[np.arange(len(D)), l]
                sig = _template_sigmas(co["X"], l, co["templates"])
                tot_irr += int((dmin > mult * sig[l]).sum())
            irr_curve[mult] = tot_irr
        stats_out["irregular_vs_gate_mult"] = irr_curve

    # ---- report text -------------------------------------------------------
    lines = [
        "THRESHOLD VALIDATION REPORT (spike_sorting.py)",
        "=" * 64,
        f"method={run['method']}, L={run['L']}, "
        f"criterion={run.get('criterion', '-')}",
        "",
        "MERGE THRESHOLD r_thr = %.2f" % merge_threshold,
        "  Rationale (self-consistency): ideally the per-channel model never"
        " treats two families on the",
        "  SAME channel as distinct if they overlap more than the within-channel"
        " maximum, so that merging",
        "  only cross-channel pairs with c >= r_thr never contradicts the"
        " per-channel family",
        "  definitions. Self-consistency holds iff within-channel max < r_thr.",
        f"  within-channel corr:  p50={q(within,50):.3f}  p90={q(within,90):.3f}  "
        f"p99={q(within,99):.3f}  max={q(within,100):.3f}",
        f"  cross-channel  corr:  p50={q(cross,50):.3f}  p90={q(cross,90):.3f}  "
        f"p95={q(cross,95):.3f}  p99={q(cross,99):.3f}  max={q(cross,100):.3f}",
        f"  => r_thr={merge_threshold:.2f} self-consistent with per-channel "
        f"separation: {bool(self_consistent)}",
        f"  => admits {int((cross >= merge_threshold).sum())}/{len(cross)} "
        f"cross-channel pairs (top {100*int((cross >= merge_threshold).sum())/max(len(cross),1):.2f}%); "
        f"{int(n_multi)} global families span >1 channel.",
        "",
        "IRREGULAR GATE (3-sigma): sigma_j = RMS member->template distance;",
        "  Chebyshev: >= 88.9% of members within 3 sigma for any distribution"
        " (Gaussian: 99.7%).",
    ]
    if "irregular_vs_gate_mult" in stats_out:
        lines.append("  irregular count vs gate multiplier (all channels): "
                     + ", ".join(f"{m}x->{v}" for m, v in stats_out["irregular_vs_gate_mult"].items()))
    lines += [
        "",
        "DBSCAN eps (wavelet method): kneedle elbow of the k-distance curve",
        "  (Satopaa et al. 2011), quantile fallback when no elbow exists; see",
        "  per-channel info['eps'] in families.npz.",
    ]
    (out_dir / f"threshold{prefix}.txt").write_text("\n".join(lines))
    print(f"Saved threshold report to: {out_dir / f'threshold{prefix}.txt'}")

    # ---- figure -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    if len(within):
        ax1.hist(within, bins=50, range=(0, 1), alpha=0.7, color="#d62728",
                 label="within-channel pairs")
    ax1.hist(cross, bins=50, range=(0, 1), alpha=0.5, color="#1f77b4",
             label="cross-channel pairs")
    ax1.axvline(merge_threshold, color="k", ls="--", lw=1.2,
                label=f"r_thr={merge_threshold:.2f}")
    ax1.set_xlabel("template |correlation|")
    ax1.set_ylabel("pairs")
    ax1.legend(fontsize=7)
    ax1.set_title("merge-threshold rationale", fontsize=9)
    if "irregular_vs_gate_mult" in stats_out:
        xs = list(stats_out["irregular_vs_gate_mult"])
        ys = list(stats_out["irregular_vs_gate_mult"].values())
        ax2.plot(xs, ys, marker="o", color="#2ca02c")
        ax2.axvline(3.0, color="k", ls="--", lw=1.2, label="gate=3")
        ax2.set_xlabel("gate multiplier (sigma)")
        ax2.set_ylabel("irregular events (all channels)")
        ax2.legend(fontsize=7)
        ax2.set_title("irregular-gate stability", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"thresholds{imgprefix}.png", dpi=150)
    plt.close(fig)
    print(f"Saved threshold figure to: {out_dir / f'thresholds{imgprefix}.png'}")
    return stats_out


def _write_agreement(run_pca: Dict[str, Any], run_wav: Dict[str, Any],
                     out_dir: Path) -> Dict[str, Any]:
    """Side-by-side agreement between pca and wavelet labelings.

    For every channel clustered by both methods, the two label vectors are
    aligned element-wise (same event order) and scored with the adjusted Rand
    index (ARI), which is 1 for identical clusterings, ~0 for random, and
    handles label permutations. A contingency (events x (pca_fam, wav_fam))
    per channel is summed globally. Writes agreement.txt into out_dir.
    """
    from sklearn.metrics import adjusted_rand_score
    out_dir.mkdir(parents=True, exist_ok=True)
    chs = sorted(set(run_pca["channels"]) & set(run_wav["channels"]))
    ari = {}
    n_ev = {}
    for ch in chs:
        la = run_pca["channel_out"][ch]["labels"]
        lb = run_wav["channel_out"][ch]["labels"]
        n_ev[ch] = len(la)
        if len(la) == len(lb) and len(np.unique(la)) > 1 and len(np.unique(lb)) > 1:
            ari[ch] = float(adjusted_rand_score(la, lb))
    ari_arr = np.array(list(ari.values()))
    lines = [
        "METHOD AGREEMENT  pca/GMM vs wavelet/DBSCAN",
        "=" * 64,
        f"channels scored: {len(ari)}",
        f"per-channel ARI:  mean={ari_arr.mean():.3f}  median={np.median(ari_arr):.3f}  "
        f"min={ari_arr.min():.3f}  max={ari_arr.max():.3f}",
        "",
        "ch  events  pca_fam  wav_fam  pca_irr  wav_irr  ARI",
    ]
    for ch in chs:
        la = run_pca["channel_out"][ch]["labels"]
        lb = run_wav["channel_out"][ch]["labels"]
        lines.append(
            f"{ch:>3} {n_ev[ch]:>6} {int(la.max())+1 if la.max()>=0 else 0:>7} "
            f"{int(lb.max())+1 if lb.max()>=0 else 0:>7} "
            f"{int((la<0).sum()):>7} {int((lb<0).sum()):>7} "
            f"{ari.get(ch, float('nan')):>5.2f}")
    (out_dir / "agreement.txt").write_text("\n".join(lines))
    print(f"Saved agreement report to: {out_dir / 'agreement.txt'}")
    return {"n_channels": len(ari), "mean_ari": float(ari_arr.mean()),
            "median_ari": float(np.median(ari_arr))}


def _run_single(npz_path: Path, method: str, args: argparse.Namespace,
                out_dir: Path) -> Dict[str, Any]:
    """Cluster, report, persist, render gallery + threshold validation."""
    run = run_sorting(
        str(npz_path), method, L=args.length,
        merge_threshold=args.merge_threshold,
        max_components=args.max_components, criterion=args.criterion,
        wavelet=args.wavelet,
        level=args.level, n_feat=args.n_features,
        min_samples=args.min_samples, eps=args.eps)
    report(run)
    suffix = "_" + method if method != "pca" else ""
    save_families(out_dir / f"families{suffix}.npz", run, npz_path)
    gen_gallery_html(run, out_dir / f"gallery{suffix}.html")
    for cid in range(len(run["components"])):
        gen_family_detail_html(run, out_dir, cid)
    validate_thresholds(run, out_dir, args.merge_threshold, suffix=suffix)
    gen_home_html(out_dir.parent)
    return run


def _latest_run_npz() -> Path:
    """Path to the most recent raw_analysis waveforms.npz (by dir mtime).

    Used as the -o default so `python spike_sorting.py` acts on the freshest
    run without requiring the user to retype the timestamped directory.
    """
    root = Path("outputs")
    candidates = sorted(root.glob("*/waveforms/waveforms.npz"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(
            f"No waveforms.npz found under {root.resolve()}/ (run "
            "raw_analysis.py first, or pass -o <path> explicitly).")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Event-waveform family discovery (fungal MEA spike sorting)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default=None,
                        help="Path to a raw_analysis waveforms.npz "
                             "(default: latest run under outputs/)")
    parser.add_argument("--method", default="both",
                        choices=["pca", "wavelet", "both"],
                        help="Feature/clustering method (default: both; runs "
                             "pca and wavelet side by side + agreement)")
    parser.add_argument("-L", "--length", type=int, default=DEFAULT_L,
                        help="Common resampled event length (default 128)")
    parser.add_argument("--max-components", type=int, default=DEFAULT_MAX_COMPONENTS,
                        help="Max GMM components tried (pca method)")
    parser.add_argument("--criterion", default="silhouette",
                        choices=["silhouette", "bic"],
                        help="GMM model-order criterion (pca method; "
                             "default: silhouette)")
    parser.add_argument("--wavelet", default=DEFAULT_WAVELET,
                        help="Mother wavelet (wavelet method)")
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL,
                        help="DWT levels (wavelet method; needs L %% 2**level == 0)")
    parser.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES,
                        help="Wavelet coefficients kept (wavelet method)")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                        help="DBSCAN min_samples (wavelet method)")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS,
                        help="DBSCAN eps; default = k-distance heuristic")
    parser.add_argument("--merge-threshold", type=float, default=DEFAULT_MERGE_THRESHOLD,
                        help="Template correlation threshold for cross-channel "
                              "merge (default 0.85)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output dir for families.npz + gallery.html "
                             "(default: <waveforms run dir>/families)")
    args = parser.parse_args()

    npz_path = _latest_run_npz() if args.output is None else Path(args.output)
    out_dir = Path(args.out_dir) if args.out_dir else (
        npz_path.parent.parent / "families")
    if args.method == "both":
        run_pca = _run_single(npz_path, "pca", args, out_dir)
        run_wav = _run_single(npz_path, "wavelet", args, out_dir)
        _write_agreement(run_pca, run_wav, out_dir)
    else:
        _run_single(npz_path, args.method, args, out_dir)


if __name__ == "__main__":
    main()
