import numpy as np
from typing import Tuple, Optional, Dict

def compute_mmd(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    if len(X) == 0 or len(Y) == 0:
        return 0.0
    
    # Ensure 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    n, d = X.shape
    m, d2 = Y.shape
    assert d == d2, "Feature dimensions must match"
    
    # Compute RBF kernel matrices
    def rbf_kernel(A, B):
        # Efficient computation using ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b^T
        A_norm = np.sum(A**2, axis=1)
        B_norm = np.sum(B**2, axis=1)
        K = np.exp(-(A_norm[:, None] + B_norm[None, :] - 2 * A @ B.T) / (2 * sigma**2))
        return K
    
    K_XX = rbf_kernel(X, X)
    K_YY = rbf_kernel(Y, Y)
    K_XY = rbf_kernel(X, Y)
    
    mmd = np.sum(K_XX) / (n * (n - 1)) + np.sum(K_YY) / (m * (m - 1)) - 2 * np.sum(K_XY) / (n * m)
    return max(0, mmd)  # Ensure non-negative

def wasserstein_distance_1d(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    # Equal sample sizes by interpolation
    n = min(len(x_sorted), len(y_sorted))
    x_resampled = np.interp(
        np.linspace(0, 1, n),
        np.linspace(0, 1, len(x_sorted)),
        x_sorted
    )
    y_resampled = np.interp(
        np.linspace(0, 1, n),
        np.linspace(0, 1, len(y_sorted)),
        y_sorted
    )
    
    return float(np.mean(np.abs(x_resampled - y_resampled)))

def waveform_loss(
    real_features: np.ndarray,
    synthetic_features: np.ndarray,
    use_mmd: bool = False,
    mmd_sigma: float = 1.0,
    lambda_mmd: float = 0.1
) -> float:
    # Compute mean feature vectors
    mu_r = np.mean(real_features, axis=0)
    mu_s = np.mean(synthetic_features, axis=0)
    
    # MSE of means
    D = len(mu_r)
    mse = np.sum((mu_r - mu_s)**2) / D
    
    if use_mmd and len(real_features) > 10 and len(synthetic_features) > 10:
        mmd = compute_mmd(real_features, synthetic_features, mmd_sigma)
        return float(mse + lambda_mmd * mmd)
    
    return float(mse)

def temporal_loss(
    real_firing_rate: float,
    synthetic_firing_rate: float,
    real_isi: np.ndarray,
    synthetic_isi: np.ndarray,
    real_cv: Optional[float] = None,
    synthetic_cv: Optional[float] = None,
    use_wasserstein: bool = True,
    gamma_wasserstein: float = 1.0,
    delta_cv: float = 0.5
) -> float:
    # Firing rate MSE
    rate_loss = (real_firing_rate - synthetic_firing_rate)**2
    
    # Wasserstein distance on ISI distributions
    wasserstein = 0.0
    if use_wasserstein and len(real_isi) > 0 and len(synthetic_isi) > 0:
        wasserstein = wasserstein_distance_1d(real_isi, synthetic_isi)
    
    # CV difference
    cv_loss = 0.0
    if real_cv is not None and synthetic_cv is not None:
        cv_loss = abs(real_cv - synthetic_cv)
    elif len(real_isi) > 1 and len(synthetic_isi) > 1:
        real_cv = np.std(real_isi) / np.mean(real_isi) if np.mean(real_isi) > 0 else 0 # type: ignore
        synthetic_cv = np.std(synthetic_isi) / np.mean(synthetic_isi) if np.mean(synthetic_isi) > 0 else 0 # type: ignore
        cv_loss = abs(real_cv - synthetic_cv) # type: ignore
    
    return float(rate_loss + gamma_wasserstein * wasserstein + delta_cv * cv_loss)

def combined_loss(
    real_features: np.ndarray,
    synthetic_features: np.ndarray,
    real_firing_rate: float,
    synthetic_firing_rate: float,
    real_isi: np.ndarray,
    synthetic_isi: np.ndarray,
    alpha: float = 1.0,
    beta: float = 10.0,
    **kwargs
) -> float:
    L_w = waveform_loss(real_features, synthetic_features, **kwargs)
    L_t = temporal_loss(
        real_firing_rate, synthetic_firing_rate,
        real_isi, synthetic_isi, **kwargs
    )
    return alpha * L_w + beta * L_t

def per_feature_mse(
    real_features: np.ndarray,
    synthetic_features: np.ndarray,
    feature_names: list
) -> Dict[str, float]:
    """Compute MSE for each feature separately."""
    mu_r = np.mean(real_features, axis=0)
    mu_s = np.mean(synthetic_features, axis=0)
    return {name: float((mu_r[i] - mu_s[i])**2) for i, name in enumerate(feature_names)}