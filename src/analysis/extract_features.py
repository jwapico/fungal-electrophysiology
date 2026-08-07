"""
extract_features.py
Self-contained script to extract waveform features and run PCA.
Saves results to outputs/features/waveform_features.npy

Run from project root:
    python src/analysis/extract_features.py
"""

import numpy as np
from pathlib import Path
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import re

# ============================================
# CONSTANTS (mirroring constants.py)
# ============================================
SAMPLE_RATE_HZ = 30000
MS_PER_SAMPLE = 1000.0 / SAMPLE_RATE_HZ
VOLTAGE_SCALE = 0.195  # int16 → μV

# Gaussian fitting constants (from classification.py)
RAW_WAVEFORM_GAUSS_P0 = [-1, 0.6, 0.1, 1, 1, 0.3]
RAW_WAVEFORM_GAUSS_LB = [-10, -10, 0.0, -10, -10, 0.1]
RAW_WAVEFORM_GAUSS_UB = [0, 10, 2.0, 10, 10, 10]
GAUSS_MAX_EVALS = 10000

DERIVATIVE_GAUSS_P0 = [-45, 0.5, 0.1, 45, 0.5, 0.1]
DERIVATIVE_GAUSS_LB = [-50, 0.0, 0.1, 0, 0.0, 0.1]
DERIVATIVE_GAUSS_UB = [0, 2.0, 5.0, 50, 2.0, 5.0]

# ============================================
# GAUSSIAN FITTING (copied from classification.py)
# ============================================
def gaussian2(x, a1, b1, c1, a2, b2, c2):
    """Sum of two Gaussians for waveform fitting."""
    return (a1 * np.exp(-((x - b1) / c1) ** 2) + 
            a2 * np.exp(-((x - b2) / c2) ** 2))

def fit_single_waveform(waveform, samps_per_ms=SAMPLE_RATE_HZ/1000, normalize=True):
    """Fit 2-Gaussian model to a single waveform. Returns 12 coefficients."""
    waveform_2d = waveform.reshape(-1, 1)
    n_samples = waveform_2d.shape[0]
    time = np.arange(n_samples) / samps_per_ms
    
    if normalize:
        max_val = np.max(np.abs(waveform_2d))
        if max_val > 0:
            waveform_2d = waveform_2d / max_val
    
    diff_waveform = np.diff(waveform_2d, axis=0)
    
    try:
        popt_raw, _ = curve_fit(
            gaussian2, time, waveform_2d.flatten(),
            p0=RAW_WAVEFORM_GAUSS_P0,
            bounds=(RAW_WAVEFORM_GAUSS_LB, RAW_WAVEFORM_GAUSS_UB),
            maxfev=GAUSS_MAX_EVALS
        )
        popt_der, _ = curve_fit(
            gaussian2, time[:-1], diff_waveform.flatten(),
            p0=DERIVATIVE_GAUSS_P0,
            bounds=(DERIVATIVE_GAUSS_LB, DERIVATIVE_GAUSS_UB),
            maxfev=GAUSS_MAX_EVALS
        )
        return np.concatenate([popt_raw, popt_der])
    except Exception:
        return np.full(12, np.nan)

def extract_snyder_features(coefs):
    """Extract Snyder features from 12 Gaussian coefficients."""
    if np.any(np.isnan(coefs)):
        return np.full(3, np.nan)
    time_to_pp = coefs[4]
    pp_duration = coefs[5]
    fall_rise = coefs[9] + coefs[6]
    return np.array([time_to_pp, pp_duration, fall_rise])

# ============================================
# NEW FEATURE EXTRACTION FUNCTIONS
# ============================================
def compute_amplitude(waveform):
    """Peak-to-peak amplitude."""
    return float(waveform.max() - waveform.min())

def compute_spike_width_fwhm(waveform, sample_rate=SAMPLE_RATE_HZ):
    """
    Compute spike width as Full Width at Half Maximum (FWHM).
    For extracellular spikes (negative deflection after negation),
    measures width at halfway between baseline and peak.
    Returns width in milliseconds.
    """
    v_min = float(waveform.min())  # Most negative (spike peak)
    v_max = float(waveform.max())  # Least negative (baseline)
    
    if v_max <= v_min:
        return np.nan
    
    half_max = v_min + 0.5 * (v_max - v_min)
    
    # Find where waveform crosses half_max
    above = waveform >= half_max
    diff = np.diff(above.astype(int))
    cross_down = np.where(diff == -1)[0]  # above → below
    cross_up = np.where(diff == 1)[0]    # below → above
    
    if len(cross_down) == 0 or len(cross_up) == 0:
        # Fallback: use first/last index below half_max
        below = waveform < half_max
        if np.any(below):
            idx = np.where(below)[0]
            return float((idx[-1] - idx[0]) * MS_PER_SAMPLE)
        return np.nan
    
    width_samples = cross_up[-1] - cross_down[0]
    return float(width_samples * MS_PER_SAMPLE)

def compute_asymmetry_index(waveform, sample_rate=SAMPLE_RATE_HZ):
    """
    Compute asymmetry index: ratio of time from start to peak vs peak to end.
    >1 means faster rise (peak closer to start).
    """
    min_idx = int(np.argmin(waveform))  # Most negative point
    n = len(waveform)
    
    if min_idx == 0 or min_idx == n - 1:
        return np.nan
    
    # Normalize: time before peak / time after peak
    return float(min_idx / (n - min_idx))

def compute_area_under_curve(waveform, sample_rate=SAMPLE_RATE_HZ):
    """Absolute area under curve (integral of |waveform|) in μV·ms."""
    time_axis = np.arange(len(waveform)) * MS_PER_SAMPLE
    return float(simpson(np.abs(waveform), x=time_axis))

def compute_snr(waveform):
    """Signal-to-noise ratio = peak-to-peak / (2 * std)."""
    peak_to_peak = waveform.max() - waveform.min()
    noise = np.std(waveform)
    return float(peak_to_peak / (2 * noise)) if noise > 0 else np.nan

# ============================================
# MAIN PIPELINE
# ============================================
def main():
    print("=" * 60)
    print("Feature Extraction and PCA Pipeline")
    print("=" * 60)
    
    # --- Load all waveform files ---
    waveform_dir = Path("outputs/waveforms")
    if not waveform_dir.exists():
        print(f"Error: {waveform_dir} not found. Run waveform extraction first.")
        return
    
    npy_files = sorted(waveform_dir.glob("ch_*_spikes.npy"))
    print(f"Found {len(npy_files)} waveform files")
    
    # --- Collect all spikes ---
    all_waveforms = []
    all_spike_times = []
    all_channel_ids = []
    all_window_sizes = []
    
    for f in npy_files:
        match = re.search(r'ch_(\d+)_spikes\.npy', f.name)
        if not match:
            continue
        ch = int(match.group(1))
        
        data = np.load(f, allow_pickle=True).item()
        waveforms = data['waveforms']
        spike_times = data['spike_times']
        window_sizes = data['window_sizes']
        
        for i in range(len(waveforms)):
            all_waveforms.append(np.asarray(waveforms[i], dtype=float))
            all_spike_times.append(float(spike_times[i]))
            all_channel_ids.append(ch)
            all_window_sizes.append(int(window_sizes[i]))
    
    n_spikes = len(all_waveforms)
    print(f"Total spikes loaded: {n_spikes}")
    
    # --- Extract new features ---
    print("\nExtracting waveform features...")
    feat_names_new = ['amplitude', 'spike_width_fwhm_ms', 'asymmetry_index', 'area_under_curve', 'snr']
    n_new = len(feat_names_new)
    features_new = np.zeros((n_spikes, n_new))
    
    for i, wf in enumerate(all_waveforms):
        if i % 100 == 0:
            print(f"  Progress: {i}/{n_spikes}...")
        features_new[i, 0] = compute_amplitude(wf)
        features_new[i, 1] = compute_spike_width_fwhm(wf)
        features_new[i, 2] = compute_asymmetry_index(wf)
        features_new[i, 3] = compute_area_under_curve(wf)
        features_new[i, 4] = compute_snr(wf)
    
    # --- Extract Snyder features (fit Gaussians) ---
    print("\nFitting Gaussians for Snyder features...")
    snyder_names = ['time_to_pp_ms', 'pp_duration_ms', 'fall_rise_contrast']
    features_snyder = np.zeros((n_spikes, 3))
    
    for i, wf in enumerate(all_waveforms):
        if i % 100 == 0:
            print(f"  Fitting: {i}/{n_spikes}...")
        coefs = fit_single_waveform(wf)
        features_snyder[i] = extract_snyder_features(coefs)
    
    # --- Combine all features ---
    all_feat_names = feat_names_new + snyder_names
    features_all = np.hstack([features_new, features_snyder])
    
    # Remove rows with NaN
    valid_mask = ~np.any(np.isnan(features_all), axis=1)
    features_clean = features_all[valid_mask]
    channels_clean = np.array(all_channel_ids)[valid_mask]
    times_clean = np.array(all_spike_times)[valid_mask]
    
    print(f"\nValid spikes after removing NaN: {features_clean.shape[0]}/{n_spikes}")
    print(f"Feature matrix shape: {features_clean.shape}")
    
    # --- PCA ---
    print("\nRunning PCA...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_clean)
    
    pca = PCA()
    pca.fit(features_scaled)
    
    print("\n" + "=" * 60)
    print("PCA RESULTS")
    print("=" * 60)
    print(f"{'PC':<6} {'Var Ratio':<12} {'Cumulative':<12}")
    print("-" * 40)
    cumsum = 0.0
    for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_))):
        cumsum = cum
        print(f"PC{i+1:<5} {var:<12.4f} {cum:<12.4f}")
    
    n_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
    n_90 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1
    print(f"\nComponents for 90% variance: {n_90}")
    print(f"Components for 95% variance: {n_95}")
    
    # --- Feature importance (loadings of PC1) ---
    print(f"\nTop 3 features contributing to PC1:")
    pc1_loadings = pca.components_[0]
    top_indices = np.argsort(np.abs(pc1_loadings))[::-1][:3]
    for idx in top_indices:
        print(f"  {all_feat_names[idx]:<30} loading: {pc1_loadings[idx]:.4f}")
    
    # --- Save results ---
    output_dir = Path("outputs/features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'features': features_clean,
        'feature_names': all_feat_names,
        'channel_ids': channels_clean,
        'spike_times': times_clean,
        'pca_components': pca.components_,
        'pca_explained_variance_ratio': pca.explained_variance_ratio_,
        'pca_mean': pca.mean_,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
    }
    
    output_file = output_dir / "waveform_features.npy"
    np.save(output_file, save_dict)
    print(f"\nSaved features to: {output_file}")
    
    # Also save CSV for inspection
    try:
        import pandas as pd
        df = pd.DataFrame(features_clean, columns=all_feat_names)
        df['channel_id'] = channels_clean
        df['spike_time_sec'] = times_clean
        csv_file = output_dir / "waveform_features.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved CSV to: {csv_file}")
    except ImportError:
        print("pandas not installed, skipping CSV export")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

if __name__ == "__main__":
    main()