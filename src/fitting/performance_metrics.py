"""
Performance metrics and visualization for optimization results.
Computes extensive metrics and generates comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

FEATURE_NAMES = [
    'amplitude', 'spike_width_fwhm_ms', 'asymmetry_index',
    'area_under_curve', 'snr', 'time_to_pp_ms',
    'pp_duration_ms', 'fall_rise_contrast'
]

def compute_all_metrics(
    real_features: np.ndarray,
    synthetic_features: np.ndarray,
    real_isi: np.ndarray,
    synthetic_isi: np.ndarray,
    real_firing_rate: float,
    synthetic_firing_rate: float,
    feature_names: List[str] = None
) -> Dict:
    """
    Compute all performance metrics comparing real vs synthetic data.
    
    Returns dict with:
        - per_feature_mse: Dict of per-feature MSE values
        - total_waveform_mse: Total waveform MSE
        - isi_wasserstein: Wasserstein distance between ISI distributions
        - isi_ks_pvalue: Kolmogorov-Smirnov test p-value
        - firing_rate_error: Absolute error in firing rate
        - firing_rate_relative_error: Relative error (%)
        - cv_real, cv_synthetic: Coefficients of variation
        - spike_count_real, spike_count_synthetic: Number of spikes
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES
    
    from .loss_functions import wasserstein_distance_1d, per_feature_mse
    
    metrics = {}
    
    # Waveform metrics
    metrics['per_feature_mse'] = per_feature_mse(real_features, synthetic_features, feature_names)
    metrics['total_waveform_mse'] = float(np.sum(list(metrics['per_feature_mse'].values())))
    
    # ISI metrics
    if len(real_isi) > 0 and len(synthetic_isi) > 0:
        metrics['isi_wasserstein'] = wasserstein_distance_1d(real_isi, synthetic_isi)
        
        # Kolmogorov-Smirnov test
        try:
            from scipy.stats import ks_2samp
            ks_stat, ks_pvalue = ks_2samp(real_isi, synthetic_isi)
            metrics['isi_ks_statistic'] = float(ks_stat)
            metrics['isi_ks_pvalue'] = float(ks_pvalue)
        except Exception:
            metrics['isi_ks_statistic'] = np.nan
            metrics['isi_ks_pvalue'] = np.nan
        
        # CV
        metrics['cv_real'] = float(np.std(real_isi) / np.mean(real_isi)) if np.mean(real_isi) > 0 else np.nan
        metrics['cv_synthetic'] = float(np.std(synthetic_isi) / np.mean(synthetic_isi)) if np.mean(synthetic_isi) > 0 else np.nan
    else:
        metrics['isi_wasserstein'] = np.nan
        metrics['isi_ks_pvalue'] = np.nan
        metrics['cv_real'] = np.nan
        metrics['cv_synthetic'] = np.nan
    
    # Firing rate metrics
    metrics['firing_rate_real'] = float(real_firing_rate)
    metrics['firing_rate_synthetic'] = float(synthetic_firing_rate)
    metrics['firing_rate_error'] = float(abs(real_firing_rate - synthetic_firing_rate))
    metrics['firing_rate_relative_error'] = float(
        abs(real_firing_rate - synthetic_firing_rate) / real_firing_rate * 100
    ) if real_firing_rate > 0 else np.nan
    
    # Spike counts
    metrics['spike_count_real'] = len(real_features)
    metrics['spike_count_synthetic'] = len(synthetic_features)
    
    return metrics

def visualize_comparison(
    real_features: np.ndarray,
    synthetic_features: np.ndarray,
    real_isi: np.ndarray,
    synthetic_isi: np.ndarray,
    real_firing_rate: float,
    synthetic_firing_rate: float,
    real_spike_times: np.ndarray,
    synthetic_spike_times: np.ndarray,
    real_avg_waveform: Optional[np.ndarray] = None,
    synthetic_avg_waveform: Optional[np.ndarray] = None,
    output_dir: Path = None,
    prefix: str = "comparison"
) -> None:
    """
    Generate extensive visualization comparing real vs synthetic data.
    Saves multiple PNG files to output_dir.
    """
    if output_dir is None:
        output_dir = Path("outputs/fitting/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feature_names = FEATURE_NAMES
    
    # 1. Per-feature comparison bar plot
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    ax = ax.flatten()
    
    mu_r = np.mean(real_features, axis=0)
    mu_s = np.mean(synthetic_features, axis=0)
    
    for i, (name, ax_i) in enumerate(zip(feature_names, ax)):
        ax_i.bar([0, 1], [mu_r[i], mu_s[i]], color=['steelblue', 'orange'])
        ax_i.set_title(name, fontsize=10)
        ax_i.set_xticks([0, 1])
        ax_i.set_xticklabels(['Real', 'Synth'], fontsize=8)
        ax_i.tick_params(labelsize=8)
    
    plt.suptitle('Feature Comparison: Real vs Synthetic', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_features.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. ISI distribution comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if len(real_isi) > 0:
        ax1.hist(real_isi, bins=50, alpha=0.6, color='steelblue', density=True, label='Real')
    if len(synthetic_isi) > 0:
        ax1.hist(synthetic_isi, bins=50, alpha=0.6, color='orange', density=True, label='Synthetic')
    ax1.set_xlabel('ISI (ms)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('ISI Distribution Comparison', fontsize=14)
    ax1.legend()
    ax1.spines[['top', 'right']].set_visible(False)
    
    # ISI cumulative distribution
    if len(real_isi) > 0:
        from scipy.stats import cumfreq
        real_cf = cumfreq(real_isi, numbins=100)
        ax2.plot(real_cf.lowerlimit + np.arange(real_cf.cumcount.size) * real_cf.binsize,
                real_cf.cumcount / len(real_isi), 'b-', label='Real', linewidth=2)
    if len(synthetic_isi) > 0:
        synth_cf = cumfreq(synthetic_isi, numbins=100)
        ax2.plot(synth_cf.lowerlimit + np.arange(synth_cf.cumcount.size) * synth_cf.binsize,
                synth_cf.cumcount / len(synthetic_isi), 'r-', label='Synthetic', linewidth=2)
    ax2.set_xlabel('ISI (ms)', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('ISI Cumulative Distribution', fontsize=14)
    ax2.legend()
    ax2.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_isi.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Spike time raster
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    
    if len(real_spike_times) > 0:
        ax.plot(real_spike_times, np.zeros_like(real_spike_times), 'b|', markersize=10, label='Real', alpha=0.6)
    if len(synthetic_spike_times) > 0:
        ax.plot(synthetic_spike_times, np.ones_like(synthetic_spike_times) * 0.5, 'r|', markersize=10, label='Synthetic', alpha=0.6)
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_yticks([0, 0.5])
    ax.set_yticklabels(['Real', 'Synthetic'])
    ax.set_title(f'Spike Times Comparison (Real: {len(real_spike_times)} spikes, Synth: {len(synthetic_spike_times)} spikes)', fontsize=14)
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_raster.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Feature distribution comparison (KDE)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (name, ax_i) in enumerate(zip(feature_names, axes)):
        if len(real_features) > 0:
            from scipy.stats import gaussian_kde
            kde_r = gaussian_kde(real_features[:, i])
            x_range = np.linspace(real_features[:, i].min(), real_features[:, i].max(), 200)
            ax_i.plot(x_range, kde_r(x_range), 'b-', linewidth=2, label='Real')
        
        if len(synthetic_features) > 0:
            kde_s = gaussian_kde(synthetic_features[:, i])
            x_range = np.linspace(synthetic_features[:, i].min(), synthetic_features[:, i].max(), 200)
            ax_i.plot(x_range, kde_s(x_range), 'r--', linewidth=2, label='Synthetic')
        
        ax_i.set_title(name, fontsize=10)
        ax_i.legend(fontsize=8)
        ax_i.spines[['top', 'right']].set_visible(False)
    
    plt.suptitle('Feature Distribution Comparison (KDE)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_kde.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Average waveform overlay (if provided)
    if real_avg_waveform is not None and synthetic_avg_waveform is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        time_axis = np.arange(len(real_avg_waveform)) * (1000.0 / 30000.0)  # ms
        ax.plot(time_axis, real_avg_waveform, 'b-', linewidth=2, label='Real', alpha=0.7)
        ax.plot(time_axis, synthetic_avg_waveform, 'r--', linewidth=2, label='Synthetic', alpha=0.7)
        
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Voltage (mV)', fontsize=12)
        ax.set_title('Average Waveform Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_waveform.png", dpi=150, bbox_inches='tight')
        plt.close()

def check_parameter_plausibility(
    params: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]]
) -> Dict[str, bool]:
    """
    Check if optimized parameters are biologically plausible.
    
    Returns dict with 'is_plausible' and individual checks.
    """
    checks = {}
    
    for key, (lower, upper) in bounds.items():
        if key in params:
            checks[f'{key}_in_bounds'] = lower <= params[key] <= upper
    
    # Additional biological constraints
    if 'a' in params and 'b' in params:
        checks['a_less_than_b'] = params['a'] < params['b']  # Typical for most neurons
    
    if 'c' in params:
        checks['c_negative'] = params['c'] < 0  # Reset value typically negative
    
    if 'd' in params:
        checks['d_positive'] = params['d'] > 0  # Increment typically positive
    
    checks['is_plausible'] = all(checks.values())
    
    return checks

def save_metrics(metrics: Dict, output_path: Path) -> None:
    """Save metrics to JSON file."""
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj) if 'float' in str(type(obj)) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        else:
            return obj
    
    metrics_converted = convert(metrics)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_converted, f, indent=2)

def print_metrics_summary(metrics: Dict) -> None:
    """Print a formatted summary of metrics."""
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS SUMMARY")
    print("=" * 60)
    
    print(f"\nSpike Counts:")
    print(f"  Real:      {metrics['spike_count_real']}")
    print(f"  Synthetic: {metrics['spike_count_synthetic']}")
    
    print(f"\nFiring Rates (Hz):")
    print(f"  Real:      {metrics['firing_rate_real']:.2f}")
    print(f"  Synthetic: {metrics['firing_rate_synthetic']:.2f}")
    print(f"  Error:     {metrics['firing_rate_error']:.2f} ({metrics['firing_rate_relative_error']:.1f}%)")
    
    print(f"\nISI Statistics:")
    print(f"  Real CV:      {metrics.get('cv_real', np.nan):.3f}")
    print(f"  Synthetic CV: {metrics.get('cv_synthetic', np.nan):.3f}")
    print(f"  Wasserstein:   {metrics.get('isi_wasserstein', np.nan):.3f}")
    print(f"  KS p-value:    {metrics.get('isi_ks_pvalue', np.nan):.3f}")
    
    print(f"\nPer-Feature MSE:")
    for name, mse in metrics.get('per_feature_mse', {}).items():
        print(f"  {name:<30} {mse:.6f}")
    print(f"  {'TOTAL':<30} {metrics.get('total_waveform_mse', 0):.6f}")
    
    print("\n" + "=" * 60)