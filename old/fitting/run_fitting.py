"""
Main pipeline runner for Izhikevich parameter optimization.
Runs all optimization methods and compares results.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .loss_functions import combined_loss, waveform_loss, temporal_loss
from .synthetic_generator import generate_synthetic_dataset
from .performance_metrics import (
    compute_all_metrics, visualize_comparison,
    check_parameter_plausibility, print_metrics_summary,
    save_metrics
)
from .fit_from_scratch import (
    grid_search, random_search, differential_evolution_search,
    cma_es_search, particle_swarm_search, bayesian_optimization
)
from .fit_brian2 import (
    run_trace_fitter, run_spike_fitter, run_feature_metric_fitter
)

# Feature names
FEATURE_NAMES = [
    'amplitude', 'spike_width_fwhm_ms', 'asymmetry_index',
    'area_under_curve', 'snr', 'time_to_pp_ms',
    'pp_duration_ms', 'fall_rise_contrast'
]

def load_cluster_data(cluster_label: str, method: str = 'kmeans') -> Dict:
    """
    Load real data for a specific cluster.
    
    Args:
        cluster_label: Cluster identifier (e.g., '0', '1', etc.)
        method: Clustering method ('kmeans', 'gmm', 'dbscan')
    
    Returns:
        Dict with 'features', 'isi', 'firing_rate', 'spike_times', etc.
    """
    features_file = Path("outputs/features/waveform_features.npy")
    
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    data = np.load(features_file, allow_pickle=True).item()
    
    # Get cluster labels
    labels_key = f'{method}_labels'
    if labels_key not in data:
        raise KeyError(f"Cluster labels not found: {labels_key}")
    
    labels = data[labels_key]
    
    # Filter by cluster
    mask = labels == int(cluster_label)
    
    features = data['features'][mask]
    channel_ids = data['channel_ids'][mask]
    spike_times = data['spike_times'][mask]
    
    # Compute ISI (from all spike times in cluster, sorted)
    spike_times_sorted = np.sort(spike_times)
    isi = np.diff(spike_times_sorted) if len(spike_times_sorted) > 1 else np.array([])
    
    # Firing rate (assume 900s recording)
    duration_s = 900.0
    firing_rate = len(spike_times) / duration_s if duration_s > 0 else 0.0
    
    return {
        'features': features,
        'isi': isi,
        'firing_rate': firing_rate,
        'spike_times': spike_times,
        'channel_ids': channel_ids,
        'n_spikes': len(spike_times)
    }

def run_all_from_scratch_methods(
    real_features: np.ndarray,
    real_isi: np.ndarray,
    real_firing_rate: float,
    cluster_label: str,
    method: str,
    output_base: Path,
    duration_ms: float = 60000.0
) -> Dict:
    """Run all from-scratch optimization methods."""
    results = {}
    output_dir = output_base / method / f"cluster_{cluster_label}"
    
    print(f"\n{'#'*60}")
    print(f"RUNNING FROM-SCRATCH METHODS FOR CLUSTER {cluster_label} ({method})")
    print(f"{'#'*60}")
    
    # 1. Random Search (baseline)
    # print("\n" + "="*60)
    # print("1. RANDOM SEARCH")
    # print("="*60)
    # results['random_search'] = random_search(
    #     real_features, real_isi, real_firing_rate,
    #     n_iterations=10,
    #     duration_ms=duration_ms,
    #     output_dir=output_dir / 'random_search'
    # )
    
    # 2. Differential Evolution (main method)f
    print("\n" + "="*60)
    print("2. DIFFERENTIAL EVOLUTION")
    print("="*60)
    results['differential_evolution'] = differential_evolution_search(
        real_features, real_isi, real_firing_rate,
        duration_ms=duration_ms,
        maxiter=10, popsize=15,
        output_dir=output_dir / 'differential_evolution'
    )
    
    # 3. CMA-ES
    print("\n" + "="*60)
    print("3. CMA-ES")
    print("="*60)
    results['cma_es'] = cma_es_search(
        real_features, real_isi, real_firing_rate,
        duration_ms=duration_ms,
        budget=500,
        output_dir=output_dir / 'cma_es'
    )
    
    # 4. Particle Swarm
    print("\n" + "="*60)
    print("4. PARTICLE SWARM")
    print("="*60)
    results['particle_swarm'] = particle_swarm_search(
        real_features, real_isi, real_firing_rate,
        duration_ms=duration_ms,
        budget=500,
        output_dir=output_dir / 'particle_swarm'
    )
    
    # 5. Bayesian Optimization
    print("\n" + "="*60)
    print("5. BAYESIAN OPTIMIZATION")
    print("="*60)
    results['bayesian_optimization'] = bayesian_optimization(
        real_features, real_isi, real_firing_rate,
        duration_ms=duration_ms,
        n_calls=50,
        output_dir=output_dir / 'bayesian_optimization'
    )
    
    return results

def run_all_brian2_methods(
    real_features: np.ndarray,
    real_isi: np.ndarray,
    real_firing_rate: float,
    real_spike_times: np.ndarray,
    cluster_label: str,
    method: str,
    output_base: Path,
    duration_ms: float = 60000.0
) -> Dict:
    """Run all Brian2-based optimization methods."""
    results = {}
    output_dir = output_base / method / f"cluster_{cluster_label}"
    
    print(f"\n{'#'*60}")
    print(f"RUNNING BRIAN2 METHODS FOR CLUSTER {cluster_label} ({method})")
    print(f"{'#'*60}")
    
    # Generate average voltage trace for TraceFitter
    # (This requires careful setup - simplified here)
    
    # 1. TraceFitter
    print("\n" + "="*60)
    print("1. TRACEFITTER")
    print("="*60)
    results['trace_fitter'] = run_trace_fitter(
        real_features, real_spike_times,
        duration_ms=duration_ms,
        output_dir=output_dir / 'trace_fitter'
    )
    
    # 2. SpikeFitter
    print("\n" + "="*60)
    print("2. SPIKEFITTER")
    print("="*60)
    results['spike_fitter'] = run_spike_fitter(
        real_spike_times,
        duration_ms=duration_ms,
        output_dir=output_dir / 'spike_fitter'
    )
    
    # 3. FeatureMetric
    print("\n" + "="*60)
    print("3. FEATUREMETRIC")
    print("="*60)
    results['feature_metric'] = run_feature_metric_fitter(
        real_features, FEATURE_NAMES,
        duration_ms=duration_ms,
        output_dir=output_dir / 'feature_metric'
    )
    
    return results

def compare_and_validate(
    real_data: Dict,
    all_results: Dict,
    output_base: Path
) -> None:
    """
    Compare all optimization methods and validate results.
    Generates comparison tables and visualizations.
    """
    print(f"\n{'='*60}")
    print("COMPARING ALL METHODS")
    print(f"{'='*60}")
    
    comparison = []
    
    for method_name, result in all_results.items():
        if result is None:
            continue
        
        comparison.append({
            'method': method_name,
            'best_loss': result.get('best_loss', np.nan),
            'elapsed_time': result.get('elapsed_time', np.nan)
        })
        
        # Validate this method's parameters
        if 'best_params' in result:
            params = result['best_params']
            
            # Generate synthetic data with these params
            synth_data = generate_synthetic_dataset(params)
            
            # Compute metrics
            metrics = compute_all_metrics(
                real_data['features'], synth_data['features'],
                real_data['isi'], synth_data['isi'],
                real_data['firing_rate'], synth_data['firing_rate']
            )
            
            # Print summary
            print(f"\n--- {method_name} Validation ---")
            print_metrics_summary(metrics)
            
            # Save metrics
            method_dir = output_base / method_name
            if method_dir.exists():
                save_metrics(metrics, method_dir / "validation_metrics.json")
            
            # Visualize comparison
            visualize_comparison(
                real_data['features'], synth_data['features'],
                real_data['isi'], synth_data['isi'],
                real_data['firing_rate'], synth_data['firing_rate'],
                real_data['spike_times'], synth_data['spike_times'],
                output_dir=method_dir / "visualizations",
                prefix="validation"
            )
            
            # Check parameter plausibility
            from .fit_from_scratch import DEFAULT_BOUNDS
            plausibility = check_parameter_plausibility(params, DEFAULT_BOUNDS)
            print(f"Parameter plausibility: {plausibility}")
    
    # Save comparison table
    comparison_file = output_base / "methods_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison saved to: {comparison_file}")

def main():
    """
    Main pipeline runner.
    Usage: python -m fitting.run_fitting --cluster <label> --method <method>
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Izhikevich parameter optimization')
    parser.add_argument('--cluster', type=str, default='0', help='Cluster label to fit')
    parser.add_argument('--method', type=str, default='kmeans', choices=['kmeans', 'gmm', 'dbscan'], help='Clustering method')
    parser.add_argument('--duration', type=float, default=60000.0, help='Simulation duration (ms)')
    parser.add_argument('--skip-brian2', action='store_true', help='Skip Brian2 methods')
    args = parser.parse_args()
    
    print("=" * 60)
    print("IZHIKEVICH PARAMETER OPTIMIZATION PIPELINE")
    print("=" * 60)
    print(f"Cluster: {args.cluster}")
    print(f"Method: {args.method}")
    print(f"Duration: {args.duration} ms")
    print(f"Skip Brian2: {args.skip_brian2}")
    
    # Load cluster data
    print("\nLoading cluster data...")
    real_data = load_cluster_data(args.cluster, args.method)
    print(f"Loaded {real_data['n_spikes']} spikes for cluster {args.cluster}")
    
    # Setup output directory
    output_base = Path("outputs/fitting") / args.method / f"cluster_{args.cluster}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Run from-scratch methods
    scratch_results = run_all_from_scratch_methods(
        real_data['features'], real_data['isi'], real_data['firing_rate'],
        args.cluster, args.method, output_base, args.duration
    )
    all_results.update(scratch_results)
    
    # Run Brian2 methods (unless skipped)
    if not args.skip_brian2:
        brian2_results = run_all_brian2_methods(
            real_data['features'], real_data['isi'], real_data['firing_rate'],
            real_data['spike_times'], args.cluster, args.method, output_base, args.duration
        )
        all_results.update(brian2_results)
    else:
        print("\nSkipping Brian2 methods...")
    
    # Compare and validate
    compare_and_validate(real_data, all_results, output_base)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Results saved to: {output_base}")

if __name__ == "__main__":
    main()