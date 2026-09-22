"""
From-scratch optimization methods for Izhikevich parameters.
Includes: Grid Search, Random Search, Differential Evolution,
          CMA-ES, Particle Swarm, Bayesian Optimization.
"""

import numpy as np
from typing import Dict, Tuple, List, Callable, Optional
from pathlib import Path
import json
import time
from .synthetic_generator import simulate_with_params

# Parameter bounds (standard Izhikevich model)
DEFAULT_BOUNDS = {
    'a': (0.001, 0.5),
    'b': (0.05, 0.5),
    'c': (-90.0, -40.0),
    'd': (0.0, 20.0),
    'I_bias': (0.0, 20.0),
    'sigma': (0.0, 5.0)
}

def objective_function(
    params_vector: np.ndarray,
    param_names: List[str],
    real_features: np.ndarray,
    real_isi: np.ndarray,
    real_firing_rate: float,
    duration_ms: float = 60000.0,
    alpha: float = 1.0,
    beta: float = 10.0
) -> float:
    """
    Objective function for optimization.
    Converts parameter vector to dict, runs simulation, computes loss.
    """
    from .synthetic_generator import generate_synthetic_dataset
    from .loss_functions import combined_loss
    
    params_dict = {name: params_vector[i] for i, name in enumerate(param_names)}
    
    try:
        # Generate synthetic data
        synth_data = simulate_with_params(params_dict, duration_ms)
        
        # Check if we got enough spikes
        if len(synth_data['features']) < 2:
            return 1e6  # Penalize insufficient spikes
        
        # Compute loss
        loss = combined_loss(
            real_features, synth_data['features'],
            real_firing_rate, synth_data['firing_rate'],
            real_isi, synth_data['isi'],
            alpha=alpha, beta=beta
        )
        
        return loss
    except Exception as e:
        return 1e6  # Penalize errors

def grid_search(
    real_features: np.ndarray,
    real_isi: np.ndarray,
    real_firing_rate: float,
    param_ranges: Dict[str, List[float]],
    output_dir: Path,
    duration_ms: float = 60000.0,
    alpha: float = 1.0,
    beta: float = 10.0,
) -> Dict:
    """
    Grid search optimization (exhaustive, for small parameter spaces).
    
    Args:
        param_ranges: Dict mapping param name to list of values to try.
                      If None, uses 5 values per param from DEFAULT_BOUNDS.
    """
    if param_ranges is None:
        # Create 5-point grid for each parameter
        param_ranges = {}
        for name, (lower, upper) in DEFAULT_BOUNDS.items():
            param_ranges[name] = np.linspace(lower, upper, 5).tolist()
    
    param_names = list(param_ranges.keys())
    
    # Generate all combinations
    from itertools import product
    all_combinations = list(product(*[param_ranges[name] for name in param_names]))
    
    print(f"\n{'='*60}")
    print(f"GRID SEARCH: {len(all_combinations)} combinations")
    print(f"{'='*60}")
    
    best_loss = float('inf')
    best_params = None
    all_results = []
    
    start_time = time.time()
    
    for i, combo in enumerate(all_combinations):
        params_vec = np.array(combo)
        loss = objective_function(
            params_vec, param_names,
            real_features, real_isi, real_firing_rate,
            duration_ms, alpha, beta
        )
        
        result = {name: combo[i] for i, name in enumerate(param_names)}
        result['loss'] = loss
        all_results.append(result)
        
        if loss < best_loss:
            best_loss = loss
            best_params = {name: combo[i] for i, name in enumerate(param_names)}
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(all_combinations)}, Best loss: {best_loss:.6f}")
    
    elapsed = time.time() - start_time
    
    print(f"\nGrid Search Complete ({elapsed:.1f}s)")
    print(f"Best Loss: {best_loss:.6f}")
    print(f"Best Params: {best_params}")
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "grid_search_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'method': 'grid_search',
                'best_params': best_params,
                'best_loss': best_loss,
                'all_results': all_results,
                'elapsed_time': elapsed
            }, f, indent=2)
    
    return {
        'method': 'grid_search',
        'best_params': best_params,
        'best_loss': best_loss,
        'all_results': all_results,
        'elapsed_time': elapsed
    }

def random_search(
    real_features: np.ndarray,
    real_isi: np.ndarray,
    real_firing_rate: float,
    bounds: Dict[str, Tuple[float, float]],
    output_dir: Path,
    n_iterations: int = 1000,
    duration_ms: float = 60000.0,
    alpha: float = 1.0,
    beta: float = 10.0,
) -> Dict:
    """Random search optimization."""
    if bounds is None:
        bounds = DEFAULT_BOUNDS
    
    param_names = list(bounds.keys())
    lower_bounds = np.array([bounds[name][0] for name in param_names])
    upper_bounds = np.array([bounds[name][1] for name in param_names])
    
    print(f"\n{'='*60}")
    print(f"RANDOM SEARCH: {n_iterations} iterations")
    print(f"{'='*60}")
    
    best_loss = float('inf')
    best_params = None
    all_results = []
    loss_history = []
    
    start_time = time.time()
    
    for i in range(n_iterations):
        # Sample random parameters
        params_vec = lower_bounds + (upper_bounds - lower_bounds) * np.random.rand(len(param_names))
        
        loss = objective_function(
            params_vec, param_names,
            real_features, real_isi, real_firing_rate,
            duration_ms, alpha, beta
        )
        
        all_results.append({name: params_vec[j] for j, name in enumerate(param_names)})
        all_results[-1]['loss'] = loss
        loss_history.append(loss)
        
        if loss < best_loss:
            best_loss = loss
            best_params = {name: params_vec[j] for j, name in enumerate(param_names)}
        
        print(f"  Progress: {i+1}/{n_iterations}, Best loss: {best_loss:.6f}")
    
    elapsed = time.time() - start_time
    
    print(f"\nRandom Search Complete ({elapsed:.1f}s)")
    print(f"Best Loss: {best_loss:.6f}")
    print(f"Best Params: {best_params}")
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "random_search_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'method': 'random_search',
                'best_params': best_params,
                'best_loss': best_loss,
                'loss_history': loss_history,
                'n_iterations': n_iterations,
                'elapsed_time': elapsed
            }, f, indent=2)
        
        # Save loss history plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(loss_history, 'b-', alpha=0.5, label='All runs')
        ax.plot(np.minimum.accumulate(loss_history), 'r-', linewidth=2, label='Best so far')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Random Search Loss History', fontsize=14, fontweight='bold')
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_dir / "random_search_loss.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        'method': 'random_search',
        'best_params': best_params,
        'best_loss': best_loss,
        'loss_history': loss_history,
        'elapsed_time': elapsed
    }

def differential_evolution_search(
    real_features: np.ndarray,
    real_isi: np.ndarray,
    real_firing_rate: float,
    bounds: Dict[str, Tuple[float, float]],
    output_dir: Path,
    duration_ms: float = 60000.0,
    alpha: float = 1.0,
    beta: float = 10.0,
    maxiter: int = 100,
    popsize: int = 15,
) -> Dict:
    """Differential Evolution optimization using scipy."""
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        print("scipy not installed. Cannot run differential evolution.")
        return {}
    
    if bounds is None:
        bounds = DEFAULT_BOUNDS
    
    param_names = list(bounds.keys())
    bounds_list = [bounds[name] for name in param_names]
    
    print(f"\n{'='*60}")
    print(f"DIFFERENTIAL EVOLUTION")
    print(f"{'='*60}")
    
    def wrapped_objective(params_vec):
        return objective_function(
            params_vec, param_names,
            real_features, real_isi, real_firing_rate,
            duration_ms, alpha, beta
        )
    
    start_time = time.time()
    
    result = differential_evolution(
        wrapped_objective,
        bounds_list,
        maxiter=maxiter,
        popsize=popsize,
        disp=True,
        # atol=1e-8
    )
    
    elapsed = time.time() - start_time
    
    best_params = {name: result.x[i] for i, name in enumerate(param_names)}
    
    print(f"\nDifferential Evolution Complete ({elapsed:.1f}s)")
    print(f"Best Loss: {result.fun:.6f}")
    print(f"Best Params: {best_params}")
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "de_search_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'method': 'differential_evolution',
                'best_params': best_params,
                'best_loss': float(result.fun),
                'n_iterations': result.nit,
                'n_evaluations': result.nfev,
                'success': result.success,
                'message': result.message,
                'elapsed_time': elapsed
            }, f, indent=2)
    
    return {
        'method': 'differential_evolution',
        'best_params': best_params,
        'best_loss': float(result.fun),
        'result': result,
        'elapsed_time': elapsed
    }

def cma_es_search(
    real_features: np.ndarray,
    real_isi: np.ndarray,
    real_firing_rate: float,
    bounds: Dict[str, Tuple[float, float]],
    output_dir: Path,
    duration_ms: float = 60000.0,
    alpha: float = 1.0,
    beta: float = 10.0,
    sigma0: float = 0.5,
    budget: int = 1000,
) -> Dict:
    """CMA-ES optimization using nevergrad."""
    try:
        import nevergrad as ng
    except ImportError:
        print("nevergrad not installed. Cannot run CMA-ES.")
        return {}
    
    if bounds is None:
        bounds = DEFAULT_BOUNDS
    
    param_names = list(bounds.keys())
    
    print(f"\n{'='*60}")
    print(f"CMA-ES (using nevergrad)")
    print(f"{'='*60}")
    
    # Create parameter space
    param_dict = {}
    for name, (lower, upper) in bounds.items():
        param_dict[name] = ng.var.Scalar().bounded(lower, upper)
    
    instrumentation = ng.instrumentation.Instrumentation(**param_dict)
    optimizer = ng.optimizers.CMA(instrumentation=instrumentation, budget=budget)
    
    def wrapped_objective(**kwargs):
        params_vec = np.array([kwargs[name] for name in param_names])
        return objective_function(
            params_vec, param_names,
            real_features, real_isi, real_firing_rate,
            duration_ms, alpha, beta
        )
    
    start_time = time.time()
    
    recommendation = optimizer.minimize(wrapped_objective)
    elapsed = time.time() - start_time
    
    best_params_raw = recommendation.kwargs
    best_params = {name: best_params_raw[name] for name in param_names}
    
    # Evaluate final loss
    params_vec = np.array([best_params[name] for name in param_names])
    best_loss = objective_function(
        params_vec, param_names,
        real_features, real_isi, real_firing_rate,
        duration_ms, alpha, beta
    )
    
    print(f"\nCMA-ES Complete ({elapsed:.1f}s)")
    print(f"Best Loss: {best_loss:.6f}")
    print(f"Best Params: {best_params}")
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "cma_es_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'method': 'cma_es',
                'best_params': best_params,
                'best_loss': best_loss,
                'budget': budget,
                'elapsed_time': elapsed
            }, f, indent=2)
    
    return {
        'method': 'cma_es',
        'best_params': best_params,
        'best_loss': best_loss,
        'elapsed_time': elapsed
    }

def particle_swarm_search(
    real_features: np.ndarray,
    real_isi: np.ndarray,
    real_firing_rate: float,
    bounds: Dict[str, Tuple[float, float]],
    output_dir: Path,
    duration_ms: float = 60000.0,
    alpha: float = 1.0,
    beta: float = 10.0,
    budget: int = 1000,
) -> Dict:
    """Particle Swarm Optimization using nevergrad."""
    try:
        import nevergrad as ng
    except ImportError:
        print("nevergrad not installed. Cannot run Particle Swarm.")
        return {}
    
    if bounds is None:
        bounds = DEFAULT_BOUNDS
    
    param_names = list(bounds.keys())
    
    print(f"\n{'='*60}")
    print(f"PARTICLE SWARM (using nevergrad)")
    print(f"{'='*60}")
    
    # Create parameter space
    param_dict = {}
    for name, (lower, upper) in bounds.items():
        param_dict[name] = ng.var.Scalar().bounded(lower, upper)
    
    instrumentation = ng.instrumentation.Instrumentation(**param_dict)
    optimizer = ng.optimizers.PSO(instrumentation=instrumentation, budget=budget)
    
    def wrapped_objective(**kwargs):
        params_vec = np.array([kwargs[name] for name in param_names])
        return objective_function(
            params_vec, param_names,
            real_features, real_isi, real_firing_rate,
            duration_ms, alpha, beta
        )
    
    start_time = time.time()
    
    recommendation = optimizer.minimize(wrapped_objective)
    elapsed = time.time() - start_time
    
    best_params_raw = recommendation.kwargs
    best_params = {name: best_params_raw[name] for name in param_names}
    
    params_vec = np.array([best_params[name] for name in param_names])
    best_loss = objective_function(
        params_vec, param_names,
        real_features, real_isi, real_firing_rate,
        duration_ms, alpha, beta
    )
    
    print(f"\nParticle Swarm Complete ({elapsed:.1f}s)")
    print(f"Best Loss: {best_loss:.6f}")
    print(f"Best Params: {best_params}")
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "pso_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'method': 'particle_swarm',
                'best_params': best_params,
                'best_loss': best_loss,
                'budget': budget,
                'elapsed_time': elapsed
            }, f, indent=2)
    
    return {
        'method': 'particle_swarm',
        'best_params': best_params,
        'best_loss': best_loss,
        'elapsed_time': elapsed
    }

def bayesian_optimization(
    real_features: np.ndarray,
    real_isi: np.ndarray,
    real_firing_rate: float,
    bounds: Dict[str, Tuple[float, float]],
    output_dir: Path,
    duration_ms: float = 60000.0,
    alpha: float = 1.0,
    beta: float = 10.0,
    n_calls: int = 50,
) -> Dict:
    """Bayesian Optimization using scikit-optimize."""
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        print("scikit-optimize not installed. Cannot run Bayesian Optimization.")
        return {}
    
    if bounds is None:
        bounds = DEFAULT_BOUNDS
    
    param_names = list(bounds.keys())
    skopt_space = [Real(bounds[name][0], bounds[name][1], name=name) for name in param_names]
    
    print(f"\n{'='*60}")
    print(f"BAYESIAN OPTIMIZATION (using scikit-optimize)")
    print(f"{'='*60}")
    
    def wrapped_objective(params_list):
        params_vec = np.array(params_list)
        return objective_function(
            params_vec, param_names,
            real_features, real_isi, real_firing_rate,
            duration_ms, alpha, beta
        )
    
    start_time = time.time()
    
    result = gp_minimize(
        wrapped_objective,
        skopt_space,
        n_calls=n_calls,
        random_state=42,
        verbose=True
    )
    
    elapsed = time.time() - start_time

    if result == None:
        return {}
    
    best_params = {name: result.x[i] for i, name in enumerate(param_names)}
    
    print(f"\nBayesian Optimization Complete ({elapsed:.1f}s)")
    print(f"Best Loss: {result.fun:.6f}")
    print(f"Best Params: {best_params}")
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "bayesian_opt_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'method': 'bayesian_optimization',
                'best_params': best_params,
                'best_loss': float(result.fun),
                'n_calls': n_calls,
                'elapsed_time': elapsed
            }, f, indent=2)
    
    return {
        'method': 'bayesian_optimization',
        'best_params': best_params,
        'best_loss': float(result.fun),
        'result': result,
        'elapsed_time': elapsed
    }