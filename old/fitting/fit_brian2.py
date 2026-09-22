"""
Brian2 + brian2modelfitting optimization methods.
Includes TraceFitter, SpikeFitter, and FeatureMetric approaches.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path

def run_trace_fitter(
    real_voltage_trace: np.ndarray,
    real_time: np.ndarray,
    bounds: Dict[str, Tuple[float, float]] = None,
    optimizer: str = 'diff_evol',
    n_rounds: int = 10,
    n_sample: int = 30,
    duration_ms: float = 60000.0,
    output_dir: Path = None
) -> Dict:
    """
    Use Brian2 TraceFitter to fit parameters to voltage trace.
    
    Args:
        real_voltage_trace: Target voltage trace (mV)
        real_time: Time points for target trace (ms)
        bounds: Parameter bounds
        optimizer: 'diff_evol', 'grid_search', 'cma' (from brian2modelfitting)
        n_rounds: Number of optimization rounds
        n_sample: Sample size per round
    """
    try:
        from brian2 import NeuronGroup, StateMonitor, run, ms, mV, nS, pA, start_scope, defaultclock
        from brian2modelfitting import TraceFitter
    except ImportError:
        print("Brian2 or brian2modelfitting not installed.")
        return None
    
    if bounds is None:
        from .fit_from_scratch import DEFAULT_BOUNDS
        bounds = DEFAULT_BOUNDS
    
    print(f"\n{'='*60}")
    print(f"BRIAN2 TRACEFITTER")
    print(f"{'='*60}")
    
    # Define Izhikevich model in Brian2
    model_eqs = '''
    dv/dt = 0.04*v**2 + 5*v + 140 - u + I_input : 1
    du/dt = a*(b*v - u) : 1
    I_input : 1
    a : 1
    b : 1
    '''
    
    # Reset equations
    reset_eqs = '''
    v = c
    u = u + d
    '''
    
    start_scope()
    
    # Create neuron group
    neuron = NeuronGroup(
        1, model_eqs, method='euler',
        threshold='v > 30', reset=reset_eqs,
        namespace={'c': -65.0, 'd': 8.0}  # Default values
    )
    
    # Create monitors
    monitor = StateMonitor(neuron, ['v', 'u'], record=True)
    
    # Setup TraceFitter
    fitter = TraceFitter(
        model=model_eqs,
        reset=reset_eqs,
        input_var='I_input',
        output_var='v',
        dt=defaultclock.dt
    )
    
    # Set target trace
    fitter.set_target(real_voltage_trace, dt=defaultclock.dt)
    
    # Set parameter bounds
    param_bounds = {name: bounds[name] for name in ['a', 'b', 'c', 'd', 'I_bias']}
    
    # Run optimization
    start_time = np.datetime64('now')
    
    result = fitter.fit(
        optimizer=optimizer,
        bounds=param_bounds,
        n_rounds=n_rounds,
        n_sample=n_sample,
        sigma=0.1  # Noise parameter
    )
    
    elapsed = (np.datetime64('now') - start_time).item().total_seconds()
    
    best_params = {name: result.x[i] for i, name in enumerate(['a', 'b', 'c', 'd', 'I_bias'])}
    
    print(f"\nTraceFitter Complete")
    print(f"Best Params: {best_params}")
    print(f"Best Loss: {result.fun:.6f}")
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        from .performance_metrics import save_metrics
        save_metrics({
            'method': 'trace_fitter',
            'best_params': best_params,
            'best_loss': float(result.fun),
            'optimizer': optimizer,
            'elapsed_time': elapsed
        }, output_dir / "trace_fitter_results.json")
    
    return {
        'method': 'trace_fitter',
        'best_params': best_params,
        'best_loss': float(result.fun),
        'result': result,
        'elapsed_time': elapsed
    }

def run_spike_fitter(
    real_spike_times: np.ndarray,
    bounds: Dict[str, Tuple[float, float]] = None,
    optimizer: str = 'diff_evol',
    n_rounds: int = 10,
    n_sample: int = 30,
    duration_ms: float = 60000.0,
    output_dir: Path = None
) -> Dict:
    """
    Use Brian2 SpikeFitter to fit parameters to spike times.
    """
    try:
        from brian2modelfitting import SpikeFitter
    except ImportError:
        print("brian2modelfitting not installed.")
        return None
    
    if bounds is None:
        from .fit_from_scratch import DEFAULT_BOUNDS
        bounds = DEFAULT_BOUNDS
    
    print(f"\n{'='*60}")
    print(f"BRIAN2 SPIKEFITTER")
    print(f"{'='*60}")
    
    # Define model
    model_eqs = '''
    dv/dt = 0.04*v**2 + 5*v + 140 - u + I_input : 1
    du/dt = a*(b*v - u) : 1
    I_input : 1
    a : 1
    b : 1
    '''
    reset_eqs = '''
    v = c
    u = u + d
    '''
    
    # Setup SpikeFitter
    fitter = SpikeFitter(
        model=model_eqs,
        reset=reset_eqs,
        input_var='I_input',
        threshold='v > 30',
        refractory=0*ms,
        dt=0.5*ms
    ) # type: ignore
    
    # Set target spikes
    fitter.set_target(real_spike_times, dt=0.5*ms)
    
    # Set bounds
    param_bounds = {name: bounds[name] for name in ['a', 'b', 'c', 'd', 'I_bias']}
    
    # Run optimization
    result = fitter.fit(
        optimizer=optimizer,
        bounds=param_bounds,
        n_rounds=n_rounds,
        n_sample=n_sample
    )
    
    best_params = {name: result.x[i] for i, name in enumerate(['a', 'b', 'c', 'd', 'I_bias'])}
    
    print(f"\nSpikeFitter Complete")
    print(f"Best Params: {best_params}")
    print(f"Best Loss: {result.fun:.6f}")
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        from .performance_metrics import save_metrics
        save_metrics({
            'method': 'spike_fitter',
            'best_params': best_params,
            'best_loss': float(result.fun),
            'optimizer': optimizer
        }, output_dir / "spike_fitter_results.json")
    
    return {
        'method': 'spike_fitter',
        'best_params': best_params,
        'best_loss': float(result.fun),
        'result': result
    }

def run_feature_metric_fitter(
    real_features: np.ndarray,
    feature_names: List[str],
    bounds: Dict[str, Tuple[float, float]] = None,
    optimizer: str = 'diff_evol',
    n_rounds: int = 10,
    n_sample: int = 30,
    duration_ms: float = 60000.0,
    output_dir: Path = None
) -> Dict:
    """
    Use Brian2 FeatureMetric with eFEL features.
    """
    try:
        from brian2modelfitting import FeatureMetric
        import efel
    except ImportError:
        print("brian2modelfitting or efel not installed.")
        return None
    
    if bounds is None:
        from .fit_from_scratch import DEFAULT_BOUNDS
        bounds = DEFAULT_BOUNDS
    
    print(f"\n{'='*60}")
    print(f"BRIAN2 FEATUREMETRIC (eFEL)")
    print(f"{'='*60}")
    
    # Define model
    model_eqs = '''
    dv/dt = 0.04*v**2 + 5*v + 140 - u + I_input : 1
    du/dt = a*(b*v - u) : 1
    I_input : 1
    a : 1
    b : 1
    '''
    reset_eqs = '''
    v = c
    u = u + d
    '''
    
    # Setup FeatureMetric
    fitter = FeatureMetric(
        model=model_eqs,
        reset=reset_eqs,
        input_var='I_input',
        threshold='v > 30',
        refractory=0*ms,
        dt=0.5*ms
    )  # type: ignore
    
    # Define eFEL features to extract
    efel_features = ['AP_amplitude', 'AP_duration', 'AHP_depth']
    
    # Set target features
    target_feature_values = np.mean(real_features, axis=0)[:3]  # First 3 features
    fitter.set_target(feature_names[:3], target_feature_values)
    
    # Set bounds
    param_bounds = {name: bounds[name] for name in ['a', 'b', 'c', 'd', 'I_bias']}
    
    # Run optimization
    result = fitter.fit(
        optimizer=optimizer,
        bounds=param_bounds,
        n_rounds=n_rounds,
        n_sample=n_sample
    )
    
    best_params = {name: result.x[i] for i, name in enumerate(['a', 'b', 'c', 'd', 'I_bias'])}
    
    print(f"\nFeatureMetric Complete")
    print(f"Best Params: {best_params}")
    print(f"Best Loss: {result.fun:.6f}")
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        from .performance_metrics import save_metrics
        save_metrics({
            'method': 'feature_metric',
            'best_params': best_params,
            'best_loss': float(result.fun),
            'optimizer': optimizer,
            'efel_features': efel_features
        }, output_dir / "feature_metric_results.json")
    
    return {
        'method': 'feature_metric',
        'best_params': best_params,
        'best_loss': float(result.fun),
        'result': result
    }