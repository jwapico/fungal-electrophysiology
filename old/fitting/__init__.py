"""
Fitting package for Izhikevich parameter optimization.
Contains from-scratch and Brian2-based optimization methods.
"""

from .loss_functions import (
    waveform_loss, temporal_loss, combined_loss,
    compute_mmd, wasserstein_distance_1d
)
from .synthetic_generator import (
    simulate_izhikevich, extract_synthetic_features,
    generate_synthetic_dataset
)
from .performance_metrics import (
    compute_all_metrics, visualize_comparison,
    check_parameter_plausibility
)
from .fit_from_scratch import (
    grid_search, random_search, differential_evolution_search,
    cma_es_search, particle_swarm_search, bayesian_optimization
)
from .fit_brian2 import (
    run_trace_fitter, run_spike_fitter, run_feature_metric_fitter
)

__version__ = "1.0.0"

__all__ = [
    # Loss functions
    'waveform_loss', 'temporal_loss', 'combined_loss',
    'compute_mmd', 'wasserstein_distance_1d',
    # Synthetic generator
    'simulate_izhikevich', 'extract_synthetic_features',
    'generate_synthetic_dataset',
    # Metrics
    'compute_all_metrics', 'visualize_comparison',
    'check_parameter_plausibility',
    # From-scratch optimizers
    'grid_search', 'random_search', 'differential_evolution_search',
    'cma_es_search', 'particle_swarm_search', 'bayesian_optimization',
    # Brian2 optimizers
    'run_trace_fitter', 'run_spike_fitter', 'run_feature_metric_fitter',
]