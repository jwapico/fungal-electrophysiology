"""
Synthetic data generation from Izhikevich neuron model.
Simulates neuron, extracts spikes, computes features matching real data.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings

# Izhikevich model constants (fixed)
K1 = 0.04
K2 = 5.0
K3 = 140.0
SPIKE_THRESH = 30.0
V_REST = -70.0
V_PEAK = 30.0

# Feature extraction constants
SAMPLE_RATE_HZ = 30000
DT_MS = 0.5
MS_PER_SAMPLE = 1000.0 / SAMPLE_RATE_HZ

def simulate_izhikevich(
    a: float, b: float, c: float, d: float,
    I_bias: float, sigma: float,
    duration_ms: float = 60000.0,
    dt: float = DT_MS,
    v0: float = -70.0,
    record_voltage: bool = True
) -> Dict:
    """
    Simulate a single Izhikevich neuron.
    
    dv/dt = k1*v^2 + k2*v + k3 - u + I_total
    du/dt = a*(b*v - u)
    if v >= 30: v = c, u = u + d
    
    Args:
        a, b, c, d: Izhikevich parameters
        I_bias: Baseline input current (mV/ms)
        sigma: Noise standard deviation (mV/sqrt(ms))
        duration_ms: Simulation duration (ms)
        dt: Time step (ms)
        v0: Initial membrane potential (mV)
        record_voltage: Whether to record full voltage trace
    
    Returns:
        Dict with 'time', 'v_history', 'u_history', 'spike_times'
    """
    steps = int(duration_ms / dt)
    time = np.arange(0, duration_ms, dt)
    
    v = v0
    u = b * v  # Initial u
    
    v_history = np.zeros(steps) if record_voltage else None
    u_history = np.zeros(steps) if record_voltage else None
    spike_times = []
    
    for i in range(steps):
        # Compute input current (constant + noise)
        noise = sigma * np.random.randn() * np.sqrt(dt)
        I_total = I_bias + noise
        
        # Detect spike
        if v >= SPIKE_THRESH:
            v = c
            u = u + d
            spike_times.append(time[i])
        
        # Izhikevich update
        dv = (K1 * v**2 + K2 * v + K3 - u + I_total) * dt
        du = a * (b * v - u) * dt
        
        v += dv
        u += du
        
        if record_voltage:
            v_history[i] = v
            u_history[i] = u
    
    return {
        'time': time,
        'v_history': v_history,
        'u_history': u_history,
        'spike_times': np.array(spike_times),
        'params': {'a': a, 'b': b, 'c': c, 'd': d, 'I_bias': I_bias, 'sigma': sigma}
    }

def extract_waveform_window(
    v_trace: np.ndarray,
    spike_time_idx: int,
    window_samples: int = 30
) -> np.ndarray:
    """
    Extract a window of voltage around a spike.
    Returns window centered on spike with total length window_samples.
    """
    half_window = window_samples // 2
    start = max(0, spike_time_idx - half_window)
    end = min(len(v_trace), spike_time_idx + half_window)
    
    window = v_trace[start:end]
    
    # Pad if necessary
    if len(window) < window_samples:
        padded = np.full(window_samples, window[0] if len(window) > 0 else 0.0)
        padded[:len(window)] = window
        return padded
    
    return window

def fit_gaussian2(x: np.ndarray, y: np.ndarray, p0: list, bounds: tuple) -> np.ndarray:
    """Fit sum of two Gaussians to data."""
    try:
        from scipy.optimize import curve_fit
        
        def gaussian2(x, a1, b1, c1, a2, b2, c2):
            return a1 * np.exp(-((x - b1) / c1)**2) + a2 * np.exp(-((x - b2) / c2)**2)
        
        popt, _ = curve_fit(gaussian2, x, y, p0=p0, bounds=bounds, maxfev=10000)
        return popt
    except Exception:
        return np.full(6, np.nan)

def extract_snyder_features(coefs: np.ndarray) -> np.ndarray:
    """Extract Snyder features from 12 Gaussian coefficients."""
    if np.any(np.isnan(coefs)):
        return np.full(3, np.nan)
    time_to_pp = coefs[4]
    pp_duration = coefs[5]
    fall_rise = coefs[9] + coefs[6]
    return np.array([time_to_pp, pp_duration, fall_rise])

def compute_waveform_features(v_window: np.ndarray) -> np.ndarray:
    """
    Compute 8 features from a single waveform window.
    Features: amplitude, spike_width_fwhm, asymmetry, area, snr, 
              time_to_pp, pp_duration, fall_rise
    """
    if len(v_window) == 0:
        return np.full(8, np.nan)
    
    # Basic features
    amplitude = float(v_window.max() - v_window.min())
    
    # Spike width (FWHM)
    v_min = float(v_window.min())
    v_max = float(v_window.max())
    if v_max > v_min:
        half_max = v_min + 0.5 * (v_max - v_min)
        above = v_window >= half_max
        diff = np.diff(above.astype(int))
        cross_down = np.where(diff == -1)[0]
        cross_up = np.where(diff == 1)[0]
        if len(cross_down) > 0 and len(cross_up) > 0:
            width_samples = cross_up[-1] - cross_down[0]
            spike_width = float(width_samples * MS_PER_SAMPLE)
        else:
            spike_width = np.nan
    else:
        spike_width = np.nan
    
    # Asymmetry index
    min_idx = int(np.argmin(v_window))
    n = len(v_window)
    asymmetry = float(min_idx / (n - min_idx)) if min_idx > 0 and min_idx < n - 1 else np.nan
    
    # Area under curve
    from scipy.integrate import simpson
    time_axis = np.arange(len(v_window)) * MS_PER_SAMPLE
    area = float(simpson(np.abs(v_window), x=time_axis))
    
    # SNR
    peak_to_peak = v_window.max() - v_window.min()
    noise = np.std(v_window)
    snr = float(peak_to_peak / (2 * noise)) if noise > 0 else np.nan
    
    # Snyder features (fit Gaussians)
    n_samples = len(v_window)
    time_x = np.arange(n_samples) / (SAMPLE_RATE_HZ / 1000.0)
    
    p0_raw = [-1, 0.6, 0.1, 1, 1, 0.3]
    bounds_raw = ([-10, -10, 0.0, -10, -10, 0.1], [0, 10, 2.0, 10, 10, 10])
    p0_deriv = [-45, 0.5, 0.1, 45, 0.5, 0.1]
    bounds_deriv = ([-50, 0.0, 0.1, 0, 0.0, 0.1], [0, 2.0, 5.0, 50, 2.0, 5.0])
    
    try:
        coefs_raw = fit_gaussian2(time_x, v_window, p0_raw, bounds_raw)
        
        diff_wave = np.diff(v_window, axis=0) if v_window.ndim == 1 else np.diff(v_window)
        coefs_deriv = fit_gaussian2(time_x[:-1], diff_wave, p0_deriv, bounds_deriv)
        
        coefs_all = np.concatenate([coefs_raw, coefs_deriv])
        snyder = extract_snyder_features(coefs_all)
    except Exception:
        snyder = np.full(3, np.nan)
    
    return np.array([
        amplitude, spike_width, asymmetry, area, snr,
        snyder[0], snyder[1], snyder[2]
    ])

def extract_synthetic_features(
    simulation_result: Dict,
    feature_names: list = None
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Extract all features from a simulation result.
    
    Returns:
        features: (N_spikes, 8) array of waveform features
        isi: Inter-spike intervals (ms)
        firing_rate: Firing rate (Hz)
        spike_times: Array of spike times (ms)
    """
    v_history = simulation_result['v_history']
    spike_times = simulation_result['spike_times']
    
    if len(spike_times) < 2:
        # Not enough spikes
        return np.array([]).reshape(0, 8), np.array([]), 0.0, spike_times
    
    # Extract waveform windows around each spike
    dt = simulation_result['time'][1] - simulation_result['time'][0]
    window_ms = 3.0  # 3 ms window
    window_samples = int(window_ms / dt)
    
    # Sample a subset of spikes for faster feature extraction
    n_sample = min(100, len(spike_times))  # Limit to 100 spikes
    if len(spike_times) > n_sample:
        idx_sample = np.random.choice(len(spike_times), n_sample, replace=False)
        spike_times_sampled = spike_times[idx_sample]
    else:
        spike_times_sampled = spike_times

    features_list = []
    for index, st in enumerate(spike_times_sampled):  # Use sampled spikes
        if index % 10 == 0:
            print(f"{index} / {len(spike_times_sampled)}")
        idx = int(st / dt)
        window = extract_waveform_window(v_history, idx, window_samples)
        feats = compute_waveform_features(window)
        if not np.any(np.isnan(feats)):
            features_list.append(feats)
    
    features = np.array(features_list) if features_list else np.array([]).reshape(0, 8)
    
    # Compute ISI
    spike_times_sorted = np.sort(spike_times)
    isi = np.diff(spike_times_sorted)
    
    # Firing rate
    duration_s = simulation_result['time'][-1] / 1000.0
    firing_rate = len(spike_times) / duration_s if duration_s > 0 else 0.0

    return features, isi, firing_rate, spike_times

def generate_synthetic_dataset(
    params: Dict[str, float],
    duration_ms: float = 60000.0,
    dt: float = DT_MS
) -> Dict:
    """
    Generate a complete synthetic dataset with features.
    
    Args:
        params: Dict with keys 'a', 'b', 'c', 'd', 'I_bias', 'sigma'
        duration_ms: Simulation duration
        dt: Time step
    
    Returns:
        Dict with 'features', 'isi', 'firing_rate', 'spike_times', 'simulation'
    """
    # Run simulation
    sim = simulate_izhikevich(
        a=params['a'], b=params['b'], c=params['c'], d=params['d'],
        I_bias=params['I_bias'], sigma=params['sigma'],
        duration_ms=duration_ms, dt=dt
    )
    
    # Extract features
    features, isi, firing_rate, spike_times = extract_synthetic_features(sim)
    
    return {
        'features': features,
        'isi': isi,
        'firing_rate': firing_rate,
        'spike_times': spike_times,
        'simulation': sim
    }

def simulate_with_params(
    params_dict: Dict,
    duration_ms: float = 60000.0,
    dt: float = DT_MS
) -> Dict:
    """
    Simulate Izhikevich neuron with specific parameters.
    Returns synthetic data in the same format as generate_synthetic_dataset.
    """
    # Extract parameters
    a = params_dict.get('a', 0.02)
    b = params_dict.get('b', 0.2)
    c = params_dict.get('c', -65.0)
    d = params_dict.get('d', 8.0)
    I_bias = params_dict.get('I_bias', 10.0)
    sigma = params_dict.get('sigma', 1.0)
    
    # Simulate
    sim_result = simulate_izhikevich(
        a=a, b=b, c=c, d=d,
        I_bias=I_bias, sigma=sigma,
        duration_ms=duration_ms, dt=dt
    )
    
    # Extract features
    features, isi, firing_rate, spike_times = extract_synthetic_features(sim_result)
    
    return {
        'features': features,
        'isi': isi,
        'firing_rate': firing_rate,
        'spike_times': spike_times,
        'params': params_dict
    }