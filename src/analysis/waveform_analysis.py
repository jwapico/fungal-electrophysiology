from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
import numpy as np
import matplotlib
import argparse
import scipy
import re

from waveform_visualization import gen_spike_waveform_html, gen_channel_html
import constants

matplotlib.use('Agg')

# gaussian window sizing
GAUSS_PARAM_P0: List[float] = [-1, 0.6, 0.1, 1, 1, 0.3]
GAUSS_PARAM_LB: List[float] = [-1000, 0.5, 0, 0, 0.6, 0]
GAUSS_PARAM_UB: List[float] = [0, 0.7, 1, 1000, 1.4, 1]
GAUSS_MAX_EVALS: int = 5000
GAUSS_WINDOW_MULT: int = 8                # multiplier for gaussian σ (scales window width)

# waveform extraction parameters
SEARCH_WINDOW: int = 50                   # samples to search for true peak
INITIAL_WINDOW: int = 30                  # initial window for gaussian fitting
WINDOW_TARGET_WIDTH: int = 160            # pad/truncate all waveforms to this length
USE_GAUSSIAN_WIDTH = True                 # this will override a target width
SMOOTHING_WINDOW_LEN: int = 15            # window size for sliding mean smoothing

# Brian2 model voltage range (need for normalization)
BRIAN2_V_MIN: float = -70.0               # mV
BRIAN2_V_MAX: float = 30.0                # mV


# runs pipeline
def main(args) -> None:
    print("=" * 60)
    print("MEA Spike Analysis Pipeline")
    print("=" * 60)
    print(f"Data file: {constants.RAW_DATA_FILE}")
    print(f"Output: {constants.OUTPUT_HTML}")
    print(f"Save waveforms: {constants.SAVE_WAVEFORMS}")
    print(f"  For Brian2: {constants.SAVE_FOR_BRIAN2} (normalized to -70/30 mV)")
    print(f"  Raw μV: {constants.SAVE_RAW_MICROVOLTS}")
    print("=" * 60)

    if args.visualize_only:
        results = load_existing_waveforms(constants.WAVEFORM_OUTPUT_DIR)

        if not results:
            print("Error: No data loaded. Run without -s flag first.")
            return
    else:
        data = load_raw_data(constants.RAW_DATA_FILE)
        
        # process each channel in raw data
        results = {}
        for ch in range(constants.NUM_CHANNELS):
            result = process_channel(data, ch, normalize_for_brian2=constants.SAVE_FOR_BRIAN2)
            results[ch] = result
        
        # save waveforms if requested
        if constants.SAVE_WAVEFORMS and constants.WAVEFORM_OUTPUT_DIR:
            save_waveforms(
                results, 
                constants.WAVEFORM_OUTPUT_DIR,
                save_brian2=constants.SAVE_FOR_BRIAN2,
                save_raw=constants.SAVE_RAW_MICROVOLTS
            )
        
    # raw data in μV but normalized to mV if using Brian2
    unit_label = "mV (normalized)" if constants.SAVE_FOR_BRIAN2 else "μV"
    
    # pretty HTML visualization
    gen_spike_waveform_html(results, constants.OUTPUT_HTML, unit_label=unit_label)
    gen_channel_html()
    
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Open {constants.OUTPUT_HTML} in your browser to view waveforms")
    print("=" * 60)


# ========================================
#      core data processing functions
# ========================================

def load_raw_data(filepath: str) -> np.ndarray:
    """
    Load raw MEA binary data file.
    
    Returns:
        data: 2D array of shape (n_samples, n_channels)
    """
    print(f"Loading raw data from: {filepath}")
    data = np.memmap(filepath, dtype=constants.BINARY_DTYPE, mode='r')
    data = data.reshape(-1, constants.NUM_CHANNELS)
    print(f"  Loaded {data.shape[0]:,} samples, {data.shape[1]} channels")
    return data


def detect_spikes(voltage: np.ndarray, st_dev_scale: float = constants.ST_DEV_SCALE,
                  min_distance: int = constants.MIN_SPIKE_DISTANCE,
                  lower_factor: float = constants.LOWER_THRESHOLD_FACTOR
                  ) -> Tuple[np.ndarray, float, float]:
    """
    Detect spikes in voltage trace using prominence-based peak detection.
    
    Returns:
        peak_indices: array of spike indices
        threshold: detection threshold used
        std_dev: noise standard deviation
    """
    std_dev = np.std(voltage)
    threshold = st_dev_scale * std_dev
    
    peak_indices, _ = scipy.signal.find_peaks(
        voltage, 
        prominence=threshold,
        distance=min_distance
    )
    
    # lower threshold if no spikes found
    if len(peak_indices) == 0:
        fallback_threshold = threshold * lower_factor
        peak_indices, _ = scipy.signal.find_peaks(
            voltage,
            prominence=fallback_threshold,
            distance=min_distance
        )
        threshold = fallback_threshold
    
    return peak_indices, float(threshold), float(std_dev)


def extract_waveform(spike_idx: int, voltage: np.ndarray,
                    search_window: int = SEARCH_WINDOW,
                    initial_window: int = INITIAL_WINDOW,
                    min_window_sec: float = 0.005,
                    normalize: bool = False
                    ) -> Optional[Tuple[np.ndarray, float, int]]:
    """Extract waveform with adaptive window sizing."""
    if (spike_idx < search_window or spike_idx >= len(voltage) - search_window):
        return None
    
    init_start = spike_idx - initial_window
    init_end = spike_idx + initial_window
    spike_segment = voltage[init_start:init_end]
    
    if len(spike_segment) < 10:
        return None
    
    window_samples, _ = fit_gaussian_window(spike_segment)

    min_window_samples = int(min_window_sec * constants.SAMPLE_RATE_HZ)
    if window_samples < min_window_samples:
        return None

    start_idx, end_idx = find_peak_and_center(spike_idx, voltage, window_samples, search_window)
    
    if start_idx < 0 or end_idx >= len(voltage):
        return None
    
    waveform = voltage[start_idx:end_idx]
    
    if len(waveform) < 5:
        return None
    
    waveform = smooth_waveform(waveform, sliding_mean_smoothing, window_length=SMOOTHING_WINDOW_LEN)
    
    if normalize:
        waveform = normalize_for_brian2(waveform)
    
    # calculate time from peak position
    actual_peak_idx = (start_idx + end_idx) / 2.0
    spike_time_sec = actual_peak_idx / constants.SAMPLE_RATE_HZ
    
    return waveform, spike_time_sec, window_samples


def smooth_waveform(
    waveform: np.ndarray,
    smooth_func: Callable[..., np.ndarray],
    **kwargs
) -> np.ndarray:
    """
    smoothing with provided function.
    
    Parameters:
    -----------
    waveform : np.ndarray
        Input waveform to smooth
    smooth_func : Callable
        Smoothing function. Common options:
        - scipy.ndimage.uniform_filter1d   (simple sliding average)
        - scipy.signal.savgol_filter       (Savitzky-Golay - preserves peaks)
        - scipy.ndimage.gaussian_filter1d (Gaussian smoothing)
    **kwargs :
        Keyword arguments passed to smooth_func
        
    Returns:
    --------
    np.ndarray: Smoothed waveform
    """
    if len(waveform) < 3:
        return waveform
    
    return smooth_func(waveform, **kwargs)


def sliding_mean_smoothing(
    waveform: np.ndarray,
    window_length: int = 3
) -> np.ndarray:
    """
    Smooth waveform using sliding window mean (simple moving average).
    
    Parameters
    ----------
    waveform : np.ndarray
        Input waveform to smooth
    window_size : int
        Size of sliding window (must be odd for symmetric window). Default: 3
        
    Returns
    -------
    np.ndarray
        Smoothed waveform of same length as input
    """
    if window_length < 1:
        raise ValueError("window_size must be >= 1")
    
    if window_length % 2 == 0:
        window_length += 1
    
    if len(waveform) < window_length:
        return waveform.copy()
    
    smoothed = np.zeros_like(waveform)
    half_window = window_length // 2
    
    for i in range(len(waveform)):
        start = max(0, i - half_window)
        end = min(len(waveform), i + half_window + 1)
        smoothed[i] = np.mean(waveform[start:end])
    
    return smoothed


def process_channel(data: np.ndarray, channel: int,
                   normalize_for_brian2: bool = False
                   ) -> Dict[str, Any]:
    """Process single channel: detect spikes and extract waveforms."""
    print(f"\nProcessing channel {channel}...")
    
    voltage = -data[:, channel] * constants.VOLTAGE_SCALE
    
    peak_indices, threshold, std_dev = detect_spikes(voltage)
    n_detected = len(peak_indices)
    
    print(f"  Detected {n_detected} spikes (threshold: {threshold:.2f} μV)")
    
    waveforms = []
    spike_times = []
    window_sizes = []
    spike_indices_valid = []
    
    # lazy import yikes im lazy
    from classification import fit_single_waveform, GMMClassifier
    for spike_idx in peak_indices:
        result = extract_waveform(
            spike_idx, voltage,
            search_window=SEARCH_WINDOW,
            initial_window=INITIAL_WINDOW,
            normalize=normalize_for_brian2
        )
        
        if result:
            waveform, spike_time, win_size = result
            coefs = fit_single_waveform(waveform)

            # filter out waveforms with null coefs and time_to_pp < 0 (physically impossible - means something went wrong in waveform extraction)
            if not np.any(np.isnan(coefs)):
                features = GMMClassifier.extract_features(coefs.reshape(1, -1))
                time_to_pp = features[0, 0]
                if time_to_pp > 0:
                    waveforms.append(waveform)
                    spike_times.append(spike_time)
                    window_sizes.append(win_size)
                    spike_indices_valid.append(spike_idx)
    
    n_extracted = len(waveforms)
    print(f"  Extracted {n_extracted} waveforms")
    
    return {
        'waveforms': np.array(waveforms, dtype=object),
        'spike_times': np.array(spike_times),
        'window_sizes': np.array(window_sizes),
        'spike_indices': np.array(spike_indices_valid),
        'threshold': threshold,
        'std_dev': std_dev,
        'n_detected': n_detected,
        'n_extracted': n_extracted
    }


# ==========================
#      waveform helpers
# ==========================

def check_edge_voltage_difference(waveform, max_diff_mv):
    """Check if voltage difference between start and end of waveform is within threshold"""
    edge_diff = abs(waveform[0] - waveform[-1])
    return edge_diff <= max_diff_mv


def pad_waveform_to_length(waveform, target_length):
    if len(waveform) >= target_length:
        return waveform[:target_length]
    
    pad_needed = target_length - len(waveform)
    pad_left = pad_needed // 2
    pad_right = pad_needed - pad_left
    
    padded = np.concatenate([
        np.full(pad_left, waveform[0]),
        waveform,
        np.full(pad_right, waveform[-1])
    ])
    return padded


def pad_waveform(waveform: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or truncate waveform to target length.
    
    Pads with edge values, truncates from center if too long.
    """
    current_length = len(waveform)
    
    if current_length >= target_length:
        # Truncate: take center portion
        start = (current_length - target_length) // 2
        return waveform[start:start + target_length]
    
    # Pad with edge values
    pad_needed = target_length - current_length
    pad_left = pad_needed // 2
    pad_right = pad_needed - pad_left
    
    return np.concatenate([
        np.full(pad_left, waveform[0]),
        waveform,
        np.full(pad_right, waveform[-1])
    ])


def save_waveforms(results: Dict[int, Dict[str, Any]], 
                  output_dir: str,
                  save_brian2: bool = True,
                  save_raw: bool = False) -> None:
    """Save extracted waveforms to numpy files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving waveforms to: {output_dir}")
    
    for channel, data in results.items():
        waveforms = data['waveforms']
        spike_times = data['spike_times']
        window_sizes = data['window_sizes']
        
        if len(waveforms) == 0:
            continue
        
        save_data = {
            'waveforms': waveforms,            # Variable length arrays
            'spike_times': spike_times,       # seconds
            'window_sizes': window_sizes,     # Actual samples per spike
            'channel': channel,
            'n_waveforms': len(waveforms),
            'sample_rate': constants.SAMPLE_RATE_HZ,
            'normalized': save_brian2,
            'unit': 'mV (Brian2)' if save_brian2 else 'μV',
            'variable_length': True
        }
        
        filename = f"ch_{channel}_spikes.npy"
        np.save(output_path / filename, save_data) # type: ignore
        print(f"  Channel {channel}: {len(waveforms)} waveforms -> {filename}")


def load_existing_waveforms(output_dir: str) -> Dict[int, Dict[str, Any]]:
    """
    Load previously extracted waveforms from .npy files.
    
    Returns:
        results: dict mapping channel -> data dict with same structure as process_channel
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        return {}
    
    npy_files = sorted(output_path.glob("ch_*_spikes.npy"))
    if not npy_files:
        print(f"Warning: No .npy files found in {output_dir}")
        return {}
    
    print(f"Loading {len(npy_files)} existing waveform files...")
    
    results = {}
    for f in npy_files:
        # filename format like "ch_16_spikes.npy"
        match = re.search(r'ch_(\d+)_spikes\.npy', f.name)
        if not match:
            print(f"  Warning: Could not parse channel from {f.name}")
            continue
        
        channel = int(match.group(1))
        try:
            data = np.load(f, allow_pickle=True).item()
            
            results[channel] = {
                'waveforms': data['waveforms'],
                'spike_times': data['spike_times'],
                'window_sizes': data['window_sizes'],
                'spike_indices': np.array([]),
                'threshold': 0.0,
                'std_dev': 0.0,
                'n_detected': len(data['waveforms']),
                'n_extracted': len(data['waveforms'])
            }
            
            print(f"  Loaded channel {channel}: {len(data['waveforms'])} waveforms")
            
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
    
    return results


def get_output_filename(cluster_id, sample_rate, ratio, filter_method, percentile, output_dir="data/waveforms"):
    suffix = f"_top{percentile}" if filter_method != "none" else ""
    method_suffix = f"_{filter_method}" if filter_method != "none" else ""
    return Path(output_dir) / f"spike_waveforms_c{cluster_id}_sr{sample_rate}_ratio{ratio}{method_suffix}{suffix}.npy"


# ==================================
#    feature extraction helpers
# ==================================

def gaussian2(x: np.ndarray, a1: float, b1: float, c1: float,
              a2: float, b2: float, c2: float) -> np.ndarray:
    """
    Sum of two Gaussians for waveform fitting.
    
    Used to model spike shape and determine optimal extraction window.
    """
    return (a1 * np.exp(-((x - b1) / c1) ** 2) + 
            a2 * np.exp(-((x - b2) / c2) ** 2))


def fit_gaussian_window(spike_segment: np.ndarray, 
                        sample_rate: int = constants.SAMPLE_RATE_HZ) -> Tuple[int, bool]:
    """
    Fit two Gaussians to spike segment to determine optimal window size.
    
    Returns:
        window_samples: optimal window size in samples
        success: whether fitting succeeded
    """
    n_samples = len(spike_segment)
    time = np.arange(n_samples) / sample_rate  # convert to seconds
    
    # normalize
    spike_norm = spike_segment / (np.max(np.abs(spike_segment)) + 1e-10)
    
    try:
        popt, _ = scipy.optimize.curve_fit(
            gaussian2, time, spike_norm,
            p0=GAUSS_PARAM_P0,
            bounds=(GAUSS_PARAM_LB, GAUSS_PARAM_UB),
            maxfev=GAUSS_MAX_EVALS
        )
        c1, c2 = abs(popt[2]), abs(popt[5])
        window_ms = GAUSS_WINDOW_MULT * (c1 + c2)
        window_samples = int(window_ms * sample_rate / 1000)
        return max(window_samples, 10), True
    except Exception:
        print("gaussian fitting failed using 2 ms window")
        return int(2 * sample_rate // 1000), False


def find_peak_and_center(spike_idx: int, voltage: np.ndarray,
                         window_samples: int, 
                         search_window: int) -> Tuple[int, int]:
    """
    Find true peak within search window and center extraction around it.
    
    Returns:
        start_idx, end_idx: extraction window boundaries
    """
    search_start = spike_idx - search_window
    search_end = spike_idx + search_window
    
    # edge cases
    if search_start < 0 or search_end >= len(voltage):
        return spike_idx - window_samples, spike_idx + window_samples
    
    search_segment = voltage[search_start:search_end]
    mean_v = np.mean(search_segment)
    peak_offset = np.argmax(np.abs(search_segment - mean_v))
    actual_peak_idx = search_start + peak_offset
    
    start_idx = actual_peak_idx - window_samples // 2
    end_idx = actual_peak_idx + window_samples // 2
    
    return int(start_idx), int(end_idx)


def calculate_spike_metrics(waveforms):
    amplitudes = np.array([w.max() - w.min() for w in waveforms])
    noises = np.array([np.std(w) for w in waveforms])
    snrs = amplitudes / (noises + 1e-10)
    return amplitudes, snrs


def normalize_for_brian2(waveform: np.ndarray) -> np.ndarray:
    """
    Normalize waveform to Brian2 Izhikevich model voltage range.
    
    Maps min to -70 mV, max to 30 mV.
    """
    v_min, v_max = waveform.min(), waveform.max()
    if v_max <= v_min:
        return waveform
    
    return ((waveform - v_min) / (v_max - v_min) * 
            (BRIAN2_V_MAX - BRIAN2_V_MIN) + BRIAN2_V_MIN)


# entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='MEA Spike Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-v', '--visualize-only',
        action='store_true',
        help='Only generate HTML visualization (Data already analyzed)'
    )

    args = parser.parse_args()
    main(args)