"""
Initially based on (Gaussian feature methodology from):

Snyder AC, Morais MJ, Smith MA (2016)
Journal of Neurophysiology.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Tuple
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
import numpy as np
import argparse
import json
import joblib

from waveform_analysis import gaussian2, load_existing_waveforms

# TODO: filter out spikes where time to PP is < 0, there must be going wrong somewhere here because this should be impossible under how weve defined the model
#   code should go in analysis pipeline these points should be thrown out or analyzed to see whats wrong

# ===================
#      constants
# ===================

SAMPLE_RATE_HZ: int = 30000
BINARY_DTYPE: str = "int16"
NUM_CHANNELS: int = 64
VOLTAGE_SCALE: float = 0.195

WAVEFORM_INPUT_DIR: str = "outputs/waveforms"
BASE_OUTPUT_DIR: Path = Path("outputs/classification")

RAW_WAVEFORM_GAUSS_P0: List[float] = [-1, 0.6, 0.1, 1, 1, 0.3]
RAW_WAVEFORM_GAUSS_LB: List[float] = [-10, -10, 0.0, -10, -10, 0.1]
RAW_WAVEFORM_GAUSS_UB: List[float] = [0, 10, 2.0, 10, 10, 10]

DERIVATIVE_GAUSS_P0: List[float] = [-45, 0.5, 0.1, 45, 0.5, 0.1]
DERIVATIVE_GAUSS_LB: List[float] = [-50, 0.0, 0.1, 0, 0.0, 0.1]
DERIVATIVE_GAUSS_UB: List[float] = [0, 2.0, 5.0, 50, 2.0, 5.0]

GAUSS_MAX_EVALS: int = 10000
NORMALIZE_WAVEFORMS: bool = True

SNR_RATIO_THRESHOLD: float = 0.25
GMM_N_COMPONENTS: int = 2
GMM_N_INIT: int = 10
GMM_MAX_ITER: int = 100
GMM_RANDOM_STATE: int = 0

COLOR_FAST: str = "red"
COLOR_REGULAR: str = "blue"
HISTOGRAM_BINS: int = 30
AXIS_MARGIN_PCT: float = 0.05
MAX_WAVEFORMS_LIMIT: int = 10000

TIME_PP_LB: float = -10.0
TIME_PP_UB: float = 10.0
PP_DUR_LB: float = -10.0
PP_DUR_UB: float = 10.0


# =====================
#      dataclasses
# =====================

@dataclass
class WaveformFeatures:
    time_to_pp_ms: float
    pp_duration_ms: float
    fall_rise_contrast: float

@dataclass
class WaveformResult:
    waveform_idx: int
    p_fast: float
    p_regular: float
    predicted_class: str
    snr: float
    features: WaveformFeatures

@dataclass
class ChannelWaveforms:
    channel: int
    waveforms: List[np.ndarray]
    spike_times: List[float]
    window_sizes: List[int]


# =======================
#      main pipeline
# =======================

def main(args) -> None:    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = BASE_OUTPUT_DIR / f"run_{timestamp}"
    
    if not args.viz_only:
        X, predictions, classifier, results, output_dir = run_classification_pipeline(
            input_dir=args.input_dir,
            output_dir=output_dir,
            max_waveforms=args.max_waveforms,
        )
        
        create_all_visualizations(X, predictions, classifier, output_dir / "visualizations")
    else:
        print("Loading existing results for visualization...")
        with open(output_dir / "classification_results.json") as f:
            data = json.load(f)
        
        features = np.array([
            [w["time_to_pp_ms"], w["pp_duration_ms"], w["fall_rise_contrast"]]
            for w in data["waveforms"]
        ])
        predictions = np.array([w["predicted_class"] for w in data["waveforms"]])
        
        classifier = joblib.load(output_dir / "classifier_gmm.joblib")
        
        create_all_visualizations(features, predictions, classifier, output_dir / "visualizations")
    
    print(f"\nOutput saved to: {output_dir}")
    print("\nDone!")


def run_classification_pipeline(
    input_dir: str = WAVEFORM_INPUT_DIR,
    output_dir: Optional[Path] = None,
    max_waveforms: int = MAX_WAVEFORMS_LIMIT,
    save_classifier: bool = True,
) -> Tuple[np.ndarray, np.ndarray, GMMClassifier, List[WaveformResult], Path]:
    """Run complete classification pipeline.
    
    Returns:
        features: (n_waveforms x 3) feature array
        predictions: predicted class labels
        classifier: fitted GMM classifier
        results: list of WaveformResult objects
        output_dir: Path to output directory
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = BASE_OUTPUT_DIR / f"run_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading waveforms from {input_dir}...")
    channels_data = load_waveforms_from_directory(input_dir)
    print(f"  Loaded {len(channels_data)} channels")
    
    all_waveforms, avg_channel_waveforms = collect_all_waveforms(channels_data, max_waveforms)
    n_waveforms = min(len(all_waveforms), max_waveforms)
    print(f"  Processing {n_waveforms} waveforms...")
    
    print("Fitting waveform parameters...")
    features_list = []
    snr_list = []
    valid_indices = []
    
    for i, wf in enumerate(all_waveforms[:max_waveforms]):
        coefs = fit_single_waveform(wf)
        
        if not np.any(np.isnan(coefs)):
            features_list.append(coefs)
            snr_list.append(GMMClassifier.compute_snr(wf))
            valid_indices.append(i)
        
        if (i + 1) % 500 == 0:
            print(f"    Processed {i + 1} / {n_waveforms} waveforms...")
    
    features = np.array(features_list)
    snr_values = np.array(snr_list)
    
    print("Training GMM classifier...")
    X = GMMClassifier.extract_features(features)
    classifier = GMMClassifier().fit(X, snr=snr_values)
    
    if save_classifier:
        joblib.dump(classifier, output_dir / "classifier_gmm.joblib")
        print(f"  Classifier saved to {output_dir / 'classifier_gmm.joblib'}")
    
    print("Classifying all waveforms...")
    probs = classifier.predict_proba(X)
    predictions = classifier.get_labels(X)
    
    results = []
    for i in range(len(X)):
        time_to_pp_ms=float(X[i, 0])
        pp_duration_ms=float(X[i, 1])
        fall_rise_contrast=float(X[i, 2])

        results.append(WaveformResult(
            waveform_idx=i,
            p_fast=float(probs[i, 0]),
            p_regular=float(probs[i, 1]),
            predicted_class=predictions[i],
            snr=float(snr_values[i]),
            features=WaveformFeatures(
                time_to_pp_ms=time_to_pp_ms,
                pp_duration_ms=pp_duration_ms,
                fall_rise_contrast=fall_rise_contrast,
            ),
        ))
    
    run_avg_channel_classification(avg_channel_waveforms, output_dir, save_classifier)
    save_results_json(results, output_dir)
    return X, predictions, classifier, results, output_dir


# =================================
#      classifiers and fitting
# =================================

# for future extension and use in data science final project
class Classifier(Protocol):
    """defines classifier interface for easy swapping of classification methods"""
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit the classifier to training data."""
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        ...
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for samples in X."""
        ...


class GMMClassifier:
    """Gaussian Mixture Model classifier.
    
    Implements the method from Snyder et al. 2016:
    - Fit 2-component GMM on high-SNR waveforms
    - Use 3 waveform features
    - Identify fast-spiking by smaller time_to_pp
    """
    
    def __init__(self):
        self.model: Optional[GaussianMixture] = None
        self.fast_idx: int = 0
        self.slow_idx: int = 1
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        snr: Optional[np.ndarray] = None,
        snr_ratio: float = SNR_RATIO_THRESHOLD,
    ) -> "GMMClassifier":
        """Fit GMM to high-SNR waveforms.
        
        Args:
            X: (n_samples x 3) feature array
            snr: (n_samples,) SNR values for filtering
            snr_ratio: bottom percentile to filter out (default 25%)
        """
        if snr is not None:
            sorted_snr = np.sort(snr)
            threshold = sorted_snr[int(len(snr) * snr_ratio)]
            ok = snr > threshold
            X_train = X[ok]
        else:
            X_train = X
        
        self.model = GaussianMixture(
            n_components=GMM_N_COMPONENTS,
            n_init=GMM_N_INIT,
            max_iter=GMM_MAX_ITER,
            random_state=GMM_RANDOM_STATE,
        )
        self.model.fit(X_train)
        
        means = self.model.means_
        self.fast_idx = int(np.argmin(means[:, 0])) # type: ignore
        self.slow_idx = 1 - self.fast_idx
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise RuntimeError("Classifier not fitted")
        
        probs = self.model.predict_proba(X)
        predictions = probs[:, self.fast_idx] > 0.5
        return np.where(predictions, 0, 1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of fast-spiking."""
        if self.model is None:
            raise RuntimeError("Classifier not fitted")
        
        probs = self.model.predict_proba(X)
        return probs[:, [self.fast_idx, self.slow_idx]]
    
    def get_labels(self, X: np.ndarray) -> np.ndarray:
        """Return class labels for features."""
        preds = self.predict(X)
        return np.where(preds == 0, "fast", "regular")
    
    @classmethod
    def extract_features(cls, coef_vals: np.ndarray) -> np.ndarray:
        """Extract classification features from fitted parameters.
        
        Per Snyder et al. 2016, the features used are:
        - time_to_pp = coef[4] (time to afterhyperpolarization peak)
        - pp_duration = coef[5] (width of afterhyperpolarization)
        - fall_rise = coef[9] + coef[6] (fall rate + rise rate)
        
        Returns:
            features: (n_samples x 3) array [time_to_pp, pp_dur, fall_rise]
        """
        time_to_pp = coef_vals[:, 4]
        pp_duration = coef_vals[:, 5]
        fall_rise = coef_vals[:, 9] + coef_vals[:, 6]
        
        return np.column_stack([time_to_pp, pp_duration, fall_rise])

    @classmethod
    def compute_snr(cls, waveform: np.ndarray) -> float:
        """Compute signal-to-noise ratio.
        
        SNR = peak_to_peak / (2 * noise_std)
        """
        if waveform.size == 0:
            return float('nan')
        
        peak_to_peak = waveform.max() - waveform.min()
        noise = np.std(waveform)
        
        if noise == 0:
            return float('nan')
        
        return peak_to_peak / (2 * noise)


def fit_waveform_parameters(
    avg_waveforms: np.ndarray,
    samps_per_ms: float = SAMPLE_RATE_HZ / 1000,
    normalize: bool = NORMALIZE_WAVEFORMS,
) -> np.ndarray:
    """Fit 2-Gaussian model to raw waveform and derivative waveform.
    
    Per Snyder et al. 2016: fit separate 2-Gaussian models to:
    - Raw waveform (6 params: A1, B1, C1, A2, B2, C2)
    - Derivative waveform (6 params)
    
    Returns:
        coef_vals: (n_neurons x 12) array of fitted parameters
    """
    n_samples, n_neurons = avg_waveforms.shape
    
    time = np.arange(n_samples) / samps_per_ms
    
    if normalize:
        max_vals = np.max(np.abs(avg_waveforms), axis=0)
        max_vals[max_vals == 0] = 1
        avg_waveforms = avg_waveforms / max_vals
    
    diff_waveforms = np.diff(avg_waveforms, axis=0)
    coef_vals = np.zeros((n_neurons, 12))
    
    for i in range(n_neurons):
        raw = avg_waveforms[:, i]
        deriv = diff_waveforms[:, i]
        
        try:
            popt_raw, _ = curve_fit(
                gaussian2, time, raw,
                p0=RAW_WAVEFORM_GAUSS_P0,
                bounds=(RAW_WAVEFORM_GAUSS_LB, RAW_WAVEFORM_GAUSS_UB),
                maxfev=GAUSS_MAX_EVALS,
            )
            
            popt_der, _ = curve_fit(
                gaussian2, time[:-1], deriv,
                p0=DERIVATIVE_GAUSS_P0,
                bounds=(DERIVATIVE_GAUSS_LB, DERIVATIVE_GAUSS_UB),
                maxfev=GAUSS_MAX_EVALS,
            )
            
            coef_vals[i] = np.concatenate([popt_raw, popt_der])
            
        except Exception:
            coef_vals[i] = np.nan
    
    return coef_vals


def fit_single_waveform(
    waveform: np.ndarray,
    samps_per_ms: float = SAMPLE_RATE_HZ / 1000,
    normalize: bool = NORMALIZE_WAVEFORMS,
) -> np.ndarray:
    """Fit 2-Gaussian model to a single waveform."""
    waveform_2d = waveform.reshape(-1, 1)
    return fit_waveform_parameters(waveform_2d, samps_per_ms, normalize).flatten()


# =======================================
#      data loading and saving utils
# =======================================

def load_waveforms_from_directory(input_dir: str = WAVEFORM_INPUT_DIR) -> List[ChannelWaveforms]:
    """Load extracted waveforms from numpy files.
    
    Args:
        input_dir: Directory containing ch_*_spikes.npy files
        file_pattern: Glob pattern for waveform files
        
    Returns:
        List of ChannelWaveforms objects
    """
    existing_channel_waveforms = load_existing_waveforms(input_dir)
    return [ChannelWaveforms(
        channel=channel_id,
        waveforms=list(data["waveforms"]),
        spike_times=list(data["spike_times"]),
        window_sizes=list(data["window_sizes"])
    ) for channel_id, data in existing_channel_waveforms.items()]


def collect_all_waveforms(
    channels_data: List[ChannelWaveforms],
    max_waveforms: int = MAX_WAVEFORMS_LIMIT,
) -> Tuple[List[np.ndarray], Dict[int, List[np.ndarray]]]:
    """Collect all waveforms into flat list with channel labels.
    
    Returns:
        all_waveforms: flat list of all waveforms
        channel_labels: corresponding channel for each waveform
        waveforms_by_channel: dict mapping channel -> list of waveforms
    """
    all_waveforms = []
    waveforms_by_channel: Dict[int, List[np.ndarray]] = {}
    
    for ch_data in channels_data:
        if ch_data.channel not in waveforms_by_channel:
            waveforms_by_channel[ch_data.channel] = []
        
        for wf in ch_data.waveforms:
            all_waveforms.append(wf)
            waveforms_by_channel[ch_data.channel].append(wf)
            
            if len(all_waveforms) >= max_waveforms:
                return all_waveforms, waveforms_by_channel
    
    return all_waveforms, waveforms_by_channel


def run_avg_channel_classification(
    waveforms_by_channel: Dict[int, List[np.ndarray]],
    output_dir: Path,
    save_classifier: bool = True,
) -> Tuple[np.ndarray, np.ndarray, GMMClassifier, List[int]]:
    """Run classification on channel-averaged waveforms.
    
    Returns:
        channel_X: features array
        channel_predictions: predicted classes
        channel_classifier: fitted GMM
        channel_ids: valid channel numbers
    """
    print("\nProcessing channel-averaged waveforms...")
    channel_features, channel_snr_list = [], []
    valid_channel_ids = []
    
    for ch in sorted(waveforms_by_channel.keys()):
        wfs = waveforms_by_channel[ch]
        if len(wfs) < 2:
            continue
        
        wfs_arr = [np.asarray(wf, dtype=float) for wf in wfs]
        min_len = min(len(w) for w in wfs_arr)
        wfs_arr = [w[:min_len] for w in wfs_arr]
        avg_waveform = np.mean(wfs_arr, axis=0)
        coefs = fit_single_waveform(avg_waveform)
        
        if not np.any(np.isnan(coefs)):
            channel_features.append(coefs)
            channel_snr_list.append(GMMClassifier.compute_snr(avg_waveform))
            valid_channel_ids.append(ch)
    
    channel_features_arr = np.array(channel_features)
    channel_snr_arr = np.array(channel_snr_list)
    
    print(f"  Training GMM on {len(channel_features_arr)} channel averages...")
    channel_X = GMMClassifier.extract_features(channel_features_arr)
    channel_classifier = GMMClassifier().fit(channel_X, snr=channel_snr_arr)
    
    if save_classifier:
        joblib.dump(channel_classifier, output_dir / "classifier_gmm_channels.joblib")
    
    channel_probs = channel_classifier.predict_proba(channel_X)
    channel_predictions = channel_classifier.get_labels(channel_X)
    
    print(f"  Channel classification: {sum(channel_predictions == 'fast')} fast, {sum(channel_predictions == 'regular')} regular")
    
    # Generate channel visualizations
    print("\nGenerating channel-level visualizations...")
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    plot_3d_scatter_interactive(
        channel_X, channel_predictions, channel_classifier,
        vis_dir / "feature_space_3d_channels.html",
    )
    plot_2d_scatters(
        channel_X, channel_predictions, channel_classifier,
        vis_dir / "feature_scatter_2d_channels.png",
    )
    plot_histograms(
        channel_X, channel_predictions,
        vis_dir / "feature_histograms_channels.png",
    )
    
    return channel_X, channel_predictions, channel_classifier, valid_channel_ids


def save_results_json(
    results: List[WaveformResult],
    output_dir: Path,
) -> None:
    """Save classification results to JSON."""
    
    output_data = {
        "n_waveforms": len(results),
        "n_fast": sum(1 for r in results if r.predicted_class == "fast"),
        "n_regular": sum(1 for r in results if r.predicted_class == "regular"),
        "waveforms": [
            {
                "waveform_idx": r.waveform_idx,
                "predicted_class": r.predicted_class,
                "p_fast": r.p_fast,
                "p_regular": r.p_regular,
                "snr": r.snr,
                "time_to_pp_ms": r.features.time_to_pp_ms,
                "pp_duration_ms": r.features.pp_duration_ms,
                "fall_rise_contrast": r.features.fall_rise_contrast,
            }
            for r in results
        ],
    }
    
    json_path = output_dir / "classification_results.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {json_path}")


# =======================
#      visualization
# =======================

def compute_axis_limits(
    values: np.ndarray,
    margin_pct: float = AXIS_MARGIN_PCT,
) -> Tuple[float, float]:
    """Compute axis limits with margin percentage."""
    v_min = np.nanmin(values)
    v_max = np.nanmax(values)
    
    if v_max == v_min:
        return v_min - 1, v_max + 1
    
    margin = (v_max - v_min) * margin_pct
    return v_min - margin, v_max + margin


def compute_fixed_or_data_limits(
    values: np.ndarray,
    fixed_lb: float,
    fixed_ub: float,
) -> Tuple[float, float]:
    """Compute axis limits - use fixed bounds if data falls outside, else use data."""
    v_min = np.nanmin(values)
    v_max = np.nanmax(values)
    
    if v_min < fixed_lb or v_max > fixed_ub:
        return fixed_lb, fixed_ub
    
    return v_min, v_max


def plot_3d_scatter_interactive(
    X: np.ndarray,
    predictions: np.ndarray,
    classifier: GMMClassifier,
    output_path: Path,
) -> None:
    """Create interactive 3D scatter plot using Plotly."""
    
    time_pp = X[:, 0]
    pp_dur = X[:, 1]
    fall_rise = X[:, 2]
    
    fast_mask = predictions == "fast"
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=time_pp[fast_mask],
        y=pp_dur[fast_mask],
        z=fall_rise[fast_mask],
        mode='markers',
        marker=dict(size=1, color=COLOR_FAST, opacity=0.6),
        name='Fast-spiking',
    ))
    
    fig.add_trace(go.Scatter3d(
        x=time_pp[~fast_mask],
        y=pp_dur[~fast_mask],
        z=fall_rise[~fast_mask],
        mode='markers',
        marker=dict(size=1, color=COLOR_REGULAR, opacity=0.6),
        name='Regular-spiking',
    ))
    
    means = classifier.model.means_ # type: ignore
    fig.add_trace(go.Scatter3d(
        x=[means[classifier.fast_idx, 0], means[classifier.slow_idx, 0]], # type: ignore
        y=[means[classifier.fast_idx, 1], means[classifier.slow_idx, 1]], # type: ignore
        z=[means[classifier.fast_idx, 2], means[classifier.slow_idx, 2]], # type: ignore
        mode='markers',
        marker=dict(size=2, symbol='x', color='black'),
        name='Cluster centers',
    ))
    
    xlim = compute_axis_limits(time_pp, margin_pct=0)
    ylim = compute_axis_limits(pp_dur, margin_pct=0)
    zlim = compute_axis_limits(fall_rise, margin_pct=0)
    
    fig.update_layout(
        title="3D Feature Space (drag to rotate)",
        scene=dict(
            xaxis_title="Time to PP (ms)",
            yaxis_title="PP Duration (ms)",
            zaxis_title="Fall/Rise Contrast",
            xaxis=dict(range=[xlim[0], xlim[1]]),
            yaxis=dict(range=[ylim[0], ylim[1]]),
            zaxis=dict(range=[zlim[0], zlim[1]]),
        ),
        legend=dict(x=0, y=1),
        width=900,
        height=700,
    )
    
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


def plot_2d_scatters(
    X: np.ndarray,
    predictions: np.ndarray,
    classifier: GMMClassifier,
    output_path: Path,
) -> None:
    """Create three 2D scatter plots."""
    
    import matplotlib.pyplot as plt
    
    time_pp = X[:, 0]
    pp_dur = X[:, 1]
    fall_rise = X[:, 2]
    
    fast_mask = predictions == "fast"
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    plot_pairs = [
        (time_pp, pp_dur, "Time to PP (ms)", "PP Duration (ms)", True),
        (time_pp, fall_rise, "Time to PP (ms)", "Fall/Rise Contrast", True),
        (pp_dur, fall_rise, "PP Duration (ms)", "Fall/Rise Contrast", False),
    ]
    
    means = classifier.model.means_ # type: ignore
    
    for ax, (x_vals, y_vals, xlabel, ylabel, use_fixed) in zip(axes, plot_pairs):
        ax.scatter(x_vals[fast_mask], y_vals[fast_mask],
                   c=COLOR_FAST, alpha=0.5, s=30, label='Fast-spiking', edgecolors='none')
        ax.scatter(x_vals[~fast_mask], y_vals[~fast_mask],
                   c=COLOR_REGULAR, alpha=0.5, s=30, label='Regular-spiking', edgecolors='none')
        
        if use_fixed:
            if xlabel.startswith("Time to PP"):
                xlim = compute_axis_limits(x_vals, margin_pct=0)
            else:
                xlim = compute_axis_limits(x_vals, margin_pct=0)
            
            if "PP Duration" in ylabel:
                ylim = compute_axis_limits(y_vals, margin_pct=0)
            elif "Time to PP" in ylabel:
                ylim = compute_axis_limits(y_vals, margin_pct=0)
            else:
                ylim = compute_axis_limits(y_vals, margin_pct=0)
        else:
            xlim = compute_axis_limits(x_vals, margin_pct=0)
            ylim = compute_axis_limits(y_vals, margin_pct=0)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        
        plot_decision_boundary(ax, classifier, x_vals, y_vals)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_decision_boundary(
    ax,
    classifier: GMMClassifier,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
) -> None:
    """Plot decision boundary line where P(fast) = 0.5."""
    means = classifier.model.means_ # type: ignore
    
    x1, y1 = means[classifier.fast_idx, 0], means[classifier.fast_idx, 1] # type: ignore
    x2, y2 = means[classifier.slow_idx, 0], means[classifier.slow_idx, 1] # type: ignore
    
    mid_x = (x1 + x2) / 2 # type: ignore
    mid_y = (y1 + y2) / 2 # type: ignore
    
    if x2 != x1:
        slope = (y2 - y1) / (x2 - x1) # type: ignore
        perp_slope = -1 / slope if slope != 0 else 0
    else:
        perp_slope = 0
    
    x_min, x_max = ax.get_xlim()
    x_line = np.array([x_min, x_max])
    y_line = mid_y + perp_slope * (x_line - mid_x)
    
    y_min, y_max = ax.get_ylim()
    y_line = np.clip(y_line, y_min, y_max)
    
    ax.plot(x_line, y_line, 'k--', alpha=0.7, linewidth=1.5, label='Decision boundary')


def plot_histograms(
    X: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    """Create three histograms comparing fast vs regular spiking."""
    
    import matplotlib.pyplot as plt
    
    time_pp = X[:, 0]
    pp_dur = X[:, 1]
    fall_rise = X[:, 2]
    
    fast_mask = predictions == "fast"
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    features = [
        (time_pp, "Time to PP (ms)"),
        (pp_dur, "PP Duration (ms)"),
        (fall_rise, "Fall/Rise Contrast"),
    ]
    
    for ax, (values, label) in zip(axes, features):
        xlim = compute_axis_limits(values, margin_pct=0)
        
        fast_vals = values[fast_mask]
        rs_vals = values[~fast_mask]
        
        ax.hist(fast_vals, bins=HISTOGRAM_BINS, alpha=0.6,
                color=COLOR_FAST, label='Fast-spiking', density=False)
        ax.hist(rs_vals, bins=HISTOGRAM_BINS, alpha=0.6,
                color=COLOR_REGULAR, label='Regular-spiking', density=False)
        
        ax.set_xlim(xlim)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_all_visualizations(
    X: np.ndarray,
    predictions: np.ndarray,
    classifier: GMMClassifier,
    output_dir: Path,
) -> None:
    """Generate all visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_3d_scatter_interactive(
        X, predictions, classifier,
        output_dir / "feature_space_3d.html",
    )
    
    plot_2d_scatters(
        X, predictions, classifier,
        output_dir / "feature_scatter_2d.png",
    )
    
    plot_histograms(
        X, predictions,
        output_dir / "feature_histograms.png",
    )
    
    print(f"\nVisualizations saved to: {output_dir}")


# entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neuron type classification from spike waveforms"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=WAVEFORM_INPUT_DIR,
        help="Directory containing waveform files",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: run_{timestamp})",
    )

    parser.add_argument(
        "--max-waveforms",
        type=int,
        default=MAX_WAVEFORMS_LIMIT,
        help="Maximum waveforms to process",
    )

    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="Skip processing, only generate visualizations from existing results",
    )
    
    args = parser.parse_args()

    main(args)