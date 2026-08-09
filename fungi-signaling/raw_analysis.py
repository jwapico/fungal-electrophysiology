"""  <-- paste the content of the revised file below -->
raw_analysis.py

MEA spike waveform extraction pipeline for fungal recordings.

Loads raw interleaved Intan binary MEA data, detects spikes on every
channel, extracts waveform windows around each spike, persists the
results to a single self-describing .npz file, and renders HTML views:

  1. waveforms_grid.html      - grid of every extracted spike window, per
                                channel; click a window to open the channel's
                                interactive view zoomed to that segment
  2. all_ch_spikes.html       - the full downsampled channel trace with the
                                detected spikes marked
  3. channel_N_interactive.html - per-channel plotly view (opens zoomed when
                                reached from the grid via ?t0=..&t1=..)

The raw signal is never inverted or otherwise altered: spikes are
detected as positive-going prominences in the recorded orientation
(see DETECT_TROUGHS for the opposite polarity). Waveforms are stored
unsmoothed and unnormalized, in raw microvolts.

Run from the project root:
    python raw_analysis.py
    python raw_analysis.py --data-file <path> -v
    python raw_analysis.py --window-mode adaptive
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np
import scipy.optimize
import scipy.signal

SAMPLE_RATE_HZ: int = 30000
NUM_CHANNELS: int = 64
VOLTAGE_SCALE: float = 0.195
BINARY_DTYPE: str = "int16"

RAW_DATA_FILE: str = "../data/raw_mea_bins/recording_control_0_cut800s.bin"

WAVEFORM_OUTPUT_FILE: str = "outputs/waveforms/waveforms.npz"
SPIKE_HTML_OUTPUT_FILE: str = "outputs/html/waveforms_grid.html"
CHANNEL_HTML_OUTPUT_FILE: str = "outputs/html/all_ch_spikes.html"
INTERACTIVE_HTML_PATTERN: str = "channel_{ch}_interactive.html"
PLOTLY_JS: str = "cdn"  # "cdn" (small, needs internet) | "inline" (offline, ~3.5MB/file)

ST_DEV_SCALE: float = 16.0
LOWER_THRESHOLD_FACTOR: float = 0.5
MIN_SPIKE_DISTANCE: int = 50
DETECT_TROUGHS: bool = False

FIT_SEED_WINDOW_MS: float = 10.0
SEARCH_WINDOW_SAMPLES: int = 50
GAUSS_WINDOW_MULT: int = 8
MIN_WINDOW_MS: float = 3.0
MAX_WINDOW_MS: float = 20.0
DEFAULT_WINDOW_MS: float = 10.0
GAUSS_MAX_EVALS: int = 5000

WINDOW_GAUSS_P0: List[float] = [-1, 0.6, 0.1, 1, 1, 0.3]
WINDOW_GAUSS_LB: List[float] = [-10, -10, 0.0, -10, -10, 0.1]
WINDOW_GAUSS_UB: List[float] = [0, 10, 2.0, 10, 10, 10]

# windowing policy
WINDOW_MODE: str = "fixed"       # "fixed" | "adaptive"
FIXED_WINDOW_MS: float = 6.0     # extraction window size in fixed mode

CHANNEL_DS_FACTOR: int = 10
MAX_WAVEFORMS_PER_CHANNEL: int = 200
WAVEFORMS_PER_ROW: int = 6
FIGURE_DPI: int = 80
TRACE_DPI: int = 100
RANDOM_SEED: int = 0

# interactive view
INTERACTIVE_OVERVIEW_DS: int = 200     # downsample of the full-trace overview
INTERACTIVE_SPIKE_DS: int = 4          # downsample of per-spike context segments
SPIKE_CONTEXT_MS: float = 200.0        # total +/- window embedded around each spike
INTERACTIVE_CONTEXT_MS: float = 100.0  # extra +/- shown when a window is clicked


def load_raw_data(filepath: str) -> np.ndarray:
    print(f"Loading raw data from: {filepath}")
    raw = np.memmap(filepath, dtype=BINARY_DTYPE, mode="r")
    if raw.size % NUM_CHANNELS != 0:
        raise ValueError(
            f"File contains {raw.size} samples, not divisible by {NUM_CHANNELS} "
            "channels; expected an interleaved multi-channel recording"
        )
    data = raw.reshape(-1, NUM_CHANNELS)
    print(f"  Loaded {data.shape[0]:,} samples, {data.shape[1]} channels")
    return data


def estimate_noise_std(x: np.ndarray) -> float:
    return float(np.std(x))


def estimate_noise_mad(x: np.ndarray) -> float:
    median = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - median)))


def detect_spikes(
    voltage: np.ndarray,
    st_dev_scale: float = ST_DEV_SCALE,
    min_distance: int = MIN_SPIKE_DISTANCE,
    lower_factor: float = LOWER_THRESHOLD_FACTOR,
    troughs: bool = DETECT_TROUGHS,
    noise_estimator: Callable[[np.ndarray], float] = estimate_noise_std,
) -> Tuple[np.ndarray, float, float]:
    probe = -voltage if troughs else voltage
    std_dev = noise_estimator(probe)
    threshold = st_dev_scale * std_dev

    # height gate: the candidate must itself exceed the threshold in absolute
    # amplitude, not merely rise above a neighbouring valley. Prominence alone
    # passes tiny bumps sitting between large deflections (false positives).
    peak_indices, _ = scipy.signal.find_peaks(
        probe, height=threshold, prominence=threshold, distance=min_distance
    )

    if len(peak_indices) == 0:
        fallback_threshold = threshold * lower_factor
        peak_indices, _ = scipy.signal.find_peaks(
            probe, height=fallback_threshold, prominence=fallback_threshold,
            distance=min_distance,
        )
        threshold = fallback_threshold

    return peak_indices, float(threshold), float(std_dev)


def gaussian2(x: np.ndarray, a1: float, b1: float, c1: float,
              a2: float, b2: float, c2: float) -> np.ndarray:
    return (a1 * np.exp(-((x - b1) / c1) ** 2) +
            a2 * np.exp(-((x - b2) / c2) ** 2))


def estimate_adaptive_window(spike_segment: np.ndarray,
                             sample_rate: int = SAMPLE_RATE_HZ) -> int:
    n_samples = len(spike_segment)
    time_ms = np.arange(n_samples) / (sample_rate / 1000.0)
    centered = spike_segment - np.mean(spike_segment)
    spike_norm = centered / (np.max(np.abs(centered)) + 1e-10)
    try:
        popt, _ = scipy.optimize.curve_fit(
            gaussian2, time_ms, spike_norm,
            p0=WINDOW_GAUSS_P0, bounds=(WINDOW_GAUSS_LB, WINDOW_GAUSS_UB),
            maxfev=GAUSS_MAX_EVALS,
        )
        c1, c2 = abs(popt[2]), abs(popt[5])
        window_ms = GAUSS_WINDOW_MULT * (c1 + c2)
    except Exception:
        window_ms = DEFAULT_WINDOW_MS
    window_ms = float(min(max(window_ms, MIN_WINDOW_MS), MAX_WINDOW_MS))
    return int(window_ms * sample_rate / 1000)


def find_true_peak(spike_idx: int, voltage: np.ndarray,
                   search_window: int = SEARCH_WINDOW_SAMPLES,
                   troughs: bool = DETECT_TROUGHS) -> Optional[int]:
    """Refine a detection to the extremum of the *detected* polarity.

    Aligning on the probe polarity (rather than argmax |.|) keeps the window
    centred on the lobe that was detected; argmax |.| could flip to the
    opposite-polarity lobe of a biphasic spike.
    """
    search_start = spike_idx - search_window
    search_end = spike_idx + search_window
    if search_start < 0 or search_end >= len(voltage):
        return None
    probe = -voltage if troughs else voltage
    segment = probe[search_start:search_end]
    return search_start + int(np.argmax(segment))


def extract_waveform(spike_idx: int, voltage: np.ndarray,
                     peak_indices: Optional[np.ndarray] = None,
                     window_mode: str = WINDOW_MODE,
                     fixed_window_ms: float = FIXED_WINDOW_MS,
                     seed_window_ms: float = FIT_SEED_WINDOW_MS,
                     troughs: bool = DETECT_TROUGHS,
                     sample_rate: int = SAMPLE_RATE_HZ,
                     ) -> Optional[Tuple[np.ndarray, float, int, int, int]]:
    """
    Extract a waveform window around a detected spike.

    Returns:
        (waveform, spike_time_sec, window_samples, window_start_idx, peak_idx)
        or None if too close to the trace boundary.
    """
    if window_mode == "adaptive":
        seed_samples = int(seed_window_ms * sample_rate / 1000)
        if spike_idx < seed_samples or spike_idx >= len(voltage) - seed_samples:
            return None
        seed_segment = voltage[spike_idx - seed_samples:spike_idx + seed_samples]
        window_samples = estimate_adaptive_window(seed_segment, sample_rate)
        # never let an adaptive window swallow a neighbouring detection
        if peak_indices is not None:
            others = np.abs(np.asarray(peak_indices, dtype=np.int64) - spike_idx)
            others = others[others > 0]
            if len(others) > 0:
                window_samples = min(window_samples,
                                     int(2 * others.min() * sample_rate / 1000))
    else:
        window_samples = int(fixed_window_ms * sample_rate / 1000)

    window_samples = int(min(max(window_samples, MIN_WINDOW_MS * sample_rate / 1000),
                             MAX_WINDOW_MS * sample_rate / 1000))

    true_peak = find_true_peak(spike_idx, voltage, troughs=troughs)
    if true_peak is None:
        return None

    half = window_samples // 2
    start_idx = true_peak - half
    end_idx = start_idx + window_samples
    if start_idx < 0 or end_idx > len(voltage):
        return None

    waveform = voltage[start_idx:end_idx].copy()
    spike_time_sec = true_peak / sample_rate
    return waveform, spike_time_sec, window_samples, start_idx, true_peak


def process_channel(data: np.ndarray, channel: int,
                    window_mode: str = WINDOW_MODE) -> Dict[str, Any]:
    print(f"\nProcessing channel {channel}...")
    voltage = data[:, channel] * VOLTAGE_SCALE
    peak_indices, threshold, std_dev = detect_spikes(noise_estimator=estimate_noise_mad, voltage=voltage)
    print(f"  Detected {len(peak_indices)} spikes (threshold: {threshold:.2f} uV)")

    waveforms: List[np.ndarray] = []
    spike_times: List[float] = []
    window_sizes: List[int] = []
    window_starts: List[int] = []
    peak_positions: List[int] = []

    for spike_idx in peak_indices:
        result = extract_waveform(spike_idx, voltage, peak_indices,
                                  window_mode=window_mode)
        if result is None:
            continue
        waveform, spike_time, win_size, start_idx, peak_idx = result
        waveforms.append(waveform)
        spike_times.append(spike_time)
        window_sizes.append(win_size)
        window_starts.append(start_idx)
        peak_positions.append(peak_idx - start_idx)

    n_extracted = len(waveforms)
    print(f"  Extracted {n_extracted} waveforms")

    return {
        "waveforms": waveforms,
        "spike_times": np.array(spike_times),
        "window_sizes": np.array(window_sizes),
        "peak_indices": np.array(peak_positions),      # position within the window
        "window_starts": np.array(window_starts),
        "threshold": threshold,
        "std_dev": std_dev,
        "n_detected": len(peak_indices),
        "n_extracted": n_extracted,
    }


def save_waveforms(results: Dict[int, Dict[str, Any]],
                   output_file: str, source_file: str) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels = sorted(results.keys())
    n_ch = len(channels)

    waveforms = np.empty(n_ch, dtype=object)
    spike_times = np.empty(n_ch, dtype=object)
    window_sizes = np.empty(n_ch, dtype=object)
    peak_positions = np.empty(n_ch, dtype=object)
    window_starts = np.empty(n_ch, dtype=object)
    thresholds = np.zeros(n_ch)
    stds = np.zeros(n_ch)
    n_detected = np.zeros(n_ch, dtype=int)
    n_extracted = np.zeros(n_ch, dtype=int)

    for i, ch in enumerate(channels):
        d = results[ch]
        waveforms[i] = np.array(d["waveforms"], dtype=object)
        spike_times[i] = np.asarray(d["spike_times"], dtype=float)
        window_sizes[i] = np.asarray(d["window_sizes"], dtype=int)
        peak_positions[i] = np.asarray(d["peak_indices"], dtype=int)
        window_starts[i] = np.asarray(d["window_starts"], dtype=int)
        thresholds[i] = d["threshold"]
        stds[i] = d["std_dev"]
        n_detected[i] = d["n_detected"]
        n_extracted[i] = d["n_extracted"]

    payload = {
        "channels": np.asarray(channels, dtype=int),
        "waveforms": waveforms,
        "spike_times": spike_times,
        "window_sizes": window_sizes,
        "peak_positions": peak_positions,
        "window_starts": window_starts,
        "thresholds": thresholds,
        "stds": stds,
        "n_detected": n_detected,
        "n_extracted": n_extracted,
        "sample_rate": SAMPLE_RATE_HZ,
        "unit": "uV",
        "source_file": source_file,
        "min_window_ms": MIN_WINDOW_MS,
        "max_window_ms": MAX_WINDOW_MS,
        "window_mode": WINDOW_MODE,
        "fixed_window_ms": FIXED_WINDOW_MS,
        "st_dev_scale": ST_DEV_SCALE,
    }
    np.savez_compressed(output_path, **payload)
    print(f"\nSaved waveforms to: {output_file}")


def load_waveforms(output_file: str) -> Dict[int, Dict[str, Any]]:
    output_path = Path(output_file)
    if not output_path.exists():
        print(f"Error: Output file does not exist: {output_file}")
        return {}
    data = np.load(output_path, allow_pickle=True)
    channels = data["channels"].tolist()
    results: Dict[int, Dict[str, Any]] = {}
    for i, ch in enumerate(channels):
        cid = int(ch)
        peak_pos = np.asarray(data["peak_positions"][i], dtype=int)
        results[cid] = {
            "waveforms": list(data["waveforms"][i]),
            "spike_times": np.asarray(data["spike_times"][i]),
            "window_sizes": np.asarray(data["window_sizes"][i]),
            "peak_indices": peak_pos,
            "window_starts": np.asarray(data["window_starts"][i]),
            "threshold": float(data["thresholds"][i]),
            "std_dev": float(data["stds"][i]),
            "n_detected": int(data["n_detected"][i]),
            "n_extracted": int(data["n_extracted"][i]),
        }
        print(f"  Loaded channel {cid}: "
              f"{results[cid]['n_extracted']} waveforms")
    return results


def _html_head(title: str) -> List[str]:
    return [
        "<!DOCTYPE html>", "<html>", "<head>",
        "    <title>MEA Spike Waveforms</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }",
        "        h1 { color: #333; }",
        "        .channel-section { margin-bottom: 40px; background: white; padding: 20px; "
        "border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "        .channel-header { background: #e8e8e8; padding: 15px; margin: -20px -20px 15px -20px; "
        "border-radius: 8px 8px 0 0; }",
        "        .channel-header h2 { margin: 0 0 10px 0; }",
        "        .stats { font-size: 13px; color: #666; }",
        "        .channel-section img { max-width: 100%; height: auto; }",
        "        .channel-section img.grid-img { max-width: none; cursor: pointer; }",
        "    </style>",
        "</head>", "<body>",
        f"    <h1>{title}</h1>",
    ]


def _write_html(output_file: str, html_parts: List[str]) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_parts))
    print(f"  Saved HTML to: {output_file}")


def _figure_to_base64(fig: Figure, dpi: int = FIGURE_DPI,
                      tight: bool = True) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight" if tight else None)
    plt.close(fig)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img


def _window_time_axis(waveform: np.ndarray,
                      sample_rate: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """Time axis in ms centred on the *window centre* (not the extremum)."""
    return (np.arange(len(waveform)) - len(waveform) // 2) * (1000.0 / sample_rate)


def gen_spike_waveform_html(results: Dict[int, Dict[str, Any]],
                            output_file: str,
                            interactive_pattern: str = INTERACTIVE_HTML_PATTERN,
                            max_waveforms: int = MAX_WAVEFORMS_PER_CHANNEL,
                            per_row: int = WAVEFORMS_PER_ROW,
                            dpi: int = FIGURE_DPI,
                            context_ms: float = INTERACTIVE_CONTEXT_MS) -> None:
    print(f"\nGenerating waveform grid HTML: {output_file}")
    rng = np.random.default_rng(RANDOM_SEED)

    html_parts = _html_head("MEA Spike Waveform Grid")
    html_parts.append("    <p>Every extracted spike window, per channel. "
                      "Time in ms, centred on the window; the vertical line marks "
                      "the detected peak. Click a window to open the channel's "
                      "interactive view zoomed to that segment.</p>")

    for ch in sorted(results.keys()):
        data = results[ch]
        waveforms = data["waveforms"]
        if len(waveforms) == 0:
            continue

        times = data["spike_times"]
        sizes = data["window_sizes"]
        peak_pos = data["peak_indices"]

        if len(waveforms) > max_waveforms:
            sel = np.sort(rng.choice(len(waveforms), max_waveforms, replace=False))
        else:
            sel = np.arange(len(waveforms))
        sub_wf = [waveforms[i] for i in sel]
        sub_t = times[sel]
        sub_size = sizes[sel]
        sub_peak = peak_pos[sel]

        n = len(sub_wf)
        n_rows = int(np.ceil(n / per_row))
        fig, axes = plt.subplots(
            n_rows, per_row, figsize=(per_row * 2.0, n_rows * 1.2), squeeze=False)

        for ax, (wf, pk) in zip(axes.flat, zip(sub_wf, sub_peak)):
            t_axis = _window_time_axis(wf)
            ax.plot(t_axis, wf, linewidth=0.6, color="#1f77b4")
            line_ms = (pk - len(wf) // 2) * (1000.0 / SAMPLE_RATE_HZ)
            ax.axvline(line_ms, color="k", linewidth=0.4, alpha=0.4)
            ax.tick_params(labelsize=5)
            ax.set_yticks([])
        for ax in axes.flat[n:]:
            ax.set_visible(False)

        fig.tight_layout()
        # save WITHOUT bbox_inches="tight" so ax positions map 1:1 to PNG pixels
        img = _figure_to_base64(fig, dpi=dpi, tight=False)
        W = int(fig.get_size_inches()[0] * dpi)
        H = int(fig.get_size_inches()[1] * dpi)

        areas = []
        for j, ax in enumerate(axes.flat):
            if j >= n:
                break
            pos = ax.get_position()
            x0, y0 = int(pos.x0 * W), int((1 - pos.y1) * H)
            x1, y1 = int(pos.x1 * W), int((1 - pos.y0) * H)
            t = float(sub_t[j])
            margin = (float(sub_size[j]) * 1000.0 / SAMPLE_RATE_HZ / 2.0 + context_ms) / 1000.0
            t0 = max(0.0, t - margin)
            t1 = t + margin
            href = (f"{interactive_pattern.format(ch=ch)}"
                    f"?t0={t0:.4f}&t1={t1:.4f}")
            areas.append(
                f'        <area shape="rect" coords="{x0},{y0},{x1},{y1}" '
                f'href="{href}" target="_blank" '
                f'title="ch{ch} spike t={t:.3f}s">')

        map_name = f"wavemap_{ch}"
        html_parts.append(f"""
        <div class="channel-section">
            <div class="channel-header">
                <h2>Channel {ch}</h2>
                <div class="stats">
                    <strong>Detected:</strong> {data['n_detected']} |
                    <strong>Extracted:</strong> {data['n_extracted']} |
                    <strong>Threshold:</strong> {data['threshold']:.2f} uV |
                    <strong>Noise sigma:</strong> {data['std_dev']:.2f} uV
                </div>
            </div>
            <img class="grid-img" width="{W}" height="{H}"
                 src="data:image/png;base64,{img}" usemap="#{map_name}"
                 alt="Channel {ch} waveforms">
            <map name="{map_name}">
{chr(10).join(areas)}
            </map>
        </div>
        """)

    html_parts.append("</body></html>")
    _write_html(output_file, html_parts)


def gen_channel_html(results: Dict[int, Dict[str, Any]],
                     output_file: str, raw_file: str,
                     ds_factor: int = CHANNEL_DS_FACTOR) -> None:
    print(f"\nGenerating full-trace HTML: {output_file}")
    data = load_raw_data(raw_file)
    html_parts = _html_head("MEA Channel Traces")
    html_parts.append("    <p>Full recording per channel with detected "
                      "spikes marked (threshold shown dashed).</p>")

    for ch in sorted(results.keys()):
        res = results[ch]
        voltage = data[:, ch] * VOLTAGE_SCALE
        voltage_ds = voltage[::ds_factor] if ds_factor > 0 else voltage
        time_ds = (np.arange(len(voltage_ds)) /
                   (SAMPLE_RATE_HZ / ds_factor) if ds_factor > 0
                   else np.arange(len(voltage_ds)) / SAMPLE_RATE_HZ)

        # refined peak indices are stored per-window; reconstruct absolute indices
        abs_peaks = np.asarray(res["window_starts"]) + np.asarray(res["peak_indices"])
        peak_times = abs_peaks / SAMPLE_RATE_HZ
        peak_voltages = voltage[abs_peaks]

        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(time_ds, voltage_ds, color="blue", linewidth=0.4)
        ax.scatter(peak_times, peak_voltages, s=6, c="red", zorder=3)
        ax.axhline(res["threshold"], color="orange", linewidth=0.8,
                   linestyle="--", alpha=0.8)
        ax.set_title(f"Channel {ch} - {res['n_extracted']} extracted spikes "
                     f"(threshold: {res['threshold']:.2f} uV)")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Voltage (uV)")
        fig.tight_layout()
        img = _figure_to_base64(fig, dpi=TRACE_DPI)
        html_parts.append(f"""
        <div class="channel-section">
            <div class="channel-header">
                <h2>Channel {ch}</h2>
                <div class="stats">
                    <strong>Detected:</strong> {res['n_detected']} |
                    <strong>Extracted:</strong> {res['n_extracted']} |
                    <strong>Threshold:</strong> {res['threshold']:.2f} uV |
                    <strong>Noise sigma:</strong> {res['std_dev']:.2f} uV
                </div>
            </div>
            <img src="data:image/png;base64,{img}" alt="Channel {ch} trace">
        </div>
        """)

    html_parts.append("</body></html>")
    _write_html(output_file, html_parts)


def gen_channel_interactive_html(results: Dict[int, Dict[str, Any]],
                                 channel: int, raw_file: str, output_file: str,
                                 overview_ds: int = INTERACTIVE_OVERVIEW_DS,
                                 spike_ds: int = INTERACTIVE_SPIKE_DS,
                                 context_ms: float = SPIKE_CONTEXT_MS,
                                 plotly_js: str = PLOTLY_JS) -> None:
    """One self-contained plotly view per channel with click-zoom support.

    The URL query parameters t0/t1 select the initial x-axis range, so the
    waveform grid can deep-link ("channel_5_interactive.html?t0=..&t1=..").
    """
    import plotly.graph_objects as go

    res = results[channel]
    print(f"\nGenerating interactive HTML for channel {channel}: {output_file}")
    data = load_raw_data(raw_file)
    voltage = data[:, channel] * VOLTAGE_SCALE
    n_samples = len(voltage)

    abs_peaks = np.asarray(res["window_starts"]) + np.asarray(res["peak_indices"])

    fig = go.Figure()
    # full-trace overview (coarse)
    if overview_ds > 1:
        t_over = np.arange(0, n_samples, overview_ds) / SAMPLE_RATE_HZ
        v_over = voltage[::overview_ds]
    else:
        t_over = np.arange(n_samples) / SAMPLE_RATE_HZ
        v_over = voltage
    fig.add_trace(go.Scatter(x=t_over, y=v_over, mode="lines",
                             name="overview", line=dict(color="#9ecae1", width=1),
                             hovertemplate="t=%{x:.3f}s<br>%{y:.1f}uV",
                             hoverlabel=dict(bgcolor="#9ecae1")))

    # higher-resolution context segments around each spike, joined with NaNs
    half = int(context_ms / 2000.0 * SAMPLE_RATE_HZ)
    ctx_t: List[float] = []
    ctx_v: List[float] = []
    for pk in abs_peaks:
        s0, s1 = max(0, pk - half), min(n_samples, pk + half)
        ctx_v.append(np.round(voltage[s0:s1:spike_ds], 1))
        ctx_t.append(np.round(np.arange(s0, s1, spike_ds) / SAMPLE_RATE_HZ, 4))
    if ctx_t:
        ctx_v = np.concatenate(ctx_v).tolist()
        ctx_t = np.concatenate(ctx_t).tolist()
        fig.add_trace(go.Scatter(x=ctx_t, y=ctx_v, mode="lines",
                                 name="spike context", showlegend=False,
                                 hoverinfo="skip",
                                 line=dict(color="#1f77b4", width=1)))

    fig.add_trace(go.Scatter(x=abs_peaks / SAMPLE_RATE_HZ,
                             y=voltage[abs_peaks], mode="markers",
                             name="detected spike",
                             marker=dict(color="red", size=7, symbol="x"),
                             hovertemplate="spike t=%{x:.3f}s<br>%{y:.1f}uV"))

    fig.update_layout(
        title=f"Channel {channel} - {res['n_extracted']} spikes "
              f"(threshold {res['threshold']:.2f} uV)",
        xaxis_title="Time (seconds)", yaxis_title="Voltage (uV)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, div_id="interactive",
                   include_plotlyjs=plotly_js)

    with open(output_path, "r") as fh:
        html = fh.read()
    script = """
<script>
(function () {
  var p = new URLSearchParams(window.location.search);
  var t0 = p.get('t0'), t1 = p.get('t1');
  if (t0 !== null && t1 !== null) {
    Plotly.relayout('interactive', {'xaxis.range': [parseFloat(t0), parseFloat(t1)]});
  }
})();
</script>
"""
    html = html.replace("</body>", script + "</body>")
    with open(output_path, "w") as fh:
        fh.write(html)
    print(f"  Saved interactive HTML to: {output_file}")


def main(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("MEA Spike Waveform Pipeline")
    print("=" * 60)
    print(f"Data file:  {args.data_file}")
    print(f"Waveforms:  {args.output}")
    print(f"Window:     {args.window_mode} "
          f"({args.fixed_window_ms} ms)" if args.window_mode == "fixed"
          else f"Window:     adaptive")
    print("=" * 60)

    if args.visualize_only:
        results = load_waveforms(args.output)
        if not results:
            print("Error: No data loaded. Run without -v first.")
            return
    else:
        data = load_raw_data(args.data_file)
        results: Dict[int, Dict[str, Any]] = {}
        for ch in range(data.shape[1]):
            results[ch] = process_channel(data, ch, window_mode=args.window_mode)
        save_waveforms(results, args.output, source_file=args.data_file)

    out_dir = Path(args.spike_html).parent
    interactive_pattern = INTERACTIVE_HTML_PATTERN
    for ch in sorted(results.keys()):
        if results[ch]["n_extracted"] > 0:
            gen_channel_interactive_html(
                results, ch, args.data_file,
                str(out_dir / interactive_pattern.format(ch=ch)))

    gen_spike_waveform_html(results, args.spike_html,
                            interactive_pattern=interactive_pattern)
    gen_channel_html(results, args.channel_html, raw_file=args.data_file)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Open {args.spike_html} in your browser")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MEA spike waveform extraction and visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-d", "--data-file", type=str, default=RAW_DATA_FILE,
                        help="Path to raw MEA binary recording")
    parser.add_argument("-o", "--output", type=str, default=WAVEFORM_OUTPUT_FILE,
                        help="Output .npz path for extracted waveforms")
    parser.add_argument("-s", "--spike-html", type=str, default=SPIKE_HTML_OUTPUT_FILE,
                        help="Output HTML path for the waveform grid")
    parser.add_argument("-c", "--channel-html", type=str, default=CHANNEL_HTML_OUTPUT_FILE,
                        help="Output HTML path for the full-trace view")
    parser.add_argument("-w", "--window-mode", type=str, default=WINDOW_MODE,
                        choices=["fixed", "adaptive"],
                        help="Window sizing: fixed (default) or adaptive (Gaussian)")
    parser.add_argument("--fixed-window-ms", type=float, default=FIXED_WINDOW_MS,
                        help="Window size in ms when --window-mode=fixed")
    parser.add_argument("-v", "--visualize-only", action="store_true",
                        help="Only render HTML from previously extracted waveforms")
    args = parser.parse_args()
    main(args)