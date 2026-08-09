"""
raw_analysis.py  (rev 3 - event-based)

Replaces per-peak spike detection with EVENT segmentation:

  - noise floor via robust estimator (MAD default: the low envelope gate must
    not be inflated by the very events we are segmenting - on ch40 std=8.7uV
    inflates the gate to 52uV and truncates the small envelope oscillations,
    while MAD=2.9uV keeps them)
  - envelope gate: contiguous |voltage| > gate  ->  excursion
  - excursions separated by <= EVENT_GAP_MS are merged into ONE event
  - an event is kept only if its dominant |deflection| exceeds the spike gate
  - window = [onset - pre_pad, offset + post_pad], clamped to
    [MIN_WINDOW_MS, MAX_WINDOW_MS]; oversized events are centered on the
    dominant peak

Each event yields one waveform window spanning the WHOLE oscillatory event
(variable number of oscillations), aligned on the dominant peak.
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np
import scipy.ndimage
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
PLOTLY_JS: str = "cdn"

# ---------------- event segmentation ----------------
EVENT_NOISE_ESTIMATOR: Callable[[np.ndarray], float] = \
    lambda x: float(1.4826 * np.median(np.abs(x - np.median(x))))  # MAD
EVENT_GATE_SCALE: float = 5.0    # low envelope gate  (x noise)
SPIKE_GATE_SCALE: float = 16.0   # high gate: an event's dominant peak must clear this
EVENT_GAP_MS: float = 5.0        # merge excursions <= this apart into one event
MIN_EVENT_MS: float = 1.5        # discard envelope blips shorter than this
OSCILLATION_MIN_DISTANCE: int = 30  # min samples between oscillation peaks (1 ms)

# ---------------- window extraction ----------------
PRE_PAD_MS: float = 2.0
POST_PAD_MS: float = 2.0
MIN_WINDOW_MS: float = 3.0
MAX_WINDOW_MS: float = 60.0

# ---------------- visualization ----------------
CHANNEL_DS_FACTOR: int = 10
MAX_WAVEFORMS_PER_CHANNEL: int = 200
WAVEFORMS_PER_ROW: int = 6
FIGURE_DPI: int = 80
TRACE_DPI: int = 100
RANDOM_SEED: int = 0

INTERACTIVE_OVERVIEW_DS: int = 200
INTERACTIVE_SPIKE_DS: int = 4
SPIKE_CONTEXT_MS: float = 200.0
INTERACTIVE_CONTEXT_MS: float = 100.0


class Event(NamedTuple):
    onset: int
    offset: int
    main: int            # dominant (largest |deflection|) sample
    n_oscillations: int


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


def detect_events(
    voltage: np.ndarray,
    noise_estimator: Callable[[np.ndarray], float] = EVENT_NOISE_ESTIMATOR,
    gate_scale: float = EVENT_GATE_SCALE,
    spike_gate_scale: float = SPIKE_GATE_SCALE,
    gap_ms: float = EVENT_GAP_MS,
    min_event_ms: float = MIN_EVENT_MS,
    osc_min_distance: int = OSCILLATION_MIN_DISTANCE,
    sample_rate: int = SAMPLE_RATE_HZ,
) -> Tuple[List[Event], float, float, float]:
    """Segment the trace into multi-oscillation spike events.

    Returns:
        (events, noise, gate, spike_gate)
    """
    noise = float(noise_estimator(voltage))
    gate = gate_scale * noise
    spike_gate = spike_gate_scale * noise

    idx = np.flatnonzero(np.abs(voltage) > gate)
    if len(idx) == 0:
        return [], noise, gate, spike_gate

    run_breaks = np.flatnonzero(np.diff(idx) > 1)
    run_starts = np.r_[idx[0], idx[run_breaks + 1]]
    run_ends = np.r_[idx[run_breaks], idx[-1]]

    # merge contiguous excursions separated by <= gap_samples
    gap_samples = int(gap_ms * sample_rate / 1000)
    merged: List[List[int]] = []
    for s, e in zip(run_starts, run_ends):
        if merged and (s - merged[-1][1] - 1) <= gap_samples:
            merged[-1][1] = e
        else:
            merged.append([int(s), int(e)])

    min_event_samples = int(min_event_ms * sample_rate / 1000)
    events: List[Event] = []
    for s, e in merged:
        if e - s + 1 < min_event_samples:
            continue
        seg = voltage[s:e + 1]
        main = s + int(np.argmax(np.abs(seg)))
        if abs(voltage[main]) < spike_gate:
            continue
        n_osc = len(scipy.signal.find_peaks(
            np.abs(seg), height=gate, distance=osc_min_distance)[0])
        events.append(Event(onset=s, offset=e, main=main, n_oscillations=n_osc))

    return events, noise, gate, spike_gate


def extract_event_window(
    ev: Event, voltage: np.ndarray,
    pre_pad_ms: float = PRE_PAD_MS,
    post_pad_ms: float = POST_PAD_MS,
    min_window_ms: float = MIN_WINDOW_MS,
    max_window_ms: float = MAX_WINDOW_MS,
    sample_rate: int = SAMPLE_RATE_HZ,
) -> Optional[Tuple[np.ndarray, float, int, int, int]]:
    """Window around a whole event, aligned on the dominant peak.

    Returns:
        (waveform, event_time_sec, window_samples, window_start_idx, peak_idx)
        or None if the trace is too short to hold the window.
    """
    pre = int(pre_pad_ms * sample_rate / 1000)
    post = int(post_pad_ms * sample_rate / 1000)
    min_w = int(min_window_ms * sample_rate / 1000)
    max_w = int(max_window_ms * sample_rate / 1000)

    natural_w = (ev.offset + post) - (ev.onset - pre) + 1
    if natural_w > max_w:
        w = max_w
        start = ev.main - w // 2
    elif natural_w < min_w:
        w = min_w
        start = ev.main - w // 2
    else:
        w = natural_w
        start = ev.onset - pre

    if w <= 0:
        return None
    start = int(max(0, min(start, len(voltage) - w)))
    end = start + w

    waveform = voltage[start:end].copy()
    return (waveform, ev.main / sample_rate, w, start, ev.main - start)


def process_channel(data: np.ndarray, channel: int) -> Dict[str, Any]:
    print(f"\nProcessing channel {channel}...")
    voltage = data[:, channel] * VOLTAGE_SCALE
    events, noise, gate, spike_gate = detect_events(voltage)
    print(f"  Found {len(events)} events (gate {gate:.2f} uV, "
          f"spike gate {spike_gate:.2f} uV, noise {noise:.2f} uV)")

    waveforms: List[np.ndarray] = []
    event_times: List[float] = []
    window_sizes: List[int] = []
    window_starts: List[int] = []
    peak_positions: List[int] = []
    n_oscillations: List[int] = []
    amplitudes: List[float] = []

    for ev in events:
        result = extract_event_window(ev, voltage)
        if result is None:
            continue
        waveform, etime, win_size, start_idx, peak_idx = result
        waveforms.append(waveform)
        event_times.append(etime)
        window_sizes.append(win_size)
        window_starts.append(start_idx)
        peak_positions.append(peak_idx)
        n_oscillations.append(ev.n_oscillations)
        amplitudes.append(float(voltage[ev.main]))

    n_extracted = len(waveforms)
    print(f"  Extracted {n_extracted} waveforms")

    return {
        "waveforms": waveforms,
        "spike_times": np.array(event_times),
        "window_sizes": np.array(window_sizes, dtype=int),
        "peak_indices": np.array(peak_positions, dtype=int),
        "window_starts": np.array(window_starts, dtype=int),
        "n_oscillations": np.array(n_oscillations, dtype=int),
        "amplitudes": np.array(amplitudes, dtype=float),
        "threshold": spike_gate,
        "gate": gate,
        "std_dev": noise,
        "n_events": len(events),
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
    n_osc = np.empty(n_ch, dtype=object)
    amplitudes = np.empty(n_ch, dtype=object)
    thresholds = np.zeros(n_ch)
    gates = np.zeros(n_ch)
    stds = np.zeros(n_ch)
    n_events = np.zeros(n_ch, dtype=int)
    n_extracted = np.zeros(n_ch, dtype=int)

    for i, ch in enumerate(channels):
        d = results[ch]
        waveforms[i] = np.array(d["waveforms"], dtype=object)
        spike_times[i] = np.asarray(d["spike_times"], dtype=float)
        window_sizes[i] = np.asarray(d["window_sizes"], dtype=int)
        peak_positions[i] = np.asarray(d["peak_indices"], dtype=int)
        window_starts[i] = np.asarray(d["window_starts"], dtype=int)
        n_osc[i] = np.asarray(d["n_oscillations"], dtype=int)
        amplitudes[i] = np.asarray(d["amplitudes"], dtype=float)
        thresholds[i] = d["threshold"]
        gates[i] = d["gate"]
        stds[i] = d["std_dev"]
        n_events[i] = d["n_events"]
        n_extracted[i] = d["n_extracted"]

    payload = {
        "channels": np.asarray(channels, dtype=int),
        "waveforms": waveforms,
        "spike_times": spike_times,
        "window_sizes": window_sizes,
        "peak_positions": peak_positions,
        "window_starts": window_starts,
        "n_oscillations": n_osc,
        "amplitudes": amplitudes,
        "thresholds": thresholds,
        "gates": gates,
        "stds": stds,
        "n_events": n_events,
        "n_extracted": n_extracted,
        "sample_rate": SAMPLE_RATE_HZ,
        "unit": "uV",
        "source_file": source_file,
        "min_window_ms": MIN_WINDOW_MS,
        "max_window_ms": MAX_WINDOW_MS,
        "pre_pad_ms": PRE_PAD_MS,
        "post_pad_ms": POST_PAD_MS,
        "event_gate_scale": EVENT_GATE_SCALE,
        "spike_gate_scale": SPIKE_GATE_SCALE,
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
        results[cid] = {
            "waveforms": list(data["waveforms"][i]),
            "spike_times": np.asarray(data["spike_times"][i]),
            "window_sizes": np.asarray(data["window_sizes"][i]),
            "peak_indices": np.asarray(data["peak_positions"][i], dtype=int),
            "window_starts": np.asarray(data["window_starts"][i], dtype=int),
            "n_oscillations": np.asarray(data["n_oscillations"][i], dtype=int),
            "amplitudes": np.asarray(data["amplitudes"][i], dtype=float),
            "threshold": float(data["thresholds"][i]),
            "gate": float(data["gates"][i]),
            "std_dev": float(data["stds"][i]),
            "n_events": int(data["n_events"][i]),
            "n_extracted": int(data["n_extracted"][i]),
        }
        print(f"  Loaded channel {cid}: {results[cid]['n_extracted']} waveforms")
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
    html_parts.append("    <p>Every extracted event window, per channel. "
                      "Time in ms, centred on the window; the vertical line marks "
                      "the dominant peak. Click a window to open the channel's "
                      "interactive view zoomed to that event.</p>")

    for ch in sorted(results.keys()):
        data = results[ch]
        waveforms = data["waveforms"]
        if len(waveforms) == 0:
            continue

        times = data["spike_times"]
        sizes = data["window_sizes"]
        peak_pos = data["peak_indices"]
        n_osc = data["n_oscillations"]

        if len(waveforms) > max_waveforms:
            sel = np.sort(rng.choice(len(waveforms), max_waveforms, replace=False))
        else:
            sel = np.arange(len(waveforms))
        sub_wf = [waveforms[i] for i in sel]
        sub_t = times[sel]
        sub_size = sizes[sel]
        sub_peak = peak_pos[sel]
        sub_osc = n_osc[sel]

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
                f'title="ch{ch} event t={t:.3f}s, {int(sub_osc[j])} oscillations">')

        map_name = f"wavemap_{ch}"
        html_parts.append(f"""
        <div class="channel-section">
            <div class="channel-header">
                <h2>Channel {ch}</h2>
                <div class="stats">
                    <strong>Events:</strong> {data['n_events']} |
                    <strong>Extracted:</strong> {data['n_extracted']} |
                    <strong>Spike gate:</strong> {data['threshold']:.2f} uV |
                    <strong>Noise (MAD):</strong> {data['std_dev']:.2f} uV
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
                      "events marked (spike gate shown dashed).</p>")

    for ch in sorted(results.keys()):
        res = results[ch]
        voltage = data[:, ch] * VOLTAGE_SCALE
        voltage_ds = voltage[::ds_factor] if ds_factor > 0 else voltage
        time_ds = (np.arange(len(voltage_ds)) /
                   (SAMPLE_RATE_HZ / ds_factor) if ds_factor > 0
                   else np.arange(len(voltage_ds)) / SAMPLE_RATE_HZ)

        abs_peaks = np.asarray(res["window_starts"]) + np.asarray(res["peak_indices"])
        peak_times = abs_peaks / SAMPLE_RATE_HZ
        peak_voltages = voltage[abs_peaks]

        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(time_ds, voltage_ds, color="blue", linewidth=0.4)
        ax.scatter(peak_times, peak_voltages, s=6, c="red", zorder=3)
        ax.axhline(res["threshold"], color="orange", linewidth=0.8,
                   linestyle="--", alpha=0.8)
        ax.set_title(f"Channel {ch} - {res['n_extracted']} events "
                     f"(spike gate: {res['threshold']:.2f} uV)")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Voltage (uV)")
        fig.tight_layout()
        img = _figure_to_base64(fig, dpi=TRACE_DPI)
        html_parts.append(f"""
        <div class="channel-section">
            <div class="channel-header">
                <h2>Channel {ch}</h2>
                <div class="stats">
                    <strong>Events:</strong> {res['n_events']} |
                    <strong>Extracted:</strong> {res['n_extracted']} |
                    <strong>Spike gate:</strong> {res['threshold']:.2f} uV |
                    <strong>Noise (MAD):</strong> {res['std_dev']:.2f} uV
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
    import plotly.graph_objects as go

    res = results[channel]
    print(f"\nGenerating interactive HTML for channel {channel}: {output_file}")
    data = load_raw_data(raw_file)
    voltage = data[:, channel] * VOLTAGE_SCALE
    n_samples = len(voltage)

    abs_peaks = np.asarray(res["window_starts"]) + np.asarray(res["peak_indices"])

    fig = go.Figure()
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
                             name="dominant peak",
                             marker=dict(color="red", size=7, symbol="x"),
                             hovertemplate="t=%{x:.3f}s<br>%{y:.1f}uV"))

    fig.update_layout(
        title=f"Channel {channel} - {res['n_extracted']} events "
              f"(spike gate {res['threshold']:.2f} uV)",
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
    print("MEA Spike Waveform Pipeline (event-based)")
    print("=" * 60)
    print(f"Data file:  {args.data_file}")
    print(f"Waveforms:  {args.output}")
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
            results[ch] = process_channel(data, ch)
        save_waveforms(results, args.output, source_file=args.data_file)

    out_dir = Path(args.spike_html).parent
    for ch in sorted(results.keys()):
        if results[ch]["n_extracted"] > 0:
            gen_channel_interactive_html(
                results, ch, args.data_file,
                str(out_dir / INTERACTIVE_HTML_PATTERN.format(ch=ch)))

    gen_spike_waveform_html(results, args.spike_html,
                            interactive_pattern=INTERACTIVE_HTML_PATTERN)
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
    parser.add_argument("-v", "--visualize-only", action="store_true",
                        help="Only render HTML from previously extracted waveforms")
    args = parser.parse_args()
    main(args)
