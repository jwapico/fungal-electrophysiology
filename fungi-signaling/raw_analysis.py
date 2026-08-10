"""
raw_analysis.py  (rev 4 - event-based, timestamped runs, tile grid)

Pipeline for MEA compound-event waveform extraction (fungal recordings).

Each channel k of the recording v_k (in microvolts) is segmented into
multi-oscillation "events" and one waveform window is extracted per event:

  1. Robust noise floor:    sigma_k = 1.4826 * median |v_k - median(v_k)|
     (MAD; 1.4826 = 1 / Phi^{-1}(3/4) so sigma_k = sigma for Gaussian noise).
     The low envelope gate must not be inflated by the very events being
     segmented, hence MAD (robust) rather than std (inflated by events).
  2. Envelope gate:         b[n] = 1{|v_k[n]| >= G * sigma_k},  G = EVENT_GATE_SCALE.
  3. Excursions:            maximal runs [s_j, e_j] with b == 1.
  4. Gap merge:             runs with s_{j+1} - e_j - 1 <= tau_gap become ONE event
                            (tau_gap = EVENT_GAP_MS = 5 ms) -> events [S_i, E_i].
  5. Event gate:            keep i iff duration >= MIN_EVENT_MS  AND
                            |v_k[m_i]| >= S * sigma_k  (S = SPIKE_GATE_SCALE),
                            m_i = argmax_{[S_i,E_i]} |v_k|  (dominant deflection).
   6. Oscillation count:     count_oscillations(v_k[S_i:E_i], sigma_k):
                             major alternating turning points (swing >=
                             OSC_MIN_SWING_SIGMAS * sigma, spacing >=
                             OSC_MIN_PERIOD_MS) -> ceil((K-1)/2) cycles.
  7. Window:                W_i = clip((E_i+p_post) - (S_i-p_pre) + 1,
                            MIN_WINDOW_MS, MAX_WINDOW_MS); if clipped, centered
                            on m_i; clamped to trace bounds. Peak offset
                            r_i = m_i - start_i.

Output layout: each run writes into its own timestamped directory
   outputs/<YYYY-MM-DD_HH-MM-SS>/
     waveforms/waveforms.npz        (persisted events, one row per channel)
     html/waveforms_grid.html       (flex CSS grid of per-event tiles)
     html/all_ch_spikes.html        (full per-channel traces with events)
     html/interactive_ch_views/channel_N_interactive.html  (click-to-zoom)
     run_meta.json                  (parameters + per-channel summary)

Run from the project root:
    python raw_analysis.py
    python raw_analysis.py --data-file <path>
    python raw_analysis.py -o outputs/<ts>/waveforms/waveforms.npz -v   # re-render
"""
from __future__ import annotations

import argparse
import base64
import datetime
import io
import json
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

# --------------------------------------------------------------------------
# output layout (every run gets its own timestamped directory so iterations
# can be tracked; -v re-renders a previous run's .npz in place)
# --------------------------------------------------------------------------
OUTPUT_ROOT: str = "outputs"
TIMESTAMP_FORMAT: str = "%Y-%m-%d_%H-%M-%S"
WAVEFORM_REL_PATH: str = "waveforms/waveforms.npz"
SPIKE_HTML_REL_PATH: str = "html/waveforms_grid.html"
CHANNEL_HTML_REL_PATH: str = "html/all_ch_spikes.html"
INTERACTIVE_HTML_PATTERN: str = "channel_{ch}_interactive.html"
INTERACTIVE_HTML_DIR: str = "interactive_ch_views"
RUN_META_FILENAME: str = "run_meta.json"
PLOTLY_JS: str = "cdn"

# ---------------- event segmentation ----------------
def estimate_noise_mad(x: np.ndarray) -> float:
    """Robust noise scale: sigma = 1.4826 * MAD(x)."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))

EVENT_NOISE_ESTIMATOR: Callable[[np.ndarray], float] = estimate_noise_mad
EVENT_GATE_SCALE: float = 5.0    # low envelope gate  (x noise)
SPIKE_GATE_SCALE: float = 7.0    # high gate: an event's dominant peak must clear this
EVENT_GAP_MS: float = 5.0        # merge excursions <= this apart into one event
MIN_EVENT_MS: float = 1.5        # discard envelope blips shorter than this
OSC_MIN_SWING_SIGMAS: float = 5.0  # oscillation swing must exceed this x noise (uV)
OSC_MIN_PERIOD_MS: float = 1.0     # min spacing between counted turning points

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
TILE_DPI: int = 120          # dpi of the per-event grid tiles
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
    """Load interleaved int16 MEA binary as a memmap of shape (T, 64).

    The raw file stores sample y[n] for the interleaved stream; the channel
    index is k = n mod 64, the time index is t = n // 64. Reshaping the flat
    array into (-1, NUM_CHANNELS) de-interleaves it in one step:
        Y[t, k] = y[64*t + k],   t = 0..T-1,  k = 0..63.
    Memory-mapped (no full load) so that 900 s * 30 kHz * 64 * 2 bytes
    (approx. 3.4 GB) streams through without exhausting RAM.
    """
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
    """Classical noise scale:  sigma = sqrt( (1/N) * sum_n (x_n - mean)^2 ).

    Biased upward when large events are present in x (the events are part of
    the variance), which inflates the gate and truncates event envelopes.
    """
    return float(np.std(x))


def estimate_noise_mad(x: np.ndarray) -> float:
    """Robust noise scale (median absolute deviation).

        sigma_hat = 1.4826 * median_n |x_n - median_m x_m|

    The constant 1.4826 = 1 / Phi^{-1}(3/4) makes sigma_hat an unbiased
    estimate of the Gaussian standard deviation sigma, while remaining
    insensitive to the (rare, large) events that inflate the std.
    """
    median = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - median)))


def count_oscillations(
    seg: np.ndarray,
    noise: float,
    swing_sigmas: float = OSC_MIN_SWING_SIGMAS,
    min_period_ms: float = OSC_MIN_PERIOD_MS,
    sample_rate: int = SAMPLE_RATE_HZ,
) -> int:
    """Count the major oscillations inside one event's excursion segment.

    Formal description
    ------------------
    Let {e_k} be the local extrema (peaks and troughs) of the segment in time
    order, and s_k = |x[e_{k+1}] - x[e_k]| the swing between consecutive
    extrema. A turning point is *significant* when

        (i) it alternates sign with the previous significant one, and
        (ii) its swing to that neighbour exceeds s_min = c * sigma, where
             sigma is the channel's robust noise scale (OSC_MIN_SWING_SIGMAS
             x sigma) and
        (iii) it is spaced >= OSC_MIN_PERIOD_MS from the previous one.

    Keeping K significant turning points, the oscillation count is

        N = max(1, ceil((K - 1) / 2)),

    i.e. the number of full (plus any trailing half) cycles spanned by the
    significant turning points. Notes:

      * The swing threshold is *absolute in noise units*, not a fraction of
        the largest swing, so a single dominant deflection cannot dwarf the
        other oscillations (the failure mode of a max-relative threshold).
      * Requiring sign alternation prevents counting ripple that sits on a
        plateau (e.g. tiny bumps inside a long negative dip).
      * Counting runs over the *excursion* [onset, offset], NOT the padded
        window, so pre/post-event baseline micro-activity is not counted.
    """
    x = np.asarray(seg, dtype=float)
    if len(x) < 3:
        return 1
    peaks = scipy.signal.find_peaks(x)[0]
    troughs = scipy.signal.find_peaks(-x)[0]
    ext = sorted([(int(i), 1) for i in peaks] + [(int(i), -1) for i in troughs])
    if not ext:
        return 1
    idx = np.array([e[0] for e in ext], dtype=float)
    pol = np.array([e[1] for e in ext])
    vals = np.array([x[e[0]] for e in ext], dtype=float)
    s_min = swing_sigmas * noise
    min_period = min_period_ms * sample_rate / 1000.0

    # pass A: sign alternation + swing threshold
    kept = []
    for k in range(len(ext)):
        if not kept:
            kept.append(k)
            continue
        p = kept[-1]
        if pol[p] == pol[k]:
            continue
        if abs(vals[k] - vals[p]) < s_min:
            continue
        kept.append(k)
    # pass B: minimum spacing between turning points (merge the weaker swing)
    kept_b = []
    for k in kept:
        if kept_b and (idx[k] - idx[kept_b[-1]]) < min_period:
            prev_swing = (abs(vals[kept_b[-1]] - vals[kept_b[-2]])
                          if len(kept_b) >= 2 else 1e18)
            cur_swing = abs(vals[k] - vals[kept_b[-1]])
            if cur_swing > prev_swing:
                kept_b.pop()
            else:
                continue
        kept_b.append(k)
    return max(1, int(np.ceil((len(kept_b) - 1) / 2)))


def detect_events(
    voltage: np.ndarray,
    noise_estimator: Callable[[np.ndarray], float] = EVENT_NOISE_ESTIMATOR,
    gate_scale: float = EVENT_GATE_SCALE,
    spike_gate_scale: float = SPIKE_GATE_SCALE,
    gap_ms: float = EVENT_GAP_MS,
    min_event_ms: float = MIN_EVENT_MS,
    swing_sigmas: float = OSC_MIN_SWING_SIGMAS,
    min_period_ms: float = OSC_MIN_PERIOD_MS,
    sample_rate: int = SAMPLE_RATE_HZ,
) -> Tuple[List[Event], float, float, float]:
    """Segment the trace v into multi-oscillation events.

    Formal description
    ------------------
    Let sigma = noise_estimator(v), gate = G*sigma, spike_gate = S*sigma.

    1. Excursion set:  E = { n : |v[n]| >= gate }.
    2. Runs: maximal contiguous intervals [s_j, e_j] within E.
    3. Merge: run j+1 is joined to run j iff
           s_{j+1} - e_j - 1 <= gap_samples,   gap_samples = tau_gap * fs/1000
       producing events [S_i, E_i].
    4. Event gate: keep i iff
           (E_i - S_i + 1) >= min_event_samples  AND
           |v[m_i]| >= spike_gate,   m_i = argmax_{n in [S_i,E_i]} |v[n]|.
       m_i is the "dominant deflection" (largest |voltage| in the event).
    5. Oscillation count:  N_i = count_oscillations(v[S_i:E_i], sigma), the
       number of major alternating turning points (swing >= OSC_MIN_SWING_SIGMAS
       x sigma, spacing >= OSC_MIN_PERIOD_MS) halved into full cycles
       (see count_oscillations for the formal definition).

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
        n_osc = count_oscillations(seg, noise)
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

    Formal description
    ------------------
    Let pre = p_pre*fs/1000, post = p_post*fs/1000, w_min = MIN_WINDOW_MS,
    w_max = MAX_WINDOW_MS (in samples). The natural window spans

        W_nat = (E_i + post) - (S_i - pre) + 1.

    The extracted size is

        W_i = min(max(W_nat, w_min), w_max),

    and the left edge is
        start_i = S_i - pre            if w_min <= W_nat <= w_max,
        start_i = m_i - floor(W_i/2)   if W_nat was clipped
                                         (centers the window on the
                                          dominant deflection m_i),
    then clamped so [start_i, start_i + W_i) lies within the trace.

    Returns (waveform, t_i, W_i, start_i, r_i) with t_i = m_i/fs the event
    time and r_i = m_i - start_i the peak offset within the window, or None
    if W_i <= 0.
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
    """Full per-channel pipeline: detect events, extract one window each.

    For channel k: v_k = data[:, k] * q (q = VOLTAGE_SCALE uV/LSB), then
    detect_events() + extract_event_window() per event. Persisted features:

      * spike_times    - event time t_i = m_i / fs           (seconds)
      * window_sizes   - W_i (samples)
      * peak_indices   - r_i = m_i - start_i (dominant-peak offset in window)
      * window_starts  - start_i (absolute sample index)
      * n_oscillations - count_oscillations() result (major cycles)
      * amplitudes     - signed dominant deflection v_k[m_i]  (uV)
    """
    print(f"\nProcessing channel {channel}...")
    voltage = data[:, channel] * VOLTAGE_SCALE
    events, noise, gate, spike_gate = detect_events(voltage=voltage)
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
    """Persist all channels to a single self-describing .npz archive.

    Arrays are stored as N_ch object arrays (one row per channel) because
    event counts and window lengths vary per channel:
        waveforms[i] : M_i x W_j  (variable-length rows kept as Python list
                                    of np.ndarray, boxed in an object array)
    plus per-channel integer/float arrays (spike_times, window_sizes, ...)
    and scalar metadata (fs, unit, parameter constants, source file) so the
    archive is fully self-describing for later analysis (spike_sorting.py).
    """
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
    """Inverse of save_waveforms(): reconstruct the per-channel dict from .npz.

    Repopulates each channel's "waveforms", "spike_times", "window_sizes",
    "peak_indices", "window_starts", "n_oscillations", "amplitudes",
    "threshold", "gate", "std_dev", "n_events", "n_extracted" so a previous
    run can be re-rendered (-v) without re-reading the raw binary.
    """
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
        "        .tile-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); "
        "gap: 6px; }",
        "        .tile { display: block; background: white; border: 1px solid #ddd; "
        "border-radius: 4px; text-decoration: none; color: inherit; }",
        "        .tile:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.25); border-color: #888; }",
        "        .tile img { display: block; width: 100%; height: auto; }",
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


def _tile_figure(waveform: np.ndarray, start_idx: int, peak_idx: int,
                 n_osc: int, sample_rate: int = SAMPLE_RATE_HZ) -> Figure:
    """Build one small tile figure for a single event window.

    Axes:
      * x-axis = absolute recording time in seconds, x[n] = (start_idx + n)/fs,
        so the dominant peak appears at its true recording time t_i = m_i/fs.
      * y-axis = window voltage (uV), with ticks drawn at the window's min
        and max values (dashed horizontal lines) so peak-to-peak is read
        directly off the tile.
      * "N osc" label in the top-right corner of the axes: the computed
        oscillation count for manual verification.
    """
    fig, ax = plt.subplots(figsize=(1.7, 1.15))
    t = (start_idx + np.arange(len(waveform))) / sample_rate
    ax.plot(t, waveform, linewidth=0.7, color="#1f77b4")
    ax.axvline((start_idx + peak_idx) / sample_rate, color="k",
               linewidth=0.5, alpha=0.5, linestyle="--")
    ymin, ymax = float(waveform.min()), float(waveform.max())
    for yv in (ymin, ymax):
        ax.axhline(yv, color="#d62728", linewidth=0.4, alpha=0.6, linestyle=":")
    ax.set_yticks([ymin, ymax])
    ax.set_yticklabels([f"{ymin:.0f}", f"{ymax:.0f}"], fontsize=5)
    ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))
    ax.tick_params(labelsize=5, length=2)
    ax.set_xticks([t[0], t[-1]])
    ax.set_xticklabels([f"{t[0]:.2f}", f"{t[-1]:.2f}"], fontsize=5)
    ax.text(0.985, 0.985, f"{n_osc} osc", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color="#555")
    fig.tight_layout(pad=0.15)
    return fig


def gen_spike_waveform_html(results: Dict[int, Dict[str, Any]],
                            output_file: str,
                            interactive_pattern: str = INTERACTIVE_HTML_PATTERN,
                            interactive_dir: str = INTERACTIVE_HTML_DIR,
                            max_waveforms: int = MAX_WAVEFORMS_PER_CHANNEL,
                            dpi: int = TILE_DPI,
                            context_ms: float = INTERACTIVE_CONTEXT_MS) -> None:
    """Render the flex CSS-grid of per-event tiles, one tile per waveform.

    Each tile is a small standalone PNG (see _tile_figure) wrapped in an
    <a> that deep-links to the channel's interactive view zoomed on that
    event. The grid uses CSS auto-fill so tiles reflow with the browser
    width (responsive/flex layout); no image maps are needed.
    """
    print(f"\nGenerating waveform grid HTML: {output_file}")
    rng = np.random.default_rng(RANDOM_SEED)

    html_parts = _html_head("MEA Spike Waveform Grid")
    html_parts.append("    <p>Every extracted event window, per channel. "
                      "x-axis is absolute recording time (s); the red dotted "
                      "lines mark the window min/max voltage; the dashed line "
                      "is the dominant peak; the corner number is the computed "
                      "oscillation count. Click a tile to open the channel's "
                      "interactive view zoomed to that event.</p>")

    for ch in sorted(results.keys()):
        data = results[ch]
        waveforms = data["waveforms"]
        if len(waveforms) == 0:
            continue

        times = data["spike_times"]
        peak_pos = data["peak_indices"]
        starts = data["window_starts"]
        n_osc = data["n_oscillations"]

        if len(waveforms) > max_waveforms:
            sel = np.sort(rng.choice(len(waveforms), max_waveforms, replace=False))
        else:
            sel = np.arange(len(waveforms))
        sub_wf = [waveforms[i] for i in sel]
        sub_peak = peak_pos[sel]
        sub_start = starts[sel]
        sub_osc = n_osc[sel]

        tiles = []
        for j, wf in enumerate(sub_wf):
            fig = _tile_figure(wf, int(sub_start[j]), int(sub_peak[j]),
                               int(sub_osc[j]))
            img = _figure_to_base64(fig, dpi=dpi, tight=False)
            t = float(times[sel[j]])
            margin = (len(wf) / (2.0 * SAMPLE_RATE_HZ) + context_ms / 1000.0)
            t0 = max(0.0, t - margin)
            t1 = t + margin
            href = (f"{interactive_dir}/{interactive_pattern.format(ch=ch)}"
                    f"?t0={t0:.4f}&t1={t1:.4f}")
            tiles.append(
                f'        <a class="tile" href="{href}" target="_blank" '
                f'title="ch{ch} event t={t:.3f}s, {int(sub_osc[j])} osc">\n'
                f'          <img src="data:image/png;base64,{img}" '
                f'alt="ch{ch} t={t:.3f}s">\n'
                f'        </a>')

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
            <div class="tile-grid">
{chr(10).join(tiles)}
            </div>
        </div>
        """)

    html_parts.append("</body></html>")
    _write_html(output_file, html_parts)


def gen_channel_html(results: Dict[int, Dict[str, Any]],
                     output_file: str, raw_file: str,
                     ds_factor: int = CHANNEL_DS_FACTOR) -> None:
    """Render one full-trace figure per channel (downsampled overview).

    The trace is decimated by CHANNEL_DS_FACTOR (y -> y[::ds]) for a light
    PNG; detected dominant peaks are overlaid as red dots and the spike gate
    as a dashed line. Purely visual, no analysis.
    """
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
    """One self-contained plotly view per channel, with click-to-zoom.

    The plot stacks (1) a heavily downsampled full-trace overview
    (decimation INTERACTIVE_OVERVIEW_DS), (2) high-resolution context
    segments around every detected peak (windowed +/- SPIKE_CONTEXT_MS,
    decimation INTERACTIVE_SPIKE_DS, joined with NaN gaps), and (3) red
    markers at the dominant peaks. The URL query ?t0=..&t1=.. selects the
    initial x-axis range (used by the grid tiles) via a JS snippet injected
    before </body>:  Plotly.relayout('interactive', {'xaxis.range': [t0, t1]}).
    """
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


def _resolve_run_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    """Determine the run directory and all output paths.

    Normal run: creates a fresh timestamped run dir
        <OUTPUT_ROOT>/<YYYY-MM-DD_HH-MM-SS>/
    so every invocation is archived for iteration tracking (old runs are
    never overwritten).

    Visualize-only (-v): re-derives the run dir from the supplied .npz path.
    If the npz lives at <run_dir>/waveforms/waveforms.npz the run dir is
    npz.parent.parent; otherwise npz.parent is used. HTML is (re)written
    into <run_dir>/html/, so a previous run can be re-rendered in place.

    Returns (run_dir, npz_path, grid_path, channel_path, interactive_dir).
    """
    if args.visualize_only:
        if not args.output:
            raise ValueError("-v requires -o <path-to-waveforms.npz>")
        npz_path = Path(args.output)
        if npz_path.parent.name == "waveforms" and npz_path.name == "waveforms.npz":
            run_dir = npz_path.parent.parent
        else:
            run_dir = npz_path.parent
    else:
        run_dir = (Path(args.out_root)
                   / datetime.datetime.now().strftime(TIMESTAMP_FORMAT))
        run_dir.mkdir(parents=True, exist_ok=True)
        npz_path = run_dir / WAVEFORM_REL_PATH

    html_dir = run_dir / "html"
    grid_path = args.spike_html or str(html_dir / "waveforms_grid.html")
    channel_path = args.channel_html or str(html_dir / "all_ch_spikes.html")
    return run_dir, npz_path, grid_path, channel_path, html_dir


def _write_run_meta(run_dir: Path, args: argparse.Namespace,
                    results: Dict[int, Dict[str, Any]], npz_path: Path) -> None:
    """Write run_meta.json: parameters, timestamps and per-channel summary.

    Mirrors the constants in this module so any archived run can be fully
    reconstructed / cross-referenced during writing of the methods section.
    """
    per_channel = {}
    for ch in sorted(results.keys()):
        d = results[ch]
        per_channel[str(ch)] = {
            "n_events": int(d["n_events"]),
            "n_extracted": int(d["n_extracted"]),
            "spike_gate_uV": round(float(d["threshold"]), 3),
            "envelope_gate_uV": round(float(d["gate"]), 3),
            "noise_mad_uV": round(float(d["std_dev"]), 3),
        }
    meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_file": args.data_file,
        "waveforms_npz": str(npz_path),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "voltage_scale_uv_per_lsb": VOLTAGE_SCALE,
        "parameters": {
            "event_gate_scale": EVENT_GATE_SCALE,
            "spike_gate_scale": SPIKE_GATE_SCALE,
            "event_gap_ms": EVENT_GAP_MS,
            "min_event_ms": MIN_EVENT_MS,
            "osc_min_swing_sigmas": OSC_MIN_SWING_SIGMAS,
            "osc_min_period_ms": OSC_MIN_PERIOD_MS,
            "noise_estimator": EVENT_NOISE_ESTIMATOR.__name__ if hasattr(
                EVENT_NOISE_ESTIMATOR, "__name__") else "lambda(MAD)",
            "pre_pad_ms": PRE_PAD_MS,
            "post_pad_ms": POST_PAD_MS,
            "min_window_ms": MIN_WINDOW_MS,
            "max_window_ms": MAX_WINDOW_MS,
        },
        "channels": per_channel,
        "totals": {
            "events": int(sum(d["n_events"] for d in results.values())),
            "extracted": int(sum(d["n_extracted"] for d in results.values())),
        },
    }
    (run_dir / RUN_META_FILENAME).write_text(
        json.dumps(meta, indent=2))
    print(f"Saved run metadata to: {run_dir / RUN_META_FILENAME}")


def main(args: argparse.Namespace) -> None:
    run_dir, npz_path, grid_path, channel_path, html_dir = _resolve_run_paths(args)
    interactive_dir = html_dir / INTERACTIVE_HTML_DIR

    print("=" * 60)
    print("MEA Spike Waveform Pipeline (event-based)")
    print("=" * 60)
    print(f"Data file:  {args.data_file}")
    print(f"Run dir:    {run_dir}")
    print(f"Waveforms:  {npz_path}")
    print("=" * 60)

    if args.visualize_only:
        results = load_waveforms(npz_path)
        if not results:
            print("Error: No data loaded. Run without -v first.")
            return
    else:
        data = load_raw_data(args.data_file)
        results: Dict[int, Dict[str, Any]] = {}
        for ch in range(data.shape[1]):
            results[ch] = process_channel(data, ch)
        save_waveforms(results, npz_path, source_file=args.data_file)
        _write_run_meta(run_dir, args, results, npz_path)

    for ch in sorted(results.keys()):
        if results[ch]["n_extracted"] > 0:
            gen_channel_interactive_html(
                results, ch, args.data_file,
                str(interactive_dir / INTERACTIVE_HTML_PATTERN.format(ch=ch)))

    gen_spike_waveform_html(results, grid_path,
                            interactive_pattern=INTERACTIVE_HTML_PATTERN,
                            interactive_dir=INTERACTIVE_HTML_DIR)
    gen_channel_html(results, channel_path, raw_file=args.data_file)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Open {grid_path} in your browser")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MEA spike waveform extraction and visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-d", "--data-file", type=str, default=RAW_DATA_FILE,
                        help="Path to raw MEA binary recording")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output .npz path; default <out-root>/<ts>/"
                             "waveforms/waveforms.npz (in -v mode, required: "
                             "path to a previous run's .npz)")
    parser.add_argument("--out-root", type=str, default=OUTPUT_ROOT,
                        help="Directory under which timestamped runs are stored")
    parser.add_argument("-s", "--spike-html", type=str, default=None,
                        help="Output HTML path for the waveform grid "
                             "(default <run-dir>/html/waveforms_grid.html)")
    parser.add_argument("-c", "--channel-html", type=str, default=None,
                        help="Output HTML path for the full-trace view "
                             "(default <run-dir>/html/all_ch_spikes.html)")
    parser.add_argument("-v", "--visualize-only", action="store_true",
                        help="Only render HTML from previously extracted "
                             "waveforms (-o points at a previous .npz)")
    args = parser.parse_args()
    main(args)
