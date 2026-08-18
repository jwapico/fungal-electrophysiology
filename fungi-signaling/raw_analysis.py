"""
raw_analysis.py  (rev 5 - persisted raw + smoothed waveforms)

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
                             #major alternating turning points (swing >=
                             OSC_MIN_SWING_SIGMAS * sigma, spacing >=
                             OSC_MIN_PERIOD_MS) -> ceil((K-1)/2) cycles.
  7. Window:                W_i = (E_i-S_i+1) + 2*pad, with
                             pad = max(PAD_FRACTION*(E_i-S_i+1), MIN_PAD_S*fs),
                             i.e. symmetric padding proportional to the
                             excursion length but never less than MIN_PAD_S on
                             each side; clamped to MIN_WINDOW_MS but NOT to an
                             upper bound (no max clipping). The window is
                             aligned on the excursion (start at S_i - pre),
                             with peak offset r_i = m_i - start_i.
  8. Smoothing:             every raw window is smoothed with a zero-phase
                            Savitzky-Golay filter (SMOOTH_METHOD, every width
                            in SMOOTH_WINDOWS_MS, polyorder SMOOTH_POLYORDER)
                            and ALL variants are PERSISTED ALONGSIDE the raw
                            one. Downstream analysis (spike_sorting.py) and
                            every visualization read the arrays from the npz;
                            nothing is re-derived at render time.

Output layout: each run writes into its own timestamped directory
   outputs/<YYYY-MM-DD_HH-MM-SS>/
     waveforms/waveforms.npz        (persisted events: raw + smoothed windows)
     html/waveforms_grid.html       (flex CSS grid of per-event tiles)
     html/all_ch_spikes.html        (full per-channel traces with events)
     html/interactive_ch_views/channel_N_interactive.html  (click-to-zoom)
     run_meta.json                  (parameters + per-channel summary)

Every run also regenerates the static entry point index.html next to this
script (fungi-signaling/index.html), which links to the newest run's grid /
all-channels / metadata pages and lists every older run. It is written by
this script (write_output_index) with plain relative links, so it works when
opened directly from disk (no server or JavaScript required).

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
import os
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np
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
EVENT_GATE_SCALE: float = 5.0    # low envelope gate  (x noise)
SPIKE_GATE_SCALE: float = 5.0    # high gate: an event's dominant peak must clear this
EVENT_GAP_MS: float = 5.0        # merge excursions <= this apart into one event
MIN_EVENT_MS: float = 0.5        # discard envelope blips shorter than this
OSC_MIN_SWING_SIGMAS: float = 5.0  # oscillation swing must exceed this x noise (uV)
OSC_MIN_PERIOD_MS: float = 1.0     # min spacing between counted turning points

# ---------------- window extraction ----------------
EXTENT_SIGMAS: float = 2.0   # window base = contiguous span where |v| > this x noise
PAD_FRACTION: float = 0.25      # pre/post pad = this fraction of the extent length
MIN_PAD_S: float = 0.02        # floor on the pre/post pad (seconds): the window
                               # always extends at least this far each side, even
                               # when the proportional pad would be smaller
MIN_WINDOW_MS: float = 3.0

# ---------------- visualization ----------------
CHANNEL_DS_FACTOR: int = 10
SPIKE_WINDOWS_LIMIT: int = 50  # default # of waveform tiles shown per channel in
                               # the grid; a page toggle reveals all of them
FIGURE_DPI: int = 80
TILE_DPI: int = 120          # dpi of the per-event grid tiles
TRACE_DPI: int = 100

INTERACTIVE_OVERVIEW_DS: int = 200
INTERACTIVE_SPIKE_DS: int = 4
SPIKE_CONTEXT_MS: float = 200.0
INTERACTIVE_CONTEXT_MS: float = 100.0

# ---------------- persisted waveform smoothing ----------------
# Applied ONCE during extraction (process_channel): every raw window is
# smoothed with EVERY width in SMOOTH_WINDOWS_MS, and all variants are
# stored in the npz (one object array per width, keyed
# "smooth_waveforms_<w>ms"). Visualizations and spike_sorting.py read the
# saved arrays; smoothing is never recomputed at render time, so the
# displayed/sorted data always equals the persisted data. The Savitzky-Golay
# kernel is symmetric, hence zero-phase: peak locations cannot shift relative
# to the raw waveform.
SMOOTH_METHOD: str = "savgol"                      # "savgol" | "none"
SMOOTH_WINDOWS_MS: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)  # savgol widths (ms)
SMOOTH_POLYORDER: int = 4                          # savgol polynomial order
SMOOTH_SHOW_BY_DEFAULT: bool = False               # grid default: raw shown


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
    recording = raw.reshape(-1, NUM_CHANNELS)
    print(f"  Loaded {recording.shape[0]:,} samples, {recording.shape[1]} channels")
    return recording


def estimate_noise_mad(samples: np.ndarray) -> float:
    """Robust noise scale (median absolute deviation).

        sigma_hat = 1.4826 * median_n |x_n - median_m x_m|

    The constant 1.4826 = 1 / Phi^{-1}(3/4) makes sigma_hat an unbiased
    estimate of the Gaussian standard deviation sigma, while remaining
    insensitive to the (rare, large) events that would inflate the std.
    """
    median = float(np.median(samples))
    return float(1.4826 * np.median(np.abs(samples - median)))


def count_oscillations(
    excursion: np.ndarray,
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
    samples = np.asarray(excursion, dtype=float)
    if len(samples) < 3:
        return 1

    # All local maxima and minima, each tagged with its polarity (+1/-1).
    peaks = scipy.signal.find_peaks(samples)[0]
    troughs = scipy.signal.find_peaks(-samples)[0]
    extrema = sorted([(int(i), 1) for i in peaks] + [(int(i), -1) for i in troughs])
    if not extrema:
        return 1
    extremum_indices = np.array([e[0] for e in extrema], dtype=float)
    extremum_polarities = np.array([e[1] for e in extrema])
    extremum_values = np.array([samples[e[0]] for e in extrema], dtype=float)
    min_swing = swing_sigmas * noise
    min_period_samples = min_period_ms * sample_rate / 1000.0

    # pass A: sign alternation + absolute swing threshold. The threshold is
    # in noise units (not a fraction of the largest swing), so a single
    # dominant deflection cannot dwarf the other oscillations.
    pass_a_indices = []
    for candidate_idx in range(len(extrema)):
        if not pass_a_indices:
            pass_a_indices.append(candidate_idx)
            continue
        last_kept_idx = pass_a_indices[-1]
        if extremum_polarities[last_kept_idx] == extremum_polarities[candidate_idx]:
            continue  # same sign: ripple on a plateau, not an oscillation
        if abs(extremum_values[candidate_idx] - extremum_values[last_kept_idx]) \
                < min_swing:
            continue
        pass_a_indices.append(candidate_idx)

    # pass B: minimum spacing between turning points; when two are too close,
    # keep the one with the larger swing and drop the other.
    kept_indices = []
    for candidate_idx in pass_a_indices:
        if kept_indices and (extremum_indices[candidate_idx]
                             - extremum_indices[kept_indices[-1]]) < min_period_samples:
            previous_swing = (abs(extremum_values[kept_indices[-1]]
                                  - extremum_values[kept_indices[-2]])
                              if len(kept_indices) >= 2 else 1e18)
            current_swing = abs(extremum_values[candidate_idx]
                                - extremum_values[kept_indices[-1]])
            if current_swing > previous_swing:
                kept_indices.pop()  # the candidate is stronger: replace
            else:
                continue  # the earlier extremum wins
        kept_indices.append(candidate_idx)

    # K significant turning points span K-1 swings => K-1 half cycles.
    return max(1, int(np.ceil((len(kept_indices) - 1) / 2)))


def detect_events(
    voltage: np.ndarray,
    noise_estimator: Callable[[np.ndarray], float] = estimate_noise_mad,
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

    # Samples whose |amplitude| clears the low envelope gate.
    above_gate_indices = np.flatnonzero(np.abs(voltage) > gate)
    if len(above_gate_indices) == 0:
        return [], noise, gate, spike_gate

    # Maximal contiguous runs within the excursion set.
    run_breaks = np.flatnonzero(np.diff(above_gate_indices) > 1)
    run_starts = np.r_[above_gate_indices[0], above_gate_indices[run_breaks + 1]]
    run_ends = np.r_[above_gate_indices[run_breaks], above_gate_indices[-1]]

    # Merge runs separated by <= gap_samples into a single event.
    gap_samples = int(gap_ms * sample_rate / 1000)
    merged_runs: List[List[int]] = []
    for run_start, run_end in zip(run_starts, run_ends):
        if merged_runs and (run_start - merged_runs[-1][1] - 1) <= gap_samples:
            merged_runs[-1][1] = run_end
        else:
            merged_runs.append([int(run_start), int(run_end)])

    min_event_samples = int(min_event_ms * sample_rate / 1000)
    events: List[Event] = []
    for run_start, run_end in merged_runs:
        if run_end - run_start + 1 < min_event_samples:
            continue  # envelope blip shorter than MIN_EVENT_MS: not an event
        excursion = voltage[run_start:run_end + 1]
        dominant_idx_in_excursion = int(np.argmax(np.abs(excursion)))
        dominant_sample = run_start + dominant_idx_in_excursion
        if abs(voltage[dominant_sample]) < spike_gate:
            continue  # no deflection strong enough to be a real spike
        n_oscillations = count_oscillations(excursion, noise)
        events.append(Event(onset=run_start, offset=run_end,
                            main=dominant_sample,
                            n_oscillations=n_oscillations))

    return events, noise, gate, spike_gate


def extract_event_window(
    event: Event,
    voltage: np.ndarray,
    noise: float,
    extent_sigmas: float = EXTENT_SIGMAS,
    pad_fraction: float = PAD_FRACTION,
    min_pad_s: float = MIN_PAD_S,
    min_window_ms: float = MIN_WINDOW_MS,
    sample_rate: int = SAMPLE_RATE_HZ,
) -> Optional[Tuple[np.ndarray, float, int, int, int]]:
    """Window around a whole event, centered on the dominant peak.

    Formal description
    ------------------
    The *extent* is the contiguous span of samples around the event's
    excursion [S_i, E_i] on which |v| stays above a LOW threshold:

        [L_i, R_i] = maximal run containing [S_i, E_i] with |v| > e*sigma,

    e = EXTENT_SIGMAS (default 2, vs the 5-sigma detection gate). This
    captures the low-amplitude leading/trailing oscillations that fall below
    the detection gate but are still part of the event.

    The pad is proportional to the extent (adaptive: small spikes get small
    margins, large spikes get large ones) but never below a floor:

        pre = post = max(round(PAD_FRACTION * L_ext), round(MIN_PAD_S * fs)),
        L_ext = R_i - L_i + 1.

    The natural window spans

        W_nat = L_ext + pre + post
              = L_ext + 2*max(round(PAD_FRACTION*L_ext), round(MIN_PAD_S*fs)).

    The extracted size is

        W_i = max(W_nat, w_min),

    with w_min = MIN_WINDOW_MS in samples. There is NO upper clipping: a
    large extent is allowed to keep its full, proportionate window.

    The window is CENTERED on the dominant deflection m_i:

        start_i = m_i - floor(W_i/2),

    so every event's peak lands at the same relative position (family sorting
    requires this alignment), then clamped so [start_i, start_i + W_i) lies
    within the trace.

    Returns (waveform, t_i, W_i, start_i, r_i) with t_i = m_i/fs the event
    time and r_i = m_i - start_i the peak offset within the window (should be
    ~W_i/2), or None if W_i <= 0.
    """
    extent_start = event.onset
    extent_end = event.offset
    extent_threshold = extent_sigmas * noise
    # Extend the excursion left/right while |v| stays above the low threshold;
    # this recovers the low-amplitude leading/trailing oscillations.
    while extent_start > 0 and abs(voltage[extent_start - 1]) > extent_threshold:
        extent_start -= 1
    while extent_end < len(voltage) - 1 and abs(voltage[extent_end + 1]) > extent_threshold:
        extent_end += 1
    extent_len = extent_end - extent_start + 1

    # Adaptive symmetric padding: proportional to the extent, floored by
    # MIN_PAD_S so short events still get a usable baseline on each side.
    pad = max(int(round(pad_fraction * extent_len)),
              int(round(min_pad_s * sample_rate)))
    min_window = int(min_window_ms * sample_rate / 1000)

    natural_window = extent_len + 2 * pad
    window_len = max(natural_window, min_window)
    if window_len <= 0:
        return None
    # Center the window on the dominant deflection so every event's peak lands
    # at the same relative position (required for family sorting).
    start = event.main - window_len // 2
    start = int(max(0, min(start, len(voltage) - window_len)))
    end = start + window_len

    waveform = voltage[start:end].copy()
    return (waveform, event.main / sample_rate, window_len, start,
            event.main - start)


def process_channel(data: np.ndarray, channel: int) -> Dict[str, Any]:
    """Full per-channel pipeline: detect events, extract one window each.

    For channel k: v_k = data[:, k] * q (q = VOLTAGE_SCALE uV/LSB), then
    detect_events() (MAD-based noise sigma) + extract_event_window() per
    event. Every raw window is additionally smoothed with smooth_waveform()
    at EVERY width in SMOOTH_WINDOWS_MS (Savitzky-Golay, polyorder
    SMOOTH_POLYORDER); the raw window and all smoothed variants are returned
    and later persisted, so downstream analysis and every visualization read
    one source of truth (the saved arrays) and never re-derive anything at
    render time.

    Persisted features per event:
      * spike_times       - event time t_i = m_i / fs          (seconds)
      * window_sizes      - W_i (samples)
      * peak_indices      - r_i = m_i - start_i (dominant-peak offset)
      * window_starts     - start_i (absolute sample index)
      * n_oscillations    - count_oscillations() result (major cycles)
      * amplitudes        - signed dominant deflection v_k[m_i]  (uV)
    """
    print(f"\nProcessing channel {channel}...")
    voltage = data[:, channel] * VOLTAGE_SCALE
    events, noise, gate, spike_gate = detect_events(voltage)
    print(f"  Found {len(events)} events (gate {gate:.2f} uV, "
          f"spike gate {spike_gate:.2f} uV, noise {noise:.2f} uV)")

    waveforms: List[np.ndarray] = []
    smooth_waveforms: Dict[float, List[np.ndarray]] = {
        window_ms: [] for window_ms in SMOOTH_WINDOWS_MS}
    event_times: List[float] = []
    window_sizes: List[int] = []
    window_starts: List[int] = []
    peak_positions: List[int] = []
    n_oscillations: List[int] = []
    amplitudes: List[float] = []

    for event in events:
        result = extract_event_window(event, voltage, noise=noise)
        if result is None:
            continue
        waveform, event_time, window_size, start_idx, peak_idx = result
        waveforms.append(waveform)
        # Smooth once at extraction time, at every persisted width; the
        # variants are stored alongside the raw window (single source of
        # truth for downstream analyses and every visualization).
        for window_ms in SMOOTH_WINDOWS_MS:
            smooth_waveforms[window_ms].append(
                smooth_waveform(waveform, window_ms=window_ms))
        event_times.append(event_time)
        window_sizes.append(window_size)
        window_starts.append(start_idx)
        peak_positions.append(peak_idx)
        n_oscillations.append(event.n_oscillations)
        amplitudes.append(float(voltage[event.main]))

    n_extracted = len(waveforms)
    print(f"  Extracted {n_extracted} waveforms")

    return {
        "waveforms": waveforms,
        "smooth_waveforms": smooth_waveforms,
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
        # smoothing parameters used at extraction time, so re-renders can
        # label the persisted data truthfully even if constants change later
        "smooth_method": SMOOTH_METHOD,
        "smooth_windows_ms": list(SMOOTH_WINDOWS_MS),
        "smooth_polyorder": SMOOTH_POLYORDER,
    }


def save_waveforms(results: Dict[int, Dict[str, Any]],
                   output_file: str, source_file: str) -> None:
    """Persist all channels to a single self-describing .npz archive.

    Arrays are stored as N_ch object arrays (one row per channel) because
    event counts and window lengths vary per channel:
        waveforms[i]             : raw windows of channel i (list of np.ndarray)
        smooth_waveforms_<w>ms[i]: Savitzky-Golay smoothed windows of channel
                                   i at width w (one key per width in
                                   SMOOTH_WINDOWS_MS)
    plus per-channel integer/float arrays (spike_times, window_sizes, ...),
    the smoothing widths actually used, and scalar metadata (fs, unit,
    parameter constants, source file). The archive is therefore fully
    self-describing for later analysis (spike_sorting.py) and re-rendering
    (-v): every downstream consumer reads the saved arrays, never re-derives
    them.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels = sorted(results.keys())
    n_ch = len(channels)

    waveforms = np.empty(n_ch, dtype=object)
    smooth_waveforms_by_width = {
        window_ms: np.empty(n_ch, dtype=object)
        for window_ms in SMOOTH_WINDOWS_MS}
    spike_times = np.empty(n_ch, dtype=object)
    window_sizes = np.empty(n_ch, dtype=object)
    peak_positions = np.empty(n_ch, dtype=object)
    window_starts = np.empty(n_ch, dtype=object)
    oscillation_counts = np.empty(n_ch, dtype=object)
    amplitudes = np.empty(n_ch, dtype=object)
    thresholds = np.zeros(n_ch)
    gates = np.zeros(n_ch)
    noise_stds = np.zeros(n_ch)
    n_events = np.zeros(n_ch, dtype=int)
    n_extracted = np.zeros(n_ch, dtype=int)

    for channel_idx, channel in enumerate(channels):
        channel_data = results[channel]
        waveforms[channel_idx] = np.array(channel_data["waveforms"], dtype=object)
        for window_ms in SMOOTH_WINDOWS_MS:
            smooth_waveforms_by_width[window_ms][channel_idx] = np.array(
                channel_data["smooth_waveforms"][window_ms], dtype=object)
        spike_times[channel_idx] = np.asarray(channel_data["spike_times"], dtype=float)
        window_sizes[channel_idx] = np.asarray(channel_data["window_sizes"], dtype=int)
        peak_positions[channel_idx] = np.asarray(channel_data["peak_indices"], dtype=int)
        window_starts[channel_idx] = np.asarray(channel_data["window_starts"], dtype=int)
        oscillation_counts[channel_idx] = np.asarray(
            channel_data["n_oscillations"], dtype=int)
        amplitudes[channel_idx] = np.asarray(channel_data["amplitudes"], dtype=float)
        thresholds[channel_idx] = channel_data["threshold"]
        gates[channel_idx] = channel_data["gate"]
        noise_stds[channel_idx] = channel_data["std_dev"]
        n_events[channel_idx] = channel_data["n_events"]
        n_extracted[channel_idx] = channel_data["n_extracted"]

    payload = {
        "channels": np.asarray(channels, dtype=int),
        "waveforms": waveforms,
        "spike_times": spike_times,
        "window_sizes": window_sizes,
        "peak_positions": peak_positions,
        "window_starts": window_starts,
        "n_oscillations": oscillation_counts,
        "amplitudes": amplitudes,
        "thresholds": thresholds,
        "gates": gates,
        "stds": noise_stds,
        "n_events": n_events,
        "n_extracted": n_extracted,
        "sample_rate": SAMPLE_RATE_HZ,
        "unit": "uV",
        "source_file": source_file,
        "min_window_ms": MIN_WINDOW_MS,
        "min_pad_s": MIN_PAD_S,
        "pad_fraction": PAD_FRACTION,
        "extent_sigmas": EXTENT_SIGMAS,
        "event_gate_scale": EVENT_GATE_SCALE,
        "spike_gate_scale": SPIKE_GATE_SCALE,
        "smooth_method": SMOOTH_METHOD,
        "smooth_windows_ms": np.asarray(list(SMOOTH_WINDOWS_MS), dtype=float),
        "smooth_polyorder": SMOOTH_POLYORDER,
        "smooth_show_by_default": SMOOTH_SHOW_BY_DEFAULT,
    }
    for window_ms in SMOOTH_WINDOWS_MS:
        payload[f"smooth_waveforms_{int(window_ms)}ms"] = \
            smooth_waveforms_by_width[window_ms]
    np.savez_compressed(output_path, **payload)
    print(f"\nSaved waveforms to: {output_file}")


def load_waveforms(output_file: str) -> Dict[int, Dict[str, Any]]:
    """Inverse of save_waveforms(): reconstruct the per-channel dict from .npz.

    Repopulates each channel's "waveforms", "smooth_waveforms" (a dict
    mapping each persisted smoothing width in ms -> list of smoothed arrays),
    "spike_times", "window_sizes", "peak_indices", "window_starts",
    "n_oscillations", "amplitudes", "threshold", "gate", "std_dev",
    "n_events", "n_extracted" and the smoothing parameters recorded in the
    archive, so a previous run can be re-rendered (-v) without re-reading the
    raw binary and without re-deriving any waveform.

    Backward compatibility: archives written before multi-width smoothing
    (rev < 6) contain a single "smooth_waveforms" array for the single width
    stored in "smooth_window_ms"; those are loaded with that width only.
    Archives that predate smoothing entirely (no "smooth_waveforms" key)
    are loaded with the smoothed copies mirroring the raw one and a warning
    is printed.
    """
    output_path = Path(output_file)
    if not output_path.exists():
        print(f"Error: Output file does not exist: {output_file}")
        return {}
    archive = np.load(output_path, allow_pickle=True)
    channels = archive["channels"].tolist()

    # Smoothing widths recorded in the archive; fall back to the current
    # module constants only when the archive does not carry them (legacy).
    smooth_windows_ms = list(archive["smooth_windows_ms"]) \
        if "smooth_windows_ms" in archive.files else list(SMOOTH_WINDOWS_MS)
    smooth_method = str(archive["smooth_method"]) if "smooth_method" in archive.files \
        else SMOOTH_METHOD
    smooth_polyorder = int(archive["smooth_polyorder"]) \
        if "smooth_polyorder" in archive.files else SMOOTH_POLYORDER

    # Multi-width archives (rev 6+) store one key per width. Legacy archives
    # store a single "smooth_waveforms" array under the width in
    # "smooth_window_ms"; archives predating smoothing have neither.
    has_multi_width = any(f"smooth_waveforms_{int(w)}ms" in archive.files
                          for w in smooth_windows_ms)
    has_legacy_smooth = "smooth_waveforms" in archive.files
    legacy_width: float = float(SMOOTH_WINDOWS_MS[0])
    if has_legacy_smooth and not has_multi_width:
        legacy_width = float(archive["smooth_window_ms"]) \
            if "smooth_window_ms" in archive.files else float(SMOOTH_WINDOWS_MS[0])
        smooth_windows_ms = [legacy_width]
        print(f"  [note] legacy archive: single smoothing width {legacy_width} ms "
              "(re-run extraction to persist all widths)")
    if not has_multi_width and not has_legacy_smooth:
        print("  [warning] archive predates smoothing; smoothed arrays will "
              "mirror raw. Re-run extraction to persist smoothed waveforms.")

    results: Dict[int, Dict[str, Any]] = {}
    for channel_idx, channel in enumerate(channels):
        channel_id = int(channel)
        raw_waveforms = list(archive["waveforms"][channel_idx])
        if has_multi_width:
            smooth_waveforms = {
                float(w): list(archive[f"smooth_waveforms_{int(w)}ms"][channel_idx])
                for w in smooth_windows_ms}
        elif has_legacy_smooth:
            smooth_waveforms = {
                legacy_width: list(archive["smooth_waveforms"][channel_idx])}
        else:
            smooth_waveforms = {float(w): raw_waveforms for w in smooth_windows_ms}
        results[channel_id] = {
            "waveforms": raw_waveforms,
            "smooth_waveforms": smooth_waveforms,
            "spike_times": np.asarray(archive["spike_times"][channel_idx]),
            "window_sizes": np.asarray(archive["window_sizes"][channel_idx]),
            "peak_indices": np.asarray(archive["peak_positions"][channel_idx],
                                       dtype=int),
            "window_starts": np.asarray(archive["window_starts"][channel_idx],
                                        dtype=int),
            "n_oscillations": np.asarray(archive["n_oscillations"][channel_idx],
                                         dtype=int),
            "amplitudes": np.asarray(archive["amplitudes"][channel_idx], dtype=float),
            "threshold": float(archive["thresholds"][channel_idx]),
            "gate": float(archive["gates"][channel_idx]),
            "std_dev": float(archive["stds"][channel_idx]),
            "n_events": int(archive["n_events"][channel_idx]),
            "n_extracted": int(archive["n_extracted"][channel_idx]),
            "smooth_method": smooth_method,
            "smooth_windows_ms": smooth_windows_ms,
            "smooth_polyorder": smooth_polyorder,
        }
        print(f"  Loaded channel {channel_id}: "
              f"{results[channel_id]['n_extracted']} waveforms")
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


def _save_figure_png(fig: Figure, output_path: Path, dpi: int = FIGURE_DPI,
                     tight: bool = True) -> None:
    """Render a figure to a PNG file (used for per-event grid tiles).

    Rendered with the same bbox/dpi handling as _figure_to_base64 so the
    on-disk tiles look identical to the embedded ones; the array data that
    produced them is the persisted npz array (no loss).
    """
    fig.savefig(output_path, format="png", dpi=dpi,
                bbox_inches="tight" if tight else None)
    plt.close(fig)


def smooth_waveform(waveform: np.ndarray,
                    method: str = SMOOTH_METHOD,
                    window_ms: float = SMOOTH_WINDOWS_MS[0],
                    polyorder: int = SMOOTH_POLYORDER,
                    sample_rate: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """Zero-phase Savitzky-Golay smoothing applied at extraction time.

    For every sample, fit a least-squares polynomial of degree `polyorder`
    to the symmetric window around it and take the fitted value at the
    center. The kernel is symmetric, so the filter is zero-phase by
    construction: peak locations cannot shift relative to the raw waveform.
    The polynomial fit tracks the smooth macro deflection and discards
    high-frequency noise, keeping peak amplitudes ~intact (a moving average
    or FIR low-pass would flatten peaks and round corners).

    Called once per (window, width) in process_channel(); the smoothed
    results are persisted next to the raw window in the npz (single source
    of truth) and are read back by the visualizations - they are never
    recomputed at render time. `method == "none"` returns the input
    unchanged.
    """
    if method != "savgol":
        return waveform
    window_len = int(round(window_ms * sample_rate / 1000)) | 1  # force odd
    window_len = max(window_len, polyorder + 1)
    if window_len % 2 == 0:
        window_len += 1
    if window_len < 3 or window_len >= len(waveform):
        return waveform
    return np.asarray(scipy.signal.savgol_filter(waveform, window_len, polyorder),
                       dtype=float)


def _tile_figure(waveform: np.ndarray, start_idx: int, peak_idx: int,
                 n_osc: int, sample_rate: int = SAMPLE_RATE_HZ,
                 ylim: Optional[Tuple[float, float]] = None) -> Figure:
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
    time_axis = (start_idx + np.arange(len(waveform))) / sample_rate
    ax.plot(time_axis, waveform, linewidth=0.7, color="#1f77b4")
    ax.axvline((start_idx + peak_idx) / sample_rate, color="k",
               linewidth=0.5, alpha=0.5, linestyle="--")
    ymin, ymax = (float(ylim[0]), float(ylim[1])) if ylim is not None \
        else (float(waveform.min()), float(waveform.max()))
    for value in (ymin, ymax):
        ax.axhline(value, color="#d62728", linewidth=0.4, alpha=0.6, linestyle=":")
    ax.set_yticks([ymin, ymax])
    ax.set_yticklabels([f"{ymin:.0f}", f"{ymax:.0f}"], fontsize=5)
    ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))
    ax.tick_params(labelsize=5, length=2)
    ax.set_xticks([time_axis[0], time_axis[-1]])
    ax.set_xticklabels([f"{time_axis[0]:.4f}", f"{time_axis[-1]:.4f}"], fontsize=5)
    ax.text(0.985, 0.985, f"{n_osc} osc", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color="#555")
    fig.tight_layout(pad=0.15)
    return fig


def gen_spike_waveform_html(results: Dict[int, Dict[str, Any]],
                            output_file: str,
                            interactive_pattern: str = INTERACTIVE_HTML_PATTERN,
                            interactive_dir: str = INTERACTIVE_HTML_DIR,
                            spike_windows_limit: int = SPIKE_WINDOWS_LIMIT,
                            dpi: int = TILE_DPI,
                            context_ms: float = INTERACTIVE_CONTEXT_MS) -> None:
    """Render the flex CSS-grid of per-event tiles, one tile per waveform.

    Each tile is a small standalone PNG (see _tile_figure) wrapped in an
    <a> that deep-links to the channel's interactive view zoomed on that
    event. The grid uses CSS auto-fill so tiles reflow with the browser
    width (responsive/flex layout); no image maps are needed.

    Every tile embeds the raw window plus one PNG per persisted smoothing
    width (all read from the npz via results[]; never recomputed). A radio
    selector at the top chooses which variant is displayed, toggled via a
    body class.

    A per-channel checkbox (checked by default) applies a CSS class that
    hides every tile beyond the first `spike_windows_limit` for THAT channel
    only, and unchecking it reveals all of them. The limit value is injected
    into the CSS from the module constant SPIKE_WINDOWS_LIMIT, so the visible
    cutoff always follows the code.
    """
    print(f"\nGenerating waveform grid HTML: {output_file}")

    # Per-event tiles are written as PNG files next to the grid (kept out of
    # the HTML so the page stays small and all smoothing variants are
    # available without inflating the file to gigabytes).
    tiles_dir = Path(output_file).parent / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    html_parts = _html_head("MEA Spike Waveform Grid")
    # Smoothing parameters come from the persisted archive metadata so they
    # always describe the data actually displayed, even on a -v re-render of
    # an older run made with different constants.
    first_channel_data = (results[sorted(results.keys())[0]]
                          if results else None)
    smooth_method = (first_channel_data["smooth_method"] if first_channel_data
                     else SMOOTH_METHOD)
    smooth_windows_ms = (list(first_channel_data["smooth_windows_ms"])
                         if first_channel_data else list(SMOOTH_WINDOWS_MS))
    smooth_polyorder = (first_channel_data["smooth_polyorder"]
                        if first_channel_data else SMOOTH_POLYORDER)
    default_variant = "raw" if not SMOOTH_SHOW_BY_DEFAULT else \
        f"{int(smooth_windows_ms[0])}ms"

    # One radio per smoothing width, plus "raw"; body class drives display.
    variant_options = [
        f'<label><input type="radio" name="smooth-variant" value="raw" '
        f'{"checked" if default_variant == "raw" else ""} '
        f'onchange="document.body.className = \'show-raw\'"> Raw</label>']
    for window_ms in smooth_windows_ms:
        tag = f"{int(window_ms)}ms"
        checked = "checked" if default_variant == tag else ""
        variant_options.append(
            f'<label><input type="radio" name="smooth-variant" value="{tag}" '
            f'{checked} '
            f'onchange="document.body.className = \'show-{tag}\'"> '
            f'{int(window_ms)} ms</label>')

    html_parts.append(f"""
    <style>
        .control-bar {{ margin: 12px 0 18px 0; font-size: 13px; color: #333;
                        background: #fff; padding: 10px 14px; border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .control-bar label {{ margin-right: 14px; }}
        .channel-section:not(.show-all) .spike-tiles .tile:nth-child(n+{spike_windows_limit + 1}) {{ display: none; }}
        .channel-section.show-all .spike-tiles .tile:nth-child(n+{spike_windows_limit + 1}) {{ display: block; }}
        .spike-tiles .tile img {{ display: none; }}
        body:not([class*="show-"]) .spike-tiles .tile img.{'raw' if default_variant == 'raw' else 's' + default_variant} {{ display: block; }}
        body.show-raw .spike-tiles .tile img.raw {{ display: block; }}
        body.show-1ms .spike-tiles .tile img.s1ms {{ display: block; }}
        body.show-2ms .spike-tiles .tile img.s2ms {{ display: block; }}
        body.show-4ms .spike-tiles .tile img.s4ms {{ display: block; }}
        body.show-8ms .spike-tiles .tile img.s8ms {{ display: block; }}
    </style>
    <div class="control-bar">
        <strong>Display:</strong>
        {chr(10) + "        ".join(variant_options)}
        <span class="meta" style="margin-left: 12px;">({smooth_method}, poly {smooth_polyorder}; raw is the truth)</span>
    </div>""")
    html_parts.append("    <p>Every extracted event window, per channel. "
                      "x-axis is absolute recording time (s); the red dotted "
                      "lines mark the window min/max voltage; the dashed line "
                      "is the dominant peak; the corner number is the computed "
                      "oscillation count. Click a tile to open the channel's "
                      "interactive view zoomed to that event. All traces are "
                      "the exact persisted waveforms from the npz (raw and "
                      "every Savitzky-Golay width are computed once at "
                      "extraction time).</p>")

    for channel in sorted(results.keys()):
        channel_data = results[channel]
        waveforms = channel_data["waveforms"]
        if len(waveforms) == 0:
            continue
        smooth_waveforms = channel_data["smooth_waveforms"]
        spike_times = channel_data["spike_times"]
        peak_indices = channel_data["peak_indices"]
        window_starts = channel_data["window_starts"]
        oscillation_counts = channel_data["n_oscillations"]
        env_gate = EVENT_GATE_SCALE * channel_data["std_dev"]
        spk_gate = SPIKE_GATE_SCALE * channel_data["std_dev"]

        tiles = []
        for index, raw_waveform in enumerate(waveforms):
            ylim = (float(raw_waveform.min()), float(raw_waveform.max()))
            tile_prefix = f"ch{channel:02d}_e{index:04d}"
            fig = _tile_figure(raw_waveform, int(window_starts[index]),
                               int(peak_indices[index]),
                               int(oscillation_counts[index]), ylim=ylim)
            _save_figure_png(fig, tiles_dir / f"{tile_prefix}_raw.png",
                             dpi=dpi, tight=False)
            # Smoothed tiles render the SAVED smoothed arrays, not fresh
            # smooth_waveform() calls: the archive is the single source of truth.
            img_entries = [f'          <img class="raw" src="tiles/{tile_prefix}_raw.png" '
                           f'alt="ch{channel} t={float(spike_times[index]):.4f}s (raw)">']
            for window_ms in smooth_windows_ms:
                tag = f"{int(window_ms)}ms"
                smooth_variant = smooth_waveforms[window_ms][index]
                fig = _tile_figure(smooth_variant, int(window_starts[index]),
                                   int(peak_indices[index]),
                                   int(oscillation_counts[index]), ylim=ylim)
                _save_figure_png(fig, tiles_dir / f"{tile_prefix}_{tag}.png",
                                 dpi=dpi, tight=False)
                img_entries.append(
                    f'          <img class="s{tag}" src="tiles/{tile_prefix}_{tag}.png" '
                    f'alt="ch{channel} t={float(spike_times[index]):.4f}s ({tag})">')
            event_time = float(spike_times[index])
            margin = (len(raw_waveform) / (2.0 * SAMPLE_RATE_HZ)
                      + context_ms / 1000.0)
            t0 = max(0.0, event_time - margin)
            t1 = event_time + margin
            href = (f"{interactive_dir}/{interactive_pattern.format(ch=channel)}"
                    f"?t0={t0:.4f}&t1={t1:.4f}")
            tiles.append(
                f'        <a class="tile" href="{href}" target="_blank" '
                f'title="ch{channel} event t={event_time:.4f}s, '
                f'{int(oscillation_counts[index])} osc">\n'
                + "\n".join(img_entries) + "\n"
                f'        </a>')

        # Per-channel limit checkbox (only shown when it does anything).
        limit_toggle = ""
        if len(tiles) > spike_windows_limit:
            limit_toggle = f"""
                <div class="stats">
                    <label><input type="checkbox" class="ch-limit" checked
                        onchange="this.closest('.channel-section').classList.toggle('show-all', !this.checked)">
                        Limit to first <strong>{spike_windows_limit}</strong> windows
                        (uncheck to show all {len(tiles)})</label>
                </div>"""

        html_parts.append(f"""
        <div class="channel-section">
            <div class="channel-header">
                <h2>Channel {channel}</h2>
                <div class="stats">
                    <strong>Events:</strong> {channel_data['n_events']} |
                    <strong>Extracted:</strong> {channel_data['n_extracted']} |
                    <strong>Envelope gate ({EVENT_GATE_SCALE:.0f}x noise):</strong> {env_gate:.2f} uV |
                    <strong>Spike gate ({SPIKE_GATE_SCALE:.0f}x noise):</strong> {spk_gate:.2f} uV |
                    <strong>Noise (MAD):</strong> {channel_data['std_dev']:.2f} uV
                </div>
                {limit_toggle}
            </div>
            <div class="spike-tiles tile-grid">
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
    PNG; detected dominant peaks are overlaid as red dots and the two gates
    (envelope gate = EVENT_GATE_SCALE x noise, spike gate = SPIKE_GATE_SCALE
    x noise, both derived from the module constants) are drawn as dashed
    lines. Purely visual, no analysis.
    """
    print(f"\nGenerating full-trace HTML: {output_file}")
    raw_data = load_raw_data(raw_file)
    html_parts = _html_head("MEA Channel Traces")
    html_parts.append("    <p>Full recording per channel with detected "
                      "events marked (envelope and spike gates shown dashed).</p>")

    for channel in sorted(results.keys()):
        channel_data = results[channel]
        voltage = raw_data[:, channel] * VOLTAGE_SCALE
        downsampled_voltage = voltage[::ds_factor] if ds_factor > 0 else voltage
        downsampled_time = (np.arange(len(downsampled_voltage)) /
                            (SAMPLE_RATE_HZ / ds_factor) if ds_factor > 0
                            else np.arange(len(downsampled_voltage)) / SAMPLE_RATE_HZ)

        # Dominant-peak positions: start of the extracted window + offset of
        # the peak inside it (both persisted in the npz).
        absolute_peaks = (np.asarray(channel_data["window_starts"])
                          + np.asarray(channel_data["peak_indices"]))
        peak_times = absolute_peaks / SAMPLE_RATE_HZ
        peak_voltages = voltage[absolute_peaks]

        env_gate = EVENT_GATE_SCALE * channel_data["std_dev"]
        spk_gate = SPIKE_GATE_SCALE * channel_data["std_dev"]

        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(downsampled_time, downsampled_voltage, color="blue", linewidth=0.4)
        ax.scatter(peak_times, peak_voltages, s=6, c="red", zorder=3)
        ax.axhline(env_gate, color="orange", linewidth=0.9, linestyle="--",
                   alpha=0.8, label=f"{EVENT_GATE_SCALE:.0f}x-noise envelope gate "
                                    f"({env_gate:.2f} uV)")
        ax.axhline(-env_gate, color="orange", linewidth=0.9, linestyle="--",
                   alpha=0.8)
        ax.axhline(spk_gate, color="purple", linewidth=0.7, linestyle=":",
                   alpha=0.8, label=f"{SPIKE_GATE_SCALE:.0f}x-noise spike gate "
                                    f"({spk_gate:.2f} uV)")
        ax.axhline(-spk_gate, color="purple", linewidth=0.7, linestyle=":",
                   alpha=0.8)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"Channel {channel} - {channel_data['n_extracted']} events "
                     f"(envelope gate: {env_gate:.2f} uV, "
                     f"spike gate: {spk_gate:.2f} uV)")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Voltage (uV)")
        fig.tight_layout()
        img = _figure_to_base64(fig, dpi=TRACE_DPI)
        html_parts.append(f"""
        <div class="channel-section">
            <div class="channel-header">
                <h2>Channel {channel}</h2>
                <div class="stats">
                    <strong>Events:</strong> {channel_data['n_events']} |
                    <strong>Extracted:</strong> {channel_data['n_extracted']} |
                    <strong>Envelope gate ({EVENT_GATE_SCALE:.0f}x noise):</strong> {env_gate:.2f} uV |
                    <strong>Spike gate ({SPIKE_GATE_SCALE:.0f}x noise):</strong> {spk_gate:.2f} uV |
                    <strong>Noise (MAD):</strong> {channel_data['std_dev']:.2f} uV
                </div>
            </div>
            <img src="data:image/png;base64,{img}" alt="Channel {channel} trace">
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
    decimation INTERACTIVE_SPIKE_DS, joined with NaN gaps so plotly draws
    no connecting line across the spaces between segments), and (3) red
    markers at the dominant peaks. The URL query ?t0=..&t1=.. selects the
    initial x-axis range (used by the grid tiles) via a JS snippet injected
    before </body>:  Plotly.relayout('interactive', {'xaxis.range': [t0, t1]}).
    """
    import plotly.graph_objects as go

    channel_data = results[channel]
    print(f"\nGenerating interactive HTML for channel {channel}: {output_file}")
    raw_data = load_raw_data(raw_file)
    voltage = raw_data[:, channel] * VOLTAGE_SCALE
    n_samples = len(voltage)

    absolute_peaks = (np.asarray(channel_data["window_starts"])
                      + np.asarray(channel_data["peak_indices"]))

    fig = go.Figure()
    if overview_ds > 1:
        overview_time = np.arange(0, n_samples, overview_ds) / SAMPLE_RATE_HZ
        overview_voltage = voltage[::overview_ds]
    else:
        overview_time = np.arange(n_samples) / SAMPLE_RATE_HZ
        overview_voltage = voltage
    fig.add_trace(go.Scatter(x=overview_time, y=overview_voltage, mode="lines",
                             name="overview", line=dict(color="#9ecae1", width=1),
                             hovertemplate="t=%{x:.3f}s<br>%{y:.1f}uV",
                             hoverlabel=dict(bgcolor="#9ecae1")))

    # High-resolution context segments around each dominant peak, separated
    # by NaN so consecutive segments are not bridged by a connecting line.
    half_context_samples = int(context_ms / 2000.0 * SAMPLE_RATE_HZ)
    context_times: List[float] = []
    context_voltages: List[float] = []
    for peak_sample in absolute_peaks:
        segment_start = max(0, peak_sample - half_context_samples)
        segment_end = min(n_samples, peak_sample + half_context_samples)
        segment_time = (np.arange(segment_start, segment_end, spike_ds)
                        / SAMPLE_RATE_HZ)
        context_times.extend(np.round(segment_time, 4).tolist())
        context_voltages.extend(
            np.round(voltage[segment_start:segment_end:spike_ds], 1).tolist())
        context_times.append(np.nan)
        context_voltages.append(np.nan)
    if context_times:
        fig.add_trace(go.Scatter(x=context_times, y=context_voltages, mode="lines",
                                 name="spike context", showlegend=False,
                                 hoverinfo="skip",
                                 line=dict(color="#1f77b4", width=1)))

    fig.add_trace(go.Scatter(x=absolute_peaks / SAMPLE_RATE_HZ,
                             y=voltage[absolute_peaks], mode="markers",
                             name="dominant peak",
                             marker=dict(color="red", size=7, symbol="x"),
                             hovertemplate="t=%{x:.3f}s<br>%{y:.1f}uV"))

    env_gate = EVENT_GATE_SCALE * channel_data["std_dev"]
    spk_gate = SPIKE_GATE_SCALE * channel_data["std_dev"]
    fig.add_hline(y=env_gate, line_color="orange", line_width=1, line_dash="dash",
                  name=f"{EVENT_GATE_SCALE:.0f}x-noise envelope gate", showlegend=True)
    fig.add_hline(y=-env_gate, line_color="orange", line_width=1, line_dash="dash")
    fig.add_hline(y=spk_gate, line_color="purple", line_width=1, line_dash="dot",
                  name=f"{SPIKE_GATE_SCALE:.0f}x-noise spike gate", showlegend=True)
    fig.add_hline(y=-spk_gate, line_color="purple", line_width=1, line_dash="dot")

    fig.update_layout(
        title=f"Channel {channel} - {channel_data['n_extracted']} events "
              f"(envelope gate {env_gate:.2f} uV, spike gate {spk_gate:.2f} uV)",
        xaxis_title="Time (seconds)", yaxis_title="Voltage (uV)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, div_id="interactive",
                   include_plotlyjs=plotly_js)

    # Inject the URL-driven zoom: the grid deep-links here with ?t0&t1.
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


def _resolve_run_paths(args: argparse.Namespace) -> Tuple[Path, Path, str, str, Path]:
    """Determine the run directory and all output paths.

    Normal run: creates a fresh timestamped run dir
        <OUTPUT_ROOT>/<YYYY-MM-DD_HH-MM-SS>/
    so every invocation is archived for iteration tracking (old runs are
    never overwritten).

    Visualize-only (-v): re-derives the run dir from the supplied .npz path.
    If the npz lives at <run_dir>/waveforms/waveforms.npz the run dir is
    npz.parent.parent; otherwise npz.parent is used. HTML is (re)written
    into <run_dir>/html/, so a previous run can be re-rendered in place.

    Returns (run_dir, npz_path, grid_path, channel_path, html_dir).
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
    grid_path: str = args.spike_html or str(html_dir / "waveforms_grid.html")
    channel_path: str = args.channel_html or str(html_dir / "all_ch_spikes.html")
    return run_dir, npz_path, grid_path, channel_path, html_dir


def _write_run_meta(run_dir: Path, args: argparse.Namespace,
                    results: Dict[int, Dict[str, Any]], npz_path: Path) -> None:
    """Write run_meta.json: parameters, timestamps and per-channel summary.

    Mirrors the constants in this module so any archived run can be fully
    reconstructed / cross-referenced during writing of the methods section.
    """
    per_channel = {}
    for channel in sorted(results.keys()):
        channel_data = results[channel]
        per_channel[str(channel)] = {
            "n_events": int(channel_data["n_events"]),
            "n_extracted": int(channel_data["n_extracted"]),
            "spike_gate_uV": round(float(channel_data["threshold"]), 3),
            "envelope_gate_uV": round(float(channel_data["gate"]), 3),
            "noise_mad_uV": round(float(channel_data["std_dev"]), 3),
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
            "pad_fraction": PAD_FRACTION,
            "min_pad_s": MIN_PAD_S,
            "min_window_ms": MIN_WINDOW_MS,
            "extent_sigmas": EXTENT_SIGMAS,
            "smooth_method": SMOOTH_METHOD,
            "smooth_windows_ms": list(SMOOTH_WINDOWS_MS),
            "smooth_polyorder": SMOOTH_POLYORDER,
            "smooth_show_by_default": SMOOTH_SHOW_BY_DEFAULT,
        },
        "channels": per_channel,
        "totals": {
            "events": int(sum(channel_data["n_events"]
                              for channel_data in results.values())),
            "extracted": int(sum(channel_data["n_extracted"]
                                 for channel_data in results.values())),
        },
    }
    (run_dir / RUN_META_FILENAME).write_text(
        json.dumps(meta, indent=2))
    print(f"Saved run metadata to: {run_dir / RUN_META_FILENAME}")


def _find_run_dirs(output_root: Path) -> List[Path]:
    """All timestamped run directories under output_root, newest first.

    A run dir is a direct child whose name parses as TIMESTAMP_FORMAT
    (e.g. 2026-08-15_12-00-00). The fixed-width timestamp compares
    lexicographically in chronological order, so sorting on the name alone
    puts the newest run first.
    """
    runs = []
    for candidate in output_root.iterdir():
        if not candidate.is_dir():
            continue
        try:
            datetime.datetime.strptime(candidate.name, TIMESTAMP_FORMAT)
        except ValueError:
            continue
        runs.append(candidate)
    runs.sort(key=lambda run: run.name, reverse=True)
    return runs


def write_output_index(output_path: Path, output_root: Path) -> None:
    """Regenerate the static entry-point index.html for all runs.

    The index is written to output_path/index.html (the script directory),
    while the runs it links to live under output_root. Called at the end of
    every run (fresh extraction and -v re-render), so the newest run's pages
    are always one click away when the index is opened from disk -- plain
    relative links, no server or JavaScript required. The newest run is
    listed first with links to its waveform grid, all-channels view and run
    metadata; every older run follows with the same links.
    """
    runs = _find_run_dirs(output_root)
    latest = runs[0] if runs else None

    def run_link(run_dir: Path, rel_path: str) -> str:
        # Relative to the index file's location (output_path/index.html), so
        # it stays correct regardless of where output_root sits relative to it.
        target = run_dir / rel_path
        return os.path.relpath(target, output_path).replace(os.sep, "/")

    def format_timestamp(run_name: str) -> str:
        return (datetime.datetime.strptime(run_name, TIMESTAMP_FORMAT)
                .strftime("%Y-%m-%d %H:%M:%S"))

    latest_section = ""
    if latest is not None:
        latest_grid = run_link(latest, "html/waveforms_grid.html")
        latest_channels = run_link(latest, "html/all_ch_spikes.html")
        latest_meta = run_link(latest, RUN_META_FILENAME)
        latest_section = f"""
    <div class="panel">
        <h2>Latest run</h2>
        <p class="meta">Newest run: <strong>{format_timestamp(latest.name)}</strong></p>
        <p class="latest-links">
            <a href="{latest_grid}">Waveform grid</a>
            <a href="{latest_channels}">All channels</a>
            <a href="{latest_meta}">run_meta.json</a>
        </p>
    </div>"""

    run_items = ""
    for run in runs:
        fmt = format_timestamp(run.name)
        grid = run_link(run, "html/waveforms_grid.html")
        channels = run_link(run, "html/all_ch_spikes.html")
        meta = run_link(run, RUN_META_FILENAME)
        run_items += (
            f'        <li>{fmt} &mdash; '
            f'<a href="{grid}">grid</a>, '
            f'<a href="{channels}">all channels</a>, '
            f'<a href="{meta}">run_meta.json</a></li>\n')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MEA Spike Waveform Outputs</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; color: #333; }}
    h1 {{ color: #333; }}
    .panel {{ background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
              padding: 20px; margin-bottom: 24px; }}
    .panel h2 {{ margin: 0 0 12px 0; }}
    a {{ color: #1f77b4; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    ul {{ padding-left: 20px; }}
    .meta {{ font-size: 13px; color: #666; }}
    .latest-links {{ font-size: 15px; }}
    .latest-links a {{ margin-right: 18px; }}
</style>
</head>
<body>
    <h1>MEA Spike Waveform Outputs</h1>
    <p class="meta">Regenerated by raw_analysis.py on every run. Old runs are
       kept and listed below; the newest run is always linked first.</p>
{latest_section}
    <div class="panel">
        <h2>All runs</h2>
        <ul>
{run_items}        </ul>
    </div>
</body>
</html>
"""
    (output_path / "index.html").write_text(html)
    print(f"Updated output index: {output_path / 'index.html'}")


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
        results = load_waveforms(str(npz_path))
        if not results:
            print("Error: No data loaded. Run without -v first.")
            return
    else:
        data = load_raw_data(args.data_file)
        results: Dict[int, Dict[str, Any]] = {}
        for channel in range(data.shape[1]):
            results[channel] = process_channel(data, channel)
        save_waveforms(results, str(npz_path), source_file=args.data_file)
        _write_run_meta(run_dir, args, results, npz_path)

    for channel in sorted(results.keys()):
        if results[channel]["n_extracted"] > 0:
            gen_channel_interactive_html(
                results, channel, args.data_file,
                str(interactive_dir / INTERACTIVE_HTML_PATTERN.format(ch=channel)))

    gen_spike_waveform_html(results, str(grid_path),
                            interactive_pattern=INTERACTIVE_HTML_PATTERN,
                            interactive_dir=INTERACTIVE_HTML_DIR)
    gen_channel_html(results, str(channel_path), raw_file=args.data_file)

    # Regenerate the static index.html next to this script (relative links
    # into outputs/, so it works when opened straight from disk).
    write_output_index(Path(os.path.dirname(os.path.abspath(__file__))), Path(args.out_root))

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
