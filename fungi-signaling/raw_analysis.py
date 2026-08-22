"""
raw_analysis.py  (rev 5 - persisted raw + smoothed waveforms)

Run from the project root:
    python raw_analysis.py
    python raw_analysis.py --data-file <path>
    python raw_analysis.py -o outputs/<ts>/waveforms/waveforms.npz -v   (re-render without re-running all analyses)

Pipeline for MEA compound-event waveform extraction (fungal spike recordings).

Each channel k of the recording v_k (in microvolts) is segmented into
events (spikes are polyphasic) and one waveform window is extracted per event:

 - N    = total number of samples per channel
 - v_k  = voltage trace for channel k, indexed by sample n = 0, ..., N-1
 - fs   = sample rate in Hz (SAMPLE_RATE_HZ)
 - k    = channel index,  k = 0, ..., 63

  1. Robust noise floor:    To detect spiking events, we first calculate the noise floor using MAD (median absolute deviation):
                                - sigma_k = 1.4826 * median_n |v_k[n] - median_m v_k[m]|
                                  where n, m range over all N samples of channel k,
                                  and 1.4826 = 1 / Phi^{-1}(3/4) so sigma_k is an
                                  unbiased estimate of sigma for Gaussian noise.
                                - The low envelope gate must not be inflated by the very events being
                                  segmented, hence MAD (robust) rather than std (inflated by events).

  2. Envelope gate:         Events are voltage fluctuations rising above 5 times the noise floor:
                                - Binary mask b[n] = 1{ |v_k[n]| > G * sigma_k },  n = 0, ..., N-1
                                  G = EVENT_GATE_SCALE = 5.0

  3. Excursions:            Excursions are contiguous runs of samples above the envelope gate:
                                - Excursion set E = { n : b[n] = 1 } is decomposed
                                  into J contiguous runs [s_j, e_j], j = 0, ..., J-1 where:
                                    - b[n] = 1 for all n in [s_j, e_j],
                                     - the sample before the run is below the gate:
                                       b[s_j - 1] = 0,  or  s_j = 0  (no sample at index -1;
                                       this case only arises when b[0] = 1, i.e. the trace
                                       itself starts above the gate),
                                    - the sample after the run is below the gate:
                                      b[e_j + 1] = 0,  or  e_j = N - 1  (run ends at trace end),
                                    - runs are disjoint and ordered: e_j < s_{j+1}.

  4. Gap merge:             Since spikes are polyphasic, a single spike may consist of multiple disjoint excursions - we must merge them:   
                                - for consecutive runs j and j+1, define the inter-run gap
                                  gap_j = s_{j+1} - e_j - 1   (count # of samples between consecutive excursions)
                                  Merge j+1 into j iff  gap_j <= tau_gap,  where
                                    - tau_gap = int(EVENT_GAP_MS * fs / 1000) samples,
                                      EVENT_GAP_MS = 5.0 ms,  fs = SAMPLE_RATE_HZ.
                                  Merging is greedy left-to-right; each merged run absorbs
                                  subsequent runs while the gap condition holds, producing
                                  M merged events [S_i, E_i],  i = 0,...,M-1.

  5. Event gate:            Prune events that are too short or too weak:
                                keep event i iff
                                    D_i >= D_min   AND   |v_k[m_i]| >= S * sigma_k,
                                where
                                    D_i     = E_i - S_i + 1                        (duration in samples)
                                    D_min   = int(MIN_EVENT_MS * fs / 1000)        (minimum duration in samples)
                                    S       = SPIKE_GATE_SCALE = 5.0               (S and ^ determined through visual inspection/experimentation)
                                    m_i     = argmax_{n in [S_i, E_i]} |v_k[n]|    (dominant deflection, sample index)

  6. Window:                Expand excursion to capture rising/falling oscillations and add padding - extract a centered window showing all dynamics:
                                Grow the excursion [S_i, E_i] into the extent [L_i, R_i] by expanding outward while
                                    |v_k[n]| > e * sigma_k,  e = EXTENT_SIGMAS = 2.0:       (lower sigma for extent - weaker oscillations)
                                    L_i = min{ n <= S_i : |v_k[n]| <= e*sigma_k } + 1,      (or 0 if none)
                                    R_i = max{ n >= E_i : |v_k[n]| <= e*sigma_k } - 1.      (or N-1 if none)
                                Then add padding scaled to window length, bounded by MIN_PAD_S:
                                    L_ext = R_i - L_i + 1,
                                    pad   = max( round(PAD_FRACTION * L_ext), round(MIN_PAD_S * fs) ),
                                    W_i   = max( L_ext + 2*pad, w_min ),
                                    where w_min = int(MIN_WINDOW_MS * fs / 1000).
                                The window is centered on the dominant deflection m_i:
                                    start_i = clamp( m_i - floor(W_i/2),  0,  N - W_i ),    (ensure windows cant extend past the trace boundaries)
                                    r_i     = m_i - start_i                                 (peak offset, ~ W_i/2).
                                The extracted waveform is x[n] = v_k[start_i + n],  n = 0, ..., W_i - 1.
                                        - In summary, place a W_i-sample window symmetrically around
                                          m_i so the peak is in the middle. If the window would extend
                                          before the trace start (start_i < 0), pin start_i to 0; if it would
                                          extend past the trace end (start_i > N - W_i), pin start_i to N - W_i.
                                          The peak offset r_i = m_i - start_i tells you where the dominant
                                          deflection lands inside the extracted window (~W_i/2 when centered,
                                          shifts toward the edge when clamped at a trace boundary).
                                            - Symmetric padding proportional to the extent length but never less than MIN_PAD_S on each side
                                            - Clamped to MIN_WINDOW_MS but not to an upper bound

   7. Smoothing:            Apply Savitzky-Golay smoothing to all windows - persisted alongsied the raw windows
                                for each width w in SMOOTH_WINDOWS_MS = (1.0, 2.0, 4.0, 8.0) ms derive kernel length
                                    L = round( w * fs / 1000 ),  forced odd  (L |= 1),
                                    L = max( L, p + 1 ),  (L must be > p: a degree-p polynomial needs at least p + 1 data points to fit)
                                where p = SMOOTH_POLYORDER = 4.
                                The Savitzky-Golay smoothed signal is
                                    y[n] = sum_{j = -L//2}^{L//2}  c_j * x[n + j],
                                where c_j are the least-squares polynomial coefficients,
                                depending on L and p only (precomputed by scipy.signal.savgol_filter)
                                    - In summary: for each output sample y[n], take a symmetric window of 
                                      L input samples centered at n, fit a degree-p polynomial to those
                                      L points in a least-squares sense, and evaluate that polynomial at the center. 
                                        - The c_j are the closed-form weights that implement this fit as a single convolution.
                                      Because the window is symmetric, the fit does not shift peaks in time (zero-phase). 
                                    - Applied once at extraction time; all variants are persisted alongside the raw window.

Output layout: each run writes into its own timestamped directory
    index.html                                                 (links to newest run, links all outputs from all runs)
    outputs/<YYYY-MM-DD_HH-MM-SS>/
        waveforms/waveforms.npz                                (persisted events: raw + smoothed windows)
        html/waveforms_grid.html                               (flex CSS grid of per-event tiles)
        html/all_ch_spikes.html                                (full per-channel traces with events)
        html/interactive_ch_views/channel_N_interactive.html   (click-to-zoom)
        run_meta.json                                          (parameters + per-channel summary)

waveforms.npz schema (np.savez_compressed):
    Scalar metadata (saved as 0-d arrays or Python scalars):
      sample_rate               int                   fs (Hz)
      unit                      str                   "uV"
      source_file               str                   path to the raw binary
      min_window_ms             float                 MIN_WINDOW_MS
      min_pad_s                 float                 MIN_PAD_S
      pad_fraction              float                 PAD_FRACTION
      extent_sigmas             float                 EXTENT_SIGMAS
      event_gate_scale          float                 EVENT_GATE_SCALE
      spike_gate_scale          float                 SPIKE_GATE_SCALE
      smooth_method             str                   "savgol" or "none"
      smooth_windows_ms         float64 (W,)          persisted smoothing widths in ms
      smooth_polyorder          int                   Savitzky-Golay polynomial order
      smooth_show_by_default    bool                  grid default variant

    Per-channel scalar arrays (dtype=float64 or int64, shape=(N_ch,)):
      thresholds                float64 (N_ch,)        spike gate per channel (S * sigma_k)
      gates                     float64 (N_ch,)        envelope gate per channel (G * sigma_k)
      stds                      float64 (N_ch,)        noise MAD per channel (sigma_k)
      n_events                  int64   (N_ch,)        detected events before window extraction
      n_extracted               int64   (N_ch,)        windows successfully extracted

    Per-channel object arrays (dtype=object, shape=(N_ch,)).
    Each element is a sub-array for one channel; event counts and window lengths vary per channel, so these are stored as object arrays:
      channels                  int64   (N_ch,)         channel IDs, e.g. [0, 1, ..., 63]

      waveforms                 object  (N_ch,)         waveforms[i] = object array of M_i raw windows, each float64 (W_i,)

      smooth_waveforms_<w>ms    object  (N_ch,)         same but Savitzky-Golay smoothed at
                                                        width w ms; one key per width in
                                                        SMOOTH_WINDOWS_MS, e.g.
                                                        smooth_waveforms_4ms

      spike_times               object  (N_ch,)         spike_times[i] = float64 (M_i,)
                                                        event times in seconds

      window_sizes              object  (N_ch,)         window_sizes[i] = int (M_i,)
                                                        window length in samples

      peak_positions            object  (N_ch,)         peak_positions[i] = int (M_i,)
                                                        offset of dominant peak within window
                                                        (r_i = m_i - start_i)

      window_starts             object  (N_ch,)         window_starts[i] = int (M_i,)
                                                        absolute sample index of window start

      amplitudes                object  (N_ch,)         amplitudes[i] = float64 (M_i,)
                                                        signed dominant deflection v_k[m_i] (uV)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
import numpy as np
import scipy.signal

from visualization_tools import (
    INTERACTIVE_HTML_DIR,
    INTERACTIVE_HTML_PATTERN,
    gen_channel_html,
    gen_channel_interactive_html,
    gen_spike_waveform_html,
    write_output_index,
)

# ---------------- dataset constants ----------------
SAMPLE_RATE_HZ: int = 30000
NUM_CHANNELS: int = 64
VOLTAGE_SCALE: float = 0.195
BINARY_DTYPE: str = "int16"
RAW_DATA_FILE: str = "../data/raw_mea_bins/recording_control_0_cut800s.bin"

# ---------------- event segmentation ----------------
EVENT_GATE_SCALE: float = 5.0    # low envelope gate  (x noise)
SPIKE_GATE_SCALE: float = 5.0    # high gate: an event's dominant peak must clear this
EVENT_GAP_MS: float = 5.0        # merge excursions <= this apart into one event
MIN_EVENT_MS: float = 0.5        # discard envelope blips shorter than this

# ---------------- window extraction ----------------
EXTENT_SIGMAS: float = 2.0   # window base = contiguous span where |v| > this x noise
PAD_FRACTION: float = 0.25      # pre/post pad = this fraction of the extent length
MIN_PAD_S: float = 0.02        # floor on the pre/post pad (seconds): the window always extends at least this far each side
MIN_WINDOW_MS: float = 3.0

# --------- persisted waveform smoothing ------------
SMOOTH_METHOD: str = "savgol"                      # "savgol" | "none"
SMOOTH_WINDOWS_MS: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)  # savgol widths (ms)
SMOOTH_POLYORDER: int = 4                          # savgol polynomial order
SMOOTH_SHOW_BY_DEFAULT: bool = False               # grid default: raw shown

# ------------------ output paths -------------------
OUTPUT_ROOT: str = "outputs"
TIMESTAMP_FORMAT: str = "%Y-%m-%d_%H-%M-%S"
WAVEFORM_REL_PATH: str = "waveforms/waveforms.npz"
SPIKE_HTML_REL_PATH: str = "html/waveforms_grid.html"
CHANNEL_HTML_REL_PATH: str = "html/all_ch_spikes.html"
RUN_META_FILENAME: str = "run_meta.json"


class Event(NamedTuple):
    onset: int
    offset: int
    center: int


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

    raw_data = load_raw_data(args.data_file)
    if args.visualize_only:
        results = load_waveforms(str(npz_path))
        if not results:
            print("Error: No data loaded. Run without -v first.")
            return
    else:
        results: Dict[int, Dict[str, Any]] = {}
        for channel in range(raw_data.shape[1]):
            results[channel] = process_channel(raw_data, channel)

        save_waveforms(results, str(npz_path), source_file=args.data_file)
        _write_run_meta(run_dir, args, results, npz_path)

    for channel in sorted(results.keys()):
        if results[channel]["n_extracted"] > 0:
            voltage = raw_data[:, channel] * VOLTAGE_SCALE
            gen_channel_interactive_html(
                results, channel, voltage,
                str(interactive_dir / INTERACTIVE_HTML_PATTERN.format(ch=channel)),
                sample_rate=SAMPLE_RATE_HZ,
                event_gate_scale=EVENT_GATE_SCALE,
                spike_gate_scale=SPIKE_GATE_SCALE
            )

    gen_spike_waveform_html(
        results, str(grid_path),
        interactive_pattern=INTERACTIVE_HTML_PATTERN,
        interactive_dir=INTERACTIVE_HTML_DIR,
        sample_rate=SAMPLE_RATE_HZ,
        event_gate_scale=EVENT_GATE_SCALE,
        spike_gate_scale=SPIKE_GATE_SCALE,
        smooth_method_default=SMOOTH_METHOD,
        smooth_windows_ms_default=SMOOTH_WINDOWS_MS,
        smooth_polyorder_default=SMOOTH_POLYORDER,
        smooth_show_by_default=SMOOTH_SHOW_BY_DEFAULT
    )
    
    gen_channel_html(results, str(channel_path), raw_data,
                     sample_rate=SAMPLE_RATE_HZ,
                     voltage_scale=VOLTAGE_SCALE,
                     event_gate_scale=EVENT_GATE_SCALE,
                     spike_gate_scale=SPIKE_GATE_SCALE
    )
    
    write_output_index(Path(os.path.dirname(os.path.abspath(__file__))), Path(args.out_root), TIMESTAMP_FORMAT, RUN_META_FILENAME)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Open {grid_path} in your browser")
    print("=" * 60)


def load_raw_data(filepath: str) -> np.ndarray:
    """Load interleaved int16 MEA binary as a memmap of shape (T, 64).

    The raw file stores sample y[n] for the interleaved stream; the channel
    index is k = n mod 64, the time index is t = n // 64. Reshaping the flat
    array into (-1, NUM_CHANNELS) de-interleaves it in one step:
        Y[t, k] = y[64*t + k],   t = 0..T-1,  k = 0..63.
    Memory-mapped (no full load) so that 3.4 GB of data 
    (900 s * 30 kHz * 64 * 2 bytes) streams through without exhausting RAM.
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


def detect_events(
    voltage: np.ndarray,
    noise_estimator: Callable[[np.ndarray], float] = estimate_noise_mad,
    gate_scale: float = EVENT_GATE_SCALE,
    spike_gate_scale: float = SPIKE_GATE_SCALE,
    gap_ms: float = EVENT_GAP_MS,
    min_event_ms: float = MIN_EVENT_MS,
    sample_rate: int = SAMPLE_RATE_HZ,
) -> Tuple[List[Event], float, float, float]:
    """Segment the trace v into events.

    Formal description
    ------------------
    Let sigma = noise_estimator(v), gate = G*sigma, spike_gate = S*sigma.

    1. Excursion set:  E = { n : |v[n]| > gate }.
    2. Runs: maximal contiguous intervals [s_j, e_j] within E.
    3. Merge: run j+1 is joined to run j iff
           s_{j+1} - e_j - 1 <= gap_samples,   gap_samples = tau_gap * fs/1000
       producing events [S_i, E_i].
    4. Event gate: keep i iff
           (E_i - S_i + 1) >= min_event_samples  AND
           |v[m_i]| >= spike_gate,   m_i = argmax_{n in [S_i, E_i]} |v[n]|.
       m_i is the "dominant deflection" (largest |voltage| in the event).

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
        events.append(Event(onset=run_start, offset=run_end, center=dominant_sample))

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
    pad = max(int(round(pad_fraction * extent_len)), int(round(min_pad_s * sample_rate)))
    min_window = int(min_window_ms * sample_rate / 1000)

    natural_window = extent_len + 2 * pad
    window_len = max(natural_window, min_window)
    if window_len <= 0:
        return None
    
    # Center the window on the dominant deflection so every event's peak lands
    # at the same relative position (required for family sorting).
    start = event.center - window_len // 2
    start = int(max(0, min(start, len(voltage) - window_len)))
    end = start + window_len

    waveform = voltage[start:end].copy()
    return (waveform, event.center / sample_rate, window_len, start, event.center - start)


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
      * amplitudes        - signed dominant deflection v_k[m_i]  (uV)
    """
    print(f"\nProcessing channel {channel}...")
    voltage = data[:, channel] * VOLTAGE_SCALE
    events, noise, gate, spike_gate = detect_events(voltage)
    print(f"  Found {len(events)} events (gate {gate:.2f} uV, spike gate {spike_gate:.2f} uV, noise {noise:.2f} uV)")

    waveforms: List[np.ndarray] = []
    smooth_waveforms: Dict[float, List[np.ndarray]] = { window_ms: [] for window_ms in SMOOTH_WINDOWS_MS }
    event_times: List[float] = []
    window_sizes: List[int] = []
    window_starts: List[int] = []
    peak_positions: List[int] = []
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
            smooth_waveforms[window_ms].append(smooth_waveform(waveform, window_ms=window_ms))
        event_times.append(event_time)
        window_sizes.append(window_size)
        window_starts.append(start_idx)
        peak_positions.append(peak_idx)
        amplitudes.append(float(voltage[event.center]))

    n_extracted = len(waveforms)
    print(f"  Extracted {n_extracted} waveforms")

    return {
        "waveforms": waveforms,
        "smooth_waveforms": smooth_waveforms,
        "spike_times": np.array(event_times),
        "window_sizes": np.array(window_sizes, dtype=int),
        "peak_indices": np.array(peak_positions, dtype=int),
        "window_starts": np.array(window_starts, dtype=int),
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


def save_waveforms(results: Dict[int, Dict[str, Any]], output_file: str, source_file: str) -> None:
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
    smooth_waveforms_by_width = { window_ms: np.empty(n_ch, dtype=object) for window_ms in SMOOTH_WINDOWS_MS }
    spike_times = np.empty(n_ch, dtype=object)
    window_sizes = np.empty(n_ch, dtype=object)
    peak_positions = np.empty(n_ch, dtype=object)
    window_starts = np.empty(n_ch, dtype=object)
    amplitudes = np.empty(n_ch, dtype=object)
    thresholds = np.zeros(n_ch)
    gates = np.zeros(n_ch)
    noise_stds = np.zeros(n_ch)
    n_events = np.zeros(n_ch, dtype=int)
    n_extracted = np.zeros(n_ch, dtype=int)

    for channel_idx, channel in enumerate(channels):
        channel_data = results[channel]
        waveforms[channel_idx] = np.array(channel_data["waveforms"], dtype=object)
        spike_times[channel_idx] = np.asarray(channel_data["spike_times"], dtype=float)
        window_sizes[channel_idx] = np.asarray(channel_data["window_sizes"], dtype=int)
        peak_positions[channel_idx] = np.asarray(channel_data["peak_indices"], dtype=int)
        window_starts[channel_idx] = np.asarray(channel_data["window_starts"], dtype=int)
        amplitudes[channel_idx] = np.asarray(channel_data["amplitudes"], dtype=float)
        thresholds[channel_idx] = channel_data["threshold"]
        gates[channel_idx] = channel_data["gate"]
        noise_stds[channel_idx] = channel_data["std_dev"]
        n_events[channel_idx] = channel_data["n_events"]
        n_extracted[channel_idx] = channel_data["n_extracted"]
        for window_ms in SMOOTH_WINDOWS_MS:
            smooth_waveforms_by_width[window_ms][channel_idx] = np.array(channel_data["smooth_waveforms"][window_ms], dtype=object)

    payload = {
        "channels": np.asarray(channels, dtype=int),
        "waveforms": waveforms,
        "spike_times": spike_times,
        "window_sizes": window_sizes,
        "peak_positions": peak_positions,
        "window_starts": window_starts,
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
        payload[f"smooth_waveforms_{int(window_ms)}ms"] = smooth_waveforms_by_width[window_ms]

    np.savez_compressed(output_path, **payload)
    print(f"\nSaved waveforms to: {output_file}")


def load_waveforms(output_file: str) -> Dict[int, Dict[str, Any]]:
    """Inverse of save_waveforms(): reconstruct the per-channel dict from .npz.

    Repopulates each channels data from a previous run so it 
    can be re-rendered (-v) without re-reading the
    raw binary and without re-deriving any waveform.
    """
    output_path = Path(output_file)
    if not output_path.exists():
        print(f"Error: Output file does not exist: {output_file}")
        return {}
    archive = np.load(output_path, allow_pickle=True)
    channels = archive["channels"].tolist()

    # Smoothing widths recorded in the archive; fall back to the current
    # module constants only when the archive does not carry them (legacy).
    smooth_windows_ms = list(archive["smooth_windows_ms"]) if "smooth_windows_ms" in archive.files else list(SMOOTH_WINDOWS_MS)
    smooth_method = str(archive["smooth_method"]) if "smooth_method" in archive.files else SMOOTH_METHOD
    smooth_polyorder = int(archive["smooth_polyorder"]) if "smooth_polyorder" in archive.files else SMOOTH_POLYORDER

    # Multi-width archives (rev 6+) store one key per width. Legacy archives
    # store a single "smooth_waveforms" array under the width in
    # "smooth_window_ms"; archives predating smoothing have neither.
    has_multi_width = any(f"smooth_waveforms_{int(w)}ms" in archive.files for w in smooth_windows_ms)
    has_legacy_smooth = "smooth_waveforms" in archive.files
    legacy_width: float = float(SMOOTH_WINDOWS_MS[0])
    if has_legacy_smooth and not has_multi_width:
        legacy_width = float(archive["smooth_window_ms"]) if "smooth_window_ms" in archive.files else float(SMOOTH_WINDOWS_MS[0])
        smooth_windows_ms = [legacy_width]
        print(f"  [note] legacy archive: single smoothing width {legacy_width} ms (re-run extraction to persist all widths)")
    if not has_multi_width and not has_legacy_smooth:
        print("  [warning] archive predates smoothing; smoothed arrays will mirror raw. Re-run extraction to persist smoothed waveforms.")

    results: Dict[int, Dict[str, Any]] = {}
    for channel_idx, channel in enumerate(channels):
        channel_id = int(channel)
        raw_waveforms = list(archive["waveforms"][channel_idx])
        if has_multi_width:
            smooth_waveforms = { float(w): list(archive[f"smooth_waveforms_{int(w)}ms"][channel_idx]) for w in smooth_windows_ms }
        elif has_legacy_smooth:
            smooth_waveforms = { legacy_width: list(archive["smooth_waveforms"][channel_idx]) }
        else:
            smooth_waveforms = { float(w): raw_waveforms for w in smooth_windows_ms }

        results[channel_id] = {
            "waveforms": raw_waveforms,
            "smooth_waveforms": smooth_waveforms,
            "spike_times": np.asarray(archive["spike_times"][channel_idx]),
            "window_sizes": np.asarray(archive["window_sizes"][channel_idx]),
            "peak_indices": np.asarray(archive["peak_positions"][channel_idx], dtype=int),
            "window_starts": np.asarray(archive["window_starts"][channel_idx], dtype=int),
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
        print(f"  Loaded channel {channel_id}: {results[channel_id]['n_extracted']} waveforms")

    return results


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
    return np.asarray(scipy.signal.savgol_filter(waveform, window_len, polyorder), dtype=float)


def _resolve_run_paths(args: argparse.Namespace) -> Tuple[Path, Path, str, str, Path]:
    """Determine the run directory and all output paths.

    Normal run: creates a fresh timestamped run dir
        <OUTPUT_ROOT>/<YYYY-MM-DD_HH-MM-SS>/
    so every invocation is archived for iteration tracking (old runs are
    never overwritten).

    Visualize-only (-v): re-derives the run dir from the supplied .npz path.
    If the npz lives at <run_dir>/waveforms/waveforms.npz the run dir is
    npz.parent.parent; otherwise npz.parent is used. HTML is (re)written
    into <run_dir>/html/, so a previous run can be rendered in place.

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
        run_dir = (Path(args.out_root) / datetime.datetime.now().strftime(TIMESTAMP_FORMAT))
        run_dir.mkdir(parents=True, exist_ok=True)
        npz_path = run_dir / WAVEFORM_REL_PATH

    html_dir = run_dir / "html"
    grid_path: str = args.spike_html or str(html_dir / "waveforms_grid.html")
    channel_path: str = args.channel_html or str(html_dir / "all_ch_spikes.html")
    return run_dir, npz_path, grid_path, channel_path, html_dir


def _write_run_meta(run_dir: Path, args: argparse.Namespace, results: Dict[int, Dict[str, Any]], npz_path: Path) -> None:
    """Write run_meta.json: parameters, timestamps and per-channel summary.

    Mirrors the constants in this module so any archived run can be fully reconstructed / cross-referenced.
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
            "events": int(sum(channel_data["n_events"] for channel_data in results.values())),
            "extracted": int(sum(channel_data["n_extracted"] for channel_data in results.values())),
        },
    }

    (run_dir / RUN_META_FILENAME).write_text(json.dumps(meta, indent=2))
    print(f"Saved run metadata to: {run_dir / RUN_META_FILENAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MEA spike waveform extraction and visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-d", "--data-file", type=str, default=RAW_DATA_FILE,
                        help="Path to raw MEA binary recording")

    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output .npz path; default <out-root>/<ts>/waveforms/waveforms.npz (in -v mode, required: path to a previous run's .npz)")

    parser.add_argument("--out-root", type=str, default=OUTPUT_ROOT,
                        help="Directory under which timestamped runs are stored")

    parser.add_argument("-s", "--spike-html", type=str, default=None,
                        help="Output HTML path for the waveform grid (default <run-dir>/html/waveforms_grid.html)")

    parser.add_argument("-c", "--channel-html", type=str, default=None,
                        help="Output HTML path for the full-trace view (default <run-dir>/html/all_ch_spikes.html)")

    parser.add_argument("-v", "--visualize-only", action="store_true",
                        help="Only render HTML from previously extracted waveforms (-o points at a previous .npz)")
                             
    args = parser.parse_args()
    main(args)
