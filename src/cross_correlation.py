"""
cross_correlation.py  (pairwise cross-correlograms for MEA spike trains)

Run from the project root:
    python cross_correlation.py                           # latest run
    python cross_correlation.py --npz path/to/waveforms.npz
    python cross_correlation.py --lag-max 200             # ±200 ms range

Pipeline for computing directed pairwise cross-correlograms between all
channels of an 8x8 MEA recording, using spike times extracted by
raw_analysis.py.

Notation:
    K       = 64 channels on the MEA grid (8 rows x 8 columns)
    s_k     = spike-train for channel k:  s_k = {t_1, t_2, ..., t_{M_k}}
              where each t_a is a spike time in seconds
    fs      = sample rate in Hz (SAMPLE_RATE_HZ = 30000)
    B       = total number of lag bins
    Δ_j     = width of bin j in ms  (non-uniform: fine near 0, coarse beyond)
    τ_max   = maximum absolute lag in ms  (LAG_MAX_MS)

  1. Non-uniform binning:
         The lag axis is split into two regimes so that fine-timescale
         propagation (sub-millisecond) is resolved without wasting bins
         on the long-tail where counts are sparse:

         - Fine bins:   Δ = FINE_BIN_MS ms,  for |lag| ≤ FINE_CUTOFF_MS
         - Coarse bins: Δ = COARSE_BIN_MS ms, for |lag| > FINE_CUTOFF_MS

         Edges are constructed symmetrically about 0:

             pos_edges = unique(arange(0, cutoff + Δ_fine/2, Δ_fine)
                                ∪ arange(cutoff, τ_max + Δ_coarse/2, Δ_coarse))
             bin_edges = [-pos_edges[::-1], pos_edges[1:]]      (mirror, exclude 0 dup)

         B = len(bin_edges) - 1 bins total, bin_widths[j] = bin_edges[j+1] - bin_edges[j].

  2. Directed cross-correlogram (channel i → channel j):
         For every spike of the source channel i at time t_a (in ms),
         compute the lag to every spike of the target channel j:

             Δt = t_b - t_a        for all t_b ∈ s_j

         and bin each lag into the non-uniform grid:

             C_{i→j}[b] = |{ (t_a, t_b) : Δt ∈ [bin_edges[b], bin_edges[b+1]) }|

         This is a permissive "all-to-all" correlogram: every source spike
         generates a full set of lagged target counts.  The result is an
         integer array of length B.

  3. Summed cross-correlogram (channel i → all others):
         Aggregate the directed CCGs from channel i to every other
         channel j ≠ i:

             S_i[b] = Σ_{j ≠ i}  C_{i→j}[b]

         S_i[b] measures the total number of spikes across all 63 target
         channels that fall in lag bin b relative to source spikes of
         channel i.  Dividing by M_i (the source spike count) yields an
         average per-source-spike response.

  4. Adjacency analysis:
         For each channel i, the four direct grid neighbors (up/down/left/
         right) are identified.  The adjacency page shows the summed CCG
         and individual directed CCGs to each neighbor, both at full lag
         range and zoomed to ±ZOOM_OFFSET_MS for inspecting short-latency
         propagation.

Output (in <run_dir>/cross_corr/):
    cross_correlograms.npz    raw CCG data: (K, K, B) int array, bin_edges, n_spikes
    ccg_grid.html             one full-width card per channel with zoomed histograms
    adjacency/<ch>_adjacency.html   per-channel page: summed + neighbor CCGs

cross_correlograms.npz schema (np.savez_compressed):
    correlograms     int64   (K, K, B)   directed CCGs; [i, j, :] = C_{i→j}
    bin_edges        float64 (B+1,)      non-uniform bin edges in ms
    n_spikes         int64   (K,)        spike count per channel
    source_npz       str                 path to input waveforms.npz
    params           str                 dict of analysis parameters
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure



# ---------------- analysis constants ----------------
LAG_MAX_MS: float = 10000.0               # maximum absolute time lag (ms)
FINE_BIN_MS: float = 0.01                 # bin width for short lags (ms)
COARSE_BIN_MS: float = 1.0                # bin width for long lags (ms)
FINE_CUTOFF_MS: float = 50.0              # boundary between fine and coarse bins (ms)
NUM_CHANNELS: int = 64                    # 8x8 MEA
SAMPLE_RATE_HZ: int = 30000               # for converting spike times to ms
GRID_ROWS: int = 8                        # MEA grid rows
GRID_COLS: int = 8                        # MEA grid columns

# ---------------- visualization constants ----------------
CCG_FIG_DPI: int = 100                    # resolution for full-range CCG figures
CCG_FIG_WIDTH: float = 14.0               # full-range figure width (inches)
CCG_FIG_HEIGHT: float = 3.0               # full-range figure height (inches)
ZOOM_OFFSET_MS: float = 5.0               # ±offset from center for zoomed plots
ZOOM_FIG_WIDTH: float = 14.0              # zoomed figure width (inches)
ZOOM_FIG_HEIGHT: float = 3.0              # zoomed figure height (inches)
ADJACENCY_FIG_DPI: int = 120              # resolution for adjacency figures
ADJACENCY_FIG_WIDTH: float = 16.0         # adjacency figure width (inches)
ADJACENCY_FIG_HEIGHT: float = 10.0        # adjacency figure height (inches)
ZOOM_LEVELS: List[int] = [5, 10]          # zoom window half-widths in ms (dropdown options)


def _build_bins(lag_max_ms: float = LAG_MAX_MS,
                fine_bin_ms: float = FINE_BIN_MS,
                coarse_bin_ms: float = COARSE_BIN_MS,
                fine_cutoff_ms: float = FINE_CUTOFF_MS) -> Tuple[np.ndarray, np.ndarray]:
    """Build non-uniform bin edges: fine bins near zero, coarse bins beyond.

    Constructs a symmetric lag axis with high resolution near lag zero
    (where direct spike propagation produces sharp peaks) and coarser
    resolution at longer lags (where counts are sparse).  The positive
    half-axis is built first, then mirrored about zero.

    pos_edges = unique(arange(0, cutoff + Δ_f/2, Δ_f)
                       ∪ arange(cutoff, τ_max + Δ_c/2, Δ_c))
    bin_edges = [-pos_edges[::-1], pos_edges[1:]]

    Parameters
    ----------
    lag_max_ms : float
        Maximum absolute lag in ms.  The bin axis spans [-lag_max_ms, +lag_max_ms].
    fine_bin_ms : float
        Bin width in ms for the fine regime (|lag| ≤ fine_cutoff_ms).
    coarse_bin_ms : float
        Bin width in ms for the coarse regime (|lag| > fine_cutoff_ms).
    fine_cutoff_ms : float
        Boundary in ms between fine and coarse binning.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (bin_edges, bin_widths) where bin_edges has shape (B+1,) and
        bin_widths has shape (B,) giving the width of each bin in ms.
    """
    # Positive half-axis: fine bins near zero, coarse bins beyond
    fine_edges = np.arange(0, fine_cutoff_ms + fine_bin_ms / 2, fine_bin_ms)
    coarse_edges = np.arange(fine_cutoff_ms, lag_max_ms + coarse_bin_ms / 2, coarse_bin_ms)
    pos_edges = np.unique(np.concatenate([fine_edges, coarse_edges]))
    pos_edges = pos_edges[pos_edges <= lag_max_ms]

    # Ensure the last edge reaches exactly lag_max_ms
    if pos_edges[-1] < lag_max_ms:
        pos_edges = np.append(pos_edges, lag_max_ms)

    # Mirror: [-pos_edges[::-1], pos_edges] excluding the duplicate at 0
    full_edges = np.concatenate([-pos_edges[::-1][:-1], pos_edges])
    bin_widths = np.diff(full_edges)
    return full_edges, bin_widths


def cross_correlogram(spike_times_source: np.ndarray,
                      spike_times_target: np.ndarray,
                      bin_edges: np.ndarray,
                      bin_widths: np.ndarray) -> np.ndarray:
    """Compute the directed cross-correlogram from source to target channel.

    For every spike of the source channel at time t_a, compute the lag
    to every spike of the target channel and bin each lag:

        C_{source→target}[b] = |{ (t_a, t_b) : t_b - t_a ∈ [edges[b], edges[b+1]) }|

    This is an all-to-all correlogram: each source spike generates a full
    set of lagged target counts.  The result is an integer histogram of
    length B = len(bin_widths).

    Parameters
    ----------
    spike_times_source : np.ndarray
        1-D array of source spike times in seconds.
    spike_times_target : np.ndarray
        1-D array of target spike times in seconds.
    bin_edges : np.ndarray
        Non-uniform bin edges in ms, shape (B+1,).  Symmetric about zero.
    bin_widths : np.ndarray
        Width of each bin in ms, shape (B,).

    Returns
    -------
    np.ndarray
        Integer array of shape (B,) giving the spike count in each lag bin.
    """
    if len(spike_times_source) == 0 or len(spike_times_target) == 0:
        return np.zeros(len(bin_widths), dtype=int)

    # Convert spike times from seconds to ms for direct comparison with bin edges
    source_ms = spike_times_source * 1000.0
    target_ms = spike_times_target * 1000.0

    counts = np.zeros(len(bin_widths), dtype=int)
    lag_min = bin_edges[0]
    lag_max = bin_edges[-1]

    # For each source spike, compute lags to all target spikes and bin them
    for source_spike_ms in source_ms:
        lags_to_target = target_ms - source_spike_ms

        # Only consider lags within the full bin range
        in_range = (lags_to_target >= lag_min) & (lags_to_target <= lag_max)
        if not np.any(in_range):
            continue

        lags_in_range = lags_to_target[in_range]

        # digitize returns 1-indexed bin; subtract 1 for 0-indexed
        bin_indices = np.digitize(lags_in_range, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_widths) - 1)

        for bin_index in bin_indices:
            counts[bin_index] += 1

    return counts


def load_spike_times(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load spike times and channel indices from a waveforms npz archive.

    Reads the spike_times object array and channels integer array
    produced by raw_analysis.py's save_waveforms() function.

    Parameters
    ----------
    npz_path : str
        Path to the waveforms .npz file (e.g. outputs/<ts>/waveforms/waveforms.npz).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (spike_times, channels) where spike_times is an object array of
        length K, each element a 1-D float64 array of spike times in
        seconds, and channels is a 1-D int64 array of channel indices.
    """
    archive = np.load(npz_path, allow_pickle=True)
    return archive["spike_times"], archive["channels"]


def compute_all_cross_correlograms(
    spike_times: np.ndarray,
    channels: np.ndarray,
    lag_max_ms: float = LAG_MAX_MS,
    fine_bin_ms: float = FINE_BIN_MS,
    coarse_bin_ms: float = COARSE_BIN_MS,
    fine_cutoff_ms: float = FINE_CUTOFF_MS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute directed cross-correlograms for all 64x63 channel pairs.

    Iterates over every ordered pair (source, target) with source ≠ target
    and computes the directed CCG using non-uniform binning.  The result
    is a 3-D array where correlograms[source, target, :] is the CCG from
    source to target.  Diagonal entries are left as zeros.

    Parameters
    ----------
    spike_times : np.ndarray
        Object array of length K, each element a 1-D float64 array of
        spike times in seconds.
    channels : np.ndarray
        1-D int64 array of channel indices (0-63).
    lag_max_ms : float
        Maximum absolute lag in ms.
    fine_bin_ms : float
        Bin width in ms for short lags.
    coarse_bin_ms : float
        Bin width in ms for long lags.
    fine_cutoff_ms : float
        Boundary in ms between fine and coarse binning.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (correlograms, bin_edges, n_spikes) where correlograms is int64
        of shape (K, K, B), bin_edges is float64 of shape (B+1,), and
        n_spikes is int64 of shape (K,) giving the spike count per channel.
    """
    # Build the non-uniform bin axis
    bin_edges, bin_widths = _build_bins(lag_max_ms, fine_bin_ms, coarse_bin_ms, fine_cutoff_ms)
    total_bins = len(bin_widths)
    total_channels = len(channels)

    # Allocate output: (K, K, B) for all directed pairs
    correlograms = np.zeros((NUM_CHANNELS, NUM_CHANNELS, total_bins), dtype=int)
    n_spikes = np.zeros(NUM_CHANNELS, dtype=int)

    # Iterate over all ordered pairs (source, target)
    for source_idx in range(total_channels):
        source_channel = channels[source_idx]
        source_times = spike_times[source_idx]
        n_spikes[source_channel] = len(source_times)

        for target_idx in range(total_channels):
            target_channel = channels[target_idx]
            if source_channel == target_channel:
                continue

            target_times = spike_times[target_idx]
            correlograms[source_channel, target_channel] = cross_correlogram(
                source_times, target_times, bin_edges, bin_widths
            )

    return correlograms, bin_edges, n_spikes


def save_cross_correlograms(output_path: str,
                            correlograms: np.ndarray,
                            bin_edges: np.ndarray,
                            n_spikes: np.ndarray,
                            npz_path: str,
                            params: Dict[str, float]) -> None:
    """Persist cross-correlograms and metadata to an npz archive.

    Saves the full (K, K, B) correlogram array, bin edges, per-channel
    spike counts, the path to the source npz, and a string-encoded dict
    of analysis parameters.

    Parameters
    ----------
    output_path : str
        Destination path for the .npz file (e.g. <run_dir>/cross_corr/cross_correlograms.npz).
    correlograms : np.ndarray
        Directed CCGs, int64 of shape (K, K, B).
    bin_edges : np.ndarray
        Non-uniform bin edges in ms, float64 of shape (B+1,).
    n_spikes : np.ndarray
        Per-channel spike counts, int64 of shape (K,).
    npz_path : str
        Path to the input waveforms.npz (stored as metadata).
    params : Dict[str, float]
        Analysis parameters (lag_max_ms, fine_bin_ms, etc.).
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        correlograms=correlograms,
        bin_edges=bin_edges,
        n_spikes=n_spikes,
        source_npz=npz_path,
        params=str(params),
    )
    print(f"Saved cross-correlograms to: {output}")


def _figure_to_svg(fig: Figure) -> str:
    """Convert a matplotlib figure to a CSS-scalable SVG string.

    Renders the figure to SVG, strips XML/DOCTYPE declarations and fixed
    width/height attributes from the root <svg> tag, then injects
    width="100%" so the SVG scales with its container and responds to
    browser zoom (ctrl+/ctrl-).

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure to render.

    Returns
    -------
    str
        Clean SVG markup with width="100%" and viewBox preserved.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    svg = buf.read().decode("utf-8")
    plt.close(fig)

    # Strip XML declaration and DOCTYPE that matplotlib prepends
    svg = re.sub(r'<\?xml[^>]*>\s*', '', svg)
    svg = re.sub(r'<!DOCTYPE[^>]*>\s*', '', svg)

    # Remove fixed width/height from root <svg> tag
    svg = re.sub(r'(<svg\b[^>]*?)\s+width="[^"]*"', r'\1', svg, count=1)
    svg = re.sub(r'(<svg\b[^>]*?)\s+height="[^"]*"', r'\1', svg, count=1)

    # Inject width="100%" so CSS and browser zoom control scaling
    svg = svg.replace('<svg ', '<svg width="100%" ', 1)
    return svg


def _get_adjacent_channels(channel: int) -> List[int]:
    """Return the direct grid neighbors (up/down/left/right) of a channel.

    On the 8x8 MEA grid, each channel has at most 4 neighbors.  Edge
    and corner channels have fewer.

    Parameters
    ----------
    channel : int
        Channel index (0-63).  Row = channel // 8, Col = channel % 8.

    Returns
    -------
    List[int]
        List of neighbor channel indices (0-3 elements).
    """
    row, col = channel // GRID_COLS, channel % GRID_COLS
    neighbors: List[int] = []

    # Check all four cardinal directions
    for row_delta, col_delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbor_row = row + row_delta
        neighbor_col = col + col_delta
        if 0 <= neighbor_row < GRID_ROWS and 0 <= neighbor_col < GRID_COLS:
            neighbors.append(neighbor_row * GRID_COLS + neighbor_col)

    return neighbors


def _get_direction_label(source_channel: int, target_channel: int) -> str:
    """Return a human-readable direction label between two grid channels.

    Parameters
    ----------
    source_channel : int
        Source channel index (0-63).
    target_channel : int
        Target channel index (0-63).

    Returns
    -------
    str
        One of "up", "down", "left", "right".
    """
    source_row, source_col = source_channel // GRID_COLS, source_channel % GRID_COLS
    target_row, target_col = target_channel // GRID_COLS, target_channel % GRID_COLS

    if target_row < source_row:
        return "up"
    elif target_row > source_row:
        return "down"
    elif target_col < source_col:
        return "left"
    else:
        return "right"


def _make_zoom_html_controls() -> Tuple[str, str]:
    """Build the HTML controls (checkbox + dropdown) and JS toggle logic.

    Returns
    -------
    Tuple[str, str]
        (controls_html, script_html) where controls_html is the HTML for
        the checkbox and dropdown, and script_html is the JS that toggles
        visibility of .full-range and .zoom-panel elements.
    """
    controls = (
        "    <div class='controls'>\n"
        "        <label><input type='checkbox' id='toggleFull'> Show full-range lags</label>\n"
        "        <label>Zoom: <select id='zoomSelect'>\n"
        "            <option value='5' selected>±5 ms</option>\n"
        "            <option value='10'>±10 ms</option>\n"
        "        </select></label>\n"
        "    </div>"
    )
    script = (
        "    <script>\n"
        "        document.getElementById('toggleFull').addEventListener('change', function() {\n"
        "            document.querySelectorAll('.full-range').forEach(el => {\n"
        "                el.style.display = this.checked ? 'block' : 'none';\n"
        "            });\n"
        "        });\n"
        "        document.getElementById('zoomSelect').addEventListener('change', function() {\n"
        "            document.querySelectorAll('.zoom-panel').forEach(el => {\n"
        "                el.classList.remove('active');\n"
        "            });\n"
        "            document.querySelectorAll('.zoom-' + this.value).forEach(el => {\n"
        "                el.classList.add('active');\n"
        "            });\n"
        "        });\n"
        "    </script>"
    )
    return controls, script


def _make_zoom_svg(data: np.ndarray,
                   bin_centers: np.ndarray,
                   zoom_half_width: int,
                   title: str,
                   color: str,
                   fig_width: float = ZOOM_FIG_WIDTH,
                   fig_height: float = ZOOM_FIG_HEIGHT) -> str:
    """Render a zoomed CCG histogram as a CSS-scalable SVG string.

    Extracts bins within ±zoom_half_width ms of zero, plots them with
    integer x-axis ticks, and returns the SVG markup.

    Parameters
    ----------
    data : np.ndarray
        CCG counts, shape (B,).
    bin_centers : np.ndarray
        Bin center positions in ms, shape (B,).
    zoom_half_width : int
        Half-width of the zoom window in ms (e.g. 5 for ±5 ms).
    title : str
        Plot title string.
    color : str
        Matplotlib color for the fill and line.
    fig_width : float
        Figure width in inches.
    fig_height : float
        Figure height in inches.

    Returns
    -------
    str
        SVG markup string.
    """
    # Extract bins within the zoom window
    zoom_mask = (bin_centers >= -zoom_half_width) & (bin_centers <= zoom_half_width)
    zoom_centers = bin_centers[zoom_mask]
    zoom_data = data[zoom_mask]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.fill_between(zoom_centers, zoom_data, alpha=0.4, color=color)
    ax.plot(zoom_centers, zoom_data, linewidth=1.0, color=color)
    ax.axvline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlim(-zoom_half_width, zoom_half_width)
    ax.set_xticks(range(-zoom_half_width, zoom_half_width + 1))
    ax.set_xlabel("Lag (ms)", fontsize=11)
    ax.set_ylabel("Spike count", fontsize=11)
    ax.set_title(title, fontsize=12, pad=6)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.6)
    return _figure_to_svg(fig)


def gen_ccg_grid_html(correlograms: np.ndarray,
                      bin_edges: np.ndarray,
                      n_spikes: np.ndarray,
                      output_path: str) -> None:
    """Generate a single-page HTML with one full-width card per channel.

    Each card displays:
      - A header with channel number, spike count, and grid neighbors
      - Zoomed histograms at ±5 ms and ±10 ms (switchable via dropdown)
      - An optional full-range summed CCG (toggled by a checkbox)

    Channels with zero spikes get a placeholder card.  Clicking a card
    opens the corresponding adjacency page (ch<NN>_adjacency.html).

    The HTML uses CSS-controlled SVG scaling (width="100%" + viewBox)
    so all plots scale with browser zoom.

    Parameters
    ----------
    correlograms : np.ndarray
        Directed CCGs, int64 of shape (K, K, B).
    bin_edges : np.ndarray
        Non-uniform bin edges in ms, float64 of shape (B+1,).
    n_spikes : np.ndarray
        Per-channel spike counts, int64 of shape (K,).
    output_path : str
        Destination path for the HTML file.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Build the shared controls and JS
    controls_html, script_html = _make_zoom_html_controls()

    # Start building the HTML document
    html = [
        "<!DOCTYPE html>", "<html>", "<head>",
        "    <title>Cross-Correlograms</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }",
        "        h1 { color: #333; }",
        "        .meta { font-size: 13px; color: #666; }",
        "        .controls { margin: 12px 0 20px 0; display: flex; gap: 20px; align-items: center; }",
        "        .channel-card { background: white; padding: 20px; margin-bottom: 24px; "
        "border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); cursor: pointer; "
        "text-decoration: none; color: inherit; display: block; }",
        "        .channel-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.2); }",
        "        .channel-header { background: #e8e8e8; padding: 15px; margin: -20px -20px 15px -20px; "
        "border-radius: 8px 8px 0 0; }",
        "        .channel-header h2 { margin: 0 0 6px 0; }",
        "        .stats { font-size: 13px; color: #666; }",
        "        svg { display: block; width: 100%; height: auto; }",
        "        .full-range, .zoom-panel { display: none; }",
        "        .zoom-panel.active { display: block; }",
        "    </style>",
        "</head>", "<body>",
        "    <h1>Cross-Correlograms</h1>",
        controls_html,
        "    <p class='meta'>Summed cross-correlogram from each channel to all 63 others. "
        "Positive lag = target fires after source. "
        "<b>Y-axis</b>: total spike count across all 63 target channels at each lag, "
        "accumulated over all source spikes. Divide by the source spike count to get "
        "an average per-source-spike response. Click a card to open its adjacency page.</p>",
        script_html,
    ]

    # Generate one card per channel
    for channel in range(NUM_CHANNELS):
        spike_count = int(n_spikes[channel])
        adj_file = f"adjacency/ch{channel:02d}_adjacency.html"

        # Placeholder card for channels with no spikes
        if spike_count == 0:
            html.append(
                f"    <a class='channel-card' href='{adj_file}'>"
                f"<div class='channel-header'><h2>Channel {channel}</h2>"
                f"<div class='stats'>0 spikes</div></div></a>"
            )
            continue

        # Summed CCG: aggregate directed CCGs from this channel to all 63 others
        summed_ccg = correlograms[channel].sum(axis=0)

        # --- Full range CCG (hidden by default, toggled by checkbox) ---
        fig_full, ax_full = plt.subplots(figsize=(CCG_FIG_WIDTH, CCG_FIG_HEIGHT))
        ax_full.fill_between(bin_centers, summed_ccg, alpha=0.4, color="steelblue")
        ax_full.plot(bin_centers, summed_ccg, linewidth=0.8, color="steelblue")
        ax_full.axvline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_full.set_xlim(bin_edges[0], bin_edges[-1])
        ax_full.set_xlabel("Lag (ms)", fontsize=11)
        ax_full.set_ylabel("Spike count", fontsize=11)
        ax_full.set_title(
            f"Channel {channel} — {spike_count} spikes — Summed CCG (all lags)",
            fontsize=12, pad=6,
        )
        ax_full.tick_params(labelsize=9)
        ax_full.spines["top"].set_visible(False)
        ax_full.spines["right"].set_visible(False)
        fig_full.tight_layout(pad=0.6)
        svg_full = _figure_to_svg(fig_full)

        # --- Zoomed histograms at each zoom level ---
        svg_zooms: Dict[int, str] = {}
        for zoom_width in ZOOM_LEVELS:
            svg_zooms[zoom_width] = _make_zoom_svg(
                summed_ccg, bin_centers, zoom_width,
                f"Channel {channel} — ±{zoom_width} ms", "steelblue",
            )

        # Build zoom panel divs; first zoom level is active by default
        neighbors = _get_adjacent_channels(channel)
        zoom_divs = ""
        for level_index, zoom_width in enumerate(ZOOM_LEVELS):
            active_class = " active" if level_index == 0 else ""
            zoom_divs += (
                f"<div class='zoom-panel zoom-{zoom_width}{active_class}'>"
                f"{svg_zooms[zoom_width]}</div>"
            )

        # Card is a clickable link to the adjacency page
        neighbor_list = ", ".join(str(nb) for nb in neighbors)
        html.append(
            f"    <a class='channel-card' href='{adj_file}'>"
            f"<div class='channel-header'><h2>Channel {channel}</h2>"
            f"<div class='stats'>{spike_count} spikes | "
            f"Neighbors: {neighbor_list}</div></div>"
            f"<div class='full-range'>{svg_full}</div>"
            f"{zoom_divs}"
            f"</a>"
        )

    html.extend(["</body>", "</html>"])
    output.write_text("\n".join(html))
    print(f"Saved CCG grid to: {output}")


def gen_adjacency_html(correlograms: np.ndarray,
                       bin_edges: np.ndarray,
                       n_spikes: np.ndarray,
                       output_dir: str) -> None:
    """Generate per-channel HTML files showing CCGs with direct neighbors.

    For each channel with ≥1 spike, produces an HTML page containing:
      - Navigation links (back to grid, prev/next channel)
      - The summed CCG (dark orange) at full lag range
      - Individual directed CCGs to each grid neighbor (steelblue)
      - Zoomed panels at ±5 ms and ±10 ms with a dropdown toggle

    Each subplot is rendered as a CSS-scalable SVG.  The zoomed panels
    are inline (not modal) and switchable via a dropdown at the top.

    Parameters
    ----------
    correlograms : np.ndarray
        Directed CCGs, int64 of shape (K, K, B).
    bin_edges : np.ndarray
        Non-uniform bin edges in ms, float64 of shape (B+1,).
    n_spikes : np.ndarray
        Per-channel spike counts, int64 of shape (K,).
    output_dir : str
        Directory to write per-channel HTML files (e.g. <run_dir>/cross_corr/adjacency/).
    """
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Build the shared controls and JS for zoom/full-range toggling
    controls_html, script_html = _make_zoom_html_controls()

    # Generate one HTML page per active channel
    for channel in range(NUM_CHANNELS):
        if n_spikes[channel] == 0:
            continue

        neighbors = _get_adjacent_channels(channel)

        # --- Build subplot data: summed CCG first, then individual neighbors ---
        subplot_entries: List[Tuple[str, str, str, np.ndarray]] = []

        # Summed CCG: aggregate from this channel to all others
        summed_ccg = correlograms[channel].sum(axis=0)
        subplot_entries.append((
            "summed", f"Ch {channel} \u2192 All (summed)", "darkorange", summed_ccg,
        ))

        # Individual neighbor CCGs with directional labels
        for neighbor in neighbors:
            direction = _get_direction_label(channel, neighbor)
            label = f"Ch {channel} \u2192 Ch {neighbor} ({direction})"
            subplot_entries.append((
                f"ch{neighbor}", label, "steelblue", correlograms[channel, neighbor],
            ))

        # --- Generate full-range SVGs for each subplot ---
        svg_full_range: List[str] = []
        for key, label, color, data in subplot_entries:
            fig, ax = plt.subplots(figsize=(12, 2.5))
            ax.fill_between(bin_centers, data, alpha=0.4, color=color)
            ax.plot(bin_centers, data, linewidth=0.8, color=color)
            ax.axvline(0, color="red", linewidth=0.7, linestyle="--", alpha=0.5)
            ax.set_xlim(bin_edges[0], bin_edges[-1])
            ax.set_title(label, fontsize=11, pad=4)
            ax.set_ylabel("Spike count", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            # Only show x-axis label on the bottom subplot
            is_last_subplot = (key == subplot_entries[-1][0])
            if key == "summed" or is_last_subplot:
                ax.set_xlabel("Lag (ms)", fontsize=10)
            fig.tight_layout(pad=0.5)
            svg_full_range.append(_figure_to_svg(fig))

        # --- Generate zoomed SVGs at each zoom level for each subplot ---
        svg_zoom_panels: Dict[int, List[str]] = {}
        for zoom_width in ZOOM_LEVELS:
            svg_zoom_panels[zoom_width] = []
            for key, label, color, data in subplot_entries:
                zoomed_svg = _make_zoom_svg(
                    data, bin_centers, zoom_width,
                    f"{label} — ±{zoom_width} ms", color,
                    fig_width=8, fig_height=3,
                )
                svg_zoom_panels[zoom_width].append(zoomed_svg)

        # --- Navigation links ---
        previous_channel: int | None = None
        next_channel: int | None = None
        for candidate in range(channel - 1, -1, -1):
            if n_spikes[candidate] > 0:
                previous_channel = candidate
                break
        for candidate in range(channel + 1, NUM_CHANNELS):
            if n_spikes[candidate] > 0:
                next_channel = candidate
                break

        nav_parts = ['<a href="../ccg_grid.html">Back to grid</a>']
        if previous_channel is not None:
            nav_parts.append(
                f'<a href="ch{previous_channel:02d}_adjacency.html">\u2190 Ch {previous_channel}</a>'
            )
        if next_channel is not None:
            nav_parts.append(
                f'<a href="ch{next_channel:02d}_adjacency.html">Ch {next_channel} \u2190</a>'
            )
        nav_html = " &nbsp;|&nbsp; ".join(nav_parts)

        # --- Build subplot divs with full-range and zoom panels ---
        subplot_divs: List[str] = []
        for subplot_idx, (key, label, color, _) in enumerate(subplot_entries):
            # Full-range panel (hidden by default)
            full_range_div = (
                f"<div class='full-range'>"
                f"{svg_full_range[subplot_idx]}</div>"
            )
            # Zoom panels for each zoom level; first level is active
            zoom_divs = ""
            for level_idx, zoom_width in enumerate(ZOOM_LEVELS):
                active_class = " active" if level_idx == 0 else ""
                zoom_divs += (
                    f"<div class='zoom-panel zoom-{zoom_width}{active_class}'>"
                    f"{svg_zoom_panels[zoom_width][subplot_idx]}</div>"
                )
            subplot_divs.append(
                f"    <div class='subplot'>\n"
                f"    <div class='subplot-label'>{label}</div>\n"
                f"    {full_range_div}\n"
                f"    {zoom_divs}\n"
                f"    </div>"
            )

        spike_count = int(n_spikes[channel])
        neighbor_list = ", ".join(str(nb) for nb in neighbors)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Channel {channel} Adjacency CCGs</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .meta {{ font-size: 13px; color: #666; }}
        .controls {{ margin: 12px 0 20px 0; display: flex; gap: 20px; align-items: center; }}
        .nav {{ font-size: 13px; margin-bottom: 12px; }}
        .nav a {{ color: #1f77b4; text-decoration: none; margin-right: 12px; }}
        .nav a:hover {{ text-decoration: underline; }}
        .subplot {{ padding: 4px; margin-bottom: 10px; border-radius: 4px;
                    background: white; border: 1px solid #eee; }}
        .subplot-label {{ font-size: 12px; font-weight: bold; color: #444; margin-bottom: 4px; }}
        .subplot svg {{ display: block; width: 100%; height: auto; }}
        .full-range, .zoom-panel {{ display: none; }}
        .zoom-panel.active {{ display: block; }}
    </style>
</head>
<body>
    <h1>Channel {channel} &mdash; Adjacency Cross-Correlograms</h1>
    <div class="nav">{nav_html}</div>
    {controls_html}
    <p class="meta">{spike_count} spikes |
    Neighbors: {neighbor_list} |
    Positive lag = neighbor fires after channel {channel}</p>

{''.join(subplot_divs)}

{script_html}
</body>
</html>"""

        html_path = output_directory / f"ch{channel:02d}_adjacency.html"
        html_path.write_text(html)

    print(f"Saved adjacency HTMLs to: {output_directory}")


def main(args: argparse.Namespace) -> None:
    """Entry point: compute and save pairwise cross-correlograms.

    Locates the waveforms npz (from CLI arg or latest run), computes
    directed CCGs for all channel pairs, saves the npz archive, and
    generates the grid HTML and per-channel adjacency HTMLs.
    """
    # Resolve the input npz path: explicit CLI arg, or auto-find latest run
    npz_path = Path(args.npz) if args.npz else Path()
    if not npz_path.is_file():
        output_root = Path(args.out_root)
        if not output_root.is_dir():
            output_root = Path(__file__).resolve().parent / args.out_root
        if output_root.is_dir():
            runs = sorted([d for d in output_root.iterdir() if d.is_dir()], reverse=True)
            for run in runs:
                candidate = run / "waveforms" / "waveforms.npz"
                if candidate.exists():
                    npz_path = candidate
                    break
        if not npz_path.is_file():
            print("Error: No waveforms npz found. Run raw_analysis.py first.")
            sys.exit(1)

    print("=" * 60)
    print("Cross-Correlation Analysis")
    print("=" * 60)
    print(f"Input:  {npz_path}")
    print(f"Lag:    ±{args.lag_max} ms")
    print(f"Bins:   {args.fine_bin} ms (0-{args.fine_cutoff} ms), "
          f"{args.coarse_bin} ms ({args.fine_cutoff}-{args.lag_max} ms)")
    print("=" * 60)

    # Load spike times from the waveforms archive
    computation_start = time.time()
    spike_times, channels = load_spike_times(str(npz_path))
    print(f"Loaded spike times for {len(channels)} channels")

    # Compute directed CCGs for all 64x63 ordered pairs
    correlograms, bin_edges, n_spikes = compute_all_cross_correlograms(
        spike_times, channels,
        lag_max_ms=args.lag_max,
        fine_bin_ms=args.fine_bin,
        coarse_bin_ms=args.coarse_bin,
        fine_cutoff_ms=args.fine_cutoff,
    )

    elapsed = time.time() - computation_start
    print(f"Computed {NUM_CHANNELS * (NUM_CHANNELS - 1)} directed cross-correlograms in {elapsed:.1f}s")
    print(f"Bin edges: {len(bin_edges)} edges, {len(bin_edges) - 1} bins")
    print(f"Spike counts: min={n_spikes.min()}, max={n_spikes.max()}, "
          f"mean={n_spikes.mean():.0f}")

    # Output directory: <run_dir>/cross_corr/
    run_dir = npz_path.parent.parent
    cross_corr_dir = run_dir / "cross_corr"
    cross_corr_dir.mkdir(parents=True, exist_ok=True)

    # Save npz with analysis parameters
    analysis_params: Dict[str, float] = {
        "lag_max_ms": args.lag_max,
        "fine_bin_ms": args.fine_bin,
        "coarse_bin_ms": args.coarse_bin,
        "fine_cutoff_ms": args.fine_cutoff,
        "sample_rate_hz": float(SAMPLE_RATE_HZ),
    }
    npz_output = cross_corr_dir / "cross_correlograms.npz"
    save_cross_correlograms(
        str(npz_output), correlograms, bin_edges, n_spikes,
        str(npz_path), analysis_params,
    )

    # Generate CCG grid HTML (one card per channel)
    grid_path = cross_corr_dir / "ccg_grid.html"
    gen_ccg_grid_html(correlograms, bin_edges, n_spikes, str(grid_path))

    # Generate per-channel adjacency HTMLs
    adjacency_dir = cross_corr_dir / "adjacency"
    gen_adjacency_html(correlograms, bin_edges, n_spikes, str(adjacency_dir))

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Output: {cross_corr_dir}")
    print(f"  npz:       {npz_output}")
    print(f"  grid:      {grid_path}")
    print(f"  adjacency: {adjacency_dir}/")
    print("=" * 60)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the cross-correlation pipeline.

    Parameters
    ----------
    argv : list[str] | None
        Argument list to parse.  None uses sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: npz, out_root, lag_max,
        fine_bin, coarse_bin, fine_cutoff.
    """
    parser = argparse.ArgumentParser(
        description="Pairwise cross-correlograms for MEA spike trains")
    parser.add_argument("--npz", type=str, default="",
                        help="Path to waveforms.npz (auto-finds latest run if empty)")
    parser.add_argument("--out-root", type=str, default="outputs",
                        help="Root directory for run outputs")
    parser.add_argument("--lag-max", type=float, default=LAG_MAX_MS,
                        help=f"Maximum absolute lag in ms (default: {LAG_MAX_MS})")
    parser.add_argument("--fine-bin", type=float, default=FINE_BIN_MS,
                        help=f"Bin width for short lags in ms (default: {FINE_BIN_MS})")
    parser.add_argument("--coarse-bin", type=float, default=COARSE_BIN_MS,
                        help=f"Bin width for long lags in ms (default: {COARSE_BIN_MS})")
    parser.add_argument("--fine-cutoff", type=float, default=FINE_CUTOFF_MS,
                        help=f"Boundary between fine and coarse bins in ms (default: {FINE_CUTOFF_MS})")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
