"""
visualization_tools.py - HTML/PNG visualization for MEA spike waveforms.

Extracted from raw_analysis.py to separate rendering from computation.
All functions accept pre-computed data and produce HTML files or PNG images.
No analysis or signal processing occurs here.

Shared analysis constants (SAMPLE_RATE_HZ, VOLTAGE_SCALE, EVENT_GATE_SCALE,
SPIKE_GATE_SCALE) are passed as parameters with defaults to avoid circular
imports with raw_analysis.py.
"""

from __future__ import annotations

import base64
import datetime
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np

# ---------------- visualization constants ----------------
CHANNEL_DS_FACTOR: int = 10
SPIKE_WINDOWS_LIMIT: int = 50  # default # of waveform tiles shown per channel in the grid; a page toggle reveals all of them
FIGURE_DPI: int = 80
TILE_DPI: int = 120          # dpi of the per-event grid tiles
TRACE_DPI: int = 100
INTERACTIVE_OVERVIEW_DS: int = 200
INTERACTIVE_SPIKE_DS: int = 4
SPIKE_CONTEXT_MS: float = 200.0
INTERACTIVE_CONTEXT_MS: float = 100.0
PLOTLY_JS: str = "cdn"
INTERACTIVE_HTML_DIR: str = "interactive_ch_views"
INTERACTIVE_HTML_PATTERN: str = "channel_{ch}_interactive.html"


def _html_head(title: str) -> List[str]:
    """Return the opening HTML tags with a standard stylesheet.

    Parameters
    ----------
    title : str
        Page title and h1 heading text.

    Returns
    -------
    List[str]
        List of HTML strings to start the document.
    """
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
    """Join HTML parts and write to a file.

    Parameters
    ----------
    output_file : str
        Output path for the HTML file.
    html_parts : List[str]
        List of HTML strings to join with newlines.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_parts))
    print(f"  Saved HTML to: {output_file}")


def _figure_to_base64(fig: Figure, dpi: int = FIGURE_DPI,
                      tight: bool = True) -> str:
    """Encode a matplotlib figure as a base64 PNG string.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to render.
    dpi : int
        Resolution in dots per inch.
    tight : bool
        If True, use bbox_inches='tight' to trim whitespace.

    Returns
    -------
    str
        Base64-encoded PNG image data suitable for inline HTML img tags.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight" if tight else None)
    plt.close(fig)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img


def _save_figure_png(fig: Figure, output_path: Path, dpi: int = FIGURE_DPI,
                     tight: bool = True) -> None:
    """Render a matplotlib figure to a PNG file.

    Used for writing per-event grid tiles to disk so the HTML page stays
    small and all smoothing variants are available without inflating the
    file to gigabytes.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to render.
    output_path : Path
        Destination path for the PNG file.
    dpi : int
        Resolution in dots per inch.
    tight : bool
        If True, use bbox_inches='tight' to trim whitespace.
    """
    fig.savefig(output_path, format="png", dpi=dpi,
                bbox_inches="tight" if tight else None)
    plt.close(fig)


def _tile_figure(waveform: np.ndarray, start_idx: int, peak_idx: int,
                 sample_rate: int = 30000,
                 ylim: Optional[Tuple[float, float]] = None) -> Figure:
    """Build a small tile figure for a single event window.

    Creates a compact plot with x-axis in absolute recording time (seconds)
    and y-axis showing the window voltage. The dominant peak is marked with
    a dashed vertical line; red dotted horizontal lines mark the window
    min/max so peak-to-peak amplitude is readable directly from the tile.

    Parameters
    ----------
    waveform : np.ndarray
        1-D extracted window in uV.
    start_idx : int
        Absolute sample index where the window starts in the full trace.
    peak_idx : int
        Offset of the dominant peak within the window.
    sample_rate : int
        Sampling rate in Hz.
    ylim : Optional[Tuple[float, float]]
        Fixed y-axis bounds (min, max). If None, computed from the waveform.

    Returns
    -------
    Figure
        Matplotlib figure object (caller must close or save).
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
    fig.tight_layout(pad=0.15)
    return fig


def gen_spike_waveform_html(results: Dict[int, Dict[str, Any]],
                            output_file: str,
                            interactive_pattern: str = INTERACTIVE_HTML_PATTERN,
                            interactive_dir: str = INTERACTIVE_HTML_DIR,
                            spike_windows_limit: int = SPIKE_WINDOWS_LIMIT,
                            dpi: int = TILE_DPI,
                            context_ms: float = INTERACTIVE_CONTEXT_MS,
                            sample_rate: int = 30000,
                            event_gate_scale: float = 5.0,
                            spike_gate_scale: float = 5.0,
                            smooth_method_default: str = "savgol",
                            smooth_windows_ms_default: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
                            smooth_polyorder_default: int = 4,
                            smooth_show_by_default: bool = False) -> None:
    """Render a flex CSS grid of per-event waveform tiles.

    Each tile is a standalone PNG (see _tile_figure) wrapped in a link to the
    channel's interactive view zoomed on that event. A radio selector at the
    top toggles between raw and smoothed variants via a CSS body class. A
    per-channel checkbox limits visible tiles to the first N.

    Parameters
    ----------
    results : Dict[int, Dict[str, Any]]
        Per-channel feature dictionaries (from load_waveforms or process_channel).
    output_file : str
        Path to write the HTML grid file.
    interactive_pattern : str
        Filename pattern for interactive HTML files (with {ch} placeholder).
    interactive_dir : str
        Directory containing interactive HTML files (relative to HTML root).
    spike_windows_limit : int
        Default number of tiles shown per channel before toggling.
    dpi : int
        Resolution for tile PNGs.
    context_ms : float
        Context margin (ms) around each event for the interactive deep-link.
    sample_rate : int
        Sampling rate in Hz.
    event_gate_scale : float
        Envelope gate multiplier (for display labels).
    spike_gate_scale : float
        Spike gate multiplier (for display labels).
    smooth_method_default : str
        Fallback smoothing method if results are empty.
    smooth_windows_ms_default : Tuple[float, ...]
        Fallback smoothing widths if results are empty.
    smooth_polyorder_default : int
        Fallback polynomial order if results are empty.
    smooth_show_by_default : bool
        If True, default to the first smoothed variant instead of raw.
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
                     else smooth_method_default)
    smooth_windows_ms = (list(first_channel_data["smooth_windows_ms"])
                         if first_channel_data else list(smooth_windows_ms_default))
    smooth_polyorder = (first_channel_data["smooth_polyorder"]
                        if first_channel_data else smooth_polyorder_default)
    default_variant = "raw" if not smooth_show_by_default else \
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
                      "is the dominant peak. Click a tile to open the channel's "
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
        env_gate = event_gate_scale * channel_data["std_dev"]
        spk_gate = spike_gate_scale * channel_data["std_dev"]

        tiles = []
        for index, raw_waveform in enumerate(waveforms):
            ylim = (float(raw_waveform.min()), float(raw_waveform.max()))
            tile_prefix = f"ch{channel:02d}_e{index:04d}"
            fig = _tile_figure(raw_waveform, int(window_starts[index]),
                               int(peak_indices[index]), sample_rate=sample_rate, ylim=ylim)
            _save_figure_png(fig, tiles_dir / f"{tile_prefix}_raw.png",
                             dpi=dpi, tight=False)
            # Smoothed tiles render the SAVED smoothed arrays, not fresh
            # smooth_waveform() calls: the archive is the single source of truth.
            img_entries = [f'          <img class="raw" loading="lazy" src="tiles/{tile_prefix}_raw.png" '
                           f'alt="ch{channel} t={float(spike_times[index]):.4f}s (raw)">']
            for window_ms in smooth_windows_ms:
                tag = f"{int(window_ms)}ms"
                smooth_variant = smooth_waveforms[window_ms][index]
                fig = _tile_figure(smooth_variant, int(window_starts[index]),
                                   int(peak_indices[index]), sample_rate=sample_rate, ylim=ylim)
                _save_figure_png(fig, tiles_dir / f"{tile_prefix}_{tag}.png",
                                 dpi=dpi, tight=False)
                img_entries.append(
                    f'          <img class="s{tag}" loading="lazy" src="tiles/{tile_prefix}_{tag}.png" '
                    f'alt="ch{channel} t={float(spike_times[index]):.4f}s ({tag})">')
            event_time = float(spike_times[index])
            margin = (len(raw_waveform) / (2.0 * sample_rate)
                      + context_ms / 1000.0)
            t0 = max(0.0, event_time - margin)
            t1 = event_time + margin
            href = (f"{interactive_dir}/{interactive_pattern.format(ch=channel)}"
                    f"?t0={t0:.4f}&t1={t1:.4f}")
            tiles.append(
                f'        <a class="tile" href="{href}" target="_blank" '
                f'title="ch{channel} event t={event_time:.4f}s">\n'
                + "\n".join(img_entries) + "\n"
                f'        </a>')

        # Per-channel limit checkbox (only shown when it does anything).
        limit_toggle = ""
        if len(tiles) > spike_windows_limit:
            limit_toggle = f"""
                <div class="stats">
                    <label><input type="checkbox" class="ch-limit" checked
                        onchange="this.closest('.channel-section').classList.toggle('show-all', !this.checked)">
                        Limit to first <strong>{spike_windows_limit}</strong> windows</label>
                </div>"""

        html_parts.append(f"""
        <div class="channel-section">
            <div class="channel-header">
                <h2>Channel {channel}</h2>
                <div class="stats">
                    <strong>Events:</strong> {channel_data['n_events']} |
                    <strong>Extracted:</strong> {channel_data['n_extracted']} |
                    <strong>Envelope gate ({event_gate_scale:.0f}x noise):</strong> {env_gate:.2f} uV |
                    <strong>Spike gate ({spike_gate_scale:.0f}x noise):</strong> {spk_gate:.2f} uV |
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
                     output_file: str, raw_data: np.ndarray,
                     ds_factor: int = CHANNEL_DS_FACTOR,
                     sample_rate: int = 30000,
                     voltage_scale: float = 0.195,
                     event_gate_scale: float = 5.0,
                     spike_gate_scale: float = 5.0,
                     trace_dpi: int = TRACE_DPI) -> None:
    """Render a full-trace overview for every channel (downsampled PNG).

    Produces a compact overview where each channel's full recording is
    decimated by `ds_factor` and plotted as a line with detected dominant
    peaks overlaid as red dots. The two gate thresholds (event and spike)
    are shown as dashed lines for visual confirmation of the gating
    criteria.

    Parameters
    ----------
    results : Dict[int, Dict[str, Any]]
        Per-channel feature dictionaries.
    output_file : str
        Path to write the HTML overview file.
    raw_data : np.ndarray
        Raw int16 array of shape (T, 64) from load_raw_data().
    ds_factor : int
        Decimation factor for the trace PNGs.
    sample_rate : int
        Sampling rate in Hz.
    voltage_scale : float
        Conversion factor from LSB to uV.
    event_gate_scale : float
        Envelope gate multiplier (for display labels).
    spike_gate_scale : float
        Spike gate multiplier (for display labels).
    trace_dpi : int
        Resolution for trace PNGs.
    """
    print(f"\nGenerating full-trace HTML: {output_file}")
    html_parts = _html_head("MEA Channel Traces")
    html_parts.append("    <p>Full recording per channel with detected "
                      "events marked (envelope and spike gates shown dashed).</p>")

    for channel in sorted(results.keys()):
        channel_data = results[channel]
        voltage = raw_data[:, channel] * voltage_scale
        downsampled_voltage = voltage[::ds_factor] if ds_factor > 0 else voltage
        downsampled_time = (np.arange(len(downsampled_voltage)) / (sample_rate / ds_factor)
                            if ds_factor > 0
                            else np.arange(len(downsampled_voltage)) / sample_rate)

        # Dominant-peak positions: start of the extracted window + offset of
        # the peak inside it (both persisted in the npz).
        absolute_peaks = (np.asarray(channel_data["window_starts"])
                          + np.asarray(channel_data["peak_indices"]))
        peak_times = absolute_peaks / sample_rate
        peak_voltages = voltage[absolute_peaks]

        env_gate = event_gate_scale * channel_data["std_dev"]
        spk_gate = spike_gate_scale * channel_data["std_dev"]

        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(downsampled_time, downsampled_voltage, color="blue", linewidth=0.4)
        ax.scatter(peak_times, peak_voltages, s=6, c="red", zorder=3)
        ax.axhline(env_gate, color="orange", linewidth=0.9, linestyle="--",
                   alpha=0.8, label=f"{event_gate_scale:.0f}x-noise envelope gate "
                                    f"({env_gate:.2f} uV)")
        ax.axhline(-env_gate, color="orange", linewidth=0.9, linestyle="--",
                   alpha=0.8)
        ax.axhline(spk_gate, color="purple", linewidth=0.7, linestyle=":",
                   alpha=0.8, label=f"{spike_gate_scale:.0f}x-noise spike gate "
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
        img = _figure_to_base64(fig, dpi=trace_dpi)
        html_parts.append(f"""
        <div class="channel-section">
            <div class="channel-header">
                <h2>Channel {channel}</h2>
                <div class="stats">
                    <strong>Events:</strong> {channel_data['n_events']} |
                    <strong>Extracted:</strong> {channel_data['n_extracted']} |
                    <strong>Envelope gate ({event_gate_scale:.0f}x noise):</strong> {env_gate:.2f} uV |
                    <strong>Spike gate ({spike_gate_scale:.0f}x noise):</strong> {spk_gate:.2f} uV |
                    <strong>Noise (MAD):</strong> {channel_data['std_dev']:.2f} uV
                </div>
            </div>
            <img src="data:image/png;base64,{img}" alt="Channel {channel} trace">
        </div>
        """)

    html_parts.append("</body></html>")
    _write_html(output_file, html_parts)


def gen_channel_interactive_html(results: Dict[int, Dict[str, Any]],
                                 channel: int, voltage: np.ndarray,
                                 output_file: str,
                                 overview_ds: int = INTERACTIVE_OVERVIEW_DS,
                                 spike_ds: int = INTERACTIVE_SPIKE_DS,
                                 context_ms: float = SPIKE_CONTEXT_MS,
                                 plotly_js: str = PLOTLY_JS,
                                 sample_rate: int = 30000,
                                 event_gate_scale: float = 5.0,
                                 spike_gate_scale: float = 5.0) -> None:
    """Render a self-contained interactive plotly view for a single channel.

    Stacks three layers: (1) a downsampled full-trace overview, (2) high-
    resolution context segments around every detected peak joined with NaN
    gaps so plotly draws no connecting lines across inter-segment spaces,
    and (3) red markers at the dominant peaks. A JS snippet reads URL query
    params `?t0=..&t1=..` to set the initial x-axis range (used by the grid
    tile deep-links).

    Parameters
    ----------
    results : Dict[int, Dict[str, Any]]
        Per-channel feature dictionaries.
    channel : int
        Channel index to render.
    voltage : np.ndarray
        1-D voltage array (uV) for the channel.
    output_file : str
        Path to write the interactive HTML file.
    overview_ds : int
        Decimation factor for the full-trace overview.
    spike_ds : int
        Decimation factor for high-res spike context segments.
    context_ms : float
        Context window (ms) around each peak.
    plotly_js : str
        Plotly JS library content (base64-encoded).
    sample_rate : int
        Sampling rate in Hz.
    event_gate_scale : float
        Envelope gate multiplier (for display labels).
    spike_gate_scale : float
        Spike gate multiplier (for display labels).
    """
    import plotly.graph_objects as go

    channel_data = results[channel]
    print(f"\nGenerating interactive HTML for channel {channel}: {output_file}")
    n_samples = len(voltage)

    absolute_peaks = (np.asarray(channel_data["window_starts"])
                      + np.asarray(channel_data["peak_indices"]))

    fig = go.Figure()
    if overview_ds > 1:
        overview_time = np.arange(0, n_samples, overview_ds) / sample_rate
        overview_voltage = voltage[::overview_ds]
    else:
        overview_time = np.arange(n_samples) / sample_rate
        overview_voltage = voltage
    fig.add_trace(go.Scatter(x=overview_time, y=overview_voltage, mode="lines",
                             name="overview", line=dict(color="#9ecae1", width=1),
                             hovertemplate="t=%{x:.3f}s<br>%{y:.1f}uV",
                             hoverlabel=dict(bgcolor="#9ecae1")))

    # High-resolution context segments around each dominant peak, separated
    # by NaN so consecutive segments are not bridged by a connecting line.
    half_context_samples = int(context_ms / 2000.0 * sample_rate)
    context_times: List[float] = []
    context_voltages: List[float] = []
    for peak_sample in absolute_peaks:
        segment_start = max(0, peak_sample - half_context_samples)
        segment_end = min(n_samples, peak_sample + half_context_samples)
        segment_time = (np.arange(segment_start, segment_end, spike_ds)
                        / sample_rate)
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

    fig.add_trace(go.Scatter(x=absolute_peaks / sample_rate,
                             y=voltage[absolute_peaks], mode="markers",
                             name="dominant peak",
                             marker=dict(color="red", size=7, symbol="x"),
                             hovertemplate="t=%{x:.3f}s<br>%{y:.1f}uV"))

    env_gate = event_gate_scale * channel_data["std_dev"]
    spk_gate = spike_gate_scale * channel_data["std_dev"]
    fig.add_hline(y=env_gate, line_color="orange", line_width=1, line_dash="dash",
                  name=f"{event_gate_scale:.0f}x-noise envelope gate", showlegend=True)
    fig.add_hline(y=-env_gate, line_color="orange", line_width=1, line_dash="dash")
    fig.add_hline(y=spk_gate, line_color="purple", line_width=1, line_dash="dot",
                  name=f"{spike_gate_scale:.0f}x-noise spike gate", showlegend=True)
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


def _find_run_dirs(output_root: Path, timestamp_format: str) -> List[Path]:
    """List all timestamped run directories under output_root (newest first).

    A run dir is a direct child whose name parses as `timestamp_format`
    (e.g. 2026-08-15_12-00-00). The fixed-width timestamp compares
    lexicographically in chronological order, so sorting on name alone
    puts the newest run first.

    Parameters
    ----------
    output_root : Path
        Root directory containing timestamped run subdirectories.
    timestamp_format : str
        strptime-compatible format string for run directory names.

    Returns
    -------
    List[Path]
        Run directories sorted newest-first.
    """
    runs = []
    for candidate in output_root.iterdir():
        if not candidate.is_dir():
            continue
        try:
            datetime.datetime.strptime(candidate.name, timestamp_format)
        except ValueError:
            continue
        runs.append(candidate)
    runs.sort(key=lambda run: run.name, reverse=True)
    return runs


def write_output_index(output_path: Path, output_root: Path,
                       timestamp_format: str, run_meta_filename: str) -> Path:
    """Regenerate the static entry-point index.html for all runs.

    Writes `output_path/index.html` with relative links to every archived
    run's waveform grid, all-channels view, and run metadata. The newest
    run is listed first. Called at the end of every run (fresh extraction
    and -v re-render) so the latest pages are always one click away from
    the index opened from disk -- plain relative links, no server or JS.

    Parameters
    ----------
    output_path : Path
        Directory where index.html is written (the script directory).
    output_root : Path
        Root directory containing timestamped run subdirectories.
    timestamp_format : str
        strptime-compatible format string for run directory names.
    run_meta_filename : str
        Filename for the run metadata JSON (e.g. "run_meta.json").

    Returns
    -------
    Path
        Path to the generated index.html file.
    """
    runs = _find_run_dirs(output_root, timestamp_format)
    latest = runs[0] if runs else None

    def run_link(run_dir: Path, rel_path: str) -> str:
        # Relative to the index file's location (output_path/index.html), so
        # it stays correct regardless of where output_root sits relative to it.
        target = run_dir / rel_path
        return os.path.relpath(target, output_path).replace(os.sep, "/")

    def format_timestamp(run_name: str) -> str:
        return (datetime.datetime.strptime(run_name, timestamp_format)
                .strftime("%Y-%m-%d %H:%M:%S"))

    latest_section = ""
    if latest is not None:
        latest_grid = run_link(latest, "html/waveforms_grid.html")
        latest_channels = run_link(latest, "html/all_ch_spikes.html")
        latest_meta = run_link(latest, run_meta_filename)
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
        meta = run_link(run, run_meta_filename)
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
    
    index_filepath = output_path / "index.html"
    index_filepath.write_text(html)
    print(f"Updated output index: {index_filepath}")
    return index_filepath
