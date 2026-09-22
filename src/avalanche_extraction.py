"""
avalanche_extraction.py  Gaussian-mixture fitting of CCG latency humps

Run from the project root:
    python avalanche_extraction.py --npz path/to/cross_correlograms.npz
    python avalanche_extraction.py --sigma 2.5          # wider extraction windows
    python avalanche_extraction.py --max-k 6            # allow more humps

Fits a variable number of Gaussians to each side of every channel's
summed cross-correlogram, then overlays the fits on the raw histograms
so you can visually judge how well each hump is captured.

The number of components k is chosen by BIC among candidates from
n_detected_peaks to n_detected_peaks + 2 (clamped at MAX_K).

Output (in <run_dir>/cross_corr/avalanche/):
    chNN.png          per-channel fit overlay
    index.html        gallery linking all channel PNGs

Constants at the top are the only things you should need to tune.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class SideFit(TypedDict):
    """Gaussian-mixture fit for one side of a summed CCG.

    Attributes
    ----------
    k : int
        Number of fitted Gaussians.
    params : np.ndarray
        Flat parameter vector
        [amp_0, mu_0, sigma_0, ..., baseline].
    bic : float
        BIC of the best fit.
    rss : float
        Residual sum of squares.
    amplitudes : np.ndarray
        (k,) amplitudes.
    mus : np.ndarray
        (k,) centres in ms.
    sigmas : np.ndarray
        (k,) standard deviations in ms.
    baseline : float
        Flat baseline level.
    fitted : np.ndarray
        Model curve at ALL bin centres.
    """
    k: int
    params: np.ndarray
    bic: float
    rss: float
    amplitudes: np.ndarray
    mus: np.ndarray
    sigmas: np.ndarray
    baseline: float
    fitted: np.ndarray


class ChannelFit(TypedDict):
    """Per-channel fit results plus the macro-resolution histogram.

    Attributes
    ----------
    pos : SideFit
        Fit on the positive-lag side.
    neg : SideFit
        Fit on the negative-lag side.
    macro_centers : np.ndarray
        Working bin centres.
    macro_counts : np.ndarray
        Rebinned counts.
    macro_smooth : np.ndarray
        Smoothed counts.
    """
    pos: SideFit
    neg: SideFit
    macro_centers: np.ndarray
    macro_counts: np.ndarray
    macro_smooth: np.ndarray

# ======================================================================
#  TUNABLE CONSTANTS  (edit these; everything else is derived)
# ======================================================================

FINE_BIN_MS = 0.01
FINE_CUTOFF_MS = 50.0
REBIN_MS = 0.1
SIGMA_SMOOTH_MS = 0.15
DEAD_ZONE_MS = 1.0
MIN_PEAK_PROMINENCE_FRACTION = 0.05
MIN_PEAK_PROMINENCE_COUNT = 3.0
MIN_PEAK_SEPARATION_MS = 0.3
MAX_K = 5
SIGMA_MULTIPLIER = 2.0
PLOT_XRANGE_MS = 10.0
ZOOM_XRANGE_MS = 5.0
NUM_CHANNELS = 64
GRID_ROWS = 8
GRID_COLS = 8

# ======================================================================
#  Gaussian model
# ======================================================================


def gaussian_model(x: np.ndarray, *params: float) -> np.ndarray:
    """Sum of Gaussians plus a flat baseline.

    Each Gaussian is parameterized as (amplitude, centre, sigma).
    Parameters are packed sequentially:
        [amp_0, mu_0, sigma_0, amp_1, mu_1, sigma_1, ..., baseline]

    Parameters
    ----------
    x : np.ndarray
        Bin centres in ms, shape (N,).
    *params : float
        Packed Gaussian parameters plus trailing baseline.

    Returns
    -------
    np.ndarray
        Model values at each x, shape (N,).
    """
    n_gauss = (len(params) - 1) // 3
    baseline = params[-1]
    y = np.full_like(x, baseline, dtype=float)
    for i in range(n_gauss):
        amp, mu, sigma = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        y = y + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return y


def _initial_guesses(centers: np.ndarray, peak_indices: np.ndarray,
                     smoothed: np.ndarray, baseline: float,
                     n_gauss: int) -> List[float]:
    """Build initial parameter vector from detected peak locations.

    Parameters
    ----------
    centers : np.ndarray
        Working bin centres in ms.
    peak_indices : np.ndarray
        Indices of detected peaks (sorted by prominence descending).
    smoothed : np.ndarray
        Smoothed counts at working resolution.
    baseline : float
        Baseline level (median of the far background).
    n_gauss : int
        Number of Gaussians to initialise.

    Returns
    -------
    List[float]
        Flat parameter list [amp_0, mu_0, sigma_0, ..., baseline].
    """
    params: List[float] = []
    for i in range(min(n_gauss, len(peak_indices))):
        idx = int(peak_indices[i])
        mu = float(centers[idx])
        amp = max(float(smoothed[idx]) - baseline, 1.0)
        sigma = 0.5  # start narrow; let the optimiser broaden
        params.extend([amp, mu, sigma])
    # pad if fewer peaks than requested n_gauss
    while len(params) < 3 * n_gauss:
        params.extend([1.0, 0.0, 1.0])
    params.append(baseline)
    return params


def _make_bounds(n_gauss: int, center_min: float, center_max: float,
                 baseline: float) -> Tuple[np.ndarray, np.ndarray]:
    """Lower and upper bounds for the parameter vector."""
    lo: List[float] = []
    hi: List[float] = []
    for _ in range(n_gauss):
        lo.extend([0.0, center_min, 0.1])
        hi.extend([np.inf, center_max, 15.0])
    lo.append(0.0)
    hi.append(max(baseline * 20, 50.0))
    return np.array(lo), np.array(hi)


def _bic(n: int, k: int, rss: float) -> float:
    """Bayesian Information Criterion (lower is better).

    Parameters
    ----------
    n : int
        Number of data points.
    k : int
        Number of free parameters.
    rss : float
        Residual sum of squares.
    """
    if rss <= 0:
        return -np.inf
    return n * np.log(rss / n) + k * np.log(n)


def _detect_peaks(counts: np.ndarray, centers: np.ndarray,
                  bin_width_ms: float) -> np.ndarray:
    """Detect local maxima on both half-axes, return peak indices
    in the full (both-sides) array sorted by descending smoothed value."""
    half_baseline = float(np.median(counts[np.abs(centers) >= 20.0])) \
        if np.any(np.abs(centers) >= 20.0) else 1.0
    min_prom = max(MIN_PEAK_PROMINENCE_COUNT,
                   MIN_PEAK_PROMINENCE_FRACTION * half_baseline)
    peaks, props = find_peaks(counts, prominence=min_prom,
                              distance=int(round(MIN_PEAK_SEPARATION_MS / bin_width_ms)))
    if len(peaks) == 0:
        return np.array([], dtype=int)
    order = np.argsort(props["prominences"])[::-1]
    return peaks[order]


def fit_side(counts: np.ndarray, centers: np.ndarray,
             bin_width_ms: float,
             dead_zone_mask: Optional[np.ndarray] = None) -> SideFit:
    """Fit Gaussians to one half of the CCG (+ or - lag side).

    Parameters
    ----------
    counts : np.ndarray
        Working-resolution counts for ONE side.
    centers : np.ndarray
        Corresponding bin centres.
    bin_width_ms : float
        Working bin width in ms.
    dead_zone_mask : np.ndarray of bool, optional
        True for bins to EXCLUDE from fitting (e.g. near-zero bins).
        If None, all bins are used.

    Returns
    -------
    SideFit keys:
        k         int           number of fitted Gaussians
        params    np.ndarray    flat parameter vector
        bic       float         BIC of the best fit
        rss       float         residual sum of squares
        amplitudes np.ndarray   (k,) amplitudes
        mus       np.ndarray   (k,) centres in ms
        sigmas    np.ndarray   (k,) standard deviations in ms
        baseline  float
        fitted    np.ndarray    model curve at ALL bin centres
    """
    n = len(counts)
    if n == 0 or counts.sum() == 0:
        return {"k": 0, "params": np.array([0.0]),
                "bic": np.inf, "rss": 0.0,
                "amplitudes": np.array([]), "mus": np.array([]),
                "sigmas": np.array([]), "baseline": 0.0,
                "fitted": np.zeros(n)}

    dz = (np.zeros(n, dtype=bool) if dead_zone_mask is None else dead_zone_mask)

    fit_mask = ~dz
    fit_counts = counts[fit_mask]
    fit_centers = centers[fit_mask]
    n_fit = int(fit_mask.sum())

    if n_fit == 0 or fit_counts.sum() == 0:
        return {"k": 0, "params": np.array([0.0]),
                "bic": np.inf, "rss": 0.0,
                "amplitudes": np.array([]), "mus": np.array([]),
                "sigmas": np.array([]), "baseline": 0.0,
                "fitted": np.full(n, float(np.median(counts)))}

    baseline = float(np.median(fit_counts))
    peak_idx = _detect_peaks(fit_counts, fit_centers, bin_width_ms)
    n_peaks = max(len(peak_idx), 1)

    best_result: Optional[SideFit] = None

    for k in range(n_peaks, min(n_peaks + 3, MAX_K + 1)):
        x0 = _initial_guesses(fit_centers, peak_idx, fit_counts, baseline, k)
        lo, hi = _make_bounds(k, fit_centers.min(), fit_centers.max(), baseline)

        try:
            res = least_squares(
                lambda p: gaussian_model(fit_centers, *p) - fit_counts,
                x0, bounds=(lo, hi),
                max_nfev=4000, method="trf",
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

        fitted_fit = gaussian_model(fit_centers, *res.x)
        rss = float(np.sum((fitted_fit - fit_counts) ** 2))
        n_params = 3 * k + 1
        bic_val = _bic(n_fit, n_params, rss)

        amplitudes = np.array([res.x[3 * i] for i in range(k)])
        mus = np.array([res.x[3 * i + 1] for i in range(k)])
        sigmas = np.array([res.x[3 * i + 2] for i in range(k)])
        bl = float(res.x[-1])

        alive = (amplitudes > 0.1) & (sigmas > 0.05)
        if alive.sum() < k:
            amplitudes = amplitudes[alive]
            mus = mus[alive]
            sigmas = sigmas[alive]
            k_effective = int(alive.sum())
            params_alive = []
            for a, m, s in zip(amplitudes, mus, sigmas):
                params_alive.extend([a, m, s])
            params_alive.append(bl)
            fitted_fit = gaussian_model(fit_centers, *params_alive)
            rss = float(np.sum((fitted_fit - fit_counts) ** 2))
            bic_val = _bic(n_fit, 3 * k_effective + 1, rss)
            k = k_effective

        if k == 0:
            continue

        fitted_all = gaussian_model(centers, *res.x)

        candidate: SideFit = {
            "k": k, "params": res.x, "bic": bic_val, "rss": rss,
            "amplitudes": amplitudes, "mus": mus, "sigmas": sigmas,
            "baseline": bl, "fitted": fitted_all,
        }

        if best_result is None or bic_val < best_result["bic"]:
            best_result = candidate

    if best_result is None:
        return {"k": 0, "params": np.array([baseline]),
                "bic": np.inf, "rss": 0.0,
                "amplitudes": np.array([]), "mus": np.array([]),
                "sigmas": np.array([]), "baseline": baseline,
                "fitted": np.full(n, baseline)}

    return best_result


# ======================================================================
#  Fitting the full summed CCG (both sides)
# ======================================================================


def fit_channel(ccg_fine: np.ndarray, fine_centers: np.ndarray,
                macro_bin_ms: float) -> ChannelFit:
    """Rebin, smooth, fit each side independently, return results.

    Parameters
    ----------
    ccg_fine : np.ndarray
        Fine-resolution summed CCG, shape (B_f,).
    fine_centers : np.ndarray
        Fine bin centres in ms, shape (B_f,).
    macro_bin_ms : float
        Working bin width in ms.

    Returns
    -------
    ChannelFit keys:
        pos / neg       SideFit  one per lag side
        macro_centers   np.ndarray  working bin centres
        macro_counts    np.ndarray  rebinned counts
        macro_smooth    np.ndarray  smoothed counts
    """
    lo = float(fine_centers.min()) - macro_bin_ms
    hi = float(fine_centers.max()) + macro_bin_ms
    bin_edges = np.arange(lo, hi + macro_bin_ms, macro_bin_ms)
    macro_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    macro_full, _ = np.histogram(
        fine_centers, bins=bin_edges,
        weights=ccg_fine.astype(float),
    )
    sigma_bins = SIGMA_SMOOTH_MS / macro_bin_ms
    macro_smooth = gaussian_filter1d(macro_full.astype(float), sigma=sigma_bins)

    pos_mask = macro_centers > 0
    neg_mask = macro_centers < 0
    pos_dz = macro_centers[pos_mask] < DEAD_ZONE_MS
    neg_dz = macro_centers[neg_mask] > -DEAD_ZONE_MS

    pos_fit = fit_side(macro_smooth[pos_mask], macro_centers[pos_mask],
                       macro_bin_ms, dead_zone_mask=pos_dz)
    neg_fit = fit_side(macro_smooth[neg_mask], macro_centers[neg_mask],
                       macro_bin_ms, dead_zone_mask=neg_dz)

    return {"pos": pos_fit, "neg": neg_fit,
            "macro_centers": macro_centers,
            "macro_counts": macro_full, "macro_smooth": macro_smooth}


# ======================================================================
#  Visualisation
# ======================================================================


def _build_figure(channel: int, n_spikes: int,
                  fit_data: ChannelFit) -> Figure:
    """Build the single-panel matplotlib figure for a channel (no output side-effects).

    Renders the summed CCG at ±ZOOM_XRANGE_MS with confidence-interval
    lines (composite fits and individual components) overlaid on the raw
    histogram.  Returns an untouched figure; the caller saves it as PNG.

    Returns
    -------
    matplotlib.figure.Figure
        Untouched figure; the caller decides whether to save it as PNG or
        encode it as SVG.  Single panel at ±ZOOM_XRANGE_MS.  The raw
        histogram is plotted as-is (no dead-zone exclusion).
    """
    from matplotlib.figure import Figure

    macro_centers = fit_data["macro_centers"]
    macro_smooth = fit_data["macro_smooth"]
    macro_counts = fit_data["macro_counts"]
    pos_fit = fit_data["pos"]
    neg_fit = fit_data["neg"]

    half_range = ZOOM_XRANGE_MS
    mask = np.abs(macro_centers) <= half_range
    xc = macro_centers[mask]
    yc_raw = macro_counts[mask].astype(float)
    yc_sm = macro_smooth[mask]
    y_top = max(float(yc_sm.max()), float(yc_raw.max()), 1.0)

    fig = Figure(figsize=(6.5, 4.2), dpi=100)
    fig.suptitle(
        f"Channel {channel} — {n_spikes} spikes  |  "
        f"SIGMA×{SIGMA_MULTIPLIER:.1f}  |  +{pos_fit['k']} Gaussians, "
        f"-{neg_fit['k']} Gaussians",
        fontsize=11, fontweight="bold")

    ax = fig.add_subplot(111)
    ax.bar(xc, yc_raw, width=REBIN_MS, color="#c8d6e5",
           edgecolor="#aab7c4", linewidth=0.3, label="raw counts")
    ax.plot(xc, yc_sm, color="#888", linewidth=1.0, zorder=3,
            label="smoothed")

    for side_data, color, label in [
            (pos_fit, "#e74c3c", "positive lag (composite)"),
            (neg_fit, "#2980b9", "negative lag (composite)")]:
        if side_data["k"] == 0:
            continue
        composite = np.maximum(gaussian_model(xc, *side_data["params"]), 0.0)
        ax.plot(xc, composite, color=color, linewidth=2.2, zorder=4,
                label=label)
        for i in range(side_data["k"]):
            amp = side_data["amplitudes"][i]
            mu = side_data["mus"][i]
            sigma = side_data["sigmas"][i]
            baseline = side_data["baseline"]
            comp = amp * np.exp(-0.5 * ((xc - mu) / sigma) ** 2) + baseline
            ax.plot(xc, comp, color=color, linewidth=1.1, linestyle="--",
                    alpha=0.6, zorder=4)
            for offset in (-SIGMA_MULTIPLIER, +SIGMA_MULTIPLIER):
                xl = mu + offset * sigma
                if abs(xl) > half_range:
                    continue
                ax.axvline(xl, color=color, linewidth=0.8,
                           linestyle=":", alpha=0.55)
            ax.annotate(
                f"μ={mu:.2f} σ={sigma:.2f}", (mu, amp + baseline),
                textcoords="offset points", xytext=(0, 5),
                fontsize=8, color=color, ha="center")
    ax.axvline(0, color="#636e72", linewidth=0.7, linestyle="--", alpha=0.6,
               label="zero lag")
    ax.set_title(f"±{half_range:.0f} ms", fontsize=10)
    ax.set_xlim(-half_range, half_range)
    ax.set_ylim(0, y_top * 1.15)
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("count")
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout(pad=0.8)
    return fig


def _plot_channel(channel: int, n_spikes: int,
                  fit_data: ChannelFit,
                  output_path: Path) -> None:
    """Render the channel figure to a PNG overlay file.

    Parameters
    ----------
    channel : int
        Channel index.
    n_spikes : int
        Number of spikes on the channel.
    fit_data : ChannelFit
        Fit result from fit_channel.
    output_path : Path
        Destination .png path.
    """
    fig = _build_figure(channel, n_spikes, fit_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=100)
    plt.close(fig)


# ======================================================================
#  HTML gallery
# ======================================================================


def gen_index_html(output_path: Path, channel_info: List[Tuple[int, int]],
                   out_dir: Path) -> None:
    """Write index.html gallery linking per-channel PNG overlays.

    Parameters
    ----------
    output_path : Path
        Path to index.html.
    channel_info : List[Tuple[int, int]]
        List of (channel, n_spikes) for all active channels.
    out_dir : Path
        Directory holding the chNN.png files.
    """
    cards = ""
    for ch, n in channel_info:
        cards += (
            f'<div class="card">'
            f'<a href="ch{ch:02d}.png">'
            f'<img src="ch{ch:02d}.png" alt="Ch {ch}"></a>'
            f'<div class="caption">Ch {ch} ({n} spikes)</div>'
            f'</div>\n'
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Avalanche extraction — Gaussian fits</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .meta {{ font-size: 13px; color: #666; margin-bottom: 16px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
                 gap: 16px; }}
        .card {{ background: white; border-radius: 6px; overflow: hidden;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .card img {{ width: 100%; display: block; }}
        .caption {{ padding: 6px 10px; font-size: 12px; color: #444; text-align: center; }}
    </style>
</head>
<body>
    <h1>Avalanche extraction — Gaussian fit overlays</h1>
    <p class="meta">SIGMA_MULTIPLIER = {SIGMA_MULTIPLIER:.1f}  |
    REBIN = {REBIN_MS} ms  |  smooth \u03c3 = {SIGMA_SMOOTH_MS} ms  |  '
    'window \u00b1{ZOOM_XRANGE_MS:.0f} ms</p>
    <p class="meta">Red = positive-lag Gaussians (target fires after source).
    Blue = negative-lag Gaussians (source fires after target).
    Dotted lines = \u00b1{SIGMA_MULTIPLIER:.1f} \u03c3 extraction boundaries.</p>
    <div class="grid">
{cards}
    </div>
</body>
</html>"""

    output_path.write_text(html)
    print(f"Saved gallery to: {output_path}")


# ======================================================================
#  Main
# ======================================================================


def main(args: argparse.Namespace) -> None:
    """Entry point: load CCGs, fit Gaussians, render overlays."""
    ccg_npz_path = Path(args.npz) if args.npz else Path()
    if not ccg_npz_path.is_file():
        output_root = Path(args.out_root)
        if not output_root.is_dir():
            output_root = Path(__file__).resolve().parent / args.out_root
        if output_root.is_dir():
            runs = sorted([d for d in output_root.iterdir()
                           if d.is_dir()], reverse=True)
            for run in runs:
                candidate = run / "cross_corr" / "cross_correlograms.npz"
                if candidate.exists():
                    ccg_npz_path = candidate
                    break
        if not ccg_npz_path.is_file():
            print("Error: no cross_correlograms.npz found. "
                  "Run cross_correlation.py first.")
            sys.exit(1)

    # apply CLI overrides to module-level constants
    import avalanche_extraction as _self
    if args.sigma is not None:
        _self.SIGMA_MULTIPLIER = args.sigma
    if args.max_k is not None:
        _self.MAX_K = args.max_k
    if args.rebin is not None:
        _self.REBIN_MS = args.rebin

    print("=" * 60)
    print("Avalanche Extraction — Gaussian fitting")
    print("=" * 60)
    print(f"Input CCG:     {ccg_npz_path}")
    print(f"SIGMA_MULT:    {SIGMA_MULTIPLIER}")
    print(f"MAX_K:         {MAX_K}")
    print("=" * 60)

    t0 = time.time()

    archive = np.load(ccg_npz_path, allow_pickle=True)
    correlograms = archive["correlograms"]
    bin_edges = archive["bin_edges"]
    n_spikes = archive["n_spikes"]

    # build fine region
    centers_full = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    fine_mask = np.abs(centers_full) <= FINE_CUTOFF_MS
    fine_idx = np.flatnonzero(fine_mask)
    fine_start = int(fine_idx.min())
    fine_stop = int(fine_idx.max()) + 1
    fine_centers = centers_full[fine_start:fine_stop]
    print(f"Fine regime: {len(fine_centers)} bins, "
          f"±{FINE_CUTOFF_MS:.0f} ms")

    # macro bin centres
    half = FINE_CUTOFF_MS
    macro_edges = np.arange(-half - REBIN_MS, half + 2 * REBIN_MS, REBIN_MS)
    macro_centers = (macro_edges[:-1] + macro_edges[1:]) / 2.0

    out_dir = ccg_npz_path.parent / "avalanche"
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_info: List[Tuple[int, int]] = []

    for ch in range(NUM_CHANNELS):
        ns = int(n_spikes[ch])
        if ns == 0:
            continue

        summed_fine = correlograms[ch].sum(axis=0)[fine_start:fine_stop]
        fit_data = fit_channel(summed_fine, fine_centers, REBIN_MS)

        png_path = out_dir / f"ch{ch:02d}.png"
        _plot_channel(ch, ns, fit_data, png_path)
        channel_info.append((ch, ns))

        elapsed = time.time() - t0
        pos_k = fit_data["pos"]["k"]
        neg_k = fit_data["neg"]["k"]
        print(f"  ch{ch:2d}: {ns:4d} spikes | "
              f"+{pos_k} Gaussians, -{neg_k} Gaussians | {elapsed:.1f}s")

    gen_index_html(out_dir / "index.html", channel_info, out_dir)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Output: {out_dir}")
    print(f"Channels fitted: {len(channel_info)}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 60)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Gaussian-mixture fitting of CCG latency humps")
    p.add_argument("--npz", default=None,
                   help="Path to cross_correlograms.npz")
    p.add_argument("--out-root", default="outputs",
                   help="Output root directory (default: outputs/)")
    p.add_argument("--sigma", type=float,
                   help=f"Std-dev multiplier for extraction boundaries "
                        f"(default: {SIGMA_MULTIPLIER})")
    p.add_argument("--max-k", type=int,
                   help=f"Maximum number of Gaussian components "
                        f"(default: {MAX_K})")
    p.add_argument("--rebin", type=float,
                   help=f"Working bin width in ms (default: {REBIN_MS})")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
