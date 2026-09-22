"""
ccg_peaks.py  (local-maximum analysis of cross-correlogram histograms)

Run from the project root:
    python ccg_peaks.py                                # latest run
    python ccg_peaks.py --npz path/to/cross_correlograms.npz
    python ccg_peaks.py --surrogates 300               # more surrogate shuffles

Stage that consumes the directed pairwise cross-correlograms written by
cross_correlation.py and turns each histogram into a structured list of
significant local maxima (peaks).  A "peak" is a lag at which the spike
count rises above the surrounding baseline; its position is an average
latency between two channels, its prominence is a coupling-strength
measure, and its empirical p-value (from shuffled-train surrogates)
says whether it would survive random re-pairing of the same spike times.

This stage answers three questions that the raw histograms only hint at:

  1. How many distinct latencies are there, and where?
         A channel whose summed CCG has peaks at +1.4, +1.9 and +2.5 ms
         has THREE average latencies to its 63 targets, not one.  The
         list of significant peak lags is the channel's latency spectrum.

  2. Who leads and who follows?
         A directed CCG C_{i->j}[b] counts target spikes at +lag and
         source-following events at -lag.  If channel i tends to fire
         ~2 ms BEFORE the array, its summed CCG has a big +2 ms hump and
         almost no -2 ms hump.  The lead index turns this asymmetry into
         a bounded number in [-1, +1].

  3. Is a peak real, or a coincidence of nearby fires?
         Circularly shifting every channel's spike train (preserving its
         own inter-spike-interval structure) destroys cross-channel
         timing.  Peak prominences extracted from these shifted trains
         form the null distribution; each observed peak is scored against
         it and against a time-split stability check.

Notation:
    K       = 64 channels on the MEA grid (8 rows x 8 columns)
    C_{i->j} = directed cross-correlogram from channel i to channel j,
               a non-uniformly binned histogram over lag b in ms
    S_i     = summed CCG of channel i to all 63 other channels
    τ       = a lag in ms
    p_w     = peak prominence (height above the lowest valley achieved
              before a higher neighboring peak), on the smoothed CCG
    Δ_f     = fine bin width in ms (0.01) within |τ| ≤ FINE_CUTOFF_MS

  1. Restrict to the fine regime:
         All peak analysis runs on |τ| ≤ PEAK_REGION_MS where bins are
         uniform 0.01 ms wide.  Beyond PEAK_REGION_MS the 1 ms coarse
         bins make peak geometry (width, prominence) non-comparable.

  2. Rebin, smooth and detect:
         Per-channel spike counts are low, so a 0.01 ms bin often holds
         0-3 counts and the perceptible "humps" span ~0.5 ms.  The fine
         counts are therefore aggregated into REBIN_MS-wide working bins
         (0.1 ms, ~3x the 30 kHz sampling period) before detection.
         Gaussian smoothing with scale PEAK_SMOOTH_SIGMA_MS then removes
         single-bin noise without erasing the dips between close peaks.
         scipy.signal.find_peaks locates local maxima separately on the
         +τ and -τ half-axes, keeping only candidates whose prominence
         exceeds a floor based on the flat baseline (median of the
         20-50 ms zone) and which are separated by at least
         MIN_PEAK_SEPARATION_MS.

  3. Metrics:
         lead index   = (P - N) / (P + N),   P, N = counts in
                        |τ| ∈ [LEAD_WINDOW_MIN_MS, LEAD_WINDOW_MAX_MS]
         dominant lag = signed lag of the largest-prominence peak
         latency spectrum = sorted significant peak lags, per side

  4. Significance:
         split_stability = fraction of full-set peaks that recur in at
                           least one random time-half of the recording
         empirical p    = fraction of surrogate peaks whose prominence
                          reaches the observed value; surrogates are
                          circular shifts of each train inside the
                          recording duration

Output (in <run_dir>/cross_corr/):
    ccg_peaks.npz      peak lists, metrics, significance scores
    quant/index.html   hub linking the views below
    quant/delay_map.html         8x8 grid of lead index and dominant lag
    quant/matrices.html          64x64 heatmaps of lag and prominence
    quant/adjacency/<ch>_peaks.html  annotated histograms with peak labels

ccg_peaks.npz schema (np.savez_compressed):
    source_npz           str    input waveforms.npz (spike times)
    source_ccg_npz       str    input cross_correlograms.npz
    params               str    dict of peak analysis parameters
    fine_edges           float64 (B_f+1,) fine-region bin edges in ms
    fine_centers         float64 (B_f,)   fine-region bin centers in ms
    fine_bin_ms          float64 width of fine bins in ms
    n_spikes             int64   (K,)     spike count per channel
    pair_peak_lags_pos   object  (K, K)   significant +lag peaks, per pair
    pair_peak_prom_pos   object  (K, K)   prominence of each such peak
    pair_peak_lags_neg   object  (K, K)   significant -lag peaks, per pair
    pair_peak_prom_neg   object  (K, K)   prominence of each such peak
    pair_dominant_lag    float64 (K, K)   signed lag of largest peak, per pair
    pair_max_prominence  float64 (K, K)   largest peak prominence, per pair
    pair_lead_index      float64 (K, K)   count-based lead asymmetry, per pair
    pair_lead_pos_count  float64 (K, K)   +lag fine counts in the lead window
    pair_lead_neg_count  float64 (K, K)   -lag fine counts in the lead window
    sum_peak_lags_pos    object  (K,)     significant +lag peaks of S_i
    sum_peak_prom_pos    object  (K,)     prominence of each such peak
    sum_peak_pval_pos    object  (K,)     empirical p-value of each such peak
    sum_peak_lags_neg    object  (K,)     significant -lag peaks of S_i
    sum_peak_prom_neg    object  (K,)     prominence of each such peak
    sum_peak_pval_neg    object  (K,)     empirical p-value of each such peak
    lead_index           float64 (K,)     asymmetry measure in [-1, +1]
    lead_pos_count       float64 (K,)     counts in + lead window
    lead_neg_count       float64 (K,)     counts in - lead window
    dominant_lag         float64 (K,)     signed lag of largest S_i peak
    split_stability      float64 (K,)     fraction of S_i peaks seen in a half
    null_prominence      float64 (N_s,)   pooled surrogate peak prominences
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cross_correlation import (
    GRID_COLS,
    GRID_ROWS,
    NUM_CHANNELS,
    ZOOM_LEVELS,
    _figure_to_svg,
    _get_adjacent_channels,
    _get_direction_label,
)



# ---------------- analysis constants ----------------
PEAK_REGION_MS: float = 50.0              # analysis window equals the fine-bin regime (ms)
REBIN_MS: float = 0.1                     # working resolution for peak detection (ms)
PEAK_SMOOTH_SIGMA_MS: float = 0.15        # gaussian width for peak detection (ms)
MIN_PROMINENCE_COUNT: float = 3.0         # absolute floor for peak prominence (counts)
MIN_PROMINENCE_FRACTION: float = 0.05     # prominence floor as fraction of flat-baseline count
MIN_PEAK_SEPARATION_MS: float = 0.3       # minimum distance between distinct peaks (ms)
PEAK_ZONE_MIN_MS: float = 0.5             # ignore candidates closer to zero than this (ms)
BASELINE_ZONE_MS: float = 20.0            # flat background used to set the prominence floor (ms)
LEAD_WINDOW_MIN_MS: float = 1.0           # inner edge of the lead-index window (ms)
LEAD_WINDOW_MAX_MS: float = 4.0           # outer edge of the lead-index window (ms)
PEAK_MATCH_TOLERANCE_MS: float = 0.4      # how close two lags must be to count as the same (ms)
N_SURROGATES: int = 200                   # shuffled-train iterations for empirical p-values

# ---------------- visualization constants ----------------
QUANT_FIG_DPI: int = 100                  # resolution for matrix figures
QUANT_FIG_WIDTH: float = 7.0              # matrix figure width (inches)
QUANT_FIG_HEIGHT: float = 6.0             # matrix figure height (inches)
ZOOM_FIG_WIDTH: float = 10.0              # annotated zoom figure width (inches)
ZOOM_FIG_HEIGHT: float = 3.0              # annotated zoom figure height (inches)



def get_fine_region(bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Extract the uniform fine-bin region from a non-uniform lag axis.

    Given the full symmetric bin-edge array produced by cross_correlation
    (0.01 ms bins within ±PEAK_REGION_MS, 1.0 ms bins beyond), return the
    edges, centers and width of the contiguous fine sub-array plus the
    start index used to slice a full CCG into its fine portion.

    Parameters
    ----------
    bin_edges : np.ndarray
        Full non-uniform bin edges in ms, shape (B+1,).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float, int]
        (fine_edges, fine_centers, fine_bin_ms, start) where fine_edges
        has shape (B_f+1,), fine_centers has shape (B_f,), fine_bin_ms
        is the width of every fine bin, and start is the index of the
        first fine bin in the original full-length arrays.
    """
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    fine_mask = np.abs(centers) <= PEAK_REGION_MS
    bin_indices = np.flatnonzero(fine_mask)
    start = int(bin_indices.min())
    stop = int(bin_indices.max()) + 1
    fine_edges = bin_edges[start:stop + 1]
    fine_centers = centers[start:stop]
    widths = np.diff(fine_edges)
    widths = widths[widths > 1e-9]
    return fine_edges, fine_centers, float(widths[0]), start


def smooth_ccg(counts: np.ndarray, bin_width_ms: float,
               sigma_ms: float = PEAK_SMOOTH_SIGMA_MS) -> np.ndarray:
    """Apply a one-dimensional gaussian filter to a CCG count array.

    Fine bins are 0.01 ms wide, below the 30 kHz sampling period of
    0.033 ms, so raw counts form a comb where neighbouring bins cannot
    both fill.  Smoothing with a gaussian of scale sigma_ms restores the
    underlying envelope that peak detection should operate on.

    Parameters
    ----------
    counts : np.ndarray
        CCG spike counts, shape (B,).
    bin_width_ms : float
        Width of each bin in ms (must be uniform).
    sigma_ms : float
        Gaussian standard deviation in ms.

    Returns
    -------
    np.ndarray
        Smoothed copy of counts, same shape and dtype float64.
    """
    sigma_bins = sigma_ms / bin_width_ms
    return gaussian_filter1d(counts.astype(np.float64), sigma=sigma_bins)


def detect_peak_sides(smoothed: np.ndarray,
                      centers: np.ndarray,
                      bin_width_ms: float) -> Dict[str, np.ndarray]:
    """Find significant local maxima on the +lag and -lag half-axes.

    Runs scipy.signal.find_peaks on each half of the smoothed fine CCG
    separately, so the two sides can carry different prominence floors
    (their baselines need not match).  Each side's floor is

        max(MIN_PROMINENCE_COUNT,
            MIN_PROMINENCE_FRACTION * baseline)

    where baseline is the median of the smoothed counts over the flat
    |τ| ∈ [BASELINE_ZONE_MS, PEAK_REGION_MS] background.  Candidates
    closer to zero than PEAK_ZONE_MIN_MS are discarded (they belong to
    the central coincidence blob), and peaks closer together than
    MIN_PEAK_SEPARATION_MS are suppressed via find_peaks' distance rule.

    Parameters
    ----------
    smoothed : np.ndarray
        Gaussian-smoothed fine CCG counts, shape (B_f,).
    centers : np.ndarray
        Fine bin centers in ms, shape (B_f,).
    bin_width_ms : float
        Width of each fine bin in ms.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys "pos_lag", "pos_prom", "neg_lag", "neg_prom"; each array is
        sorted by descending prominence.
    """
    results: Dict[str, np.ndarray] = {
        "pos_lag": np.empty(0),
        "pos_prom": np.empty(0),
        "neg_lag": np.empty(0),
        "neg_prom": np.empty(0),
    }

    if len(centers) == 0:
        return results

    min_separation_bins = int(round(MIN_PEAK_SEPARATION_MS / bin_width_ms))
    baseline_mask = np.abs(centers) >= BASELINE_ZONE_MS

    for side_sign, side_key in [("pos", "pos"), ("neg", "neg")]:
        if side_sign == "pos":
            side_mask = centers > 0.0
        else:
            side_mask = centers < 0.0
        if not np.any(side_mask):
            continue

        zone_counts = smoothed[side_mask & baseline_mask]
        if len(zone_counts) == 0:
            baseline = 1.0
        else:
            baseline = float(np.median(zone_counts))
        min_prominence = max(MIN_PROMINENCE_COUNT, MIN_PROMINENCE_FRACTION * baseline)

        side_smoothed = smoothed[side_mask]
        side_centers = centers[side_mask]
        peaks, peak_props = find_peaks(
            side_smoothed,
            prominence=min_prominence,
            distance=min_separation_bins,
        )

        peak_lags = side_centers[peaks]
        peak_proms = peak_props["prominences"]
        away_from_zero = np.abs(peak_lags) >= PEAK_ZONE_MIN_MS
        peak_lags = peak_lags[away_from_zero]
        peak_proms = peak_proms[away_from_zero]

        if len(peak_lags) == 0:
            continue

        order = np.argsort(peak_proms)[::-1]
        results[f"{side_key}_lag"] = peak_lags[order]
        results[f"{side_key}_prom"] = peak_proms[order]

    return results


def rebin_fine(counts_fine: np.ndarray,
               fine_centers: np.ndarray,
               rebin_ms: float = REBIN_MS) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate fine 0.01 ms counts into REBIN_MS-wide working bins.

    Per-channel firing is sparse enough that a single 0.01 ms bin holds
    0-3 counts, far below the width of the real latency humps (~0.5 ms).
    Summing every fine bin into a REBIN_MS bin collects the hump's mass
    into countable units.  The working axis is symmetric about 0 with
    bin centers at ±REBIN_MS/2, ±3REBIN_MS/2, ... .

    Parameters
    ----------
    counts_fine : np.ndarray
        Fine-region CCG counts, shape (B_f,).
    fine_centers : np.ndarray
        Fine bin centers in ms, shape (B_f,).
    rebin_ms : float
        Working bin width in ms.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (macro_counts, macro_centers) with macro_centers in ms.
    """
    half_width = PEAK_REGION_MS
    macro_edges = np.round(np.arange(-half_width, half_width + rebin_ms, rebin_ms), 9)
    macro_centers = (macro_edges[:-1] + macro_edges[1:]) / 2.0
    bin_indices = np.searchsorted(macro_edges, fine_centers, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, len(macro_centers) - 1)
    macro_counts = np.bincount(bin_indices, weights=counts_fine,
                               minlength=len(macro_centers)).astype(np.int64)
    return macro_counts, macro_centers


def extract_peaks(counts: np.ndarray,
                  centers: np.ndarray,
                  bin_width_ms: float) -> Dict[str, np.ndarray]:
    """Extract significant peak lags and prominences from a CCG.

    Convenience wrapper around smooth_ccg + detect_peak_sides that
    operates on working-resolution (REBIN_MS) count arrays.

    Parameters
    ----------
    counts : np.ndarray
        Working-resolution CCG counts, shape (B_w,).
    centers : np.ndarray
        Working bin centers in ms, shape (B_w,).
    bin_width_ms : float
        Width of each working bin in ms (REBIN_MS).

    Returns
    -------
    Dict[str, np.ndarray]
        Peak lags and prominences per side (see detect_peak_sides).
    """
    smoothed = smooth_ccg(counts, bin_width_ms)
    return detect_peak_sides(smoothed, centers, bin_width_ms)


def compute_pair_peaks(correlograms: np.ndarray,
                       fine_centers: np.ndarray,
                       start: int,
                       macro_centers: np.ndarray,
                       macro_bin_ms: float,
                       total_channels: int = NUM_CHANNELS) -> Dict[str, np.ndarray]:
    """Extract peaks from every directed pair CCG.

    Applies peak detection to all K x K directed cross-correlograms,
    producing one peak list per ordered pair.  The diagonal (i == j) is
    left empty.  Also reduces each pair to its dominant (largest-
    prominence) signed lag and its largest prominence.

    Parameters
    ----------
    correlograms : np.ndarray
        Directed CCGs, int64 of shape (K, K, B).
    fine_centers : np.ndarray
        Fine bin centers in ms, shape (B_f,).
    start : int
        Index of the first fine bin in the full-length CCG arrays.
    macro_centers : np.ndarray
        Working-resolution bin centers in ms, shape (B_w,).
    macro_bin_ms : float
        Width of each working bin in ms.
    total_channels : int
        Number of channels on the MEA.

    Returns
    -------
    Dict[str, np.ndarray]
        "lags_pos"/"prom_pos"/"lags_neg"/"prom_neg" object arrays of
        shape (K, K,), each element a 1-D float64 array of peak lags or
        prominences; "dominant_lag" float64 (K, K); "max_prominence"
        float64 (K, K); "lead_index" float64 (K, K); "lead_pos_count"
        and "lead_neg_count" float64 (K, K) with the fine counts in the
        [LEAD_WINDOW_MIN_MS, LEAD_WINDOW_MAX_MS] band (see lead_index).
    """
    stop = start + len(fine_centers)
    lags_pos = np.empty((total_channels, total_channels), dtype=object)
    prom_pos = np.empty((total_channels, total_channels), dtype=object)
    lags_neg = np.empty((total_channels, total_channels), dtype=object)
    prom_neg = np.empty((total_channels, total_channels), dtype=object)
    dominant_lag = np.zeros((total_channels, total_channels))
    max_prominence = np.zeros((total_channels, total_channels))
    pair_lead_index = np.zeros((total_channels, total_channels))
    pair_lead_pos = np.zeros((total_channels, total_channels))
    pair_lead_neg = np.zeros((total_channels, total_channels))

    for source_channel in range(total_channels):
        for target_channel in range(total_channels):
            if source_channel == target_channel:
                lags_pos[source_channel, target_channel] = np.empty(0)
                prom_pos[source_channel, target_channel] = np.empty(0)
                lags_neg[source_channel, target_channel] = np.empty(0)
                prom_neg[source_channel, target_channel] = np.empty(0)
                continue

            pair_counts, _ = rebin_fine(
                correlograms[source_channel, target_channel, start:stop], fine_centers,
            )
            peaks = extract_peaks(pair_counts, macro_centers, macro_bin_ms)
            lags_pos[source_channel, target_channel] = peaks["pos_lag"]
            prom_pos[source_channel, target_channel] = peaks["pos_prom"]
            lags_neg[source_channel, target_channel] = peaks["neg_lag"]
            prom_neg[source_channel, target_channel] = peaks["neg_prom"]

            best_lag, best_prominence = dominant_peak(
                peaks["pos_lag"], peaks["pos_prom"], peaks["neg_lag"], peaks["neg_prom"],
            )
            dominant_lag[source_channel, target_channel] = best_lag
            if best_prominence is not None:
                max_prominence[source_channel, target_channel] = best_prominence

            pair_lead, pair_pos, pair_neg = lead_index(
                correlograms[source_channel, target_channel, start:stop], fine_centers,
            )
            pair_lead_index[source_channel, target_channel] = pair_lead
            pair_lead_pos[source_channel, target_channel] = pair_pos
            pair_lead_neg[source_channel, target_channel] = pair_neg

    return {
        "lags_pos": lags_pos,
        "prom_pos": prom_pos,
        "lags_neg": lags_neg,
        "prom_neg": prom_neg,
        "dominant_lag": dominant_lag,
        "max_prominence": max_prominence,
        "lead_index": pair_lead_index,
        "lead_pos_count": pair_lead_pos,
        "lead_neg_count": pair_lead_neg,
    }


def dominant_peak(lags_pos: np.ndarray, prom_pos: np.ndarray,
                  lags_neg: np.ndarray, prom_neg: np.ndarray) -> Tuple[float, Optional[float]]:
    """Return the signed lag of the largest-prominence peak.

    Combines both sides and selects the peak with the greatest
    prominence.  Returns (0.0, None) when no peaks exist.

    Parameters
    ----------
    lags_pos : np.ndarray
        Positive-side peak lags in ms.
    prom_pos : np.ndarray
        Prominence of positive-side peaks.
    lags_neg : np.ndarray
        Negative-side peak lags in ms.
    prom_neg : np.ndarray
        Prominence of negative-side peaks.

    Returns
    -------
    Tuple[float, Optional[float]]
        (dominant_lag_ms, dominant_prominence); prominence is None when
        there are no peaks.
    """
    all_lags = np.concatenate([lags_pos, lags_neg])
    all_proms = np.concatenate([prom_pos, prom_neg])
    if len(all_lags) == 0:
        return 0.0, None
    best_index = int(np.argmax(all_proms))
    return float(all_lags[best_index]), float(all_proms[best_index])


def lead_index(summed_counts: np.ndarray,
               fine_centers: np.ndarray) -> Tuple[float, float, float]:
    """Compute the signed lead asymmetry of a summed CCG.

    Counts in the +lag window [LEAD_WINDOW_MIN_MS, LEAD_WINDOW_MAX_MS]
    measure how often the array follows this channel; counts in the
    mirror -lag window measure how often it leads it.  The index

        L = (P - N) / (P + N)

    is +1 for a pure initiator, -1 for a pure follower, and 0 for a
    channel with no consistent ordering.  When P + N == 0 the index is
    undefined and returned as 0.

    Parameters
    ----------
    summed_counts : np.ndarray
        Summed fine CCG of a channel to all others, shape (B_f,).
    fine_centers : np.ndarray
        Fine bin centers in ms, shape (B_f,).

    Returns
    -------
    Tuple[float, float, float]
        (lead_index, pos_count, neg_count).
    """
    pos_mask = (fine_centers >= LEAD_WINDOW_MIN_MS) & (fine_centers <= LEAD_WINDOW_MAX_MS)
    neg_mask = (fine_centers >= -LEAD_WINDOW_MAX_MS) & (fine_centers <= -LEAD_WINDOW_MIN_MS)
    pos_count = float(summed_counts[pos_mask].sum())
    neg_count = float(summed_counts[neg_mask].sum())
    total = pos_count + neg_count
    if total == 0.0:
        return 0.0, pos_count, neg_count
    return (pos_count - neg_count) / total, pos_count, neg_count


def load_inputs(ccg_npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load the cross-correlogram archive and its source spike trains.

    Parameters
    ----------
    ccg_npz_path : str
        Path to a cross_correlograms.npz archive.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]
        (correlograms, bin_edges, n_spikes, spike_times, channels,
        waveforms_npz) where spike_times is the object array of spike
        times in seconds and waveforms_npz the path to the source file.
    """
    archive = np.load(ccg_npz_path, allow_pickle=True)
    correlograms = archive["correlograms"]
    bin_edges = archive["bin_edges"]
    n_spikes = archive["n_spikes"]
    source_npz = str(archive["source_npz"])
    waveforms = np.load(source_npz, allow_pickle=True)
    return (
        correlograms,
        bin_edges,
        n_spikes,
        waveforms["spike_times"],
        waveforms["channels"],
        source_npz,
    )


def ccg_fine_window(source_ms: np.ndarray,
                    target_ms: np.ndarray,
                    fine_edges: np.ndarray) -> np.ndarray:
    """Compute a fine-region CCG by windowed binning.

    For every source spike, finds the target spikes within
    ±PEAK_REGION_MS via binary search and bins only those lags.  This is
    far cheaper than all-to-all binning (each source spike touches a
    handful of nearby targets at these firing rates) and identical to the
    fine portion of cross_correlation.cross_correlogram.

    Parameters
    ----------
    source_ms : np.ndarray
        Source spike times in ms (any order).
    target_ms : np.ndarray
        Sorted target spike times in ms.
    fine_edges : np.ndarray
        Fine-region bin edges in ms, shape (B_f+1,).

    Returns
    -------
    np.ndarray
        Integer counts, shape (B_f,).
    """
    counts = np.zeros(len(fine_edges) - 1, dtype=int)
    if len(source_ms) == 0 or len(target_ms) == 0:
        return counts

    lower = np.searchsorted(target_ms, source_ms - PEAK_REGION_MS, side="left")
    upper = np.searchsorted(target_ms, source_ms + PEAK_REGION_MS, side="right")

    for source_spike_ms, low_idx, high_idx in zip(source_ms, lower, upper):
        if low_idx == high_idx:
            continue
        lags = target_ms[low_idx:high_idx] - source_spike_ms
        bin_indices = np.searchsorted(fine_edges, lags, side="right") - 1
        np.clip(bin_indices, 0, len(counts) - 1, out=bin_indices)
        counts += np.bincount(bin_indices, minlength=len(counts))

    return counts


def summed_ccg_from_trains(spike_times_ms: np.ndarray,
                           source_channel: int,
                           fine_edges: np.ndarray,
                           total_channels: int = NUM_CHANNELS) -> np.ndarray:
    """Compute the summed fine CCG of one channel against all the rest.

    Uses the identity that the sum of directed CCGs from channel i to
    every other channel equals the CCG of channel i against the union of
    all other channels' spike trains:

        S_i = Σ_{j ≠ i} C_{i->j} = CCG(i, ∪_{j ≠ i} s_j)

    This makes one windowed CCG call per surrogate/half instead of 63.

    Parameters
    ----------
    spike_times_ms : np.ndarray
        Object array of length K, each element a 1-D float64 ndarray of
        spike times in ms.
    source_channel : int
        Index of the source channel.
    fine_edges : np.ndarray
        Fine-region bin edges in ms, shape (B_f+1,).
    total_channels : int
        Number of channels on the MEA.

    Returns
    -------
    np.ndarray
        Integer counts, shape (B_f,).
    """
    target_parts = [train for channel, train in enumerate(spike_times_ms)
                    if channel != source_channel and len(train) > 0]
    if len(target_parts) == 0:
        return np.zeros(len(fine_edges) - 1, dtype=int)
    target_union = np.sort(np.concatenate(target_parts))
    return ccg_fine_window(spike_times_ms[source_channel], target_union, fine_edges)


def summed_peaks_from_trains(spike_times_ms: np.ndarray,
                             fine_edges: np.ndarray,
                             fine_centers: np.ndarray,
                             macro_centers: np.ndarray,
                             macro_bin_ms: float,
                             total_channels: int = NUM_CHANNELS) -> List[Dict[str, np.ndarray]]:
    """Extract summed-CCG peaks from spike trains directly.

    Recomputes each channel's summed fine CCG from the trains (rather
    than a stored npz), rebins to working resolution, and returns the
    detected peaks per channel.  Used by the split-half and surrogate
    routines which need CCGs of modified trains.

    Parameters
    ----------
    spike_times_ms : np.ndarray
        Object array of length K with per-channel spike times in ms.
    fine_edges : np.ndarray
        Fine-region bin edges in ms, shape (B_f+1,).
    fine_centers : np.ndarray
        Fine bin centers in ms, shape (B_f,).
    macro_centers : np.ndarray
        Working-resolution bin centers in ms, shape (B_w,).
    macro_bin_ms : float
        Width of each working bin in ms.
    total_channels : int
        Number of channels on the MEA.

    Returns
    -------
    List[Dict[str, np.ndarray]]
        One peak dict per channel (see detect_peak_sides), indexed by
        channel.
    """
    per_channel: List[Dict[str, np.ndarray]] = []
    for channel in range(total_channels):
        summed = summed_ccg_from_trains(spike_times_ms, channel, fine_edges, total_channels)
        macro_counts, _ = rebin_fine(summed, fine_centers)
        smoothed = smooth_ccg(macro_counts, macro_bin_ms)
        per_channel.append(detect_peak_sides(smoothed, macro_centers, macro_bin_ms))
    return per_channel


def shuffle_trains(spike_times_ms: np.ndarray,
                   duration_s: float,
                   random_generator: np.random.Generator) -> np.ndarray:
    """Circularly shift each channel's train by a random offset.

    Each train is shifted by a random amount drawn uniformly over the
    recording duration and wrapped mod duration.  This preserves every
    channel's own temporal structure (rates, bursts, inter-spike
    intervals) while destroying the cross-channel timing that creates CCG
    peaks.

    Parameters
    ----------
    spike_times_ms : np.ndarray
        Object array of length K with per-channel spike times in ms.
    duration_s : float
        Recording duration in seconds (max spike time).
    random_generator : np.random.Generator
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        Object array of shifted trains in ms.
    """
    shifted = np.empty(len(spike_times_ms), dtype=object)
    duration_ms = duration_s * 1000.0
    for channel, train in enumerate(spike_times_ms):
        if len(train) == 0:
            shifted[channel] = train.copy()
            continue
        offset_ms = random_generator.uniform(0.0, duration_ms)
        shifted[channel] = (train + offset_ms) % duration_ms
    return shifted


def surrogate_null_prominences(spike_times_ms: np.ndarray,
                               fine_edges: np.ndarray,
                               fine_centers: np.ndarray,
                               macro_centers: np.ndarray,
                               macro_bin_ms: float,
                               duration_s: float,
                               n_surrogates: int = N_SURROGATES,
                               seed: int = 0) -> np.ndarray:
    """Pool peak prominences from shuffled spike trains.

    For each surrogate the trains are circularly shifted and every
    channel's summed CCG is rebuilt, smooth-peak detected with the same
    rebinning, smoothing and prominence rules as the observed data, and
    the prominences of all detected peaks (both sides, all channels) are
    pooled.  An observed prominence is significant if it exceeds most of
    this null pool.

    Parameters
    ----------
    spike_times_ms : np.ndarray
        Object array with per-channel spike times in ms.
    fine_edges : np.ndarray
        Fine-region bin edges in ms, shape (B_f+1,).
    fine_centers : np.ndarray
        Fine bin centers in ms, shape (B_f,).
    macro_centers : np.ndarray
        Working-resolution bin centers in ms, shape (B_w,).
    macro_bin_ms : float
        Width of each working bin in ms.
    duration_s : float
        Recording duration in seconds.
    n_surrogates : int
        Number of shuffled iterations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Pooled surrogate peak prominences, shape (N_s,).
    """
    random_generator = np.random.default_rng(seed)
    pooled: List[np.ndarray] = []

    for _ in range(n_surrogates):
        shifted = shuffle_trains(spike_times_ms, duration_s, random_generator)
        per_channel_peaks = summed_peaks_from_trains(
            shifted, fine_edges, fine_centers, macro_centers, macro_bin_ms,
        )
        for peaks in per_channel_peaks:
            pooled.append(peaks["pos_prom"])
            pooled.append(peaks["neg_prom"])

    if len(pooled) == 0:
        return np.empty(0)
    return np.concatenate(pooled)


def empirical_pvalues(observed_proms: np.ndarray,
                      null_proms: np.ndarray) -> np.ndarray:
    """Score observed peak prominences against a surrogate null pool.

    The empirical p-value is the fraction of null prominences reaching
    the observed value, with pseudocount smoothing:

        p = (1 + |{q ∈ null : q ≥ p_w}|) / (1 + |null|)

    so a peak beating the whole null reports p = 1 / (|null| + 1).

    Parameters
    ----------
    observed_proms : np.ndarray
        Prominence of each observed peak.
    null_proms : np.ndarray
        Pooled surrogate prominences (the null distribution).

    Returns
    -------
    np.ndarray
        Empirical p-value per observed peak, shape matching
        observed_proms.
    """
    if len(null_proms) == 0:
        return np.full_like(observed_proms, np.nan)
    pvalues = np.empty(len(observed_proms))
    null_sorted = np.sort(null_proms)
    for index, prominence in enumerate(observed_proms):
        count = len(null_sorted) - np.searchsorted(null_sorted, prominence, side="left")
        pvalues[index] = (1.0 + count) / (1.0 + len(null_sorted))
    return pvalues


def split_half_stability(spike_times_ms: np.ndarray,
                         fine_edges: np.ndarray,
                         fine_centers: np.ndarray,
                         macro_centers: np.ndarray,
                         macro_bin_ms: float,
                         total_channels: int = NUM_CHANNELS) -> np.ndarray:
    """Measure peak recurrence across two random time halves.

    Each channel's train is cut at its own median time.  On each half,
    the summed CCG per channel is rebuilt from trains, rebinned and
    peak-detected.  A full-set observed peak counts as stable when any
    half reproduces it on the same side within PEAK_MATCH_TOLERANCE_MS.

    Parameters
    ----------
    spike_times_ms : np.ndarray
        Object array with per-channel spike times in ms.
    fine_edges : np.ndarray
        Fine-region bin edges in ms, shape (B_f+1,).
    fine_centers : np.ndarray
        Fine bin centers in ms, shape (B_f,).
    macro_centers : np.ndarray
        Working-resolution bin centers in ms, shape (B_w,).
    macro_bin_ms : float
        Width of each working bin in ms.
    total_channels : int
        Number of channels on the MEA.

    Returns
    -------
    np.ndarray
        Stability fraction per channel, shape (K,).  Channels with no
        observed peaks score nan.
    """
    half_trains: List[Tuple[np.ndarray, np.ndarray]] = []
    for channel in range(total_channels):
        train = spike_times_ms[channel]
        if len(train) == 0:
            half_trains.append((train.copy(), train.copy()))
            continue
        split_time = float(np.median(train))
        earlier = train[train <= split_time]
        later = train[train > split_time]
        half_trains.append((earlier, later))

    half_peaks_storage: List[Dict[str, np.ndarray]] = []
    for half_index in range(2):
        half_arr = np.empty(total_channels, dtype=object)
        for channel in range(total_channels):
            half_arr[channel] = half_trains[channel][half_index]
        half_peaks_storage.append(
            summed_peaks_from_trains(
                half_arr, fine_edges, fine_centers, macro_centers, macro_bin_ms, total_channels,
            )
        )

    stability = np.full(total_channels, np.nan)
    for channel in range(total_channels):
        observed_counts, _ = rebin_fine(
            summed_ccg_from_trains(spike_times_ms, channel, fine_edges, total_channels),
            fine_centers,
        )
        observed = detect_peak_sides(
            smooth_ccg(observed_counts, macro_bin_ms), macro_centers, macro_bin_ms,
        )

        total_observed = 0
        matched = 0
        for side_name in ("pos", "neg"):
            observed_lags = observed[f"{side_name}_lag"]
            total_observed += len(observed_lags)
            for observed_lag in observed_lags:
                if any(
                    np.any(np.abs(half_peaks[channel][f"{side_name}_lag"] - observed_lag)
                           <= PEAK_MATCH_TOLERANCE_MS)
                    for half_peaks in half_peaks_storage
                ):
                    matched += 1

        if total_observed > 0:
            stability[channel] = matched / total_observed

    return stability


def build_summed_metrics(correlograms: np.ndarray,
                         started_fine: int,
                         fine_centers: np.ndarray,
                         macro_centers: np.ndarray,
                         macro_bin_ms: float,
                         per_channel_peaks: Optional[List[Dict[str, np.ndarray]]] = None,
                         total_channels: int = NUM_CHANNELS) -> Dict[str, np.ndarray]:
    """Compute summed-CCG peaks, lead index and dominant lag per channel.

    The summed CCG of each channel is sliced from the stored
    correlograms (S_i = Σ_j C_{i->j}).  Uses the provided per-channel
    peak dicts when given, otherwise extracts them on working-resolution
    bins.  The lead index window stays on the fine (0.01 ms) counts.

    Parameters
    ----------
    correlograms : np.ndarray
        Directed CCGs, int64 of shape (K, K, B).
    started_fine : int
        Index of the first fine bin in the full-length CCG arrays.
    fine_centers : np.ndarray
        Fine bin centers in ms, shape (B_f,).
    macro_centers : np.ndarray
        Working-resolution bin centers in ms, shape (B_w,).
    macro_bin_ms : float
        Width of each working bin in ms.
    per_channel_peaks : Optional[List[Dict[str, np.ndarray]]]
        Precomputed peak dicts per channel; computed here if None.
    total_channels : int
        Number of channels on the MEA.

    Returns
    -------
    Dict[str, np.ndarray]
        "lags_pos"/"prom_pos"/"lags_neg"/"prom_neg" object arrays of
        shape (K,); "lead_index", "lead_pos_count", "lead_neg_count",
        "dominant_lag" float64 (K,).
    """
    stop = started_fine + len(fine_centers)
    lags_pos = np.empty(total_channels, dtype=object)
    prom_pos = np.empty(total_channels, dtype=object)
    lags_neg = np.empty(total_channels, dtype=object)
    prom_neg = np.empty(total_channels, dtype=object)
    lead_index_arr = np.zeros(total_channels)
    lead_pos_arr = np.zeros(total_channels)
    lead_neg_arr = np.zeros(total_channels)
    dominant_lag_arr = np.zeros(total_channels)

    for channel in range(total_channels):
        summed = correlograms[channel].sum(axis=0)[started_fine:stop]
        if per_channel_peaks is not None:
            peaks = per_channel_peaks[channel]
        else:
            macro_counts, _ = rebin_fine(summed, fine_centers)
            peaks = extract_peaks(macro_counts, macro_centers, macro_bin_ms)
        lags_pos[channel] = peaks["pos_lag"]
        prom_pos[channel] = peaks["pos_prom"]
        lags_neg[channel] = peaks["neg_lag"]
        prom_neg[channel] = peaks["neg_prom"]

        lead, pos_count, neg_count = lead_index(summed, fine_centers)
        lead_index_arr[channel] = lead
        lead_pos_arr[channel] = pos_count
        lead_neg_arr[channel] = neg_count

        best_lag, _ = dominant_peak(peaks["pos_lag"], peaks["pos_prom"], peaks["neg_lag"], peaks["neg_prom"])
        dominant_lag_arr[channel] = best_lag

    return {
        "lags_pos": lags_pos,
        "prom_pos": prom_pos,
        "lags_neg": lags_neg,
        "prom_neg": prom_neg,
        "lead_index": lead_index_arr,
        "lead_pos_count": lead_pos_arr,
        "lead_neg_count": lead_neg_arr,
        "dominant_lag": dominant_lag_arr,
    }


def save_peaks(output_path: str, payload: Dict[str, np.ndarray], params: Dict[str, float]) -> None:
    """Persist peak lists, metrics and significance scores to an npz.

    Parameters
    ----------
    output_path : str
        Destination path for ccg_peaks.npz.
    payload : Dict[str, np.ndarray]
        Mapping of schema keys to arrays (see the module docstring).
    params : Dict[str, float]
        Peak analysis parameters.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload, params=str(params))
    print(f"Saved peak analysis to: {output}")


def gen_delay_map_html(lead_indices: np.ndarray,
                       dominant_lags: np.ndarray,
                       n_spikes: np.ndarray,
                       output_path: str) -> None:
    """Write an 8x8 heatmap of lead index with dominant-lag labels.

    Each grid cell is colored by the channel's lead index on the
    LEAD_WINDOW_MIN_MS..LEAD_WINDOW_MAX_MS lag scale (red = initiator,
    blue = follower, white = balanced).  The cell body shows the
    dominant signed lag; the tooltip adds the spike count.  Cells link
    to the per-channel annotated adjacency page.  The most extreme
    leaders and followers are summarized below the grid.

    Parameters
    ----------
    lead_indices : np.ndarray
        Lead index per channel, shape (K,).
    dominant_lags : np.ndarray
        Dominant signed lag per channel in ms, shape (K,).
    n_spikes : np.ndarray
        Spike count per channel, shape (K,).
    output_path : str
        Destination path for delay_map.html.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    colormap = plt.get_cmap("coolwarm")

    grid_rows: List[str] = []
    for row in range(GRID_ROWS):
        cells: List[str] = []
        for col in range(GRID_COLS):
            channel = row * GRID_COLS + col
            lead = float(lead_indices[channel])
            red, green, blue, _ = colormap(0.5 * (lead + 1.0))
            color_hex = f"#{int(red * 255):02X}{int(green * 255):02X}{int(blue * 255):02X}"
            spike_count = int(n_spikes[channel])
            cells.append(
                f"<a class='cell' href='adjacency/ch{channel:02d}_peaks.html' "
                f"title='Ch {channel}: {spike_count} spikes' "
                f"style='background:{color_hex}'>"
                f"<div class='ch'>{channel}</div>"
                f"<div class='lead'>{lead:+.2f}</div>"
                f"<div class='lag'>{dominant_lags[channel]:+.1f} ms</div></a>"
            )
        grid_rows.append("        <div class='row'>" + "".join(cells) + "</div>")
    grid_html = "\n".join(grid_rows)

    order = np.argsort(lead_indices)[::-1]
    leaders = "".join(
        f"<li>Ch {channel}: lead {lead_indices[channel]:+.2f}, "
        f"dominant lag {dominant_lags[channel]:+.1f} ms</li>"
        for channel in order[:5]
        if abs(float(lead_indices[channel])) > 1e-6
    )
    followers = "".join(
        f"<li>Ch {channel}: lead {lead_indices[channel]:+.2f}, "
        f"dominant lag {dominant_lags[channel]:+.1f} ms</li>"
        for channel in order[-5:][::-1]
        if abs(float(lead_indices[channel])) > 1e-6
    )
    if not leaders:
        leaders = "<li>none</li>"
    if not followers:
        followers = "<li>none</li>"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Lead index delay map</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        h2 {{ color: #444; font-size: 16px; }}
        .meta {{ font-size: 13px; color: #666; max-width: 900px; }}
        .grid {{ display: inline-block; border: 2px solid #333; border-radius: 6px;
                 background: white; padding: 4px; }}
        .row {{ display: flex; }}
        .cell {{ display: block; width: 88px; height: 88px; margin: 2px; border-radius: 4px;
                 text-decoration: none; color: #222; padding: 6px; box-sizing: border-box; }}
        .cell:hover {{ outline: 3px solid #333; }}
        .ch {{ font-weight: bold; font-size: 14px; }}
        .lead {{ font-size: 16px; font-weight: bold; margin-top: 6px; }}
        .lag {{ font-size: 12px; margin-top: 2px; }}
        .legend {{ margin: 12px 0; display: flex; align-items: center; gap: 8px;
                   font-size: 13px; color: #555; }}
        .bar {{ width: 200px; height: 14px; border-radius: 3px;
                background: linear-gradient(90deg, #4575b4, white, #d73027); }}
        .lists {{ display: flex; gap: 60px; margin-top: 20px; font-size: 13px; }}
        .lists ul {{ margin: 4px 0 0 0; padding-left: 20px; }}
        .lists li {{ margin: 2px 0; }}
        .nav a {{ color: #1f77b4; text-decoration: none; margin-right: 12px; font-size: 13px; }}
    </style>
</head>
<body>
    <h1>Lead index delay map</h1>
    <div class="nav"><a href="index.html">Back to peak-analysis hub</a>
    <a href="matrices.html">All-pairs matrices</a></div>
    <div class="legend">
        <span>follower (-1)</span><div class="bar"></div><span>lead (+1)</span>
        &nbsp;&nbsp;|&nbsp;&nbsp; lead window {LEAD_WINDOW_MIN_MS:.0f}-{LEAD_WINDOW_MAX_MS:.0f} ms
    </div>
    <div class="grid">
{grid_html}
    </div>
    <div class="lists">
        <div><h2>Top initiators</h2><ul>{leaders}</ul></div>
        <div><h2>Top followers</h2><ul>{followers}</ul></div>
    </div>
    <p class="meta">Color: lead index = ({LEAD_WINDOW_MIN_MS:.0f}-{LEAD_WINDOW_MAX_MS:.0f} ms
    counts to the right minus counts to the left) divided by their sum.
    Red = the array tends to fire <i>after</i> this channel (initiator);
    blue = it tends to fire <i>before</i> (follower).  Number in each cell:
    dominant signed peak lag in ms.  Click a cell for its annotated page.</p>
</body>
</html>"""

    output.write_text(html)
    print(f"Saved delay map to: {output}")


def _matrix_svg(matrix: np.ndarray,
                title: str,
                cmap_name: str,
                symmetric: bool) -> str:
    """Render a K x K heatmap of a pair statistic as an SVG string.

    Zero entries (diagonal self-pairs and pairs with no detected peak)
    are masked to white.  The color scale is symmetric about zero for
    signed lags and zero-to-max for prominences.

    Parameters
    ----------
    matrix : np.ndarray
        K x K float array.
    title : str
        Plot title.
    cmap_name : str
        Matplotlib colormap name.
    symmetric : bool
        Whether to center the color map on zero (True) or start at 0
        (False).

    Returns
    -------
    str
        SVG markup.
    """
    data = np.ma.masked_where(matrix == 0.0, matrix)
    scale_max = float(np.abs(matrix).max()) if symmetric else float(matrix.max())
    if symmetric:
        scale_max = max(scale_max, 0.5)
    offset = 0.0
    if symmetric:
        offset = -scale_max

    fig, ax = plt.subplots(figsize=(QUANT_FIG_WIDTH, QUANT_FIG_HEIGHT), dpi=QUANT_FIG_DPI)
    image = ax.imshow(data, cmap=cmap_name, origin="upper", interpolation="none",
                      vmin=offset, vmax=scale_max, aspect="equal")
    ax.set_xticks(range(0, NUM_CHANNELS, GRID_ROWS))
    ax.set_yticks(range(0, NUM_CHANNELS, GRID_COLS))
    ax.set_xlabel("Target channel")
    ax.set_ylabel("Source channel")
    ax.set_title(title, fontsize=12, pad=8)
    fig.colorbar(image, ax=ax, shrink=0.8, pad=0.02)
    fig.tight_layout(pad=0.6)
    return _figure_to_svg(fig)


def gen_matrices_html(pair_dominant_lag: np.ndarray,
                      pair_max_prominence: np.ndarray,
                      pair_lead_index: np.ndarray,
                      output_path: str) -> None:
    """Write 64x64 heatmaps of dominant lag, prominence and lead index.

    Parameters
    ----------
    pair_dominant_lag : np.ndarray
        Signed dominant lag per directed pair in ms, shape (K, K).
    pair_max_prominence : np.ndarray
        Largest peak prominence per directed pair, shape (K, K).
    pair_lead_index : np.ndarray
        Count-based lead asymmetry per directed pair in [-1, +1],
        shape (K, K).
    output_path : str
        Destination path for matrices.html.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lag_svg = _matrix_svg(pair_dominant_lag, "Dominant peak lag per directed pair (ms)",
                          "coolwarm", symmetric=True)
    prominence_svg = _matrix_svg(pair_max_prominence,
                                 "Largest peak prominence per directed pair (counts)",
                                 "viridis", symmetric=False)
    lead_svg = _matrix_svg(pair_lead_index,
                           "Lead index per directed pair (count asymmetry, +1 = initiator)",
                           "coolwarm", symmetric=True)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>All-pairs peak matrices</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .meta {{ font-size: 13px; color: #666; max-width: 900px; }}
        .matrix {{ background: white; padding: 14px; border-radius: 8px; margin-bottom: 20px;
                   box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        svg {{ display: block; width: 100%; height: auto; }}
        .nav {{ margin-bottom: 12px; font-size: 13px; }}
        .nav a {{ color: #1f77b4; text-decoration: none; margin-right: 12px; }}
    </style>
</head>
<body>
    <h1>All-pairs peak matrices</h1>
    <div class="nav"><a href="index.html">Back to peak-analysis hub</a>
    <a href="delay_map.html">Delay map</a></div>
    <p class="meta">Row = source channel, column = target channel.  Cell (i, j)
    reads the directed pair i&rarr;j.  In the lag and prominence matrices, red
    cells mean channel j tends to fire <i>after</i> channel i at their dominant
    peak lag; blue means the reverse.  White cells (diagonal, or pairs with no
    significant peak) are masked.  Per-pair spike counts are usually too small
    for a resolvable peak, so those matrices stay mostly empty; the lead-index
    matrix instead sums the <i>counts</i> in the 1-4 ms band on each side, so it
    shows the directed asymmetry (red = source fires ~2 ms before target,
    blue = target fires first) even where no single peak clears the floor.</p>
    <div class="matrix">{lag_svg}</div>
    <div class="matrix">{prominence_svg}</div>
    <div class="matrix">{lead_svg}</div>
</body>
</html>"""

    output.write_text(html)
    print(f"Saved matrices to: {output}")


def _make_zoom_dropdown() -> Tuple[str, str]:
    """Build a zoom-only dropdown control (no full-range checkbox).

    Returns
    -------
    Tuple[str, str]
        (controls_html, script_html) where the script toggles active
        .zoom-panel divs according to the selected zoom width.
    """
    controls = (
        "    <div class='controls'>\n"
        "        <label>Zoom: <select id='zoomSelect'>\n"
        "            <option value='5' selected>\u00b15 ms</option>\n"
        "            <option value='10'>\u00b110 ms</option>\n"
        "        </select></label>\n"
        "    </div>"
    )
    script = (
        "    <script>\n"
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


def _make_annotated_zoom_svg(counts: np.ndarray,
                             centers: np.ndarray,
                             bin_width_ms: float,
                             zoom_half_width: int,
                             title: str,
                             color: str,
                             pos_lags: np.ndarray,
                             pos_pvals: Optional[np.ndarray],
                             neg_lags: np.ndarray,
                             neg_pvals: Optional[np.ndarray]) -> str:
    """Render a zoomed CCG with detected peaks marked and labeled.

    Plots working-resolution counts (thin line), the gaussian-smoothed
    envelope (dashed), and every detected peak within the zoom window as
    a red dot with a lag label (plus a * badge when an empirical p < 0.05
    is supplied for a summed-CCG peak).

    Parameters
    ----------
    counts : np.ndarray
        Working-resolution CCG counts, shape (B_w,).
    centers : np.ndarray
        Working bin centers in ms, shape (B_w,).
    bin_width_ms : float
        Width of each working bin in ms.
    zoom_half_width : int
        Half-width of the zoom window in ms.
    title : str
        Plot title.
    color : str
        Matplotlib color for the histogram fill.
    pos_lags : np.ndarray
        Positive-side peak lags in ms.
    pos_pvals : Optional[np.ndarray]
        Empirical p-values for positive-side peaks, or None.
    neg_lags : np.ndarray
        Negative-side peak lags in ms.
    neg_pvals : Optional[np.ndarray]
        Empirical p-values for negative-side peaks, or None.

    Returns
    -------
    str
        SVG markup.
    """
    zoom_mask = (centers >= -zoom_half_width) & (centers <= zoom_half_width)
    centers_zoom = centers[zoom_mask]
    counts_zoom = counts[zoom_mask]
    smoothed = smooth_ccg(counts, bin_width_ms)
    smoothed_zoom = smoothed[zoom_mask]

    fig, ax = plt.subplots(figsize=(ZOOM_FIG_WIDTH, ZOOM_FIG_HEIGHT))
    ax.plot(centers_zoom, counts_zoom, linewidth=1.0, color=color, alpha=0.7)
    ax.plot(centers_zoom, smoothed_zoom, linewidth=1.6, color=color, linestyle="--")
    ax.axvline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlim(-zoom_half_width, zoom_half_width)
    ax.set_xticks(range(-zoom_half_width, zoom_half_width + 1))
    ax.set_xlabel("Lag (ms)", fontsize=11)
    ax.set_ylabel("Spike count", fontsize=11)
    ax.set_title(title, fontsize=12, pad=6)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for lag_group, pvals in [(pos_lags, pos_pvals), (neg_lags, neg_pvals)]:
        if len(lag_group) == 0:
            continue
        for peak_index, lag in enumerate(lag_group):
            if abs(lag) > zoom_half_width:
                continue
            bin_index = int(np.argmin(np.abs(centers - lag)))
            y_value = float(smoothed[bin_index])
            label = f"{lag:.2f}"
            if pvals is not None and float(pvals[peak_index]) < 0.05:
                label = label + "*"
            ax.plot(lag, y_value, marker="o", markersize=5, color="red", zorder=5)
            ax.annotate(label, (lag, y_value), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color="darkred")

    fig.tight_layout(pad=0.6)
    return _figure_to_svg(fig)


def _fmt_pair_peaks(lags: np.ndarray, prominences: np.ndarray) -> str:
    """Format a side's peak list as "lag (prominence)" text.

    Parameters
    ----------
    lags : np.ndarray
        Peak lags in ms.
    prominences : np.ndarray
        Peak prominences.

    Returns
    -------
    str
        Human-readable list, or "none".
    """
    if len(lags) == 0:
        return "none"
    return ", ".join(f"{lag:+.2f} ms ({prom:.0f})" for lag, prom in zip(lags, prominences))


def _fmt_summed_peaks(lags: np.ndarray, prominences: np.ndarray,
                      pvals: np.ndarray) -> str:
    """Format a summed-CCG peak list with p-value badges.

    Parameters
    ----------
    lags : np.ndarray
        Peak lags in ms.
    prominences : np.ndarray
        Peak prominences.
    pvals : np.ndarray
        Empirical p-values.

    Returns
    -------
    str
        Human-readable list with p, or "none".
    """
    if len(lags) == 0:
        return "none"
    badges = []
    for lag, prominence, pvalue in zip(lags, prominences, pvals):
        if pvalue < 0.05:
            badge = f"{lag:+.2f} ms (p={pvalue:.3g})*"
        else:
            badge = f"{lag:+.2f} ms (p={pvalue:.3g})"
        badges.append(badge)
    return ", ".join(badges)


def gen_adjacency_peaks_html(channel: int,
                             n_spikes: np.ndarray,
                             summed_ccg: np.ndarray,
                             pair_ccgs: Dict[int, np.ndarray],
                             summary_peaks: Dict[str, np.ndarray],
                             pair_peaks: Dict[int, Dict[str, np.ndarray]],
                             pair_lead_index: np.ndarray,
                             pair_lead_pos: np.ndarray,
                             pair_lead_neg: np.ndarray,
                             fine_centers: np.ndarray,
                             macro_centers: np.ndarray,
                             macro_bin_ms: float,
                             output_dir: str) -> None:
    """Write one channel's annotated latency page.

    The page shows the summed CCG to all other channels and the directed
    CCG to each grid neighbor, each as annotation-marked zoomed
    histograms with a per-side peak readout underneath.  A header line
    reports the channel's lead index, dominant lag, latency spectra and
    split-half stability.

    Parameters
    ----------
    channel : int
        Channel index.
    n_spikes : np.ndarray
        Spike count per channel, shape (K,).
    summed_ccg : np.ndarray
        Summed fine CCG of this channel, shape (B_f,).
    pair_ccgs : Dict[int, np.ndarray]
        Neighbor -> fine CCG, directed from this channel.
    summary_peaks : Dict[str, np.ndarray]
        Summed-CCG peak objects for this channel.
    pair_peaks : Dict[int, Dict[str, np.ndarray]]
        Per-neighbor peak dicts.
    pair_lead_index : np.ndarray
        Lead index per directed pair, shape (K, K).
    pair_lead_pos : np.ndarray
        +lag fine counts in the lead window, shape (K, K).
    pair_lead_neg : np.ndarray
        -lag fine counts in the lead window, shape (K, K).
    fine_centers : np.ndarray
        Fine bin centers in ms, shape (B_f,).
    macro_centers : np.ndarray
        Working-resolution bin centers in ms, shape (B_w,).
    macro_bin_ms : float
        Width of each working bin in ms.
    output_dir : str
        Directory to write the HTML into (quant/adjacency/).
    """
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    controls_html, script_html = _make_zoom_dropdown()
    neighbors = _get_adjacent_channels(channel)

    lead = summary_peaks["lead_index"][channel]
    dominant_lag = summary_peaks["dominant_lag"][channel]
    stability = summary_peaks["split_stability"][channel]
    n_pos_peaks = len(summary_peaks["lags_pos"][channel])
    n_neg_peaks = len(summary_peaks["lags_neg"][channel])

    spectrum_text = (
        f"positive: {_fmt_summed_peaks(summary_peaks['lags_pos'][channel], summary_peaks['prom_pos'][channel], summary_peaks['pval_pos'][channel])}"
        f" | negative: {_fmt_summed_peaks(summary_peaks['lags_neg'][channel], summary_peaks['prom_neg'][channel], summary_peaks['pval_neg'][channel])}"
    )

    subplot_blocks: List[str] = []
    subplot_entries = [("summed", f"Ch {channel} \u2192 All 63 (summed)", "darkorange")]
    for neighbor in neighbors:
        direction = _get_direction_label(channel, neighbor)
        subplot_entries.append(
            (f"ch{neighbor}", f"Ch {channel} \u2192 Ch {neighbor} ({direction})", "steelblue")
        )

    for block_key, label, color in subplot_entries:
        if block_key == "summed":
            counts = summed_ccg
            peaks = summary_peaks
            pos_lags = peaks["lags_pos"][channel]
            pos_proms = peaks["prom_pos"][channel]
            pos_pvals = peaks["pval_pos"][channel]
            neg_lags = peaks["lags_neg"][channel]
            neg_proms = peaks["prom_neg"][channel]
            neg_pvals = peaks["pval_neg"][channel]
            readout = (f"peaks: {_fmt_summed_peaks(pos_lags, pos_proms, pos_pvals)}"
                       f" | {_fmt_summed_peaks(neg_lags, neg_proms, neg_pvals)}")
        else:
            neighbor = int(block_key[2:])
            counts = pair_ccgs[neighbor]
            peaks = pair_peaks[neighbor]
            pos_lags = peaks["pos_lag"]
            pos_proms = peaks["pos_prom"]
            pos_pvals = None
            neg_lags = peaks["neg_lag"]
            neg_proms = peaks["neg_prom"]
            neg_pvals = None
            pair_lead_value = pair_lead_index[channel, neighbor]
            pair_pos_count = pair_lead_pos[channel, neighbor]
            pair_neg_count = pair_lead_neg[channel, neighbor]
            if pair_pos_count + pair_neg_count == 0:
                lead_text = f"lead {pair_lead_value:+.2f} (no co-fires in 1-4 ms)"
            else:
                lead_text = (f"lead {pair_lead_value:+.2f} "
                             f"({int(pair_pos_count)} vs {int(pair_neg_count)})")
            readout = (f"peaks: {_fmt_pair_peaks(pos_lags, pos_proms)} "
                       f"| {_fmt_pair_peaks(neg_lags, neg_proms)} | {lead_text}")

        zoom_divs = ""
        for level_index, zoom_width in enumerate(ZOOM_LEVELS):
            active_class = " active" if level_index == 0 else ""
            macro_counts, _ = rebin_fine(counts, fine_centers)
            svg = _make_annotated_zoom_svg(
                macro_counts, macro_centers, macro_bin_ms, zoom_width,
                f"{label} \u2014 \u00b1{zoom_width} ms", color,
                pos_lags, pos_pvals, neg_lags, neg_pvals,
            )
            zoom_divs += f"<div class='zoom-panel zoom-{zoom_width}{active_class}'>{svg}</div>\n"

        subplot_blocks.append(
            f"    <div class='subplot'>\n"
            f"    <div class='subplot-label'>{label}</div>\n"
            f"    <div class='readout'>{readout}</div>\n"
            f"    {zoom_divs}\n"
            f"    </div>"
        )

    prev_channel, next_channel = _nav_neighbors(channel, n_spikes)
    nav_parts = ['<a href="../index.html">Back to hub</a>', '<a href="../delay_map.html">Delay map</a>']
    if prev_channel is not None:
        nav_parts.append(f'<a href="ch{prev_channel:02d}_peaks.html">\u2190 Ch {prev_channel}</a>')
    if next_channel is not None:
        nav_parts.append(f'<a href="ch{next_channel:02d}_peaks.html">Ch {next_channel} \u2192</a>')
    nav_html = " &nbsp;|&nbsp; ".join(nav_parts)

    lead_text = (f"Ch {channel} | {int(n_spikes[channel])} spikes | "
                 f"lead index {lead:+.2f} | dominant lag {dominant_lag:+.2f} ms | "
                 f"split-half stability {stability:.2f}" if not np.isnan(stability)
                 else f"Ch {channel} | {int(n_spikes[channel])} spikes | "
                 f"lead index {lead:+.2f} | dominant lag {dominant_lag:+.2f} ms | "
                 f"split-half stability n/a")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Channel {channel} \u2014 quantified latency structure</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .meta {{ font-size: 13px; color: #666; }}
        .spectrum {{ font-size: 13px; color: #333; background: #fff8dc; padding: 8px 10px;
                     border-radius: 4px; margin: 10px 0; }}
        .controls {{ margin: 12px 0; }}
        .nav {{ font-size: 13px; margin-bottom: 12px; }}
        .nav a {{ color: #1f77b4; text-decoration: none; margin-right: 12px; }}
        .subplot {{ padding: 6px; margin-bottom: 14px; border-radius: 4px;
                    background: white; border: 1px solid #eee; }}
        .subplot-label {{ font-size: 12px; font-weight: bold; color: #444; margin-bottom: 4px; }}
        .readout {{ font-size: 12px; color: #555; margin-bottom: 4px; font-family: monospace; }}
        .subplot svg {{ display: block; width: 100%; height: auto; }}
        .zoom-panel {{ display: none; }}
        .zoom-panel.active {{ display: block; }}
    </style>
</head>
<body>
    <h1>Channel {channel} \u2014 quantified latency structure</h1>
    <div class="nav">{nav_html}</div>
    <p class="meta">{lead_text}</p>
    <div class="spectrum">latency spectrum: {spectrum_text}</div>
    <p class="meta">Red dots mark significant peaks; * marks empirical p &lt; 0.05
    (surrogate shuffles). Positive lag = target fires after the source. Pair
    readouts also give the lead index: count asymmetry in the 1-4 ms band
    (&quot;+1.00 (2 vs 0)&quot; = 2 target spikes in the +band, none in the -band).
    Pairwise spikes are sparse, so most pairs show no resolvable peak even
    though their lead asymmetry is clean.</p>
    {controls_html}

{''.join(subplot_blocks)}

{script_html}
</body>
</html>"""

    html_path = output_directory / f"ch{channel:02d}_peaks.html"
    html_path.write_text(html)


def _nav_neighbors(channel: int, n_spikes: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    """Find the previous and next active channel for page navigation.

    Parameters
    ----------
    channel : int
        Current channel index.
    n_spikes : np.ndarray
        Spike count per channel, shape (K,).

    Returns
    -------
    Tuple[Optional[int], Optional[int]]
        (previous, next) active channel, or None at the edges.
    """
    previous_channel: Optional[int] = None
    next_channel: Optional[int] = None
    for candidate in range(channel - 1, -1, -1):
        if n_spikes[candidate] > 0:
            previous_channel = candidate
            break
    for candidate in range(channel + 1, NUM_CHANNELS):
        if n_spikes[candidate] > 0:
            next_channel = candidate
            break
    return previous_channel, next_channel


def gen_index_html(output_path: str, n_spikes: np.ndarray) -> None:
    """Write the hub page linking all peak-analysis views.

    Parameters
    ----------
    output_path : str
        Destination path for index.html.
    n_spikes : np.ndarray
        Spike count per channel, shape (K,).
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    active_channels = int(np.sum(n_spikes > 0))

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Peak analysis hub</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .meta {{ font-size: 13px; color: #666; max-width: 900px; }}
        .card {{ background: white; padding: 16px 20px; margin-bottom: 14px;
                 border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                 display: block; text-decoration: none; color: #333; }}
        .card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.2); }}
        .card h2 {{ margin: 0 0 6px 0; font-size: 18px; }}
        .card p {{ margin: 0; font-size: 13px; color: #666; }}
    </style>
</head>
<body>
    <h1>Peak analysis hub</h1>
    <p class="meta">{active_channels} channels with spikes.  This stage extracts
    significant local maxima from every cross-correlogram, scores their
    prominence against shuffled-train surrogates, and measures lead-follow
    asymmetry.  All scales are in ms unless noted.</p>

    <a class="card" href="delay_map.html"><h2>Lead index delay map</h2>
    <p>8x8 heatmap of who fires ~{LEAD_WINDOW_MIN_MS:.0f}-{LEAD_WINDOW_MAX_MS:.0f} ms
    before the array (initiators, red) vs. after (followers, blue).</p></a>

    <a class="card" href="matrices.html"><h2>All-pairs lag &amp; prominence matrices</h2>
    <p>64x64 heatmaps of the dominant peak lag and largest peak prominence
    for every directed pair.</p></a>

    <a class="card" href="adjacency/ch00_peaks.html"><h2>Annotated adjacency pages</h2>
    <p>Per-channel histograms with detected peaks marked, labeled, and rated
    with empirical p-values.</p></a>
</body>
</html>"""

    output.write_text(html)
    print(f"Saved peak-analysis hub to: {output}")


def main(args: argparse.Namespace) -> None:
    """Entry point: extract CCG peaks, score them, and render views.

    Locates a cross_correlograms.npz (CLI arg or latest run), loads the
    stored CCGs and the source spike trains, detects peaks on the fine
    regime for every directed pair and every summed CCG, computes lead
    indices, split-half stability and surrogate p-values, then saves
    ccg_peaks.npz and the quant HTML views.
    """
    ccg_npz_path = Path(args.npz) if args.npz else Path()
    if not ccg_npz_path.is_file():
        output_root = Path(args.out_root)
        if not output_root.is_dir():
            output_root = Path(__file__).resolve().parent / args.out_root
        if output_root.is_dir():
            runs = sorted([d for d in output_root.iterdir() if d.is_dir()], reverse=True)
            for run in runs:
                candidate = run / "cross_corr" / "cross_correlograms.npz"
                if candidate.exists():
                    ccg_npz_path = candidate
                    break
        if not ccg_npz_path.is_file():
            print("Error: No cross_correlograms.npz found. Run cross_correlation.py first.")
            sys.exit(1)

    print("=" * 60)
    print("CCG Peak Analysis")
    print("=" * 60)
    print(f"Input CCG:  {ccg_npz_path}")
    print(f"Surrogates: {args.surrogates}")
    print("=" * 60)

    computation_start = time.time()

    correlograms, bin_edges, n_spikes, spike_times_s, channels, source_npz = load_inputs(
        str(ccg_npz_path)
    )
    spike_times_ms = np.array(
        [np.asarray(train, dtype=np.float64) * 1000.0 for train in spike_times_s], dtype=object
    )
    duration_s = 0.0
    for train in spike_times_s:
        if len(train) > 0:
            duration_s = max(duration_s, float(train.max()))
    print(f"Loaded spike times for {len(channels)} channels, "
          f"duration {duration_s:.0f} s")

    fine_edges, fine_centers, fine_bin_ms, start = get_fine_region(bin_edges)
    print(f"Fine regime: {len(fine_centers)} bins of {fine_bin_ms} ms "
          f"within \u00b1{PEAK_REGION_MS:.0f} ms")
    _, macro_centers = rebin_fine(np.zeros(len(fine_centers)), fine_centers)
    macro_bin_ms: float = REBIN_MS
    print(f"Working bins: {len(macro_centers)} bins of {macro_bin_ms} ms")

    pair_peak_data = compute_pair_peaks(
        correlograms, fine_centers, start, macro_centers, macro_bin_ms,
    )
    print(f"Detected directed-pair peaks in {time.time() - computation_start:.1f}s")

    summary_data = build_summed_metrics(
        correlograms, start, fine_centers, macro_centers, macro_bin_ms,
    )
    print(f"Summed-CCG metrics in {time.time() - computation_start:.1f}s")

    stability = split_half_stability(
        spike_times_ms, fine_edges, fine_centers, macro_centers, macro_bin_ms,
    )
    print(f"Split-half stability in {time.time() - computation_start:.1f}s")

    null_proms = surrogate_null_prominences(
        spike_times_ms, fine_edges, fine_centers, macro_centers, macro_bin_ms,
        duration_s, n_surrogates=args.surrogates,
    )
    print(f"Surrogate null ({len(null_proms)} pooled prominences) in "
          f"{time.time() - computation_start:.1f}s")

    pval_pos = np.empty(NUM_CHANNELS, dtype=object)
    pval_neg = np.empty(NUM_CHANNELS, dtype=object)
    for channel in range(NUM_CHANNELS):
        pval_pos[channel] = empirical_pvalues(summary_data["prom_pos"][channel], null_proms)
        pval_neg[channel] = empirical_pvalues(summary_data["prom_neg"][channel], null_proms)

    run_dir = ccg_npz_path.parent.parent
    analysis_params: Dict[str, float] = {
        "peak_region_ms": PEAK_REGION_MS,
        "peak_smooth_sigma_ms": PEAK_SMOOTH_SIGMA_MS,
        "min_prominence_count": MIN_PROMINENCE_COUNT,
        "min_prominence_fraction": MIN_PROMINENCE_FRACTION,
        "min_peak_separation_ms": MIN_PEAK_SEPARATION_MS,
        "peak_zone_min_ms": PEAK_ZONE_MIN_MS,
        "baseline_zone_ms": BASELINE_ZONE_MS,
        "lead_window_min_ms": LEAD_WINDOW_MIN_MS,
        "lead_window_max_ms": LEAD_WINDOW_MAX_MS,
        "peak_match_tolerance_ms": PEAK_MATCH_TOLERANCE_MS,
        "rebin_ms": REBIN_MS,
        "n_surrogates": float(args.surrogates),
    }

    payload: Dict[str, np.ndarray] = {
        "source_npz": source_npz,
        "source_ccg_npz": str(ccg_npz_path),
        "fine_edges": fine_edges,
        "fine_centers": fine_centers,
        "fine_bin_ms": np.asarray(fine_bin_ms),
        "n_spikes": n_spikes,
        "pair_peak_lags_pos": pair_peak_data["lags_pos"],
        "pair_peak_prom_pos": pair_peak_data["prom_pos"],
        "pair_peak_lags_neg": pair_peak_data["lags_neg"],
        "pair_peak_prom_neg": pair_peak_data["prom_neg"],
        "pair_dominant_lag": pair_peak_data["dominant_lag"],
        "pair_max_prominence": pair_peak_data["max_prominence"],
        "pair_lead_index": pair_peak_data["lead_index"],
        "pair_lead_pos_count": pair_peak_data["lead_pos_count"],
        "pair_lead_neg_count": pair_peak_data["lead_neg_count"],
        "sum_peak_lags_pos": summary_data["lags_pos"],
        "sum_peak_prom_pos": summary_data["prom_pos"],
        "sum_peak_pval_pos": pval_pos,
        "sum_peak_lags_neg": summary_data["lags_neg"],
        "sum_peak_prom_neg": summary_data["prom_neg"],
        "sum_peak_pval_neg": pval_neg,
        "lead_index": summary_data["lead_index"],
        "lead_pos_count": summary_data["lead_pos_count"],
        "lead_neg_count": summary_data["lead_neg_count"],
        "dominant_lag": summary_data["dominant_lag"],
        "split_stability": stability,
        "null_prominence": null_proms,
    }

    npz_output = run_dir / "cross_corr" / "ccg_peaks.npz"
    save_peaks(str(npz_output), payload, analysis_params)

    quant_dir = run_dir / "cross_corr" / "quant"
    gen_index_html(str(quant_dir / "index.html"), n_spikes)
    gen_delay_map_html(
        summary_data["lead_index"], summary_data["dominant_lag"], n_spikes,
        str(quant_dir / "delay_map.html"),
    )
    gen_matrices_html(
        pair_peak_data["dominant_lag"], pair_peak_data["max_prominence"],
        pair_peak_data["lead_index"],
        str(quant_dir / "matrices.html"),
    )

    stop = start + len(fine_centers)
    adjacency_quant_dir = quant_dir / "adjacency"
    for channel in range(NUM_CHANNELS):
        if n_spikes[channel] == 0:
            continue
        neighbors = _get_adjacent_channels(channel)
        summed_ccg = correlograms[channel].sum(axis=0)[start:stop]
        pair_ccgs = {neighbor: correlograms[channel, neighbor, start:stop]
                     for neighbor in neighbors}
        pair_peaks = {
            neighbor: {
                "pos_lag": pair_peak_data["lags_pos"][channel, neighbor],
                "pos_prom": pair_peak_data["prom_pos"][channel, neighbor],
                "neg_lag": pair_peak_data["lags_neg"][channel, neighbor],
                "neg_prom": pair_peak_data["prom_neg"][channel, neighbor],
            }
            for neighbor in neighbors
        }
        summary_peaks = {
            "lead_index": summary_data["lead_index"],
            "dominant_lag": summary_data["dominant_lag"],
            "split_stability": stability,
            "lags_pos": summary_data["lags_pos"],
            "prom_pos": summary_data["prom_pos"],
            "pval_pos": pval_pos,
            "lags_neg": summary_data["lags_neg"],
            "prom_neg": summary_data["prom_neg"],
            "pval_neg": pval_neg,
        }
        gen_adjacency_peaks_html(
            channel, n_spikes, summed_ccg, pair_ccgs, summary_peaks, pair_peaks,
            pair_peak_data["lead_index"], pair_peak_data["lead_pos_count"],
            pair_peak_data["lead_neg_count"],
            fine_centers, macro_centers, macro_bin_ms, str(adjacency_quant_dir),
        )

    elapsed = time.time() - computation_start
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Output: {quant_dir}")
    print(f"  npz:   {npz_output}")
    print(f"  quant: {quant_dir}/")
    print(f"  elapsed: {elapsed:.1f}s")
    print("=" * 60)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the CCG peak analysis.

    Parameters
    ----------
    argv : Optional[List[str]]
        Argument list to parse.  None uses sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: npz, out_root, surrogates.
    """
    parser = argparse.ArgumentParser(
        description="Peak extraction from pairwise cross-correlograms")
    parser.add_argument("--npz", type=str, default="",
                        help="Path to cross_correlograms.npz (auto-finds latest run if empty)")
    parser.add_argument("--out-root", type=str, default="outputs",
                        help="Root directory for run outputs")
    parser.add_argument("--surrogates", type=int, default=N_SURROGATES,
                        help=f"Number of shuffled-train iterations (default: {N_SURROGATES})")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())