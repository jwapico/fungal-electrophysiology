"""
avalanche_analysis.py  (criticality / avalanche statistics)

Tests whether the fungal MEA activity shows the signatures of a system
poised near a critical point, using the "neuronal avalanche" framework
developed for mammalian cortex (Beggs & Plenz 2003, J Neurosci 23:11167;
Haldeman & Beggs 2005, Phys Rev Lett 94:058101).

The analysis works on the PERSISTED event times in a run's waveforms.npz
(the same arrays spike_sorting.py reads), so it never re-derives events:

  1. Raster:            each channel's event times are binned at width
                        DELTA_MS. An avalanche is a maximal run of
                        consecutive non-empty bins (standard Beggs-Plenz).
  2. Size distribution: P(size) over all avalanches; fitted with a maximum
                        likelihood power law (Clauset-Shalizi-Newman 2009,
                        SIAM Rev 51:661). Critical prediction: alpha ~ 3/2.
  3. Branching ratio:   sigma = <n_active(t+1) | n_active(t)> / n_active(t),
                        the expected number of "descendant" active bins one
                        time step later. Critical prediction: sigma ~ 1.
  4. Duration:          P(duration), with critical prediction ~ 2.0.
  5. Surrogate control: the SAME statistics on time-shuffled rasters
                        (per-channel event times uniformly re-drawn). If the
                        shuffled data reproduces the observed power law,
                        the signal is a thresholded stochastic artifact and
                        NOT criticality (Touboul & Destexhe 2010, PLoS ONE
                        5:e8982).

The test is the comparison, at fixed bin width, between the real and the
shuffled rasters: only a significant difference (and exponents consistent
with 3/2, 2, sigma ~ 1) supports the critical-point hypothesis.

Run from the project root:
    python avalanche_analysis.py -o outputs/<ts>/waveforms/waveforms.npz
    python avalanche_analysis.py -o ... -b 250        # 250 ms bins
    python avalanche_analysis.py -o ... -b 250 -s 200 # 200 surrogates
    python avalanche_analysis.py -o ... --min-events 30  # per-channel floor
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

DEFAULT_BIN_MS: float = 250.0      # Beggs-Plenz bin width for fungal timescales
DEFAULT_N_SURROGATES: int = 100    # shuffled rasters for the control
MIN_CHANNEL_EVENTS: int = 1        # channels with fewer events are dropped
OUTPUT_SUBDIR: str = "avalanche"

CRIT_ALPHA: float = 1.5    # critical avalanche size exponent (Beggs & Plenz 2003)
CRIT_TAU: float = 2.0      # critical avalanche duration exponent
CRIT_SIGMA: float = 1.0    # critical branching ratio


def _load_event_times(npz_path: Path, min_events: int) -> List[np.ndarray]:
    """Return per-channel sorted event times (seconds) from a run's npz.

    Channels with fewer than `min_events` total events are excluded (they
    carry no avalanche statistics and only add dead weight to the raster).
    """
    archive = np.load(str(npz_path), allow_pickle=True)
    spike_times = archive["spike_times"]
    per_channel: List[np.ndarray] = []
    for channel in range(len(spike_times)):
        times = np.asarray(spike_times[channel], dtype=float)
        if len(times) >= min_events:
            per_channel.append(np.sort(times))
    return per_channel


def _bin_raster(event_times: Sequence[np.ndarray], bin_s: float,
                duration_s: float) -> np.ndarray:
    """Bin each channel's event times into an event-count raster.

    Raster[i, t] = number of events channel i fired in bin t. Bin width is
    given in seconds (callers convert from ms); bins run from 0 to
    `duration_s` (the full recording length, so trailing silence is binned
    and correctly splits the last avalanche).
    """
    n_bins = int(np.ceil(duration_s / bin_s))
    raster = np.zeros((len(event_times), n_bins), dtype=np.int64)
    bin_edges = np.arange(0.0, duration_s + bin_s, bin_s)
    for i, times in enumerate(event_times):
        if len(times) == 0:
            continue
        bin_idx = np.searchsorted(bin_edges, times, side="right") - 1
        np.add.at(raster[i], np.clip(bin_idx, 0, n_bins - 1), 1)
    return raster


def _find_avalanches(raster: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split the raster into avalanches; return (sizes, durations).

    Any bin with activity in >= 1 channel is "active". Avalanches are
    maximal runs of consecutive active bins separated by >= 1 silent bin.
    Duration is the run length in bins; SIZE is the total number of events
    summed across channels and bins in the run (Beggs & Plenz 2003), so a
    bin with several co-active channels contributes its full count.
    """
    active = raster.sum(axis=0)
    sizes: List[int] = []
    durations: List[int] = []
    run = 0
    run_events = 0
    for count in active:
        if count > 0:
            run += 1
            run_events += int(count)
        elif run > 0:
            sizes.append(run_events)
            durations.append(run)
            run = 0
            run_events = 0
    if run > 0:
        sizes.append(run_events)
        durations.append(run)
    return np.asarray(sizes, dtype=float), np.asarray(durations, dtype=float)


def _branching_parameter(raster: np.ndarray) -> float:
    """Expected number of active bins in bin t+1 per active bin in bin t.

    sigma = mean( n_active(t+1) / n_active(t) ) over bins t with
    n_active(t) > 0, excluding the final bin. sigma < 1 subcritical,
    sigma = 1 critical, sigma > 1 supercritical (Beggs & Plenz 2003).
    """
    active_counts = raster.sum(axis=0).astype(float)
    pairs = [(active_counts[t], active_counts[t + 1])
             for t in range(len(active_counts) - 1) if active_counts[t] > 0]
    if not pairs:
        return float("nan")
    return float(np.mean([child / parent for parent, child in pairs]))


def _discrete_powerlaw_alpha(values: np.ndarray, xmin: float) -> float:
    """MLE for a discrete power law P(x) ~ x^-alpha, x >= xmin.

    Uses the Clauset-Shalizi-Newman estimator (SIAM Rev 51, 2009): alpha is
    the unique root of

        d/dalpha ln zeta(alpha, xmin) = -mean(ln x | x >= xmin),

    where zeta is the Hurwitz zeta function and the derivative is obtained
    numerically (scipy has no analytic d/dalpha for zeta). The left side is
    strictly increasing in alpha, so the root is unique on (1, inf) whenever
    the empirical mean log exceeds ln(xmin) - 1 (see CSN eq. 3.6); outside
    that range we report NaN.
    """
    from scipy.special import zeta

    x = np.asarray(values, dtype=float)
    x = x[x >= xmin]
    if len(x) < 20:
        return float("nan")
    mean_log = float(np.mean(np.log(x)))
    lower = 1.0001

    # Feasibility: the root exists only if mean_log > ln(xmin) - 1/xmin ... the
    # exact CSN criterion is mean_log > (ln xmin) - 1 for xmin >= 1 roughly.
    # We simply extend the bracket adaptively until the objective changes sign
    # or the upper bound becomes unreasonably large.
    def objective(alpha: float) -> float:
        h = min(1e-3, (alpha - 1.0) / 8.0)      # stay off the zeta pole at alpha=1
        z_plus = zeta(alpha + h, xmin)
        z_minus = zeta(alpha - h, xmin)
        dln_zeta = float((np.log(z_plus) - np.log(z_minus)) / (2.0 * h))
        return dln_zeta + mean_log

    hi = 2.0
    while hi < 1e6 and not np.isfinite(objective(hi)):
        hi *= 2.0
    fhi = objective(hi)
    flo = objective(lower)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi >= 0.0:
        # No sign change: the data are not power-law above xmin (e.g. mean_log
        # too small -> alpha below 1, or too large -> beyond any bracket).
        if fhi < 0.0:          # root above the bracket: keep extending
            hi = 2.0
            while hi < 1e6:
                fhi = objective(hi)
                if fhi >= 0.0:
                    break
                hi *= 2.0
        if np.isfinite(fhi) and fhi >= 0.0 and flo * fhi < 0.0:
            return _brentq(objective, lower, hi)
        return float("nan")

    return _brentq(objective, lower, hi)


def _brentq(f: Callable[[float], float], lo: float, hi: float,
            tol: float = 1e-6) -> float:
    """Bracketed bisection / Brent search for a monotone root of f."""
    flo = f(lo)
    fhi = f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol or (hi - lo) < tol * 1e-3:
            return mid
        if fmid * flo < 0.0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)


def _ks_statistic(values: np.ndarray, alpha: float, xmin: float) -> float:
    """Kolmogorov-Smirnov distance between data and the fitted power law."""
    from scipy.special import zeta

    x = np.asarray(values, dtype=float)
    x = x[x >= xmin]
    if len(x) < 20 or not np.isfinite(alpha):
        return float("nan")

    # Empirical CDF above xmin.
    xs = np.sort(x)
    cdf_data = (np.arange(1, len(xs) + 1) / len(xs)).astype(float)

    # Theoretical CDF of a discrete power law: (zeta(alpha,xmin) - zeta(alpha,k+1)) / zeta(alpha,xmin).
    z0 = zeta(alpha, xmin)
    cdf_model = np.array(
        [1.0 - zeta(alpha, k + 1) / z0 for k in xs])
    return float(np.max(np.abs(cdf_data - cdf_model)))


def _fit_powerlaw(values: np.ndarray) -> Tuple[float, float, int, float]:
    """Fit a power law to `values`; return (alpha, xmin, n_tail, ks).

    xmin is chosen over the observed values to minimize the KS distance
    between the data and the MLE power law (Clauset-Shalizi-Newman 2009).
    """
    x = np.asarray(values, dtype=float)
    x = x[x >= 1.0]
    if len(x) < 20:
        return float("nan"), float("nan"), 0, float("nan")

    best = (float("nan"), float("nan"), 0, float("inf"))
    for xmin in np.unique(x):
        if (x >= xmin).sum() < 20:
            continue
        alpha = _discrete_powerlaw_alpha(x, xmin)
        if not np.isfinite(alpha):
            continue
        ks = _ks_statistic(x, alpha, xmin)
        if ks < best[3]:
            best = (alpha, xmin, int((x >= xmin).sum()), ks)
    return best


def _shuffled_raster(event_times: Sequence[np.ndarray], bin_s: float,
                     duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """A surrogate raster: every event time is re-drawn uniformly at random.

    Per-channel rates are preserved (the number of events per channel does
    not change) but all temporal correlations between and within channels are
    destroyed. If avalanches survive this, they are a binning artifact.
    """
    shuffled = []
    for times in event_times:
        n = len(times)
        if n == 0:
            shuffled.append(np.asarray([], dtype=float))
        else:
            shuffled.append(duration_s * rng.uniform(size=n))
    return _bin_raster(shuffled, bin_s, duration_s)


def _report(metrics: Dict[str, object]) -> str:
    """Format the analysis results into a compact human-readable block."""
    lines = []
    lines.append("=" * 62)
    lines.append("Criticality (avalanche) analysis")
    lines.append("=" * 62)
    lines.append(f"Channels used:            {metrics['n_channels']}")
    lines.append(f"Total events:             {metrics['n_events']}")
    lines.append(f"Bin width (ms):           {metrics['bin_ms']}")
    lines.append(f"Avalanches:               {metrics['n_avalanches']}")
    lines.append("")
    lines.append(f"Size exponent alpha:      {metrics['alpha']:.2f}   "
                 f"(critical ~ {CRIT_ALPHA})")
    lines.append(f"  xmin:                   {metrics['alpha_xmin']:.0f}  "
                 f"(n_tail={metrics['alpha_ntail']}, KS={metrics['alpha_ks']:.3f})")
    lines.append(f"Duration exponent tau:    {metrics['tau']:.2f}   "
                 f"(critical ~ {CRIT_TAU})")
    lines.append(f"Branching ratio sigma:    {metrics['sigma']:.3f}   "
                 f"(critical ~ {CRIT_SIGMA})")
    lines.append("")
    lines.append("Surrogate control (time-shuffled rasters):")
    lines.append(f"  sigma_shuf (mean):      {metrics['sigma_shuf_mean']:.3f} "
                 f"+/- {metrics['sigma_shuf_std']:.3f}")
    lines.append(f"  alpha_shuf (mean):      {metrics['alpha_shuf_mean']:.2f} "
                 f"+/- {metrics['alpha_shuf_std']:.2f}")
    if metrics.get("z_sigma") is not None:
        lines.append(f"  sigma z-score:          {metrics['z_sigma']:.2f} "
                     f"({metrics['z_sigma_note']})")
    lines.append("")
    return "\n".join(lines)


def _plot_avalanches(size_real: np.ndarray, size_shuf: np.ndarray,
                     duration_real: np.ndarray, alpha: float, xmin: float,
                     tau: float, out_path: Path) -> None:
    """Two-panel figure: size and duration distributions, real vs shuffled."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def hist_data(x: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        counts, _ = np.histogram(x, bins=bins)
        return counts, bins[:-1]

    # --- avalanche size (log-log, probability density) ---
    ax = axes[0]
    max_size = max(size_real.max() if len(size_real) else 1,
                   size_shuf.max() if len(size_shuf) else 1)
    bins = np.logspace(0, np.log10(max_size + 1), 40)
    counts_r, mids = hist_data(size_real, bins)
    density_r = counts_r / (bins[1:] - bins[:-1]) / len(size_real)
    counts_s, mids_s = hist_data(size_shuf, bins)
    density_s = counts_s / (bins[1:] - bins[:-1]) / len(size_shuf)
    mask = density_r > 0
    ax.loglog(mids[mask], density_r[mask], "o-", ms=4, label="real")
    mask_s = density_s > 0
    ax.loglog(mids_s[mask_s], density_s[mask_s], "s--", ms=4, label="shuffled")
    if np.isfinite(alpha):
        xs = np.linspace(xmin, max_size, 50)
        ax.loglog(xs, 0.1 * xs ** (-alpha), "k-", lw=1,
                  label=f"power law a={alpha:.2f}")
    ax.set_xlabel("avalanche size (bins)")
    ax.set_ylabel("probability density")
    ax.set_title("Avalanche size distribution")
    ax.legend()

    # --- duration ---
    ax = axes[1]
    max_dur = max(duration_real.max() if len(duration_real) else 1, 10)
    dbins = np.arange(1, int(max_dur) + 2) - 0.5
    counts_d, _ = np.histogram(duration_real, bins=dbins)
    mid_d = dbins[:-1] + 0.5
    mask_d = counts_d > 0
    ax.loglog(mid_d[mask_d], counts_d[mask_d] / len(duration_real), "o-",
              ms=4, label="real")
    if np.isfinite(tau):
        xs = np.linspace(1, max_dur, 50)
        ax.loglog(xs, 0.05 * xs ** (-tau), "k-", lw=1, label=f"power law tau={tau:.2f}")
    ax.set_xlabel("avalanche duration (bins)")
    ax.set_ylabel("probability")
    ax.set_title("Avalanche duration distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _compute(npz_path: Path, bin_ms: float, n_surrogates: int,
             min_events: int, seed: int) -> Dict[str, object]:
    """Run the full avalanche analysis; return a metrics dict."""
    rng = np.random.default_rng(seed)
    event_times = _load_event_times(npz_path, min_events)
    if not event_times:
        raise ValueError("no channels with >= min_events events")

    duration_s = max(t.max() for t in event_times) if event_times else 0.0
    bin_s = bin_ms / 1000.0
    raster = _bin_raster(event_times, bin_s, duration_s)

    sizes, durations = _find_avalanches(raster)
    sigma = _branching_parameter(raster)
    alpha, alpha_xmin, alpha_ntail, alpha_ks = _fit_powerlaw(sizes)
    tau, _, _, _ = _fit_powerlaw(durations)

    # --- surrogate control ---
    sigmas = np.zeros(n_surrogates)
    alphas = np.zeros(n_surrogates)
    for i in range(n_surrogates):
        shuf = _shuffled_raster(event_times, bin_s, duration_s, rng)
        s, _ = _find_avalanches(shuf)
        sigmas[i] = _branching_parameter(shuf)
        a, _, _, _ = _fit_powerlaw(s)
        alphas[i] = a

    sigma_shuf_mean = float(np.nanmean(sigmas))
    sigma_shuf_std = float(np.nanstd(sigmas))
    alpha_shuf_mean = float(np.nanmean(alphas))
    alpha_shuf_std = float(np.nanstd(alphas))
    if sigma_shuf_std > 0:
        z_sigma = (sigma - sigma_shuf_mean) / sigma_shuf_std
        z_note = ("no diff from chance" if abs(z_sigma) < 2
                  else "significant diff from chance")
    else:
        z_sigma, z_note = None, "n/a"

    return {
        "npz": str(npz_path),
        "bin_ms": bin_ms,
        "n_surrogates": n_surrogates,
        "min_events": min_events,
        "n_channels": len(event_times),
        "n_events": int(sum(len(t) for t in event_times)),
        "duration_s": float(duration_s),
        "n_avalanches": int(len(sizes)),
        "alpha": alpha,
        "alpha_xmin": alpha_xmin,
        "alpha_ntail": alpha_ntail,
        "alpha_ks": alpha_ks,
        "tau": tau,
        "sigma": sigma,
        "sigma_shuf_mean": sigma_shuf_mean,
        "sigma_shuf_std": sigma_shuf_std,
        "alpha_shuf_mean": alpha_shuf_mean,
        "alpha_shuf_std": alpha_shuf_std,
        "z_sigma": z_sigma,
        "z_sigma_note": z_note,
    }


def _sweep(npz_path: Path, bin_widths_ms: Sequence[float], min_events: int,
           seed: int) -> List[Dict[str, object]]:
    """Evaluate avalanche statistics across a range of bin widths.

    This is the Beggs-Plenz protocol in its usual form: instead of choosing
    one bin width a priori, the (sigma, alpha) pair is computed at every
    width and the trajectory is examined for a passage through the critical
    point (sigma ~ 1, alpha ~ 3/2). Criticality claims are only supported if
    the trajectory crosses near (1, 1.5) at a biologically meaningful width
    AND the surrogate sigma differs from chance at that width.
    """
    event_times = _load_event_times(npz_path, min_events)
    if not event_times:
        raise ValueError("no channels with >= min_events events")
    duration_s = max(t.max() for t in event_times)

    rows = []
    for bin_ms in bin_widths_ms:
        bin_s = bin_ms / 1000.0
        raster = _bin_raster(event_times, bin_s, duration_s)
        sizes, durations = _find_avalanches(raster)
        sigma = _branching_parameter(raster)
        alpha, xmin, ntail, ks = _fit_powerlaw(sizes)
        tau, _, _, _ = _fit_powerlaw(durations)
        rows.append({
            "bin_ms": bin_ms,
            "n_avalanches": int(len(sizes)),
            "sigma": sigma,
            "alpha": alpha,
            "alpha_xmin": xmin,
            "alpha_ntail": ntail,
            "alpha_ks": ks,
            "tau": tau,
        })
    return rows


def _plot_sweep(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    """Plot the (sigma, alpha) trajectory as bin width varies."""
    sigmas = [r["sigma"] for r in rows]
    alphas = [r["alpha"] for r in rows]
    bins = [r["bin_ms"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(sigmas, alphas, "o-", ms=5)
    ax.axvline(CRIT_SIGMA, color="r", ls="--", lw=1, label="critical sigma=1")
    ax.axhline(CRIT_ALPHA, color="g", ls="--", lw=1, label="critical alpha=1.5")
    for s, a, b in zip(sigmas, alphas, bins):
        ax.annotate(f"{b:g}ms", (s, a), textcoords="offset points",
                    xytext=(4, 4), fontsize=8)
    ax.set_xlabel("branching ratio sigma")
    ax.set_ylabel("size exponent alpha")
    ax.set_title("(sigma, alpha) trajectory vs bin width")
    ax.legend()

    ax = axes[1]
    ax.semilogx(bins, sigmas, "o-", ms=5, label="sigma")
    ax.axhline(CRIT_SIGMA, color="r", ls="--", lw=1)
    ax2 = ax.twinx()
    ax2.semilogx(bins, alphas, "s--", ms=5, color="g", label="alpha")
    ax.set_xlabel("bin width (ms)")
    ax.set_ylabel("sigma", color="b")
    ax2.set_ylabel("alpha", color="g")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    npz_path = Path(args.output)
    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")

    metrics = _compute(npz_path, bin_ms=args.bin_ms,
                       n_surrogates=args.surrogates,
                       min_events=args.min_events, seed=args.seed)
    print(_report(metrics))

    out_root = Path(args.out_dir or npz_path.parent)
    out_dir = out_root / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Bin-width sweep: the standard Beggs-Plenz trajectory test.
    if args.sweep:
        widths = np.geomspace(args.sweep_min_ms, args.sweep_max_ms,
                              args.sweep_steps)
        rows = _sweep(npz_path, list(widths), args.min_events, args.seed)
        (out_dir / "sweep.json").write_text(json.dumps(rows, indent=2))
        _plot_sweep(rows, out_dir / "bin_sweep.png")
        print("\nBin-width sweep (sigma / alpha vs bin width):")
        print(f"  {'bin_ms':>9} {'sigma':>7} {'alpha':>7} {'xmin':>5} "
              f"{'n_av':>6} {'KS':>6}")
        for r in rows:
            print(f"  {r['bin_ms']:>9.1f} {r['sigma']:>7.3f} "
                  f"{r['alpha']:>7.2f} {r['alpha_xmin']:>5.0f} "
                  f"{r['n_avalanches']:>6d} {r['alpha_ks']:>6.3f}")
        print(f"  Saved sweep JSON + figure to: {out_dir}")

    # Recompute the distributions once more for the figure (cheap: no surrogates).
    event_times = _load_event_times(npz_path, args.min_events)
    duration_s = max(t.max() for t in event_times)
    bin_s = args.bin_ms / 1000.0
    rng = np.random.default_rng(args.seed)
    size_shuf = []
    for _ in range(args.surrogates):
        shuf = _shuffled_raster(event_times, bin_s, duration_s, rng)
        s, _ = _find_avalanches(shuf)
        size_shuf.extend(s)
    raster = _bin_raster(event_times, bin_s, duration_s)
    sizes, durations = _find_avalanches(raster)
    _plot_avalanches(sizes, np.asarray(size_shuf), durations,
                     metrics["alpha"], metrics["alpha_xmin"], metrics["tau"],
                     out_dir / "avalanche_distributions.png")

    print(f"\nSaved metrics + figure to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Criticality / avalanche analysis of fungal MEA events "
                    "(Beggs-Plenz avalanche framework + surrogate control)")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to a run's waveforms/waveforms.npz")
    parser.add_argument("-b", "--bin-ms", type=float, default=DEFAULT_BIN_MS,
                        help="Avalanche bin width in ms (default %(default)s)")
    parser.add_argument("-s", "--surrogates", type=int, default=DEFAULT_N_SURROGATES,
                        help="Number of shuffled rasters for the control "
                             "(default %(default)s)")
    parser.add_argument("--min-events", type=int, default=MIN_CHANNEL_EVENTS,
                        help="Drop channels with fewer events (default %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the surrogate shuffling")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: the npz's directory)")
    parser.add_argument("--sweep", action="store_true",
                        help="Also sweep bin widths and report the "
                             "(sigma, alpha) trajectory")
    parser.add_argument("--sweep-min-ms", type=float, default=25.0,
                        help="Smallest sweep bin width in ms (default %(default)s)")
    parser.add_argument("--sweep-max-ms", type=float, default=2000.0,
                        help="Largest sweep bin width in ms (default %(default)s)")
    parser.add_argument("--sweep-steps", type=int, default=12,
                        help="Number of log-spaced sweep widths (default %(default)s)")
    args = parser.parse_args()
    main(args)
