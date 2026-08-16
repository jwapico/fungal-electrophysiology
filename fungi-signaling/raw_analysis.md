# raw_analysis.py — MEA compound-event waveform extraction pipeline

Companion document to `fungi-signaling/raw_analysis.py` (rev 5). It describes
every computation performed by the pipeline, the mathematical form of each
operation, why each design choice was made, the biological reasoning behind
the parameters, and a justification for every hardcoded constant.

The pipeline converts a raw multielectrode-array (MEA) recording of fungal
electrical activity into a set of extracted multi-oscillation "events" per
channel, one waveform window per event, and a set of human-reviewable HTML
views. The extracted waveforms are the single source of truth for every
downstream step: `spike_sorting.py` (family sorting) and all visualization.

---

## 1. Inputs and basic signal model

### 1.1 Raw binary format

The recording is a flat little-endian `int16` stream of interleaved samples:

```
y[n] = sample n of the interleaved stream
channel index   k = n mod 64
time index      t = n // 64
```

De-interleaving is a single reshape:

```
Y[t, k] = y[64*t + k],   t = 0..T-1,  k = 0..63
```

The file is memory-mapped (`np.memmap`) rather than loaded into RAM: a
900 s recording at 30 kHz with 64 channels is ~3.4 GB on disk
(900 · 30000 · 64 · 2 bytes), and streaming it avoids exhausting memory.

### 1.2 Digital-to-analog scaling

Raw ADC units are converted to microvolts:

```
v_k[t] = Y[t, k] · q,   q = VOLTAGE_SCALE = 0.195 µV/LSB
```

All downstream thresholds are expressed in microvolts; the noise estimate
and the gates derived from it are therefore physical quantities, not ADC
counts, which keeps the pipeline insensitive to the digitizer gain.

Constants in this block:

| Constant | Value | Rationale |
|---|---|---|
| `SAMPLE_RATE_HZ` | 30 000 | hardware sampling rate of the MEA system |
| `NUM_CHANNELS` | 64 | electrode layout of the array |
| `VOLTAGE_SCALE` | 0.195 µV/LSB | ADC gain of the acquisition hardware |
| `RAW_DATA_FILE` | `../data/raw_mea_bins/recording_control_0_cut800s.bin` | default dataset |

---

## 2. Noise floor estimation (MAD)

The channel noise scale is estimated once per channel with the median
absolute deviation:

```
sigma_hat(v) = 1.4826 · median_n |v[n] - median_m v[m]|
```

The constant `1.4826 = 1/Φ⁻¹(3/4)` (inverse normal CDF at the 3rd
quartile) makes `sigma_hat` a consistent, unbiased estimator of the
Gaussian standard deviation σ:

```
1.4826 · median |X| = σ    when X ~ N(0, σ²)
```

because `median|X| = σ·Φ⁻¹(3/4)`.

**Why MAD and not the standard deviation:** the recording is dominated by
quiescent baseline with occasional large excursions (the very events we
want to detect). The sample standard deviation is inflated by those
excursions and would overstate the noise, pushing the detection gates too
high and suppressing the events we seek. The MAD is robust to the few large
samples (breakdown point 50%), so it estimates the *baseline* noise, not the
contaminated mixture. This choice is biologically motivated: fungal MEA
traces contain sparse, large compound potentials, and the gate must be set
relative to the *resting* noise, not the average including the events.

---

## 3. Event detection

The trace is segmented into events by a two-gate scheme:

- a **low envelope gate** (`EVENT_GATE_SCALE · σ`) that defines candidate
  excursion intervals, and
- a **high spike gate** (`SPIKE_GATE_SCALE · σ`) that an event's dominant
  deflection must clear.

### 3.1 Excursion set

```
E = { n : |v[n]| ≥ G·σ },   G = EVENT_GATE_SCALE
```

### 3.2 Maximal runs

`E` is decomposed into maximal contiguous runs `[s_j, e_j]`. This is done
with `np.diff` on the indices where `|v| > gate`: breaks occur wherever
successive gate-crossing indices are not adjacent.

### 3.3 Gap merging

Runs that are close enough are fused into one event:

```
run j+1 merges into run j  iff  s_{j+1} - e_j - 1 ≤ τ_gap·f_s/1000
```

with `τ_gap = EVENT_GAP_MS = 5 ms`. A compound fungal potential often dips
back below the envelope gate for a few milliseconds between oscillations
(brief repolarization crossings); merging treats such a dip as part of one
compound event rather than two separate events.

### 3.4 Event gate

A merged run `[S_i, E_i]` becomes an event iff:

```
(E_i - S_i + 1) ≥ τ_min·f_s/1000            (MIN_EVENT_MS = 0.5 ms)
|v[m_i]| ≥ S·σ                               (SPIKE_GATE_SCALE = 4.0)
m_i = argmax_{n ∈ [S_i,E_i]} |v[n]|
```

The minimum duration discards envelope blips that are too short to be a
biological potential. The spike gate ensures the event contains at least
one deflection strong enough to be a genuine spike — it must clear 4× the
noise — so noise crossings that merely exceed the 5σ envelope gate but
never produce a real spike are rejected.

`m_i`, the **dominant deflection**, is the sample of largest absolute
voltage in the event. It is stored per event and later used as the
alignment anchor for the extraction window and as the event time.

---

## 4. Oscillation counting

`count_oscillations(excursion, noise)` counts the number of major
oscillations inside an event's excursion segment `v[S_i:E_i]`.

### 4.1 Extrema

All local maxima and minima of the segment are collected (via
`scipy.signal.find_peaks` on the signal and on its negation), each tagged
with its polarity (+1 peak, −1 trough).

### 4.2 Pass A — sign alternation + absolute swing

Let `{e_k}` be the extrema in time order and
`s_k = |x[e_{k+1}] - x[e_k]|` the swing between consecutive extrema. A
turning point is kept when:

1. it **alternates sign** with the previously kept one, and
2. its swing to that neighbour satisfies

```
s_k ≥ c·σ,   c = OSC_MIN_SWING_SIGMAS = 5.0
```

The swing threshold is **absolute in noise units**, not a fraction of the
largest swing. This is deliberate: a relative threshold would let one huge
dominant deflection dwarf all other oscillations in the event (a common
failure mode of "count everything above x% of the max" schemes), whereas an
absolute threshold in noise units counts every deflection that is genuinely
large relative to the noise floor. The sign-alternation rule prevents
counting ripple that sits on a plateau (small bumps within a long monotonic
dip).

### 4.3 Pass B — minimum spacing

Turning points must be at least `τ_period = OSC_MIN_PERIOD_MS = 1 ms` apart.
When two surviving points are too close, the one with the larger swing is
kept and the other dropped. This enforces a physiological refractory
minimum: distinct oscillations of a real compound potential are separated by
at least ~1 ms, so tighter alternations are treated as a single peak.

### 4.4 Cycle count

`K` significant turning points span `K-1` swings = `K-1` half cycles:

```
N = max(1, ceil((K-1)/2))
```

The count is therefore the number of full (plus any trailing half) cycles
spanned by the significant turning points, floored at 1.

**Biological validity:** oscillation counts are computed only over the
*excursion* (onset…offset), not over the padded window, so pre/post-event
baseline micro-activity is never counted as oscillation. The count is a
feature used downstream for family sorting and is stored as
`n_oscillations` per event.

---

## 5. Window extraction

Each event gets exactly one waveform window, extracted by
`extract_event_window`.

### 5.1 Extent

The *extent* is the maximal contiguous span around the excursion on which
`|v|` stays above a low threshold:

```
[L_i, R_i] = maximal run containing [S_i, E_i] with |v[n]| > e·σ
e = EXTENT_SIGMAS = 2.0
```

The detection gate is 5σ; many oscillations *belonging* to the event have
amplitudes between 2σ and 5σ and are missed by the excursion. Walking the
span outward while `|v| > 2σ` recovers these low-amplitude leading and
trailing oscillations, so the window covers the whole compound potential
rather than only its loudest part.

### 5.2 Adaptive padding

The window is padded symmetrically, proportional to the extent, but never
below a floor:

```
pre = post = max(round(PAD_FRACTION · L_ext), round(MIN_PAD_S · f_s))
L_ext = R_i - L_i + 1
```

with `PAD_FRACTION = 0.25` and `MIN_PAD_S = 0.02 s`.

The proportionality is adaptive: a small event gets a small margin, a large
one gets a larger margin, so every window carries a similar *relative* amount
of baseline around the potential. The `MIN_PAD_S` floor guarantees at least
20 ms of baseline on each side even for very short events, which is needed
for stable noise estimation and family-sorting alignment.

### 5.3 Natural and final window size

```
W_nat = L_ext + pre + post = L_ext + 2·max(round(PAD_FRACTION·L_ext), round(MIN_PAD_S·f_s))
W_i   = max(W_nat, w_min),   w_min = MIN_WINDOW_MS·f_s/1000 = 3 ms
```

There is **no upper clipping**: a large extent keeps its full, proportionate
window. Clipping at the top would truncate large compound potentials, and
since downstream alignment is to the dominant peak, oversized windows are
harmless.

### 5.4 Centering on the dominant deflection

The window is centered on the dominant deflection so every event's peak
lands at the same relative position:

```
start_i = clamp(m_i - floor(W_i/2), 0, len(v) - W_i)
waveform_i = v[start_i : start_i + W_i]
```

Alignment on the peak (not the onset) is required by family sorting:
variation in onset timing is absorbed by the padding, and peaks line up
across events. The peak offset `r_i = m_i - start_i` is stored.

Returns `(waveform, t_i, W_i, start_i, r_i)` with event time
`t_i = m_i/f_s`.

---

## 6. Smoothing (Savitzky-Golay), persisted once

Every raw window is smoothed at extraction time with a zero-phase
Savitzky-Golay filter and **both copies are stored in the npz**:

```
smooth_waveform(w) = savgol_filter(w, window_len, polyorder)
```

- `window_len = max(oddify(round(SMOOTH_WINDOW_MS · f_s/1000)), polyorder+1)`,
  with `SMOOTH_WINDOW_MS = 2.0 ms` → 60 samples, forced odd (60 → 61), and
  `SMOOTH_POLYORDER = 4`.
- If `window_len < 3` or `window_len ≥ len(waveform)`, the window is
  returned unchanged (filter not applicable).

**Why Savitzky-Golay:** for each sample, a least-squares polynomial of
degree `polyorder` is fitted to the symmetric window around it and the
fitted value at the center is kept. This tracks the smooth macro-deflection
and suppresses high-frequency noise while keeping peak amplitudes
approximately intact — a moving average or plain FIR low-pass would flatten
peaks and round corners, which is exactly what we do *not* want when the
peak is the biological signal of interest. The kernel is symmetric, hence
**zero-phase**: peak locations cannot shift relative to the raw waveform,
so alignment (`r_i`) and event times are unaffected by smoothing.

**Single source of truth:** smoothing is computed exactly once, during
`process_channel`. The smoothed arrays are persisted alongside the raw ones
and are read back by every visualization and by `spike_sorting.py`. Nothing
is re-derived at render time, so the displayed/sorted data always equals
the persisted data, even when the module constants change between a run and
its re-render.

---

## 7. Persistence (npz archive)

`save_waveforms` writes one self-describing, compressed `.npz` archive at
`outputs/<ts>/waveforms/waveforms.npz` containing:

- **per-channel object arrays** (one row per channel; object dtype because
  event counts and window lengths vary):
  - `waveforms[i]` — raw windows of channel i (list of arrays)
  - `smooth_waveforms[i]` — Savitzky-Golay smoothed copies of the same windows
  - `spike_times`, `window_sizes`, `peak_positions`, `window_starts`,
    `n_oscillations`, `amplitudes` — the per-event features
- **per-channel scalar arrays:** `thresholds` (spike gate), `gates`
  (envelope gate), `stds` (noise σ), `n_events`, `n_extracted`
- **scalar metadata:** `sample_rate`, `unit` (`"uV"`), `source_file`, and
  the full parameter set used (`min_window_ms`, `min_pad_s`, `pad_fraction`,
  `extent_sigmas`, `event_gate_scale`, `spike_gate_scale`, `smooth_method`,
  `smooth_window_ms`, `smooth_polyorder`, `smooth_show_by_default`)

`load_waveforms` is the exact inverse and also:

- reads the **smoothing parameters recorded in the archive** (so re-renders
  label the data truthfully even if module constants changed), falling back
  to the current module constants only for legacy archives that lack them;
- handles archives that predate smoothing (no `smooth_waveforms` key): the
  smoothed copy mirrors the raw one and a warning is printed.

The archive is the single source of truth for every downstream consumer.

---

## 8. Output layout and re-rendering

Each run writes into its own timestamped directory (old runs are never
overwritten):

```
outputs/<YYYY-MM-DD_HH-MM-SS>/
  waveforms/waveforms.npz         persisted raw + smoothed windows + features
  html/waveforms_grid.html        per-event tile grid
  html/all_ch_spikes.html         full-trace overview per channel
  html/interactive_ch_views/channel_N_interactive.html   plotly click-to-zoom
  run_meta.json                   parameters + per-channel summary
```

`-v` (visualize-only) re-renders the HTML from a previous run's `.npz` in
place, without re-reading the raw binary and without re-deriving any
waveform. The run directory is recovered from the `.npz` path.

### 8.1 Waveform grid (`gen_spike_waveform_html`)

Flex/CSS grid of one tile per event. Each tile embeds two PNGs — the raw
window and its **saved** smoothed copy (both from `results[]`, i.e. the
npz) — with identical y-limits so raw vs smoothed is comparable. A checkbox
toggles between them; another limits the visible tiles to the first
`SPIKE_WINDOWS_LIMIT = 50` per channel. Each tile links (`?t0=..&t1=..`)
to the channel's interactive view, pre-zoomed to that event.

### 8.2 Channel overviews

`gen_channel_html` renders one full-trace PNG per channel, decimated by
`CHANNEL_DS_FACTOR = 10`, with dominant peaks overlaid as red dots and the
±envelope/±spike gates drawn as dashed lines (gates derived from the
*persisted* noise σ of that channel).

`gen_channel_interactive_html` builds a self-contained plotly view with a
heavily decimated overview (`INTERACTIVE_OVERVIEW_DS = 200`), a set of
high-resolution context segments around each peak (windowed
`SPIKE_CONTEXT_MS = 200 ms`, decimated by `INTERACTIVE_SPIKE_DS = 4`,
joined by NaN gaps so no line bridges across segments), red peak markers,
and the two gates. A small injected script applies the URL zoom range.

### 8.3 Output index (`write_output_index`)

Every run (fresh or `-v`) regenerates `outputs/index.html`, a fully static
entry page with plain relative links: the newest run first (grid, all
channels, `run_meta.json`) followed by every older run. It works when the
file is opened directly from disk — no server and no JavaScript required.

---

## 9. Justification of the hardcoded constants

Detection and windowing parameters were deliberately made module constants
rather than user-facing knobs so that every run records them in both the
npz and `run_meta.json`, making each archived run fully reproducible and
cross-referenceable when writing a methods section.

| Constant | Value | Argument |
|---|---|---|
| `EVENT_GATE_SCALE` | 5.0 | 5σ envelope gate — near-universal spike-detection convention (Quiroga et al.); robust against noise crossings (P(|noise|>5σ) ≈ 6e-7 per sample) while capturing all real potentials |
| `SPIKE_GATE_SCALE` | 4.0 | an event must contain at least one ≥4σ deflection; rejects excursions that are purely sub-spike-threshold |
| `EVENT_GAP_MS` | 5.0 | merges intra-compound dips ≤5 ms into one event; fungal compound potentials can cross below the gate between oscillations |
| `MIN_EVENT_MS` | 0.5 | removes envelope blips too short to be biological; below ~0.5 ms a deflection is not resolvable as an action potential |
| `OSC_MIN_SWING_SIGMAS` | 5.0 | only oscillations whose swing exceeds 5σ are counted — same noise-clearance logic as the detection gate |
| `OSC_MIN_PERIOD_MS` | 1.0 | physiological minimum oscillation period; enforces refractory spacing between counted turning points |
| `EXTENT_SIGMAS` | 2.0 | captures low-amplitude (2–5σ) leading/trailing oscillations missed by the 5σ detection gate |
| `PAD_FRACTION` | 0.25 | adaptive padding: relative baseline ~12.5% of extent length per side |
| `MIN_PAD_S` | 0.02 s | floor so every window has ≥20 ms baseline each side (stable noise/alignment) |
| `MIN_WINDOW_MS` | 3.0 | absolute lower bound on window width |
| `SMOOTH_WINDOW_MS` | 2.0 | savgol window ≈ 2 ms (60 samples): smooths noise without touching spike features (~0.2–1 ms) |
| `SMOOTH_POLYORDER` | 4 | quartic fit preserves peak shape; lower orders over-flatten, higher orders track noise |
| `CHANNEL_DS_FACTOR` | 10 | overview trace decimation (visually clean, small PNGs) |
| `SPIKE_WINDOWS_LIMIT` | 50 | default tiles shown per channel in the grid; a toggle reveals all |

---

## 10. Per-event persisted features (schema)

For each event `i` of channel `k`:

| Key | Meaning |
|---|---|
| `spike_times` | event time `t_i = m_i/f_s` (s) |
| `window_sizes` | `W_i` (samples) |
| `peak_positions` | `r_i = m_i - start_i`, offset of dominant peak inside the window |
| `window_starts` | `start_i`, absolute sample index of window start |
| `n_oscillations` | `N_i` from `count_oscillations` |
| `amplitudes` | signed dominant deflection `v_k[m_i]` (µV) |
| `waveforms` | raw window (µV, `float64`) |
| `smooth_waveforms` | Savitzky-Golay smoothed copy of the window (µV, `float64`) |
| `threshold` / `gate` / `std_dev` | spike gate, envelope gate, noise σ (µV) |

All values are physical quantities in microvolts and seconds, so downstream
analyses and figures carry no unit ambiguity.
