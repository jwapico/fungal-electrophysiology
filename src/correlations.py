"""
correlations.py

Cross-channel analysis of fungal MEA recordings. Intended home for a "gamut"
of analyses (synchrony, phase-locking, propagation delays, cross-correlation
lags, etc.). Current scope (v2):

  - HEATMAP VIDEO of the voltage dynamics. The 64 channels are laid out as an
    8x8 grid in ROW-MAJOR order: channel c sits at (row = c // 8, col = c % 8),
    so the first row is channels 0..7, the second 8..15, and so on.
  - Each frame shows a per-channel voltage statistic over an analysis window:
        rms      sqrt(mean(v^2))                 (default; activity envelope)
        absmean  mean(|v|)
        range    max(v) - min(v)                 (peak-to-peak)
        mad      1.4826 * median(|v - median(v)|)
    measured in microvolts (VOLTAGE_SCALE uV/LSB).
  - Frame timing. Frames step through the recording at `--fps` frames/sec. By
    default fps is AUTO = 1000 / window_ms, so every frame covers exactly one
    window and consecutive frames tile the recording contiguously: the video
    plays in REAL TIME (its duration always equals the analyzed recording
    length, whatever fps you choose -- fps changes temporal resolution, not
    playback length). A lower fps renders fewer frames (faster) with the same
    playback duration; a larger window smooths each frame.
  - Output: MP4 via ffmpeg (streamed to stdin -- no intermediate PNG files) or
    GIF via Pillow, plus metrics.json. The mp4 codec is real-time capable and
    handles 80k+ frames; GIF does not (256 colors, memory-heavy), so mp4 is
    the default when ffmpeg is on PATH.

Channel ordering note: the layout is purely positional -- channel index c is
placed at grid cell (c // 8, c % 8). No electrode-map file is consulted; if
the MEA headstage uses a different physical arrangement this mapping is the
single place to change (GRID_ROWS / GRID_COLS / the layout helper).

Run (from fungi-signaling/):
    python correlations.py                                  # 10 ms real-time mp4
    python correlations.py --measure range --t-end 60
    python correlations.py --fps 25 --window-ms 10          # fewer frames
    python correlations.py --video gif
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import numpy as np
from PIL import Image

from raw_analysis import (
    NUM_CHANNELS,
    RAW_DATA_FILE,
    SAMPLE_RATE_HZ,
    VOLTAGE_SCALE,
    load_raw_data,
)

# ---------------------------------------------------------------------------
# layout & defaults
# ---------------------------------------------------------------------------
GRID_ROWS: int = 8
GRID_COLS: int = 8
DEFAULT_MEASURE: str = "rms"
DEFAULT_WINDOW_MS: float = 10.0
OUTPUT_ROOT: str = "outputs"
TIMESTAMP_FORMAT: str = "%Y-%m-%d_%H-%M-%S"
FRAME_DIR: str = "frames"
PIX_W: int = 800          # video pixel width (even -> yuv420p-safe)
PIX_H: int = 800          # video pixel height (even -> yuv420p-safe)

MEASURES: Tuple[str, ...] = ("rms", "absmean", "range", "mad")


def frame_measure(seg: np.ndarray, name: str) -> np.ndarray:
    """Per-channel statistic of a (window_n, 64) voltage segment -> (64,) uV.

    `seg` must be float (the raw int16 is cast before calling to avoid
    int32 overflow in the squaring for rms).
    """
    if name == "rms":
        return np.sqrt(np.mean(seg * seg, axis=0))
    if name == "absmean":
        return np.mean(np.abs(seg), axis=0)
    if name == "range":
        return np.max(seg, axis=0) - np.min(seg, axis=0)
    if name == "mad":
        med = np.median(seg, axis=0)
        return 1.4826 * np.median(np.abs(seg - med), axis=0)
    raise ValueError(f"unknown measure {name!r}; choose from {MEASURES}")


def channel_grid(measure: np.ndarray) -> np.ndarray:
    """Place a length-64 per-channel measure vector into an 8x8 grid.

    channel c -> row c // GRID_COLS, col c % GRID_COLS (row-major order).
    """
    if measure.shape != (NUM_CHANNELS,):
        raise ValueError(f"expected {NUM_CHANNELS} channels, got {measure.shape}")
    return measure.reshape(GRID_ROWS, GRID_COLS)


def auto_fps(window_ms: float) -> float:
    """Real-time frame rate: one contiguous window per frame."""
    return 1000.0 / window_ms


def _progress(done: int, total: int, what: str, t0: float) -> None:
    """Overwrite a single terminal line with % done + ETA."""
    pct = 100.0 * done / total
    elapsed = time.time() - t0
    eta = elapsed / done * (total - done) if done else 0.0
    sys.stdout.write(
        f"\r  {what}: {done}/{total} frames ({pct:5.1f}%)  "
        f"[elapsed {elapsed:6.1f}s, ETA {eta:6.1f}s]"
    )
    sys.stdout.flush()


def compute_frames(recording: np.ndarray, window_ms: float, fps: float,
                   t_start_s: float, t_end_s: float,
                   measure_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the per-channel statistic for every frame.

    Returns (frames, times): frames is (n_frames, 64) with each row the
    per-channel measure in uV; times is the frame midpoint in seconds.
    """
    fs = SAMPLE_RATE_HZ
    window_n = max(1, int(round(window_ms / 1000.0 * fs)))
    step_n = max(1, int(round(fs / fps)))

    t_start = max(0.0, t_start_s)
    t_end = min(recording.shape[0] / fs, t_end_s)
    i0 = int(t_start * fs)
    n_frames = max(1, int((t_end - t_start) * fps))

    frames = np.empty((n_frames, NUM_CHANNELS), dtype=np.float64)
    times = np.empty(n_frames, dtype=np.float64)
    t0 = time.time()
    for f in range(n_frames):
        i = i0 + f * step_n
        seg = recording[i:i + window_n].astype(np.float64)
        frames[f] = frame_measure(seg, measure_name) * VOLTAGE_SCALE
        times[f] = (i + window_n / 2.0) / fs
        if f % 200 == 0 or f == n_frames - 1:
            _progress(f + 1, n_frames, "computing", t0)
    sys.stdout.write("\n")
    return frames, times


def _set_up_axes(ax, measure_name: str, vmin: float, vmax: float, cmap, norm):
    """Configure an 8x8 heatmap whose gridlines sit exactly on tile edges."""
    im = ax.imshow(channel_grid(np.zeros(NUM_CHANNELS)), cmap=cmap, norm=norm,
                   interpolation="nearest", origin="lower", aspect="equal",
                   extent=[-0.5, GRID_COLS - 0.5, -0.5, GRID_ROWS - 0.5])
    ax.set_xlim(-0.5, GRID_COLS - 0.5)
    ax.set_ylim(-0.5, GRID_ROWS - 0.5)
    ax.set_xticks(np.arange(GRID_COLS))
    ax.set_yticks(np.arange(GRID_ROWS))
    ax.set_xticklabels([str(c) for c in range(GRID_COLS)])
    ax.set_yticklabels([f"{r * GRID_COLS}-{r * GRID_COLS + GRID_COLS - 1}"
                        for r in range(GRID_ROWS)])
    # minor ticks at tile boundaries -> gridlines aligned with tile edges
    ax.set_xticks(np.arange(-0.5, GRID_COLS), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_ROWS), minor=True)
    ax.grid(True, which="minor", color="white", linewidth=0.6, alpha=0.4)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("column")
    ax.set_ylabel("row (channels)")
    ax.set_title("", fontsize=11)
    return im


def render_video(frames: np.ndarray, times: np.ndarray, out_dir: Path,
                 measure_name: str, fps: float, vmin: float, vmax: float,
                 video: str = "mp4", dpi: int = 100,
                 keep_frames: bool = False) -> Path:
    """Render the heatmap video; returns the video path."""
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(PIX_W / dpi, PIX_H / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    im = _set_up_axes(ax, measure_name, vmin, vmax, cmap, norm)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(f"{measure_name} voltage (uV)")
    title = ax.set_title("", fontsize=11)

    out_path: Path = out_dir / "heatmap.mp4"

    frame_dir = None
    if video == "gif" or keep_frames:
        frame_dir = out_dir / FRAME_DIR
        frame_dir.mkdir(parents=True, exist_ok=True)

    proc = None
    ffmpeg_stdin_pipe: "subprocess.Popen[bytes] | None" = None
    if video == "mp4":
        if not _have_ffmpeg():
            raise RuntimeError("ffmpeg not found on PATH; use --video gif")
        out_path = out_dir / "heatmap.mp4"
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{PIX_W}x{PIX_H}", "-r", str(fps),
             "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "18", "-preset", "medium", str(out_path)],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        ffmpeg_stdin_pipe = proc

    png_paths: List[Path] = []
    n = len(frames)
    t0 = time.time()
    for f in range(n):
        im.set_data(channel_grid(frames[f]))
        title.set_text(f"t = {times[f]:6.1f} s")
        fig.canvas.draw()
        if ffmpeg_stdin_pipe is not None and ffmpeg_stdin_pipe.stdin is not None:
            canvas = cast(FigureCanvasAgg, fig.canvas)
            buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
            if buf.shape[0] % 2:      # keep yuv420p happy (even dims)
                buf = buf[:-1]
            if buf.shape[1] % 2:
                buf = buf[:, :-1]
            ffmpeg_stdin_pipe.stdin.write(buf.tobytes())
        if frame_dir is not None:
            png_path = frame_dir / f"frame_{f:06d}.png"
            fig.savefig(png_path)
            png_paths.append(png_path)
        if f % 200 == 0 or f == n - 1:
            _progress(f + 1, n, "rendering", t0)
    sys.stdout.write("\n")

    plt.close(fig)

    if video == "gif":
        out_path = out_dir / "heatmap.gif"
        ims = [Image.open(p).convert("RGB") for p in png_paths]
        ims[0].save(out_path, save_all=True, append_images=ims[1:],
                    duration=int(1000.0 / fps), loop=0)
        del ims
    elif proc is not None:
        assert proc.stdin is not None
        proc.stdin.close()  # type: ignore[union-attr]
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")
    return out_path


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def analyze(recording: np.ndarray, window_ms: float, fps: float,
            t_start_s: float, t_end_s: float, measure_name: str,
            vmin: float, vmax: float, video: str, out_dir: Path,
            dpi: int, keep_frames: bool) -> Dict[str, object]:
    """Run the heatmap-video analysis and return a summary dict."""
    frames, times = compute_frames(recording, window_ms, fps, t_start_s,
                                   t_end_s, measure_name)
    if vmin is None:
        vmin = float(np.percentile(frames, 1.0))
    if vmax is None:
        vmax = float(np.percentile(frames, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = render_video(frames, times, out_dir, measure_name, fps,
                              vmin, vmax, video=video, dpi=dpi,
                              keep_frames=keep_frames)

    summary = {
        "source": str(RAW_DATA_FILE),
        "measure": measure_name,
        "window_ms": window_ms,
        "fps": fps,
        "t_start_s": float(times[0]),
        "t_end_s": float(times[-1]),
        "n_frames": len(frames),
        "video_length_s": len(frames) / fps,
        "vmin_uv": vmin,
        "vmax_uv": vmax,
        "grid": f"{GRID_ROWS}x{GRID_COLS} row-major (c -> r=c//{GRID_COLS}, "
                f"col=c%{GRID_COLS})",
        "video": str(video_path),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def main(args: argparse.Namespace) -> None:
    recording = load_raw_data(args.raw)
    out_dir = Path(args.out_dir or OUTPUT_ROOT)
    if args.out_dir is None:
        import datetime as _dt
        out_dir = out_dir / _dt.datetime.now().strftime(TIMESTAMP_FORMAT) / "correlations"

    fps = args.fps if args.fps else auto_fps(args.window_ms)
    video = args.video
    if video == "auto":
        video = "mp4" if _have_ffmpeg() else "gif"

    summary = analyze(recording, window_ms=args.window_ms, fps=fps,
                      t_start_s=args.t_start, t_end_s=args.t_end,
                      measure_name=args.measure, vmin=args.vmin, vmax=args.vmax,
                      video=video, out_dir=out_dir, dpi=args.dpi,
                      keep_frames=args.keep_frames)

    print("=" * 60)
    print("Correlations: heatmap video")
    print("=" * 60)
    print(f"  measure:            {summary['measure']}")
    print(f"  window:             {summary['window_ms']} ms")
    print(f"  fps:                {summary['fps']:g}")
    print(f"  time range:         {summary['t_start_s']:.1f} - "
          f"{summary['t_end_s']:.1f} s")
    print(f"  frames:             {summary['n_frames']:,}")
    print(f"  video length:       {_fmt_duration(cast(float, summary['video_length_s']))}")
    print(f"  color scale:        {summary['vmin_uv']:.1f} - "
          f"{summary['vmax_uv']:.1f} uV")
    print(f"  video:              {summary['video']}")
    if args.keep_frames or video == "gif":
        print(f"  frames dir:         {out_dir / FRAME_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-channel analyses of fungal MEA recordings. v2: "
                    "8x8 heatmap video of voltage dynamics (channel c at "
                    "row c//8, col c%8), real-time by default.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-r", "--raw", type=str, default=RAW_DATA_FILE,
                        help="Path to raw MEA binary recording")
    parser.add_argument("-o", "--out-dir", type=str, default=None,
                        help="Output directory; default outputs/<ts>/correlations")
    parser.add_argument("--measure", type=str, default=DEFAULT_MEASURE,
                        choices=MEASURES,
                        help="Per-channel voltage statistic per frame "
                             "(default %(default)s)")
    parser.add_argument("-w", "--window-ms", type=float, default=DEFAULT_WINDOW_MS,
                        help="Analysis window per frame (default %(default)s ms)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Frame rate; default AUTO = 1000/window_ms "
                             "(real time, contiguous frames). Playback length "
                             "is always the recording length regardless of fps.")
    parser.add_argument("--t-start", type=float, default=0.0,
                        help="Start time in seconds (default %(default)s)")
    parser.add_argument("--t-end", type=float, default=np.inf,
                        help="End time in seconds (default: end of recording)")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Fixed color-scale minimum (uV); default = 1st pct")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Fixed color-scale maximum (uV); default = 99th pct")
    parser.add_argument("--video", type=str, default="auto",
                        choices=("auto", "gif", "mp4"),
                        help="Video format (default: mp4 if ffmpeg on PATH, "
                             "else gif)")
    parser.add_argument("--dpi", type=int, default=100,
                        help="Figure dpi (default %(default)s)")
    parser.add_argument("--keep-frames", action="store_true",
                        help="Also save individual PNG frames to frames/")
    args = parser.parse_args()
    main(args)
