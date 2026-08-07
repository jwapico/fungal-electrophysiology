from typing import Any, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import scipy.signal
import base64
import io

import constants

MAX_WAVEFORMS_PER_CHANNEL: int = 99999    # limit per channel for performance
WAVEFORMS_PER_ROW: int = 3                # HTML grid columns
FIGURE_SIZE: Tuple[int, int] = (8, 5)     # matplotlib figure (width, height)
DPI: int = 60                             # image resolution
Y_AXIS_PADDING: float = 10.0              # extra padding around waveform
CHANNEL_DS_FACTOR: int = 10               # downscaling to make large scale channel visualizations more performant
OUTPUT_DIR = "outputs/images"


def find_channel_spikes(channel: int, data: np.ndarray, ds_factor: int = CHANNEL_DS_FACTOR, min_spike_distance: int = 50):
    """
    Detect spikes on a single channel. Only used for visualization
    
    Returns: spike_indices at 30kHz sample rate
    """
    # Convert to microvolts
    voltage = -data[:, channel] * 0.195
    
    # Downsample for display
    voltage_ds = voltage[::ds_factor] if ds_factor != 0 else voltage
    
    # Detect spikes
    std_dev = np.std(voltage_ds)
    threshold = constants.ST_DEV_SCALE * std_dev
    
    peaks, _ = scipy.signal.find_peaks(voltage_ds, prominence=threshold, distance=min_spike_distance)
    
    # Fallback if no spikes
    if len(peaks) == 0:
        peaks, _ = scipy.signal.find_peaks(voltage_ds, prominence=threshold / 2, distance=min_spike_distance)
    
    # Convert back to 30kHz indices
    spike_indices_30khz = peaks * ds_factor
    
    return spike_indices_30khz, threshold, std_dev, voltage_ds


def channel_interactive_view(channel_n, ds_factor = CHANNEL_DS_FACTOR):
    # Load data
    data = np.memmap(constants.RAW_DATA_FILE, dtype=np.int16, mode='r').reshape(-1, 64)
    channel = int(channel_n)
    print(f"Showing interactive view for channel {channel}")
    
    spike_indices, threshold, std_dev, _ = find_channel_spikes(channel, data)
    print(f"  Threshold: {threshold:.2f} μV (std: {std_dev:.2f} μV)")
    print(f"  Detected: {len(spike_indices)} spikes")
    
    voltage = -data[:, channel] * 0.195
    voltage_ds = voltage[::ds_factor] if ds_factor != 0 else voltage
    time = np.arange(len(voltage_ds)) / (constants.SAMPLE_RATE_HZ // ds_factor) if ds_factor != 0 else np.arange(len(voltage_ds)) / constants.SAMPLE_RATE_HZ
    
    peak_times = spike_indices / constants.SAMPLE_RATE_HZ
    peak_voltages = voltage[spike_indices]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=voltage_ds, mode='lines', 
                             line=dict(color='blue', width=0.5),
                             name='Voltage'))
    fig.add_trace(go.Scatter(x=peak_times, y=peak_voltages, mode='markers',
                             marker=dict(color='red', size=6),
                             name='Detected spikes'))
    
    fig.update_layout(
        title=f"Channel {channel} - {len(spike_indices)} spikes",
        xaxis_title="Time (seconds)",
        yaxis_title="Voltage (μV)",
        hovermode='x unified',
        height=600
    )
    fig.show()
    return


def gen_channel_html(output_path = f"{OUTPUT_DIR}/all_ch_spikes.html"):
    # Load data
    data = np.memmap(constants.RAW_DATA_FILE, dtype=np.int16, mode='r').reshape(-1, 64)
    
    print("Processing all 64 channels...")
    all_spike_indices = []
    
    for ch in range(64):
        spike_indices, threshold, std_dev, _ = find_channel_spikes(ch, data)
        print(f"Channel {ch}: {len(spike_indices)} spikes (threshold: {threshold:.2f} μV)")
        
        save_channel_image(ch, data, spike_indices, float(threshold), float(std_dev))
        all_spike_indices.append(spike_indices)
    
    # Generate HTML
    print("\nGenerating HTML...")
    html_lines = ['<html><body>']
    for ch in range(64):
        html_lines.append(f'<h2>Channel {ch}</h2>')
        html_lines.append(f'<img src="ch_{ch}.png"><br>')
    html_lines.append('</body></html>')
    
    with open(output_path, "w") as f:
        f.write('\n'.join(html_lines))
    
    # Save spike indices
    np.save(Path(OUTPUT_DIR) / "ch_spike_times.npy", 
            np.array(all_spike_indices, dtype=object))
    
    print(f"Done! Open all_channels_spikes.html to view.")


def save_channel_image(channel: int, data: np.ndarray,
                       spike_indices: np.ndarray,
                       threshold: float, std_dev: float, ds_factor: int = CHANNEL_DS_FACTOR) -> None:
    """
    Save static matplotlib image for one channel.
    """
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    voltage = -data[:, channel] * 0.195
    voltage_ds = voltage[::ds_factor] if ds_factor != 0 else voltage
    time = np.arange(len(voltage_ds)) / (constants.SAMPLE_RATE_HZ // ds_factor) if ds_factor != 0 else np.arange(len(voltage_ds)) / constants.SAMPLE_RATE_HZ
    
    peak_times = spike_indices / constants.SAMPLE_RATE_HZ
    peak_voltages = voltage[spike_indices]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, voltage_ds, 'b-', linewidth=0.5)
    ax.scatter(peak_times, peak_voltages, c='red', s=6)
    ax.set_title(f"Channel {channel} - {len(spike_indices)} spikes (threshold: {threshold:.2f} μV)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Voltage (μV)")
    plt.savefig(output_dir / f"ch_{channel}.png", dpi=100)
    plt.close()


def create_waveform_image(waveform: np.ndarray, spike_time_sec: float,
                         window_size: int,
                         sample_rate: int = constants.SAMPLE_RATE_HZ,
                         unit_label: str = "μV") -> str:
    """Create matplotlib waveform image with ACTUAL spike width."""
    n_samples = min(len(waveform), window_size) 
    half_window = n_samples / sample_rate / 2
    
    time_sec = np.linspace(
        spike_time_sec - half_window,
        spike_time_sec + half_window,
        n_samples
    )
    
    display_waveform = waveform[:n_samples]
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(time_sec, display_waveform, 'b-', linewidth=0.8)
    
    ax.set_xlim([time_sec[0], time_sec[-1]]) #type: ignore
    y_min = display_waveform.min() - Y_AXIS_PADDING
    y_max = display_waveform.max() + Y_AXIS_PADDING
    ax.set_ylim([y_min, y_max]) #type: ignore
    
    ax.set_xlabel(f'Time (seconds)', fontsize=12)
    ax.set_ylabel(f'Voltage ({unit_label})', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Spike at {spike_time_sec:.4f}s ({n_samples} samples)', fontsize=11)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    return img_base64


def gen_spike_waveform_html(results: Dict[int, Dict[str, Any]],
                 output_path: str,
                 max_waveforms: int = MAX_WAVEFORMS_PER_CHANNEL,
                 unit_label: str = "μV") -> None:
    """Generate HTML grid visualization."""
    print(f"\nGenerating HTML visualization: {output_path}")
    
    html_parts = []
    
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
    <title>MEA Spike Waveforms</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .channel-section { margin-bottom: 40px; background: white; padding: 20px; 
                          border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .channel-header { background: #e8e8e8; padding: 15px; margin: -20px -20px 15px -20px;
                        border-radius: 8px 8px 0 0; }
        .channel-header h2 { margin: 0 0 10px 0; }
        .stats { font-size: 13px; color: #666; }
        .waveform-grid { display: flex; flex-wrap: wrap; gap: 15px; justify-content: flex-start; }
        .waveform-cell { text-align: center; background: #fafafa; padding: 10px; border-radius: 4px; }
        .waveform-cell img { border: 1px solid #ddd; border-radius: 4px; }
        .waveform-cell .label { font-size: 11px; color: #888; margin-top: 5px; }
    </style>
</head>
<body>
    <h1>MEA Spike Waveform Analysis</h1>
    <p>Raw extracellular recordings - fungal mycelium</p>
""")
    
    channels = sorted(results.keys())
    
    for ch in channels:
        data = results[ch]
        waveforms = data['waveforms']
        spike_times = data['spike_times']
        window_sizes = data['window_sizes']
        threshold = data['threshold']
        std_dev = data['std_dev']
        n_detected = data['n_detected']
        n_extracted = data['n_extracted']
        
        if len(waveforms) > max_waveforms:
            indices = np.random.choice(len(waveforms), max_waveforms, replace=False)
            waveforms = waveforms[indices]
            spike_times = spike_times[indices]
            window_sizes = window_sizes[indices]
        
        print(f"  Channel {ch}: generating {len(waveforms)} waveform images...")
        
        html_parts.append(f"""
    <div class="channel-section">
        <div class="channel-header">
            <h2>Channel {ch}</h2>
            <div class="stats">
                <strong>Detected:</strong> {n_detected} |
                <strong>Extracted:</strong> {n_extracted} |
                <strong>Threshold:</strong> {threshold:.2f} {unit_label} |
                <strong>Noise σ:</strong> {std_dev:.2f} {unit_label}
            </div>
        </div>
        <div class="waveform-grid">
""")
        for j, (w, t, ws) in enumerate(zip(waveforms, spike_times, window_sizes)):
            img = create_waveform_image(w, t, ws, unit_label=unit_label)
            html_parts.append(f"""
            <div class="waveform-cell">
                <img src="data:image/png;base64,{img}" alt="Waveform {j}">
                <div class="label">ch:{ch} #{j} @ {t:.3f}s | {ws} samples</div>
            </div>
""")
        
        html_parts.append("""
        </div>
    </div>
""")
    
    html_parts.append("</body></html>")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(html_parts))
    print(f"  Saved HTML to: {output_path}")
