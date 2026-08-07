
from typing import Dict, Optional
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from pathlib import Path

from simulation.izhikevich import IzhikevichNeuron, simulate_neuron
from analysis.constants import SAMPLE_RATE_HZ

def visualize_channel(channel: int, data: np.ndarray, 
                     spike_indices: np.ndarray,
                     threshold: float, std_dev: float, ds_factor: int = 0) -> None:
    """
    Show interactive Plotly figure for one channel.
    """
    voltage = -data[:, channel] * 0.195
    voltage_ds = voltage[::ds_factor] if ds_factor != 0 else voltage
    time = np.arange(len(voltage_ds)) / (SAMPLE_RATE_HZ // ds_factor) if ds_factor != 0 else np.arange(len(voltage_ds)) / SAMPLE_RATE_HZ
    
    peak_times = spike_indices / SAMPLE_RATE_HZ
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

def save_channel_image(channel: int, data: np.ndarray,
                       spike_indices: np.ndarray,
                       threshold: float, std_dev: float, ds_factor: int = 0, output_dir = "outputs/images/all_channel_spikes") -> None:
    """
    Save static matplotlib image for one channel.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    voltage = -data[:, channel] * 0.195
    voltage_ds = voltage[::ds_factor] if ds_factor != 0 else voltage
    time = np.arange(len(voltage_ds)) / (SAMPLE_RATE_HZ // ds_factor) if ds_factor != 0 else np.arange(len(voltage_ds)) / SAMPLE_RATE_HZ
    
    peak_times = spike_indices / SAMPLE_RATE_HZ
    peak_voltages = voltage[spike_indices]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, voltage_ds, 'b-', linewidth=0.5)
    ax.scatter(peak_times, peak_voltages, c='red', s=6)
    ax.set_title(f"Channel {channel} - {len(spike_indices)} spikes (threshold: {threshold:.2f} μV)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Voltage (μV)")
    plt.savefig(output_dir / f"ch_{channel}.png", dpi=100)
    plt.close()

def plot_single_neuron_results(
    neuron_type: str = 'RS',
    simulation_data: Optional[Dict[str, NDArray[np.float64]]] = None
):
    """Plot results for a single neuron
    
    Args:
        neuron_type: type of neuron to simulate if no data is provided
        simulation_data: optional pre-simulated data dict with 'time', 'v_history',
                         'u_history', and 'spike_times' keys.
                         If provided, skips internal simulation and uses this data directly.
    """
    if simulation_data is not None:
        time = simulation_data['time']
        v_history = simulation_data['v_history']
        u_history = simulation_data['u_history']
        spike_history = simulation_data['spike_times']
    else:
        time, v_history, u_history, spike_history = simulate_neuron(neuron_type=neuron_type).values()
    
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # Plot membrane potential
    ax1 = plt.subplot(gs[0])
    ax1.plot(time, v_history, 'b-', linewidth=1)
    ax1.set_ylabel('Membrane Potential (mV)', fontsize=12)
    ax1.set_title(f'{neuron_type} Neuron Response', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(time))
    ax1.set_ylim([-100, 40])  # type: ignore
    
    # Mark spikes
    if spike_history.any():
        ax1.plot(spike_history, [30] * len(spike_history), 'r.', markersize=8)
    
    # Plot input current
    ax2 = plt.subplot(gs[1], sharex=ax1)
    I = np.zeros_like(time)
    I[(time >= 200) & (time < 700)] = 10
    ax2.plot(time, I, 'g-', linewidth=2)
    ax2.set_ylabel('Input Current', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-1, 12])  # type: ignore
    
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.plot(time, u_history, 'r-', linewidth=1)
    ax3.set_ylabel('Recovery Variable', fontsize=12)
    ax3.set_xlabel('Time (ms)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_all_neuron_types():
    """Plot all neuron types in one figure"""
    neuron_types = ['RS', 'IB', 'CH', 'FS', 'LTS', 'RZ']
    
    fig, axes = plt.subplots(len(neuron_types), 1, figsize=(12, 14))
    
    for idx, neuron_type in enumerate(neuron_types):
        time, v_history, _, spike_history = simulate_neuron(neuron_type=neuron_type, duration=500).values()
        
        axes[idx].plot(time, v_history, 'b-', linewidth=1)
        
        spike_times = [t for t in spike_history if t < 500]
        if spike_times:
            axes[idx].plot(spike_times, [30] * len(spike_times), 'r.', markersize=6)
        
        axes[idx].axvspan(200, 400, alpha=0.2, color='green')
        axes[idx].set_ylabel(neuron_type, fontsize=10, rotation=0, labelpad=20)
        axes[idx].set_ylim([-100, 40])
        axes[idx].grid(True, alpha=0.3)
        
        if idx == len(neuron_types) - 1:
            axes[idx].set_xlabel('Time (ms)', fontsize=12)
    
    plt.suptitle('Different Types of Cortical Neurons', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_network_activity(firings_array, n_total):
    """Plot network simulation results"""
    if firings_array.ndim == 1 or len(firings_array) == 0:
        firings_array = np.empty((0, 2))
    
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1])
    
    # spike raster plot
    ax1 = plt.subplot(gs[0])
    if len(firings_array) > 0:
        ax1.plot(firings_array[:, 0], firings_array[:, 1], 'k.', markersize=0.5, alpha=0.6)
    ax1.set_ylabel('Neuron Index', fontsize=12)
    ax1.set_title('Network Activity - Spike Raster Plot', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1000]) # type: ignore
    ax1.set_ylim([0, n_total]) # type: ignore
    
    # Population firing rate
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    # Calculate firing rate in 10ms bins
    bin_width = 10
    bins = np.arange(0, 1001, bin_width)
    firing_rates = []
    
    for i in range(len(bins)-1):
        mask = (firings_array[:, 0] >= bins[i]) & (firings_array[:, 0] < bins[i+1])
        rate = np.sum(mask) / (n_total * bin_width * 0.001)  # Convert to Hz
        firing_rates.append(rate)
    
    ax2.plot(bins[:-1], firing_rates, 'b-', linewidth=2)
    ax2.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 15]) # type: ignore
    
    plt.tight_layout()
    return fig

def plot_phase_plane(
    neuron_type: str = 'RS',
    I: float = 0,
    simulation_data: Optional[Dict[str, NDArray[np.float64]]] = None
):
    """Plot phase plane analysis for a neuron
    
    Args:
        neuron_type: type of neuron ('RS', 'IB', 'CH')
        I: input current
        simulation_data: optional pre-simulated data dict with 'v_history' and 'u_history' keys.
                         If provided, skips internal simulation and uses this data for the trajectory.
    """
    params = {
        'RS': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8},
        'IB': {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4},
        'CH': {'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2},
    }
    
    if neuron_type not in params:
        neuron_type = 'RS'
    
    a, b, c, d = params[neuron_type].values()
    
    # Create phase plane grid
    v_range = np.linspace(-80, 40, 30)
    u_range = np.linspace(-20, 20, 30)
    V, U = np.meshgrid(v_range, u_range)
    
    # Compute derivatives
    dV = 0.04 * V**2 + 5 * V + 140 - U + I
    dU = a * (b * V - U)
    
    # Normalize for vector field
    magnitude = np.sqrt(dV**2 + dU**2)
    dV_norm = dV / (magnitude + 1e-10)
    dU_norm = dU / (magnitude + 1e-10)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Phase plane with vector field
    ax1.quiver(V, U, dV_norm, dU_norm, magnitude, cmap='viridis', alpha=0.6)
    ax1.set_xlabel('Membrane Potential (v)', fontsize=12)
    ax1.set_ylabel('Recovery Variable (u)', fontsize=12)
    ax1.set_title(f'{neuron_type} Neuron Phase Plane (I={I})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Nullclines
    v_null = np.linspace(-80, 40, 100)
    u_null_v = 0.04 * v_null**2 + 5 * v_null + 140 + I
    u_null_u = b * v_range
    
    ax1.plot(v_null, u_null_v, 'r-', linewidth=2, label='v-nullcline')
    ax1.plot(v_range, u_null_u, 'b-', linewidth=2, label='u-nullcline')
    ax1.legend()
    
    # Use provided data or run internal simulation
    if simulation_data is not None:
        v_vals = simulation_data['v_history'].tolist()
        u_vals = simulation_data['u_history'].tolist()
    else:
        neuron = IzhikevichNeuron(a=a, b=b, c=c, d=d)
        v_vals = []
        u_vals = []
        for _ in range(1000):
            neuron.update(I, dt=0.5)
            v_vals.append(neuron.v)
            u_vals.append(neuron.u)
    
    ax1.plot(v_vals, u_vals, 'k-', linewidth=1, alpha=0.7)
    ax1.plot(v_vals[0], u_vals[0], 'go', markersize=10, label='Start')
    ax1.plot(v_vals[-1], u_vals[-1], 'ro', markersize=10, label='End')
    
    # Time series plot
    time = np.arange(0, 500, 0.5)
    ax2.plot(time, v_vals[:len(time)], 'b-', linewidth=1)
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Membrane Potential (mV)', fontsize=12)
    ax2.set_title(f'{neuron_type} Neuron Time Series', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-100, 40])
    
    plt.tight_layout()
    return fig

def save_spike_raster(firings_array: np.ndarray, output_path: str, 
                      time_conversion_ms_to_sec: float = 1000.0) -> None:
    """
    Convert spike data to per-neuron spike times and save.
    
    Args:
        firings_array: Nx2 array with [time_ms, neuron_idx]
        output_path: path to save .npy file
        time_conversion_ms_to_sec: divide by this to convert ms to seconds
    """
    if firings_array.size == 0:
        print("Warning: No spikes to save")
        return
    
    max_neuron = int(firings_array[:, 1].max()) + 1
    
    # Create list of spike times per neuron
    spike_times_by_neuron = []
    for neuron_id in range(max_neuron):
        mask = firings_array[:, 1] == neuron_id
        times_sec = firings_array[mask, 0] / time_conversion_ms_to_sec
        spike_times_by_neuron.append(times_sec)
    
    # Save as array of arrays
    np.save(output_path, np.array(spike_times_by_neuron, dtype=object))
    print(f"Saved spike raster to {output_path}")
    print(f"  Format: {max_neuron} neurons, each with array of spike times (seconds)")