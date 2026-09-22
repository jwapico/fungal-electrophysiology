#!/usr/bin/env python3
"""
Spike raster extraction from simulation output data.

Usage:
    python -m analysis.spike_raster outputs/sim_output/3d_izh_0.json
    python -m analysis.spike_raster render/simulation_data.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def get_unique_output_path(output_dir: Path, stem: str, suffix: str) -> Path:
    """Get unique output path by incrementing filename if it already exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_path = output_dir / f"{stem}{suffix}"
    if not base_path.exists():
        return base_path
    
    counter = 1
    while True:
        new_path = output_dir / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def extract_spike_rasters_ms(sim_output_path: Path) -> tuple[list[list[float]], int]:
    """Extract spike times for all channels that spiked (times in milliseconds).
    
    Args:
        sim_output_path: Path to simulation JSON output file.
        
    Returns:
        Tuple of:
        - rasters: List of lists where each inner list contains spike times (ms) for one channel
        - duration_ms: Total simulation duration in milliseconds
    """
    with open(sim_output_path, 'r') as f:
        data = json.load(f)
    
    spikes = data.get('spikes', {})
    duration_ms = data['metadata']['duration']
    
    node_ids = sorted(int(k) for k in spikes.keys())
    max_channel = max(node_ids) + 1 if node_ids else 0
    
    rasters = [[] for _ in range(max_channel)]
    
    for node_id_str, spike_times in spikes.items():
        channel_idx = int(node_id_str)
        if spike_times:
            rasters[channel_idx] = list(spike_times)
    
    return rasters, int(duration_ms)


def save_spike_rasters(rasters: list[list[float]], output_path: Path) -> None:
    """Save spike rasters to numpy file.
    
    Args:
        rasters: List of lists containing spike times in milliseconds.
        output_path: Path to save numpy file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.array(rasters, dtype=object))
    print(f"Saved spike rasters to {output_path}")


def visualize_spike_raster(
    rasters: list[list[float]],
    output_path: Path,
    duration_ms: float = 1000.0,
    marker_size: float = 1.0,
    row_height: int = 1
) -> None:
    """Create and save spike raster visualization.
    
    Args:
        rasters: List of lists containing spike times (in milliseconds).
        output_path: Path to save the image.
        duration_ms: Total duration of simulation in milliseconds.
        marker_size: Size of spike markers.
        row_height: Pixels per row (neuron).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_channels = len(rasters)
    num_spiking = sum(1 for r in rasters if r)
    
    total_height = num_channels * row_height
    fig_height = min(50, max(8, total_height * 0.015))
    fig_width = 14
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    for channel_idx, spike_times in enumerate(rasters):
        if spike_times:
            ax.plot(spike_times, [channel_idx] * len(spike_times), 
                    'k.', markersize=marker_size)
    
    ax.set_xlim(0, duration_ms)
    ax.set_ylim(-0.5, num_channels - 0.5)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Channel (Node)', fontsize=12)
    ax.set_title(f'Spike Raster - {num_spiking} spiking channels out of {num_channels} total', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved spike raster plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract spike rasters from simulation output')
    parser.add_argument('sim_output_path', type=Path, help='Path to simulation JSON output file')
    parser.add_argument('--row-height', type=int, default=1,
                    help='Pixels per row for raster plot (default: 1)')
    parser.add_argument('--marker-size', type=float, default=1.0,
                    help='Marker size for spikes (default: 1.0)')
    args = parser.parse_args()
    
    input_path = args.sim_output_path
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    duration_ms = data['metadata']['duration']
    
    stem = input_path.stem
    output_dir = Path('outputs/rasters')
    
    rasters, duration_ms = extract_spike_rasters_ms(input_path)
    num_spiking_channels = sum(1 for r in rasters if r)
    print(f"Found {num_spiking_channels} spiking channels out of {len(rasters)} total")
    print(f"Duration: {duration_ms} ms")
    
    output_npy = get_unique_output_path(output_dir, stem, '_rasters.npy')
    save_spike_rasters(rasters, output_npy)
    
    output_png = get_unique_output_path(output_dir, stem, '_raster.png')
    visualize_spike_raster(rasters, output_png, duration_ms=duration_ms,
                          row_height=args.row_height, marker_size=args.marker_size)


if __name__ == '__main__':
    main()