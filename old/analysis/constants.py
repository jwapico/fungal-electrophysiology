from pathlib import Path

SAMPLE_RATE_HZ: int = 30000              # MEA sampling rate (samples/second)
NUM_CHANNELS: int = 64                   # Number of electrodes on MEA
VOLTAGE_SCALE: float = 0.195             # Intan RHD2164 ADC: int16 → μV conversion
BINARY_DTYPE: str = "int16"              # Raw binary file data type

RAW_DATA_FILE: str = "data/raw_mea_bins/recording_control_0_cut800s.bin"
OUTPUT_HTML: str = f"outputs/html/waveforms_grid_{len(list(Path('outputs/html').glob('*')))}.html"
WAVEFORM_OUTPUT_DIR: str = "outputs/waveforms"

SAVE_WAVEFORMS: bool = True               # Whether to save waveforms to disk
SAVE_FOR_BRIAN2: bool = False             # Normalize to -70/30 mV for model fitting
SAVE_RAW_MICROVOLTS: bool = False         # Also save raw microvolts

# spike detection parameters
ST_DEV_SCALE: float = 16.0                # std_dev multiplier for detection threshold
MIN_SPIKE_DISTANCE: int = 50              # minimum samples between spikes
LOWER_THRESHOLD_FACTOR: float = 0.5       # fallback threshold multiplier
