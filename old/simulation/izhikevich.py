from dataclasses import dataclass, asdict
from typing import Dict, Optional
from numpy.typing import NDArray
from pathlib import Path
import numpy as np
import json
import os
# TODO: use with waveform analysis pipeline to test generated parameter sets

KEYS = {"a", "b", "c", "d", "I_bias", "sigma"}
DEFAULT_PARAM_DIR = Path("outputs/parameter_sets")

@dataclass
class IzhikevichParams:
    """Izhikevich neuron parameter set.
    
    Attributes:
        a: Recovery time constant (1/ms)
        b: Recovery variable coupling
        c: Reset value for membrane potential
        d: Reset value for recovery variable
        I_bias: Baseline input current
        sigma: Noise standard deviation
        name: Optional identifier for the parameter set
    """
    a: float
    b: float
    c: float
    d: float
    I_bias: float
    sigma: float
    name: str = ""

    @classmethod
    def from_json(cls, path: Path, name: str = "") -> "IzhikevichParams":
        """Load ParamSet from a JSON file.
        
        Args:
            path: Path to the JSON file
            name: Optional name override (defaults to stem if not provided)
            
        Returns:
            ParamSet instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required keys are missing from JSON
        """
        if not path.exists():
            raise FileNotFoundError(f"Parameter file not found: {path}")
        
        with open(path, "r") as f:
            data = json.load(f)
        
        missing_keys = KEYS - set(data.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in {path}: {missing_keys}")
        
        if not name:
            name = path.stem
        
        return cls(
            a=data["a"],
            b=data["b"],
            c=data["c"],
            d=data["d"],
            I_bias=data["I_bias"],
            sigma=data["sigma"],
            name=name
        )

    def to_json(self, path: Optional[Path] = None) -> None:
        """Save ParamSet to a JSON file.
        
        Args:
            path: Path to save to. If None, uses DEFAULT_PARAM_DIR / {name}.json
            
        Raises:
            ValueError: If name is empty and no path provided
        """
        if path is None:
            if not self.name:
                raise ValueError("Either 'path' or 'name' must be provided")
            path = DEFAULT_PARAM_DIR / f"{self.name}.json"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = asdict(self)
        if not data["name"]:
            del data["name"]
        
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for use with simulation functions.
        
        Returns:
            Dict with keys: a, b, c, d, I_bias, sigma
        """
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "I_bias": self.I_bias,
            "sigma": self.sigma
        }

def load_param_set(name: str, directory: Path = DEFAULT_PARAM_DIR) -> IzhikevichParams:
    """Load a single parameter set by name.
    
    Args:
        name: Name of the parameter set (without .json extension)
        directory: Directory containing JSON files
        
    Returns:
        ParamSet instance
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = directory / f"{name}.json"
    return IzhikevichParams.from_json(path, name=name)

class IzhikevichNeuron:
    def __init__(self, a = 0.02, b = 0.2, c = -65.0, d = 2.0, v0 = -70.0, u0 = -14.0, spike_thresh = 30.0):
        """
        Args:
            a: time scale of recovery variable
            b: sensitivity of recovery variable
            c: after-spike reset value of membrane potential
            d: after-spike reset of recovery variable
            v0: initial membrane potential
            u0: initial recovery variable
        """

        self.a: float = a
        self.b: float = b
        self.c: float = c
        self.d: float = d
        self.v: float = v0
        self.u: float = u0 * b
        self.spike_thresh: float = spike_thresh
        self.spike_times = []
        
    def update(self, I: float, dt=0.5, sigma=0.0):
        """
        Args:
            I: input current
            dt: time step in ms
        
        Returns:
            bool: indicates if spike occurred
        """

        if self.v >= self.spike_thresh:
            self.v = self.c
            self.u += self.d
            return True

        noise = sigma * np.random.randn() * np.sqrt(dt)
        dv = 0.04*self.v**2 + 5*self.v + 140 - self.u + I + noise
        du = self.a*(self.b*self.v - self.u)

        self.v += dv*dt
        self.u += du*dt
        return False

def simulate_neuron(neuron_params: Optional[Dict[str, float]] = None, neuron_type = 'RS', duration = 1000, dt = 0.5, I_step = 10, step_start = 200, step_pulse_time = 500, sigma = 0.0) -> Dict[str, NDArray[np.float64]]:
    """
    Args:
        neuron_params: 'a', 'b', 'c', and 'd' neuron parameters
        neuron_type: type of neuron ('RS', 'IB', 'CH', 'FS', 'LTS', 'TC', 'RZ')
        duration: simulation duration in ms
        dt: time step in ms
        I_step: amplitude of step current
        step_start: when to apply step current
        sigma: magnitude of simulated input current noise (represents 'I' in the model)
    
    Returns:
        Dict[str, np.ndarray]: dict with 'v_history', 'u_history', and 'spike_times' arrays
    """

    if neuron_params:
        neuron = IzhikevichNeuron(**{k: v for k, v in neuron_params.items() if k != "I_bias"})
    else:
        param_types = {
            'RS': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8},
            'IB': {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4},
            'CH': {'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2},
            'FS': {'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2},
            'LTS': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2},
            'TC': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 0.05},
            'RZ': {'a': 0.1, 'b': 0.26, 'c': -65, 'd': 2}
        }
        
        if neuron_type not in param_types:
            raise ValueError(f"Unknown neuron type: {neuron_type}")

        neuron = IzhikevichNeuron(**param_types[neuron_type])
    
    # init history arrays
    steps = int(duration / dt)
    time = np.arange(0, duration, dt)
    v_history = np.zeros(steps)
    u_history = np.zeros(steps)
    spike_times = []
    
    # run simulation
    for i in range(steps):
        # simulate square pulse of input current for step_pulse_time (500ms)
        if step_start <= time[i] < step_start + step_pulse_time:
            if neuron_params:
                I = neuron_params["I_bias"]
            else:
                I = I_step
        else:
            I = 0
        
        # update neuron and append history
        spiked = neuron.update(I, dt, sigma)
        v_history[i] = neuron.v
        u_history[i] = neuron.u
        
        if spiked:
            spike_times.append(time[i])

    simulation_data = {
        "time": time,
        "v_history": v_history,
        "u_history": u_history,
        "spike_times": np.array(spike_times),
    }
    
    return simulation_data

def simulate_3d_izhikevich_network(
    output_dir: Optional[str],
    node_coords,
    edges,
    num_excitory_n,
    num_inhibitory_n,
    excitatory_params,
    inhibitory_params,
    duration=1000,
    dt=0.5,
    spike_thresh=30,
    coupling_scale = 0.05,
    sigma=0.0,
    heterogeneity=0.0,
    gamma=0.1,
    normalize_coupling=False
):
    """
    Hyphal network with mixed neuron types and diffusive coupling.

    Parameters
    ----------
    node_coords : dict
        Mapping from node id to 3D position array.
    edges : list of tuples
        Each tuple: (u, v, weight, distance, section_id).
    num_excitory_n, num_inhibitory_n : int
        Number of excitatory/inhibitory neurons (must be <= len(node_coords)).
    excitatory_params, inhibitory_params : dict
        Izhikevich parameters (a, b, c, d, optional I_bias).
    duration, dt : float
        Simulation duration and time step (ms).
    spike_thresh : float
        Spike detection threshold (mV).
    sigma : float
        Noise amplitude.
    heterogeneity : float
        scales randomness added to model parameters of nodes (default = 0) 
    gamma : float
        Additional model parameters.

    Returns
    -------
    If return_voltage is False:
        time : ndarray
            Time vector.
        firings : ndarray
            Spike times and neuron indices (Nx2 array).
        num_n : int
            Total number of neurons.
    If return_voltage is True:
        time, firings, num_n : as above.
        node_pos : dict
            Same as input node_pos.
        edges : list
            Same as input edges.
        v_history : ndarray
            Shape (num_steps, num_n). Membrane potential over time.
    """

    time = np.arange(0, duration, dt)
    num_n = len(node_coords)
    num_steps = int(duration / dt)
    assert num_excitory_n + num_inhibitory_n <= num_n, "Too many neurons requested"

    # randomly sample proportions for inhib and excit
    all_indices = np.random.permutation(num_n)
    excit_idx = all_indices[:num_excitory_n]
    inhib_idx = all_indices[num_excitory_n:num_excitory_n + num_inhibitory_n]

    # init params
    a = np.zeros(num_n)
    b = np.zeros(num_n)
    c = np.zeros(num_n)
    d = np.zeros(num_n)
    I_bias_vec = np.zeros(num_n)

    # assign excitatory
    a[excit_idx] = excitatory_params["a"]
    b[excit_idx] = excitatory_params["b"]
    c[excit_idx] = excitatory_params["c"]
    d[excit_idx] = excitatory_params["d"]
    I_bias_vec[excit_idx] = excitatory_params.get("I_bias", 0)

    # assign inhibitory
    a[inhib_idx] = inhibitory_params["a"]
    b[inhib_idx] = inhibitory_params["b"]
    c[inhib_idx] = inhibitory_params["c"]
    d[inhib_idx] = inhibitory_params["d"]
    I_bias_vec[inhib_idx] = inhibitory_params.get("I_bias", 0)

    # add randomness to params scaled by heterogeneity parameter
    a += heterogeneity * np.random.randn(num_n)
    b += heterogeneity * np.random.randn(num_n)
    c += heterogeneity * np.random.randn(num_n)
    d += heterogeneity * np.random.randn(num_n)
    I_bias_vec += heterogeneity * np.random.randn(num_n)

    # initialize random state for core izhikevich model variables u and v 
    v = -65 + 5 * np.random.randn(num_n)
    u = b * v

    neighbors = [[] for _ in range(num_n)]
    weights = [[] for _ in range(num_n)]
    v_history = np.zeros((num_steps, num_n))

    # build adjacency lists connecting neighbors and assign weights
    for node_idx1, node_idx2, w, _, _ in edges:
        neighbors[node_idx1].append(node_idx2)
        weights[node_idx1].append(w)

        neighbors[node_idx2].append(node_idx1)
        weights[node_idx2].append(w)

    # simulation loop
    firings_history = []
    for t_idx in range(num_steps):
        print(f"Step {t_idx} / {num_steps}", end="\r", flush=True)

        # check which nodes fired
        fired = np.where(v >= spike_thresh)[0]
        for neuron_idx in fired:
            firings_history.append([time[t_idx], neuron_idx])

        # reset the fired
        v[fired] = c[fired]
        u[fired] += d[fired]

        # diffusive coupling
        I_coupling = np.zeros(num_n)
        for i in range(num_n):
            for j_idx, j in enumerate(neighbors[i]):
                w_ij = weights[i][j_idx]
                I_coupling[i] += coupling_scale * w_ij * (v[j] - v[i])

            # normalize (helps prevent blow-up for networks with dense connections)
            # if neighbors[i]:
            #     I_coupling[i] /= len(neighbors[i])

        # add noise and calculate total current contributions
        noise = sigma * np.random.randn(num_n) * np.sqrt(dt)
        I_total = I_bias_vec + I_coupling + noise

        # izhikevich update step
        dv = 0.04 * v**2 + 5 * v + 140 - u + I_total - gamma * v
        du = a * (b * v - u)

        v += dv * dt
        u += du * dt

        v_history[t_idx] = v.copy()

    firings_array = np.array(firings_history) if firings_history else np.empty((0, 2))

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_filepath = output_dir + f"3d_izh_{len(list(Path(output_dir).glob('*')))}.json"
        save_output_to_file(output_filepath, time, firings_array, num_n, node_coords, edges, v_history)

    return time, firings_array, num_n, node_coords, edges, v_history

def save_output_to_file(output_path, time, firings_array, num_n, node_coords, edges, v_history):
    nodes_list = []
    for nid, pos in node_coords.items():
        nodes_list.append({
            "id": int(nid),
            "position": pos.tolist()
        })
    
    edges_list = []
    for (u, v, w, dist, sid) in edges:
        edges_list.append({
            "source": int(u),
            "target": int(v),
            "weight": float(w),
            "distance": float(dist),
            "section_id": int(sid)
        })
    
    spikes_dict = {}
    if firings_array.size > 0:
        for t, neuron_idx in firings_array:
            idx = int(neuron_idx)
            spikes_dict.setdefault(idx, []).append(float(t))
    
    voltages_list = []
    for nid in range(num_n):
        voltages_list.append(v_history[:, nid].tolist())
    
    data = {
        "metadata": {
            "duration": float(time[-1] + 0.5),
            "dt": float(time[1] - time[0]),
            "num_neurons": num_n,
            "num_steps": len(time),
            "spike_threshold": 30.0
        },
        "nodes": nodes_list,
        "edges": edges_list,
        "time": time.tolist(),
        "voltages": voltages_list,
        "spikes": spikes_dict
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported simulation data to {output_path}")
    print(f"  Nodes: {len(nodes_list)}")
    print(f"  Edges: {len(edges_list)}")
    print(f"  Time steps: {len(time)}")
    print(f"  Neurons with spikes: {len(spikes_dict)}")
    print(f"  Voltage array shape: {v_history.shape}")