from pathlib import Path

from izhikevich import IzhikevichParams, load_param_set
from parse_cyberfungi import parse_cyberfungi
from izhikevich import simulate_3d_izhikevich_network, save_output_to_file

CYBERFUNGI_XML_PATH = "data/cyberfungi/test1.mycelia.xml"
OUTPUT_DIR = "outputs/sim_output/"

PARAM_SET_A = "a"
PARAM_SET_B = "b"

def main():
    # parse cyberfungi structure and load params
    node_pos, weighted_edges, _, _ = parse_cyberfungi(CYBERFUNGI_XML_PATH)
    slow_spiking_a: IzhikevichParams = load_param_set(PARAM_SET_A)
    fast_spiking_b: IzhikevichParams = load_param_set(PARAM_SET_B)
    
    # run simulation
    time, firings_array, num_n, node_coords, edges, v_history = simulate_3d_izhikevich_network(
        output_dir=OUTPUT_DIR,
        node_coords=node_pos,
        edges=weighted_edges,
        num_excitory_n=800,
        num_inhibitory_n=200,
        excitatory_params=slow_spiking_a.to_dict(),
        inhibitory_params=fast_spiking_b.to_dict(),
        duration=60 * 1000,
        dt=0.5,
        sigma=0.1,
    )

if __name__ == "__main__":
    main()