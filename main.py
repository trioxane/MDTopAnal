import os
import pickle
from time import time
from tqdm import tqdm

import graph
from read_write import read_xyz
from crystal_structure import Structure

def get_v_connectivity(filename):

    with open(filename, 'r') as inp:
        content = inp.readlines()

    for i, line in enumerate(content):
        if line.startswith('_topol_link.multiplicity'):
            connectivity_data_start_line = i + 1

    connectivity_lines = [l.strip()
                          for l in content[connectivity_data_start_line:-1]
                          if ' v ' in l]

    vconnectivity = []
    for l in connectivity_lines:
        i, j, *_ = l.split()
        vconnectivity.append((int(i) - 1, int(j) - 1))

    return set(vconnectivity)



tic = time()
for system in (
        # 'water_md_full',
        'etolmdc',
        # 'bzammdc',
        # 'pntnmdc',
        # 'benzac02mdc',
        # 'benzac02mdc_reverse',
        # 'crotac01mdc',
        # 'glucmdc',
        # 'glucmdc_420K',
        # 'glucmdc_620K',
        # 'py1mdc',
        # 'b3mdc',
):

    selected_intermolecular_bonds = ()
    v_connectivity_only = False
    if len(selected_intermolecular_bonds) == 0:
        v_connectivity_only = True

    START_FRAME = 0
    MAX_FRAME = 50                         # max frame to be read
    STEP = 1                                  # frame read in step

    # read in the valence bond connectivity if present
    if os.path.exists(f"{system}_connectivity.cif"):
        print(f'Connectivity has been found for {system}')
        v_bonds = get_v_connectivity(f"{system}_connectivity.cif")
    else:
        v_bonds = None

    snapshot_graphs = []

    for i, frame in tqdm(enumerate(read_xyz(f'{system}.xyz')), desc=system):

        if i < START_FRAME:
            continue

        if i % STEP == 0:

            _, symbols, coordinates, lattice = frame
            structure = Structure(f"frame_{i}", lattice, symbols, coordinates, vconnectivity=v_bonds)
            structure.identify_bonds(
                r_cov_sum_tolerance=0.2,
                omega_threshold=0.25,
                check_direct=False,
                v_connectivity_only=v_connectivity_only
            )

            if 'hb' in selected_intermolecular_bonds:
                structure.find_hydrogen_bonds(
                    D_atoms=('O', 'N'),  # 'O' 'N' 'C'
                    A_atoms=('O', 'N'),
                    r_HA=2.5,
                    angle=120
                )

            simplified_net = structure.get_molecular_graph(bond_types=selected_intermolecular_bonds)
            packing_net = structure.get_molecular_graph(bond_types=('vw', 'hb'))
            centroid_VDP_data = simplified_net.get_centroid_LVDP_data(omega_threshold=0.1)
            nx_snapshot_graph = graph.NetworkxGraph(simplified_net, centroid_VDP_data, i).get_graph_data()
            snapshot_graphs.append(nx_snapshot_graph)

            if i % 500 == 0:
                with open(f"./cifs/{system}_frame_{i}_structure.cif", "w") as nf:
                    nf.write(str(structure))
                with open(f"./cifs/{system}_frame_{i}_simplified_net.cif", "w") as nf:
                    nf.write(str(simplified_net))
                with open(f"./cifs/{system}_frame_{i}_packing_net.cif", "w") as nf:
                    nf.write(str(packing_net))

        if i == MAX_FRAME:
            break

    with open(f'MDTopAnalysis_{system}.nxg', 'wb') as out:
        pickle.dump(snapshot_graphs, out)

print(f'Total calculation time: {time() - tic:.1f} s')

# TODO 1) fast HB identification implementation https://github.com/MDAnalysis/mdanalysis/blob/develop/package/MDAnalysis/analysis/hydrogenbonds/hbond_analysis.py#L792
# TODO 2) editing of the adjacency list using topology of the first frame (or predifined topology) so that no unusual molecules appear due to accidentally short interatomic contacts
# TODO 2.5) or check at each frame CN of atoms and whether they have changed from the previous frame
# TODO 3) VDP direct and solid_angle additional criteria - use or not?
# TODO 4) run sequence: - supply extended xyz trajectory and run analysis of the first frame - the ideal one that will be used for 1) connectivity calculation (valence bonds defining molecules)
# TODO 4) 2) simplified idealised structure which will be used for the template graph with all edges between centroid and its neighbours in the first coordination shell
# TODO 4) 3) this creates file {system}.vb_connectivity.g and {system}.completegraph.g