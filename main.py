import graph
from read_write import read_xyz
from geometry import Voro, calc_angles
from constants import *
from graph import PeriodicGraph
from collections import Counter

import os
import pickle
from time import time
from tqdm import tqdm


class UnitCell:

    def __init__(self, cell_geometry):

        if len(cell_geometry) == 6:
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma = cell_geometry
            self.vectors = self.calc_lattice_vectors(*cell_geometry)
        elif len(cell_geometry) == 3:
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma = self.calc_cell_parameters(cell_geometry)
            self.vectors = np.array(cell_geometry)
        else:
            raise IOError(
                """cell geometry should be specified as a list lengths of cell vectors 
                and angles between them r 3x3 matrix of cell vectors!"""
            )
        self.inv_vectors = np.linalg.inv(self.vectors)
        self.orthogonal = self.is_orthogonal(self.alpha, self.beta, self.gamma)
        self.volume = self.cal_volume(self.vectors)

    @staticmethod
    def is_orthogonal(alpha, beta, gamma, tol=1e-2):

        return all(abs(angle - 90) < tol for angle in (alpha, beta, gamma))

    @staticmethod
    def cal_volume(lattice_vectors):

        return np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2]))

    @staticmethod
    def calc_cell_parameters(lattice_vectors):

        a = np.linalg.norm(lattice_vectors[0])
        b = np.linalg.norm(lattice_vectors[1])
        c = np.linalg.norm(lattice_vectors[2])
        alpha = np.degrees(np.arccos(np.dot(lattice_vectors[1], lattice_vectors[2]) / (b * c)))
        beta = np.degrees(np.arccos(np.dot(lattice_vectors[0], lattice_vectors[2]) / (a * c)))
        gamma = np.degrees(np.arccos(np.dot(lattice_vectors[0], lattice_vectors[1]) / (a * b)))
        return a, b, c, alpha, beta, gamma

    @staticmethod
    def calc_lattice_vectors(a, b, c, alpha, beta, gamma):

        if UnitCell.is_orthogonal.orthogonal:
            vectors = np.array(
                [[a, 0, 0],
                 [0, b, 0],
                 [0, 0, c]]
            )
        else:
            alpha = np.radians(alpha % 180)
            betta = np.radians(beta % 180)
            gamma = np.radians(gamma % 180)
            c1 = c * np.cos(betta)
            c2 = c * (np.cos(alpha) - np.cos(gamma) * np.cos(betta)) / np.sin(gamma)
            c3 = np.sqrt(c * c - c1 * c1 - c2 * c2)
            vectors = np.array([[a, 0., 0.],
                                [b * np.cos(gamma), b * np.sin(gamma), 0.],
                                [c1, c2, c3]])
        return vectors

    def get_cart_coord(self, fract_coords):

        n = len(fract_coords)
        if self.orthogonal:
            cart_coords = fract_coords * np.tile([self.a, self.b, self.c], (n, 1))
        else:
            cart_coords = np.dot(fract_coords, self.vectors)
        return cart_coords

    def get_fract_coords(self, cart_coords):

        n = len(cart_coords)
        cart_coords = np.array(cart_coords)
        if self.orthogonal:
            fract_coords = cart_coords / np.tile([self.a, self.b, self.c], (n, 1))
        else:
            fract_coords = np.dot(cart_coords, self.inv_vectors)
        return fract_coords


class Structure:

    def __init__(self, name, unit_cell, symbols, cart_coords=[], fract_coords=[], vconnectivity=None):

        self.name = name
        self.unit_cell = UnitCell(unit_cell)
        self.symbols = symbols
        self.vconnectivity = vconnectivity
        self._fill_atomic_radii_atomic_numbers()

        if len(cart_coords) != 0 and len(fract_coords) == 0:
            self.fract_coords = self.unit_cell.get_fract_coords(cart_coords) % 1
            self.cart_coords = self.unit_cell.get_cart_coord(self.fract_coords)
        elif len(cart_coords) == 0 and len(fract_coords) != 0:
            self.fract_coords = np.array(fract_coords) % 1
            self.cart_coords = self.unit_cell.get_cart_coord(self.fract_coords)
        else:
            self.cart_coords = cart_coords
            self.fract_coords = cart_coords
        self.adjacency_list = []
        self.molecular_groups = []

    def _fill_atomic_radii_atomic_numbers(self):
        self.atomic_numbers = np.array([get_number(s) for s in self.symbols])
        self.atomic_radii = get_covalent_radii(self.atomic_numbers)

    @staticmethod
    def calc_coord_and_translation(fract_coord, prec=1e-4):

        fract_coord = [round(c) if abs(c) < prec or abs(abs(c % 1) - 1) < prec else c for c in fract_coord]
        translation = np.array([0, 0, 0])
        new_coord = np.array([0., 0., 0.])
        for i in range(len(new_coord)):
            new_coord[i], translation[i] = (lambda x: (x % 1, - int(x - (x % 1))))(fract_coord[i])
        return new_coord, translation

    def get_atom_fract_coord(self, atom_index, translation):

        return self.fract_coords[atom_index] + translation

    def get_atom_cart_coord(self, atom_index, translation):
        
        return self.cart_coords[atom_index] + np.dot(translation, self.unit_cell.vectors)

    def get_atom_neighbors(self,  atom_index, translation, bond_types={}):

        return [[i, (t[0] + translation[0], t[1] + translation[1], t[2] + translation[2]), b_t]
                for i, t, b_t in self.adjacency_list[atom_index] if len(bond_types) == 0 or b_t in bond_types]

    @staticmethod
    def get_formula(symbol_list):

        symbol_counts = Counter(symbol_list)
        formula = ''
        for symbol, count in symbol_counts.items():
            if count == 1:
                formula += symbol
            else:
                formula += f"{symbol}{count}"
        return formula

    def get_centroid_LVDP_data(self, omega_threshold=0.05):
        """
        For a set of centroids return characteristics of the lattice VDP
        Args:
            omega_threshold: threshold solid angle; small VDP faces will be ignored
        """
        a, b, c = self.unit_cell.vectors
        translations = [(i, j, k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]
        translations.sort(key=lambda t: (abs(t[0]), abs(t[1]), abs(t[2])))
        t_vectors = [i * a + j * b + k * c for i, j, k in translations]
        extended_coordinates = [c + t for t in t_vectors for c in self.cart_coords]

        points_VDP_data = Voro(extended_coordinates, len(self.cart_coords)).get_VDP_data(omega_threshold=omega_threshold)

        return points_VDP_data

    def identify_bonds(self, tol=0.2, omega_threshold=0.25, check_direct=False):

        a, b, c = self.unit_cell.vectors
        translations = [(i, j, k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]
        translations.sort(key=lambda t: (abs(t[0]), abs(t[1]), abs(t[2])))
        t_vectors = [i * a + j * b + k * c for i, j, k in translations]
        extended_coordinates = [c + t for t in t_vectors for c in self.cart_coords]
        atom_index_translation = [(i, t) for t in translations for i in range(len(self.cart_coords))]

        contacts = Voro(extended_coordinates, len(self.cart_coords)).calc_p_adjacency(
            omega_threshold=omega_threshold, check_direct=check_direct)

        adjacency_list = [[] for _ in contacts]
        for i, neighbors in enumerate(contacts):
            for j, type in neighbors:
                index, translation = atom_index_translation[j]
                # if there is no predefined valence bonds connectivity - calculate it for each frame from scratch
                if self.vconnectivity is None:
                    r = self.atomic_radii[i] + self.atomic_radii[index]
                    if (
                            type == "direct"
                            and
                            np.linalg.norm(extended_coordinates[i] - extended_coordinates[j]) < r * (1 + tol)
                    ):
                        contact_type = "v"
                    else:
                        contact_type = "vw"
                else:
                    if ((i, index) in self.vconnectivity) or ((index, i) in self.vconnectivity):
                        contact_type = "v"
                    else:
                        contact_type = "vw"

                adjacency_list[i].append((index, translation, contact_type))
                adjacency_list[index].append((i, (-translation[0], -translation[1], -translation[2]), contact_type))

        self.adjacency_list = np.array([list(set(ns)) for ns in adjacency_list], dtype=object)
        return self.adjacency_list

    def find_hydrogen_bonds(
            self,
            D_atoms=('N', 'O'),
            A_atoms=('N', 'O'),
            r_HA=2.5,
            angle=120):
        """
        In the D-H...A fragment
        D_atoms: tuple of D atoms, ('N', 'O')
        A_atoms: tuple of A atoms, ('N', 'O')
        r_HA: the maximal distance H...A, float
        angle: the minimal angle DHA, float
        """
        angle = angle * np.pi / 180
        for i, ns in enumerate(self.adjacency_list):
            for j, (a, t_a, bond_type_ha) in enumerate(ns):
                if bond_type_ha == 'vw':
                    if self.symbols[i] == "H":
                        h = i
                    elif self.symbols[j] == "H":
                        a, h, t_a = i, a, (-t_a[0], -t_a[1], -t_a[2])
                    else:
                        continue
                    a_coord = self.get_atom_cart_coord(a, t_a)
                    h_a_length = np.linalg.norm(self.cart_coords[h] - a_coord)
                    # print(self.symbols[h], self.symbols[a], h_a_length, self.fract_coords[h])
                    if self.symbols[a] in A_atoms and h_a_length < r_HA:
                        for d, t_d, bond_type_dh in self.get_atom_neighbors(h, (0, 0, 0), {"v"}):
                            d_coord = self.get_atom_cart_coord(d, t_d)
                            #print(self.symbols[a], self.symbols[h], self.symbols[a], a_b_length, h_b_length, calc_angles([a_coord], [self.cart_coords[h]], [a_coord]))
                            if (
                                    self.symbols[d] in D_atoms and
                                    calc_angles([d_coord], [self.cart_coords[h]], [a_coord])[0] > angle
                            ):
                                k, t_3, _ = self.adjacency_list[i][j]
                                self.adjacency_list[i][j] = [k, t_3, "hb"]
                                for l, (n, t_4, _) in enumerate(self.adjacency_list[k]):
                                    if n == i and t_4 == (-t_3[0], -t_3[1], t_3[2]):
                                        self.adjacency_list[k][l] = [n, t_4, "hb"]
                                        break

    def find_molecular_groups(self):

        if len(self.adjacency_list) == 0:
            self.identify_bonds()
        covalent_bonds = [((i, j), tr) for i, ns in enumerate(self.adjacency_list)
                          for j, tr, bt in ns if i <= j and bt == "v"]
        bonds = [b for b, _ in covalent_bonds]
        translations = [t for _, t in covalent_bonds]
        pg = PeriodicGraph(list(range(len(self.symbols))), bonds, translations)
        self.molecular_groups = [(group, periodicity) for group, periodicity in pg.calc_periodicity()]
        return self.molecular_groups

    def get_molecular_graph(self, bond_types=("vw", "hb")):

        if len(self.molecular_groups) == 0:
            self.find_molecular_groups()

        molecular_adjacency_list = [[] for _ in self.molecular_groups]
        centroids = [sum([self.fract_coords[i] + t for i, t in group]) / len(group)
                     for group, p in self.molecular_groups]

        cs_ts = np.array([self.calc_coord_and_translation(c) for c in centroids])
        centroids, translations = cs_ts[:, 0], np.array(cs_ts[:, 1], dtype=int)
        self.molecular_groups = [[[[a, t + translations[i]] for a, t in group], p]
                                 for i, (group, p) in enumerate(self.molecular_groups)]

        atom_molecules = {j: (i, t) for i, (mg, p) in enumerate(self.molecular_groups) for j, t in mg}
        for i, (group, p) in enumerate(self.molecular_groups):
            #print(i, self.get_formula(self.symbols[[a for a, t in group]]))
            if p != 0:
                raise Exception("There are polymer groups!")
            neighbors = [(n, t_1 + t_2) for a, t_1 in group for n, t_2, bt in self.adjacency_list[a]
                         if bt in bond_types]
            for n, t_1 in neighbors:
                mol_index, t_2 = atom_molecules[n]
                molecular_adjacency_list[i].append((mol_index, tuple(t_1 - t_2), "v"))
                molecular_adjacency_list[mol_index].append((i, tuple(t_2 - t_1), "v"))
        molecular_adjacency_list = [list(set(ns)) for ns in molecular_adjacency_list]
        mol_graph = Structure(f"{self.name}_simplified",
                              self.unit_cell.vectors,
                              ["Rn"] * len(centroids),
                              fract_coords=centroids)
        mol_graph.adjacency_list = molecular_adjacency_list
        return mol_graph

    def __str__(self):

        data = "data_" + self.name + '\n'
        data += "_cell_length_a                      " + str(self.unit_cell.a) + '\n'
        data += "_cell_length_b                      " + str(self.unit_cell.b) + '\n'
        data += "_cell_length_c                      " + str(self.unit_cell.c) + '\n'
        data += "_cell_angle_alpha                   " + str(self.unit_cell.alpha) + '\n'
        data += "_cell_angle_beta                    " + str(self.unit_cell.beta) + '\n'
        data += "_cell_angle_gamma                   " + str(self.unit_cell.gamma) + '\n'
        data += "_symmetry_space_group_name_H-M   P1\n"
        data += "_symmetry_Int_Tables_number      1\n"
        data += ("loop_\n"
                 + "_space_group_symop.id\n"
                 + "_space_group_symop.operation_xyz\n"
                 + "1 x,y,z\n")
        if len(self.symbols) != 0:
            data += ("loop_\n"
                     + "_atom_site_label\n"
                     + "_atom_site_type_symbol\n"
                     + "_atom_site_symmetry_multiplicity\n"
                     + "_atom_site.fract_x\n"
                     + "_atom_site.fract_y\n"
                     + "_atom_site.fract_z\n"
                     + "_atom_site_occupancy\n")
            for i, s in enumerate(self.symbols):
                data += (f"{s}{i} {s} 1.0 "
                         + '%6.5f' % (self.fract_coords[i][0]) + ' '
                         + '%6.5f' % (self.fract_coords[i][1]) + ' '
                         + '%6.5f' % (self.fract_coords[i][2]) + ' 1.0\n')
            data += ("loop_\n"
                     + "_topol_atom.id\n"
                     + "_topol_atom.node_id\n"
                     + "_topol_atom.atom_label\n"
                     + "_topol_atom.element_symbol\n")
            for i, s in enumerate(self.symbols):
                data += "{} {} {} {} {} {}\n".format(i + 1, i + 1, s + str(i), s, "#",  s + str(i))
            data += ("loop_\n"
                     + "_topol_node.id\n"
                     + "_topol_node.label\n")
            for i, s in enumerate(self.symbols):
                data += f"{i+1} {s}{i}\n"
        if len(self.adjacency_list) != 0:
            data += ("loop_\n"
                     + "_topol_link.node_id_1\n"
                     + "_topol_link.node_id_2\n"
                     + "_topol_link.symop_id_1\n"
                     + "_topol_link.translation_1_x\n"
                     + "_topol_link.translation_1_y\n"
                     + "_topol_link.translation_1_z\n"
                     + "_topol_link.symop_id_2\n"
                     + "_topol_link.translation_2_x\n"
                     + "_topol_link.translation_2_y\n"
                     + "_topol_link.translation_2_z\n"
                     + "_topol_link.type\n"
                     + "_topol_link.multiplicity\n")
            for i, ns in enumerate(self.adjacency_list):
                for j, translation, bond_type in ns:
                    if i < j:
                        data += "{0:<5} {1:<5} 1 0 0 0 1 {2:>2} {3:>2} {4:>2} {5:>3} 1 \n".format(i + 1, j + 1, *translation, bond_type)
        data += "#End of " + self.name + '\n'
        return data


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
        # 'etolmdc',
        # 'bzammdc',
        'pntnmdc',
        # 'benzac02mdc',
        # 'benzac02mdc_reverse',
        # 'crotac01mdc',
        # 'glucmdc',
        # 'glucmdc_420K',
        # 'glucmdc_620K',
        # 'py1mdc',
        # 'b3mdc',
):

    selected_intermolecular_bonds = ('hb', )
    START_FRAME = 500
    MAX_FRAME = 1700                         # max frame to be read
    STEP = 2                                 # frame read in step

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
            structure.identify_bonds(tol=0.2, omega_threshold=0.25, check_direct=False)
            if 'hb' in selected_intermolecular_bonds:
                structure.find_hydrogen_bonds(
                    D_atoms=('O', 'N'),  # 'O' 'N' 'C'
                    A_atoms=('O', 'N'),
                    r_HA=2.5,
                    angle=120
                )
            simplified_graph = structure.get_molecular_graph(bond_types=selected_intermolecular_bonds)
            centroid_VDP_data = simplified_graph.get_centroid_LVDP_data(omega_threshold=0.1)
            nx_snapshot_graph = graph.NetworkxGraph(simplified_graph, centroid_VDP_data, i).get_graph_data()
            snapshot_graphs.append(nx_snapshot_graph)

            if i % 500 == 0:
                with open(f"./cifs/{system}_frame_{i}_structure.cif", "w") as nf:
                    nf.write(str(structure))
                with open(f"./cifs/{system}_frame_{i}_simplified.cif", "w") as nf:
                    nf.write(str(simplified_graph))

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