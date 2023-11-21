from read_write import read_xyz
from geometry import Voro
from constants import*
from graph import PeriodicGraph
from collections import Counter


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
            cart_coords = (np.tile(self.vectors[0], (n, 1)) * np.tile(fract_coords[:, 0][:, np.newaxis], (1, 3))
                           + np.tile(self.vectors[1], (n, 1)) * np.tile(fract_coords[:, 1][:, np.newaxis], (1, 3))
                           + np.tile(self.vectors[2], (n, 1)) * np.tile(fract_coords[:, 2][:, np.newaxis], (1, 3)))
        return cart_coords

    def get_fract_coords(self, cart_coords):

        n = len(cart_coords)
        cart_coords = np.array(cart_coords)
        if self.orthogonal:
            fract_coords = cart_coords / np.tile([self.a, self.b, self.c], (n, 1))
        else:
            fract_coords = (np.tile(self.inv_vectors[0], (n, 1)) * np.tile(cart_coords[:, 0][:, np.newaxis], (1, 3))
                            + np.tile(self.inv_vectors[1], (n, 1)) * np.tile(cart_coords[:, 1][:, np.newaxis], (1, 3))
                            + np.tile(self.inv_vectors[2], (n, 1)) * np.tile(cart_coords[:, 2][:, np.newaxis], (1, 3)))
        return fract_coords


class Structure:

    def __init__(self, name, unit_cell, symbols, cart_coords=[], fract_coords=[]):

        self.name = name
        self.unit_cell = UnitCell(unit_cell)
        self.symbols = symbols
        self.atomic_numbers = None
        self.atomic_radii = None
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

    def cal_bonds(self, tol=0.1):

        a, b, c = self.unit_cell.vectors
        translations = [(i, j, k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]
        translations.sort(key=lambda t: (abs(t[0]), abs(t[1]), abs(t[2])))
        t_vectors = [i * a + j * b + k * c for i, j, k in translations]
        extended_coordinates = [c + t for t in t_vectors for c in self.cart_coords]
        atom_index_translation = [(i, t) for t in translations for i in range(len(self.cart_coords))]
        contacts = Voro(extended_coordinates, len(self.cart_coords)).calc_p_adjacency()
        adjacency_list = [[] for _ in contacts]
        if self.atomic_radii is None:
            if self.atomic_numbers is None:
                self.atomic_numbers = np.array([get_number(s) for s in self.symbols])
            self.atomic_radii = get_covalent_radii(self.atomic_numbers)
        for i, neighbors in enumerate(contacts):
            for j, type in neighbors:
                index, translation = atom_index_translation[j]
                r = self.atomic_radii[i] + self.atomic_radii[index]
                if (type == "direct"
                        and np.linalg.norm(extended_coordinates[i] - extended_coordinates[j]) < r + r * tol):
                    contact_type = "v"
                else:
                    contact_type = "vw"
                adjacency_list[i].append((index, translation, contact_type))
                adjacency_list[index].append((i, (-translation[0], -translation[1], -translation[2]), contact_type))
        self.adjacency_list = np.array([list(set(ns)) for ns in adjacency_list])
        return self.adjacency_list

    def find_molecular_groups(self):

        if len(self.adjacency_list) == 0:
            self.cal_bonds()
        covalent_bonds = [((i, j), tr) for i, ns in enumerate(self.adjacency_list)
                          for j, tr, bt in ns if i <= j and bt == "v"]
        bonds = [b for b, _ in covalent_bonds]
        translations = [t for _, t in covalent_bonds]
        pg = PeriodicGraph(list(range(len(self.symbols))), bonds, translations)
        self.molecular_groups = [(group, periodicity) for group, periodicity in pg.calc_periodicity()]
        return self.molecular_groups

    def get_molecular_graph(self):

        if len(self.molecular_groups) == 0:
            self.find_molecular_groups()

        molecular_adjacency_list = [[] for _ in self.molecular_groups]
        atom_molecules = {j: (i, t) for i, (mg, p) in enumerate(self.molecular_groups) for j, t in mg}
        centroids = [sum([self.fract_coords[i] + t for i, t in group]) / len(group)
                     for group, p in self.molecular_groups]
        centroids = np.array(centroids)

        for i, (group, p) in enumerate(self.molecular_groups):
            if p != 0:
                raise Exception("There are polymer groups!")
            neighbors = [(n, t_1 + t_2) for a, t_1 in group for n, t_2, bt in self.adjacency_list[a] if bt == "vw"]
            for n, t_1 in neighbors:
                mol_index, t_2 = atom_molecules[n]
                molecular_adjacency_list[i].append((mol_index, tuple(t_1 - t_2), "v"))
                molecular_adjacency_list[mol_index].append((i, tuple(t_2 - t_1), "v"))
        molecular_adjacency_list = [list(set(ns)) for ns in molecular_adjacency_list]
        mol_graph = Structure(self.name + "_mol_graph", self.unit_cell.vectors,
                              ["Ar"] * len(centroids), fract_coords=centroids)
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
                data += (s + str(i) + ' '
                         + s + " 1.0 "
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
                     + " _topol_node.id\n"
                     + "_topol_node.label\n")
            for i, s in enumerate(self.symbols):
                data += "{} {}\n".format(i + 1, s + str(i))
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
                        data += "{} {} 1 0 0 0 1 {} {} {} {} 1 \n".format(i + 1, j + 1, *translation, bond_type)
        data += "#End of " + self.name + '\n'
        return data


for i, frame in enumerate(read_xyz("glucmdc_295K.xyz")):
    _, symbols, coordinates, lattice = frame
    structure = Structure(str(i), lattice, symbols, coordinates)
    with open("{}_graph.cif".format(i), "w") as nf:
        nf.write(str(structure.get_molecular_graph()))
    print(i, "done")

