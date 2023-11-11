import numpy as np


def read_xyz(file_name):

    with open(file_name, "r") as open_file:
        n_at = open_file.readline()
        while n_at != "":
            lattice = None
            name = open_file.readline()
            if "Lattice=" in name:
                lattice = np.array((name.split("Lattice=")[1].split("\"")[1]).split(), dtype=float).reshape(3, 3)
            atoms = [open_file.readline().strip().split() for i in range(int(n_at))]
            symbols = np.array([a[0] for a in atoms])
            coordinates = np.array([[float(c) for c in a[1:4]] for a in atoms])
            n_at = open_file.readline()
            yield name, symbols, coordinates, lattice