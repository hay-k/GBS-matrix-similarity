import numpy as np
from core.gbs import GBSDevice
import strawberryfields as sf
import networkx as nx

if __name__ == '__main__':
    # generate a random symmetric matrix
    matrix = np.random.rand(10, 10)
    matrix = matrix + matrix.T

    molecule = GBSDevice(name='test_molecule')
    molecule.encode_matrix(matrix, 5)

    print("Exact feature vector of events:", molecule.get_event_feature_vector(2, max_photons_per_mode=2))
    print("MC feature vector of events:", molecule.get_event_feature_vector(2, max_photons_per_mode=2, mc=True))

    print("Exact feature vector of orbits:", molecule.get_orbit_feature_vector(2))
    print("MC feature vector of orbits:", molecule.get_orbit_feature_vector(2, mc=True))
