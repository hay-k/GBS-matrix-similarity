import numpy as np
from itertools import permutations

from core.gbs import GBSDevice
from core.molecular_data import get_single_two_body, get_adjacency_matrix


if __name__ == '__main__':
    matrix = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])

    molecule = GBSDevice(name='test_molecule')
    molecule.encode_matrix(matrix, 5)

    print(molecule.get_feature_vector(3))
    
    
def test_molecule(mol, n_mean=5, max_photons=5):
    a,b = get_single_two_body(mol)
    adj_matrix = get_adjacency_matrix(a,b)
    print('adjacency matrix: ' + str(mol) + ': ' + str(adj_matrix.shape) + ': ' + str(adj_matrix))
    molecule = GBSDevice(name=mol)
    l, w = adj_matrix.shape
    molecule.encode_matrix(adj_matrix, n_mean, np.zeros(2 * l))
    feature_vec = molecule.get_orbit_feature_vector(max_photons)
    return feature_vec

def test_molecule_list(mol_list, n_mean=5, max_photons=5):
    output = {}
    for mol in mol_list:
        output[mol] = test_molecule(mol, n_mean, max_photons)
    return output
        