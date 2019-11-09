import numpy as np
from itertools import permutations

from core.gbs import GBSDevice


if __name__ == '__main__':
    matrix = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])

    molecule = GBSDevice(name='test_molecule')
    molecule.encode_matrix(matrix, 5)

    print(molecule.get_feature_vector(3))



