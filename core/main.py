import numpy as np
import core.molecular_data as md
from core.gbs import GBSDevice
from openfermion.utils import geometry_from_pubchem
import pubchempy


if __name__ == '__main__':
    # matrix = np.array([[1, 1, 1],
    #                    [1, 1, 1],
    #                    [1, 1, 1]])
    #
    # molecule = GBSDevice(name='test_molecule')
    # molecule.encode_matrix(matrix, 5)
    #
    # print(molecule.get_feature_vector(3))

    # print(md.get_single_two_body("H2"))
    # geometry = geometry_from_pubchem("O2")
    # print(geometry)
    name = 'O2'
    pubchempy_2d_molecule = pubchempy.get_compounds(name, 'name')
    print(pubchempy_2d_molecule[0].to_dict())
    print(geometry_from_pubchem(name))
