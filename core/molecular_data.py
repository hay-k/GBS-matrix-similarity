from openfermion.hamiltonians import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.utils import geometry_from_pubchem
import numpy as np
from core.utils import round_custom
from data.geometry_nist import geometries

mols = ["H2", "LiH", "O2", "N2", "F2", "Ne2", "Ar2", "CO", "HCN", "HNC", "CH4", "H2O", "NH3", "BH3", "H2O2", "H2CO", "HCOOH", "CH3OH", "CH3CH2OH"]

large_mols = ["Ne2"]

not_present = ["Ar2", "CO"]


def wrapper_get_single_two_body():
    for item in mols:
        if item in large_mols or item in not_present:
            continue
        a, b = get_single_two_body(item)
        adj_mat = get_molecular_matrix(a, b)
        print(str(item) + ': ' + str(adj_mat))
        # run gbs


def get_geometry(formula, source):
    """
    Get geometry for the specified molecule from the specified source. If the source is 'pubchem'
    the geometry will be retrieved from online pubchem database, if it is 'nist', then the geometry
    will be retrieved from data/geometry_nist.py.

    :param formula: The chemical formula of the molecule in capital letters. E.g. H2O
    :param source: Can be either 'pubchem' or 'nist'.
    :return: The geometry specification, e.g. for H2 - [('H', (0, 0, 0.3713970)), ('H', (0, 0, -0.3713970))]
    """
    if source == 'pubchem':
        geometry = geometry_from_pubchem(formula)
    elif source == 'nist':
        geometry = geometries.get(formula, None)
    else:
        raise ValueError("Source type {} not supported, please use 'pubchem', or 'nist'".format(source))

    if not geometry:
        raise LookupError("No geometry information was found for the specified molecule in the specified source.")

    return geometry


def get_single_two_body(formula, source, basis='sto-3g', multiplicity=1):
    """
    Calculate and return single body and two body integrals for the molecule specified with formula.

    :param formula: The chemical formula of the molecule in capital letters. E.g. H2O
    :param source: Can be either 'pubchem' or 'nist'.
    :param basis:
    :param multiplicity:
    :return: one body and two body integrals
    """
    geometry = get_geometry(formula, source)
    molecule = MolecularData(geometry, basis, multiplicity)
    molecule = run_pyscf(molecule)

    # TODO: add proper functionality to save to a file, preferably with specifiable location adn filename.
    # _molecule.save()

    return molecule.one_body_integrals, molecule.two_body_integrals


def get_single_two_body_file(molecule_file_name):
    """
    Loads the molecule from a file.

    :param molecule_file_name: Filename
    :return: Molecule
    """
    molecule = MolecularData(filename=molecule_file_name)
    molecule.load()
    # _molecule = run_pyscf(molecule)

    return molecule.one_body_integrals, molecule.two_body_integrals


def get_molecular_matrix(single_body, two_body):
    """
    Create a 2D matrix representation of the molecule based on its single body and two body integrals.
    The output of this function is suitable for encoding into a GBS device without displacements.

    :param single_body: single body integrals
    :param two_body: two body integrals
    :return: the matrix
    """
    x, y = single_body.shape
    func = np.vectorize(round_custom)
    _new_dim = x * y
    single_one_dim = single_body.reshape(_new_dim, 1)
    two_body_two_dim = func(two_body.reshape(_new_dim, _new_dim))
    idx = 0
    x, _ = two_body_two_dim.shape
    while idx < x:
        two_body_two_dim[idx][idx] = round_custom(single_one_dim[idx][0])
        idx += 1
    return two_body_two_dim


def get_molecular_matrix_and_vector(single_body, two_body):
    """
    Create a representation of the molecule with a 2D matrix and a 1D vector (dim(vector) = 2*dim(matrix)),
    based on its single body and two body integrals. The output of this function is suitable for encoding
    into a GBS device with displacements. The two body integrals are reshaped into a 2D matrix, and the one
    body integrals into a 1D vector.

    :param single_body: single body integrals
    :param two_body: two body integrals
    :return: the matrix
    """
    x, y = single_body.shape
    func = np.vectorize(round_custom)
    _new_dim = x * y
    single_one_dim = single_body.reshape(_new_dim, 1)
    two_body_two_dim = func(two_body.reshape(_new_dim, _new_dim))

    return single_one_dim, two_body_two_dim
