from openfermion.hamiltonians import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.utils import geometry_from_pubchem
import numpy as np
from core.utils import round_custom

mols = ["H2", "LiH", "O2", "N2", "F2", "Ne2", "Ar2", "CO", "HCN", "HNC", "CH4", "H2O", "NH3", "BH3", "H2O2", "H2CO",
        "HCOOH", "CH3OH", "CH3CH2OH"]

large_mols = ["Ne2"]

not_present = ["Ar2", "CO"]


def wrapper_get_single_two_body():
    for item in mols:
        if item in large_mols or item in not_present:
            continue
        a, b = get_single_two_body(item)
        adj_mat = get_adjacency_matrix(a, b)
        print(str(item) + ': ' + str(adj_mat))
        # run gbs


def get_single_two_body_h2():
    """
    1. returns single body/ two body integrals for hydrogen
    2. assumes geometry of hydrogen, basis and multiplicity. These can be made as optional arguments.
    """
    geometry = [['H', [0, 0, 0]], ['H', [0, 0, 0.74]]]  # H--H distance = 74pm
    basis = 'sto-3g'
    multiplicity = 1  # (2S+1)
    charge = 0
    h2_molecule = MolecularData(geometry, basis, multiplicity, charge)
    h2_molecule = run_pyscf(h2_molecule)
    return h2_molecule.one_body_integrals, h2_molecule.two_body_integrals


def get_single_two_body(molecule_name, basis='sto-3g', multiplicity=1):
    """
    1. same as get_single_two_body_h2, except you can use it for any molecule which exits in pubchem
    2. returns one body and two body integrals
    """
    geometry = geometry_from_pubchem(molecule_name)
    _molecule = MolecularData(geometry, basis, multiplicity)
    _molecule = run_pyscf(_molecule)
    # TODO: save the molecule after running pyscf.
    return _molecule.one_body_integrals, _molecule.two_body_integrals


def get_single_two_body_file(molecule_file_name):
    """
    1. loads the molecule from the file.
    2. This means we should write a function which saves the molecules into a file XXX: TODO
    3.
    """
    _molecule = MolecularData(filename=molecule_file_name)
    _molecule.load()
    #    _molecule = run_pyscf(_molecule)
    return _molecule.one_body_integrals, _molecule.two_body_integrals


def get_adjacency_matrix(single_body, two_body):
    """
    creates adjacency matrix for the graph representation of a molecule given its single body and two body integrals

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
