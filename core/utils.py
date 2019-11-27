'''

'''
import numpy as np
from openfermion.hamiltonians import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.utils import geometry_from_pubchem
import gbsapps
import gbsapps.sample

mols = ["H2", "LiH", "O2", "N2", "F2", "Ne2", "Ar2", "CO", "HCN", "HNC", "CH4", "H2O", "NH3", "BH3", "H2O2", "H2CO", "HCOOH", "CH3OH", "CH3CH2OH"]


def wrapper_get_single_two_body():
  for item in mols:
    a,b = get_single_two_body(item)
    print(str(item) + ': ' + str(get_adjacency_matrix(a, b)))

'''
rounds the number in the argument if it is less than threshhold value.
if num < thresh:
    round(num)
else
    num
'''
def round_custom(num, thresh=0.0001):
    if num < thresh:
        num = round(num)
        if num == -0:
            return 0
    return num    

'''
creates adjacency matrix for the graph representation of a molecule given its single body and two body integrals

input: single body and two integrals

'''
def get_adjacency_matrix(single_body, two_body):
    x,y = single_body.shape
    func =  np.vectorize(round_custom)
    _new_dim = x * y
    single_one_dim = single_body.reshape(_new_dim, 1)
    two_body_two_dim = func(two_body.reshape(_new_dim, _new_dim))
    idx = 0
    x, _ = two_body_two_dim.shape
    while idx < x:
        two_body_two_dim[idx][idx] = round_custom(single_one_dim[idx][0])
        idx += 1
    return two_body_two_dim

'''
1. returns single body/ two body integrals for hydrogen
2. assumes geometry of hydogen, basis and multiplicity. These can be made as optional arguments.
'''
def get_single_two_body_h2():
    geometry = [['H', [0, 0, 0]], ['H', [0, 0, 0.74]]] # H--H distance = 0.74pm
    basis = 'sto-3g'
    multiplicity = 1 #(2S+1)
    charge = 0
    h2_molecule = MolecularData(geometry, basis, multiplicity, charge)
    h2_molecule = run_pyscf(h2_molecule)
    return h2_molecule.one_body_integrals, h2_molecule.two_body_integrals

'''
1. same as get_single_two_body except you can use it for any molecule which exits in pubchem
2. returns one body and two body integrals
'''
def get_single_two_body(molecule_name, basis='sto-3g', multiplicity=1):
    geometry = geometry_from_pubchem(molecule_name)
    _molecule = MolecularData(geometry, basis, multiplicity)
    _molecule = run_pyscf(_molecule)
    # save the molecule after running pyscf.
    return _molecule.one_body_integrals, _molecule.two_body_integrals

'''
1. loads the molecule from the file. 
2. This means we should write a function which saves the molecules into a file XXX: TODO
3. 
'''
def get_single_two_body_file(molecule_file_name):
    _molecule = MolecularData(filename=molecule_file_name)
    _molecule.load()
#    _molecule = run_pyscf(_molecule)
    return _molecule.one_body_integrals, _molecule.two_body_integrals

'''
returns similarity between two molecules: mol1 and mol2
demonstrates how are other functions are supposed to be used.
'''
def wrapper(mol1, mol2):
    a1, b1 = get_single_two_body(mol1)
    adj1 = get_adjacency_matrix(a1, b1)
    a2, b2 = get_single_two_body(mol2)
    adj2 = get_adjacency_matrix(a2, b2)
    return calculate_similarity(adj1, adj2)

'''
calculates similarity between two adjacency matrices using hilbert-schmitz method.

'''    
def calculate_similarity(mol1_adj, mol2_adj):
    diff = mol1_adj - mol2_adj
    diff_dot = np.dot(diff, np.transpose(np.conjugate(diff)))
    if np.linalg.norm(diff_dot) == 0:
        return 1-np.sqrt(np.trace(diff_dot))
    else:
        return 1-np.sqrt(np.trace(diff_dot)/(np.trace(diff)**2))

'''
temprorary function: will be removed XXX: TODO

Cant think why it was written in the first place.
'''    
def process_fidelity(U1, U2, normalize=True):
    """
    Calculate the process fidelity given two process operators.
    """
    if normalize:
        return np.trace(np.dot(U1, U2)) / (np.trace(U1) * np.trace(U2))
    else:
        return np.trace(np.dot(U1, U2))    

'''
temprorary function: will be removed XXX: TODO

it just demonstrates how to call the gbsapps quantum sampler function.
'''    
def run_gbs(adjacency_mat, number_of_times, num_val=4, threshold=False):
    idx = 0
    while idx < number_of_times:
        print(gbsapps.sample.quantum_sampler(adjacency_mat, num_val, threshold))
        idx += 1

'''
temprorary function: will be removed XXX: TODO

this was written during the hackathon as a quick and dirty way. So it should be replaced by an algorithm.
'''        
def get_state_map(num_bits):
    if num_bits == 3:
        return {
            '11': ['110', '011', '101'],
            '2': ['002', '020', '200'],
            '22': ['022', '202', '220'],
            '13': ['013', '103', '130', '031', '301', '310'],
            '121': ['121', '211', '112'],
            '4': ['004', '040', '004'],
            '123': ['123', '213', '231', '132', '312', '321']
        }
    elif num_bits == 5:
        return {
            '11': ['00011', '00110', '01100', '11000', '00101', '01010', '10100', '01001', '10010', '10001'],
            '13': ['00031', '00013', '00301', '00103', '03010', '01030', '30100', '10300', '30010', '10030', '30001', '10003'],
            '22': ['00022', '00202', '00220', '02020', '02200', '20200', '20020', '02002', '20002'],
            '112': ['00211', '00121', '00112', '02110', '01210', '01120', '21100', '12100', '11200', '02011', '01021'],
        }