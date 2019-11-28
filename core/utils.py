import numpy as np


def round_custom(num, thresh=0.0001):
    """
    rounds the number in the argument if it is less than threshold value.
    if num < thresh:
        round(num)
    else
        num
    """
    if num < thresh:
        num = round(num)
        if num == -0:
            return 0
    return num


def calculate_similarity(mol1_adj, mol2_adj):
    """
    calculates similarity between two adjacency matrices using hilbert-schmitz method.

    """
    diff = mol1_adj - mol2_adj
    diff_dot = np.dot(diff, np.transpose(np.conjugate(diff)))
    if np.linalg.norm(diff_dot) == 0:
        return 1 - np.sqrt(np.trace(diff_dot))
    else:
        return 1 - np.sqrt(np.trace(diff_dot) / (np.trace(diff) ** 2))


# def process_fidelity(U1, U2, normalize=True):
#     """
#     Calculate the process fidelity given two process operators.
#     temporary function: will be removed XXX: TODO
#
#     Cant think why it was written in the first place.
#     """
#     if normalize:
#         return np.trace(np.dot(U1, U2)) / (np.trace(U1) * np.trace(U2))
#     else:
#         return np.trace(np.dot(U1, U2))


def get_state_map(num_bits):
    """
    temporary function: will be removed XXX: TODO

    this was written during the hackathon as a quick and dirty way. So it should be replaced by an algorithm.
    """
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
            '13': ['00031', '00013', '00301', '00103', '03010', '01030', '30100', '10300', '30010', '10030', '30001',
                   '10003'],
            '22': ['00022', '00202', '00220', '02020', '02200', '20200', '20020', '02002', '20002'],
            '112': ['00211', '00121', '00112', '02110', '01210', '01120', '21100', '12100', '11200', '02011', '01021'],
        }
