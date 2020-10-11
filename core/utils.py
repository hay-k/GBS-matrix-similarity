import numpy as np


def round_custom(num, threshold=0.0001):
    """
    Rounds the number in the argument if its absolute value is less than threshold value.
    """
    if abs(num) < threshold:
        num = round(num)
        if num == -0:
            return 0
    return num


def calculate_similarity(mol1_adj, mol2_adj):
    """
    Calculates similarity between two matrices using hilbert-schmidt method.
    """
    diff = mol1_adj - mol2_adj
    diff_dot = np.dot(diff, np.transpose(np.conjugate(diff)))
    if np.linalg.norm(diff_dot) == 0:
        return 1 - np.sqrt(np.trace(diff_dot))
    else:
        return 1 - np.sqrt(np.trace(diff_dot) / (np.trace(diff) ** 2))
