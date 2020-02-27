from core.gbs import GBSDevice
import core.molecular_data as md
import numpy as np
import os


def get_probabilities_mc(formula_list, source, n_mean, group_type, groups, samples, output_file, append=False):
    if not append and os.path.exists(output_file):
        os.remove(output_file)

    for formula in formula_list:
        single_body, two_body = md.get_single_two_body(formula, source)
        vector, matrix = md.get_molecular_matrix_and_vector(single_body, two_body)

        molecule = GBSDevice(formula)
        molecule.encode_matrix(matrix, n_mean, np.append(vector, vector))

        if group_type == 'orbit':
            get_probability_func = molecule.get_orbit_probability_mc
        else:
            get_probability_func = molecule.get_event_probability_mc

        probs = [(group, get_probability_func(group, samples=samples)) for group in groups]

        with open(output_file, 'a') as file:
            file.write("{} (n_mean = {}): {}\n".format(formula, n_mean, probs))


if __name__ == '__main__':
    get_probabilities_mc(formula_list=['H2', 'H2O', 'HCl', 'LiH', 'NaH', 'KH', 'LiOH', 'NaOH', 'KOH'],
                         source='nist',
                         n_mean=7,
                         group_type='orbit',
                         groups=[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
                         samples=10000,
                         output_file='../results/orbit_probabilities.txt',
                         append=False)
