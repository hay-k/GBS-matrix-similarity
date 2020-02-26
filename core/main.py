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
            file.write("{}: {}\n".format(formula, probs))


if __name__ == '__main__':
    molecules = ['H2', 'H2']
    source = 'nist'
    n_mean = 5
    group_type = 'event'
    events = [(2, 1), (3, 1)]
    samples = 1000
    filename = "../results/event_probabilities.txt"

    get_probabilities_mc(molecules,
                         source,
                         n_mean,
                         group_type,
                         events,
                         samples,
                         filename)

