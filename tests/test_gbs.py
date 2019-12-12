from functools import partial
from numpy import ones, mean, std
from math import factorial
from itertools import permutations
from core.gbs import GBSDevice
import unittest
import timeit


class GBSDeviceTimingTest(unittest.TestCase):

    def test_feature_vector(self):
        mode_count = 10
        max_photons = 5
        molecule = GBSDevice(name='test_molecule', mode_count=mode_count)
        orbit_reps = molecule.get_all_orbit_representatives(max_photons)

        all_patterns = []
        for orbit in orbit_reps:
            all_patterns.extend(set(permutations(orbit)))
        expected_length = 0

        for i in range(max_photons + 1):
            # count the number of all possible detection events with total photon number up to (including) max_photons
            _tmp_len = factorial(mode_count + i - 1) // (factorial(i) * factorial(mode_count - 1))
            expected_length += _tmp_len

        self.assertEqual(len(all_patterns), expected_length)

    def test_three_photons(self):
        times = []
        for matrix_dim in range(3, 15):
            matrix = ones((matrix_dim, matrix_dim))
            molecule = GBSDevice(name='test_molecule')
            molecule.encode_matrix(matrix, 1)

            func = partial(molecule.get_probability, [1, 1, 1] + [0] * (matrix_dim - 3))
            avg_timing = timeit.timeit(func, number=100) / 100
            times.append(avg_timing)

            print("Average timing for a {matrix_dim}x{matrix_dim} matrix: {timing} sec"
                  .format(matrix_dim=matrix_dim, timing=avg_timing))

        self.assertLess(std(times), mean(times))
