from functools import partial
from numpy import ones, mean, std
from core.gbs import GBSDevice
import unittest
import timeit


class GBSDeviceTimingTest(unittest.TestCase):

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
