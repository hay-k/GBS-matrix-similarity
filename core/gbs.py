import numpy as np
from thewalrus import quantum
from typing import List
from itertools import permutations


class GBSDevice:
    name: str
    mode_count: int
    cov: np.array

    def __init__(self, name: str, mode_count: int = None):
        self.name = name
        self.mode_count = mode_count
        self.cov = np.array([[], []])

    def encode_matrix(self, matrix: np.array, n_mean: float):
        """
        construct a covariance matrix out of the given matrix, and encode into the GBS device.
        If the device has self.mode_count == None at the time of this function call, then the mode count
        will be defined according to the matrix embedding.

        :param matrix: the matrix to embed into the device
        :param n_mean: mean photon number in the device
        :return: nothing
        """
        if self.mode_count and self.mode_count != 2 * len(matrix):
            raise ValueError("There are not sufficient modes in the device to encode this matrix")
        Q = quantum.gen_Qmat_from_graph(matrix, n_mean=n_mean)
        self.cov = quantum.Covmat(Q)
        self.mode_count = len(matrix)

    def get_state_vector(self):
        return quantum.state_vector(np.zeros(2 * self.mode_count), self.cov)

    def get_density_matrix(self):
        return quantum.density_matrix(np.zeros(2 * self.mode_count), self.cov)

    def get_probability(self, pattern: List[int]):
        """
        :param pattern: a list of photon counts (per mode)
        :return: the probability of the given photon counting event
        """
        amp = quantum.pure_state_amplitude(np.zeros(2 * self.mode_count), self.cov, pattern)
        return np.abs(amp) ** 2

    def get_orbit_probability(self, pattern: List[int]):
        """
        :param pattern: a list of photon counts (per mode). One of the photon count events in the orbit
        :return: the probability of the given photon counting event
        """
        def _unique_permutations():
            return set(permutations(pattern))

        perms = _unique_permutations()
        prob = 0
        for item in perms:
            prob += self.get_probability(item)

        return prob

    def get_all_orbits(self, max_photons: int):
        """
        Generate a list of photon counting events, where each event is a representative of one orbit,
        in particular, the the representative where the number of photons are in descending order.

        :param max_photons: maximum number of registered photons in a counting event
        :return: a list of photon counting events. List[List]
        """

        def _expand(pattern: List, expansion_list: List[List]):
            if pattern[0] <= pattern[1]:
                return expansion
            for i in range(1, len(pattern)):
                b = pattern.copy()
                b[0] -= 1
                if b[i] < b[i - 1]:
                    b[i] += 1
                    # TODO: eliminate the need for this duplication check
                    if b not in expansion_list:
                        expansion_list.append(b)
                    _expand(b, expansion_list)

        if max_photons > self.mode_count:
            raise NotImplementedError("max_photons larger than the number of modes not supported currently")

        expansion = []
        for n in range(max_photons + 1):
            a = [n]
            a.extend([0] * (max_photons - 1))
            expansion.append(a)
            _expand(a, expansion)

        return expansion

    def get_feature_vector(self, max_photons: int):
        orbits = self.get_all_orbits(max_photons)
        feature_vector = []
        for orbit in orbits:
            prob = self.get_orbit_probability(orbit)
            feature_vector.extend([prob])

        return feature_vector
