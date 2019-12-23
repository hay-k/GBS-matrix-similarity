import numpy as np
from thewalrus import quantum
import strawberryfields as sf
from typing import List
from sympy.utilities.iterables import multiset_permutations


class GBSDevice:

    def __init__(self, name: str):
        self.name = name
        self.mode_count = None
        self.state = None
        self.n_mean = None

    def encode_matrix(self, matrix: np.array, n_mean: float):
        """
        construct a covariance matrix out of the given matrix, and encode into the GBS device.
        If the device has self.mode_count == None at the time of this function call, then the mode count
        will be defined according to the matrix embedding.

        :param matrix: the matrix to embed into the device
        :param n_mean: mean photon number in the device
        :return: nothing
        """
        mode_count = len(matrix)
        mean_photon_per_mode = n_mean / float(mode_count)
        program = sf.Program(mode_count)
        with program.context as q:
            sf.ops.GraphEmbed(matrix, mean_photon_per_mode=mean_photon_per_mode) | q

        eng = sf.LocalEngine(backend="gaussian")
        result = eng.run(program)

        self.mode_count = mode_count
        self.n_mean = n_mean
        self.state = result.state

    def get_state_vector(self):
        return quantum.state_vector(self.state.means(),
                                    self.state.cov())

    def get_density_matrix(self):
        return quantum.density_matrix(self.state.means(),
                                      self.state.cov())

    def get_probability(self, pattern: List[int]):
        """
        :param pattern: a list of photon counts (per mode)
        :return: the probability of the given photon counting event
        """
        photons = sum(pattern)
        return self.state.fock_prob(pattern, cutoff=photons + 1)

    def get_orbit_probability_exact(self, orbit: List[int]):
        """
        :param orbit: a specification of an orbit
        :return: the exact probability of the given orbit
        """
        pattern = orbit + [0] * (self.mode_count - len(orbit))
        perms = list(multiset_permutations(pattern))
        prob = 0
        for item in perms:
            prob += self.get_probability(item)

        return prob

    def get_orbit_probability_mc(self, orbit: list, samples: int = 1000):
        """
        Calculate approximate probability of the orbit with Monte Carlo.
        Similar to function sf.apps.similarity.prob_orbit_mc()

        :param orbit: a specification of an orbit
        :param samples: number of Monte Carlo samples
        :return: the mc approximate probability of the given orbit
        """
        photons = sum(orbit)
        prob = 0
        for _ in range(samples):
            sample = sf.apps.similarity.orbit_to_sample(orbit, self.mode_count)
            prob += self.state.fock_prob(sample, cutoff=photons + 1)

        prob = prob * sf.apps.similarity.orbit_cardinality(orbit, self.mode_count) / samples

        return prob

    def get_all_orbit_representatives(self, max_photons: int):
        """
        Generate a list of photon counting events, where each event is a representative of one orbit,
        in particular, the the representative where the number of photons are in descending order.

        The function does this by first taking a pattern with all photons in the first mode,
        then generates the other representatives by subtracting a photon from the first mode
        and distributing to the other modes, keeping the descending order.

        :param max_photons: maximum number of registered photons in a counting event
        :return: a list of photon counting events. List[List]
        """
        def _expand(pattern: List, start_index: int):
            # :param start_index: the index from where to start distributing the photons in the first position
            if pattern[0] <= pattern[1]:
                return []
            pattern_list = []
            index = start_index
            for i in range(start_index, len(pattern)):
                b = pattern.copy()
                b[0] -= 1
                if b[i] < b[i - 1]:
                    b[i] += 1
                    pattern_list.append(b)
                    pattern_list.extend(_expand(b, index))
                    index += 1
            return pattern_list

        expansion = []
        for n in range(max_photons + 1):
            a = [n] + [0] * (self.mode_count - 1)
            expansion.append(a)
            expansion.extend(_expand(a, 1))

        return expansion

    def get_feature_vector_exact(self, max_photons: int):
        orbits = self.get_all_orbit_representatives(max_photons)
        feature_vector = [self.get_orbit_probability_exact(orbit) for orbit in orbits]

        return feature_vector
