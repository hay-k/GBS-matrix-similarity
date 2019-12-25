import numpy as np
from thewalrus import quantum
import strawberryfields as sf
from typing import List
from sympy.utilities.iterables import multiset_permutations


class GBSDevice:

    def __init__(self, name: str):
        self.name = name

        self.mode_count = None
        self.cov = None  # covariance matrix
        self.mu = None  # vector of means

    def encode_matrix(self, matrix: np.array, n_mean: float, mu: np.array):
        """
        construct a covariance matrix out of the given matrix, and encode into the GBS device.
        If the device has self.mode_count == None at the time of this function call, then the mode count
        will be defined according to the matrix embedding.

        :param matrix: the matrix to embed into the device
        :param n_mean: mean photon number in the device
        :param mu: vector of means
        :return: nothing
        """
        Q = quantum.gen_Qmat_from_graph(matrix, n_mean=n_mean)
        self.mode_count = len(matrix)
        self.cov = quantum.Covmat(Q)
        self.mu = mu

    def get_state_vector(self):
        return quantum.state_vector(self.mu, self.cov)

    def get_density_matrix(self, cutoff = 5):
        return quantum.density_matrix(self.mu, self.cov, cutoff=cutoff)

    def get_probability(self, pattern: List[int]):
        """
        :param pattern: a list of photon counts (per mode)
        :return: the probability of the given photon counting event
        """
        return quantum.density_matrix_element(self.mu, self.cov, pattern, pattern).real

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

    def get_event_probability_exact(self, event: tuple):
        """
        :param event: a specification of an event. Tuple of length 2
        :return: the exact probability of the given event
        """
        photons, max_photons_per_mode = event
        prob = 0
        for orbit in sf.apps.similarity.orbits(photons):
            if max(orbit) <= max_photons_per_mode:
                prob += self.get_orbit_probability_exact(orbit)

        return prob

    def get_orbit_probability_mc(self, orbit: list, samples: int = 1000):
        """
        Calculate approximate probability of the orbit with Monte Carlo.
        Similar to function sf.apps.similarity.prob_orbit_mc()

        :param orbit: a specification of an orbit
        :param samples: number of Monte Carlo samples.
        :return: the mc approximate probability of the given orbit
        """
        prob = 0
        for _ in range(samples):
            sample = sf.apps.similarity.orbit_to_sample(orbit, self.mode_count)
            prob += self.get_probability(sample)

        prob = prob * sf.apps.similarity.orbit_cardinality(orbit, self.mode_count) / samples

        return prob

    def get_event_probability_mc(self, event: tuple, samples: int = 1000):
        """
        Calculate approximate probability of the event with Monte Carlo.
        Similar to function sf.apps.similarity.prob_event_mc()

        :param event: a specification of an event. Tuple of length 2
        :param samples: number of Monte Carlo samples.
        :return: the mc approximate probability of the given event
        """
        photons, max_photons_per_mode = event

        prob = 0
        for _ in range(samples):
            sample = sf.apps.similarity.event_to_sample(photons, max_photons_per_mode, self.mode_count)
            prob += self.get_probability(sample)

        prob = prob * sf.apps.similarity.event_cardinality(photons, max_photons_per_mode, self.mode_count) / samples

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

    def get_orbit_feature_vector(self, max_photons: int, mc=False):
        """
        Calculate and return a feature vector, based on orbit probabilities

        :param max_photons: max number of photons in a detection event
        :param mc: if True, uses monte carlo simulation to. If False, calculates exact
        :return: feature vector comprised of orbit probabilities
        """
        orbit_prob = self.get_orbit_probability_mc if mc else self.get_orbit_probability_exact

        # For now, for mc=True case we use the default number of samples=1000,
        # unless we come up with a better idea for the value of samples
        return [orbit_prob(orbit)
                for n in range(max_photons + 1)
                for orbit in sf.apps.similarity.orbits(n)]

    def get_event_feature_vector(self, max_photons: int, max_photons_per_mode, mc=False):
        """
        Calculate and return a feature vector, based on event probabilities

        :param max_photons: max number of photons in a detection event
        :param max_photons_per_mode: max number of photons per mode
        :param mc: if True, uses monte carlo simulation to. If False, calculates exact
        :return: feature vector comprised of event probabilities
        """
        event_prob = self.get_event_probability_mc if mc else self.get_event_probability_exact

        # For now, for mc=True case we use the default number of samples=1000,
        # unless we come up with a better idea for the value of samples
        return [event_prob((photons, max_photons_per_mode)) for photons in range(max_photons + 1)]
