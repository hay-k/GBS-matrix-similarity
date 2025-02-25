import numpy as np
from thewalrus import quantum
import strawberryfields.apps.similarity as sfs
from typing import List
from sympy.utilities.iterables import multiset_permutations
from functools import partial


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
        for orbit in sfs.orbits(photons):
            if max(orbit) <= max_photons_per_mode:
                prob += self.get_orbit_probability_exact(orbit)

        return prob

    def get_orbit_probability_mc(self, orbit: list, samples: int = 1000):
        """
        Calculate approximate probability of the orbit with Monte Carlo.
        Similar to function sfs.prob_orbit_mc()

        :param orbit: a specification of an orbit
        :param samples: number of Monte Carlo samples.
        :return: the mc approximate probability of the given orbit
        """
        prob = 0
        for _ in range(samples):
            sample = sfs.orbit_to_sample(orbit, self.mode_count)
            prob += self.get_probability(sample)

        prob = prob * sfs.orbit_cardinality(orbit, self.mode_count) / samples

        return prob

    def get_event_probability_mc(self, event: tuple, samples: int = 1000):
        """
        Calculate approximate probability of the event with Monte Carlo.
        Similar to function sfs.prob_event_mc()

        :param event: a specification of an event. Tuple of length 2
        :param samples: number of Monte Carlo samples.
        :return: the mc approximate probability of the given event
        """
        photons, max_photons_per_mode = event

        prob = 0
        for _ in range(samples):
            sample = sfs.event_to_sample(photons, max_photons_per_mode, self.mode_count)
            prob += self.get_probability(sample)

        prob = prob * sfs.event_cardinality(photons, max_photons_per_mode, self.mode_count) / samples

        return prob

    def get_orbit_feature_vector(self, max_photons: int, mc=False, samples: int = 1000):
        """
        Calculate and return a feature vector, based on orbit probabilities

        :param max_photons: max number of photons in a detection event
        :param mc: if True, uses monte carlo simulation to. If False, calculates exact
        :param samples: number of samples for monte carlo simulation
        :return: feature vector comprised of orbit probabilities
        """
        orbit_prob = partial(self.get_orbit_probability_mc, samples=samples) if mc else self.get_orbit_probability_exact

        return [orbit_prob(orbit)
                for n in range(max_photons + 1)
                for orbit in sfs.orbits(n)]

    def get_event_feature_vector(self, max_photons: int, max_photons_per_mode, mc=False, samples: int = 1000):
        """
        Calculate and return a feature vector, based on event probabilities

        :param max_photons: max number of photons in a detection event
        :param max_photons_per_mode: max number of photons per mode
        :param mc: if True, uses monte carlo simulation to. If False, calculates exact
        :param samples: number of samples for monte carlo simulation
        :return: feature vector comprised of event probabilities
        """
        event_prob = partial(self.get_event_probability_mc, samples=samples) if mc else self.get_event_probability_exact

        return [event_prob((photons, max_photons_per_mode))
                for photons in range(max_photons + 1)]
