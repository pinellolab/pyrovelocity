import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
from sklearn.neighbors import KernelDensity


class VMF:
    """
    A class for creating and handling von Mises-Fisher distributions in N dimensions.

    Attributes:
        mu (array-like): The mean direction of the distribution.
        kappa (float): The concentration parameter of the distribution.
        dim (int): The number of dimensions.
    """

    def __init__(self, mu, kappa):
        """
        Stores parameters for a VMF object.

        Args:
            mu (array-like): The mean direction of the distribution.
            kappa (float): The concentration parameter of the distribution.
        """

        self.mu = mu
        self.kappa = kappa
        self.dim = len(mu)

    def sample(self, num_samples):
        """
        Samples from the von Mises-Fisher distribution.

        Args:
            num_samples (int): The number of samples to draw.

        Returns:
            ndarray: The sampled points in the form of a num_samples by dim array.
        """
        result = np.zeros((num_samples, self.dim))

        for i in range(num_samples):
            w = self._sample_weight()
            v = self._sample_orthogonal_to()
            result[i, :] = np.sqrt(1 - w**2) * v + w * self.mu

        return result

    def _sample_weight(self):
        """
        Samples the weight component used for generating N-D vMF samples.

        Returns:
            float: The sampled weight.
        """
        b = (-2 * self.kappa + np.sqrt(4 * self.kappa**2 + (self.dim - 1) ** 2)) / (
            self.dim - 1
        )
        x = (1 - b) / (1 + b)
        c = self.kappa * x + (self.dim - 1) * np.log(1 - x**2)

        while True:
            z = np.random.beta(0.5 * (self.dim - 1), 0.5 * (self.dim - 1))
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            u = np.random.uniform(0, 1)
            if self.kappa * w + (self.dim - 1) * np.log(1 - x * w) - c >= np.log(u):
                return w

    def _sample_orthogonal_to(self):
        """
        Sample a vector orthogonal to the mean direction.

        Returns:
            ndarray: The sampled orthogonal vector.
        """
        v = np.random.randn(self.dim)
        proj_mu_v = self.mu * np.dot(self.mu, v)
        return (v - proj_mu_v) / np.linalg.norm(v - proj_mu_v)


def sample_uniform(dimension, sample_number):
    """
    Samples from a uniform distribution on the N-D unit hypersphere.

    Args:
        sample_number (int): The number of samples to draw.

    Returns:
        ndarray: The sampled points in the form of a sample_number by dimension array.
    """
    result = np.random.randn(sample_number, dimension)
    result /= np.linalg.norm(result, axis=1)[:, np.newaxis]
    return result


