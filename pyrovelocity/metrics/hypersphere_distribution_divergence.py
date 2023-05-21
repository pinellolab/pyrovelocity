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


def kl_divergence(p, q, dimension, sample_number):
    """
    Computes the Kullback-Leibler divergence between two distributions, estimated using kernel density estimation.
    This uses Scott's rule to estimate the bandwidth parameter.

    Scott, D.W. (1992) Multivariate Density Estimation. Theory, Practice and Visualization. New York: Wiley.

    Args:
        p (ndarray): Samples from the first distribution.
        q (ndarray): Samples from the second distribution.
        sample_number (float): The number of samples used to estimate the distributions.
        dimension (int): The dimension of the distributions.

    Returns:
        float: The Kullback-Leibler divergence between the first and second distributions.
        float: The bandwidth parameter for the kernel density estimation used in the Kullback-Leibler divergence computation.
    """

    bandwidth = sample_number ** (-1 / (dimension + 4))

    p_kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(p)
    q_kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(q)

    p_scores = p_kde.score_samples(p)
    q_scores = q_kde.score_samples(p)

    kl_div = np.sum(p_scores - q_scores) / len(p)
    return kl_div, bandwidth


def plot_samples_and_kde(ax, samples_vmf, samples_uniform, kappa, dim=3, bandwidth=0.1):
    if dim not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3.")

    if dim == 3:
        ax.scatter(
            samples_vmf[:, 0], samples_vmf[:, 1], samples_vmf[:, 2], color="darkgreen"
        )
        ax.set_title(rf"VMF distribution ($\kappa$={kappa})")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

    else:
        kde_vmf = KernelDensity(bandwidth=bandwidth).fit(samples_vmf)
        kde_uniform = KernelDensity(bandwidth=bandwidth).fit(samples_uniform)

        # grid
        num_points = 100
        x = np.linspace(min(samples_vmf[:, 0]), max(samples_vmf[:, 0]), num_points)
        y = np.linspace(min(samples_vmf[:, 1]), max(samples_vmf[:, 1]), num_points)
        X, Y = np.meshgrid(x, y)
        xy = np.vstack([X.ravel(), Y.ravel()]).T

        # evaluate
        Z_vmf = np.exp(kde_vmf.score_samples(xy)).reshape(X.shape)
        Z_uniform = np.exp(kde_uniform.score_samples(xy)).reshape(X.shape)

        # 2-D vMF distribution with contours
        ax.scatter(samples_vmf[:, 0], samples_vmf[:, 1], color="darkgreen")
        ax.contour(X, Y, Z_vmf, cmap="viridis")
        ax.set_title(rf"VMF distribution ($\kappa$={kappa})")
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])

