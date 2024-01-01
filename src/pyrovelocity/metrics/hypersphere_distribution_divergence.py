import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
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
        b = (
            -2 * self.kappa + np.sqrt(4 * self.kappa**2 + (self.dim - 1) ** 2)
        ) / (self.dim - 1)
        x = (1 - b) / (1 + b)
        c = self.kappa * x + (self.dim - 1) * np.log(1 - x**2)

        while True:
            z = np.random.beta(0.5 * (self.dim - 1), 0.5 * (self.dim - 1))
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            u = np.random.uniform(0, 1)
            if self.kappa * w + (self.dim - 1) * np.log(
                1 - x * w
            ) - c >= np.log(u):
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


def kl_divergence(p, q, bandwidth=None):
    """
    Computes the Kullback-Leibler divergence between two distributions, estimated using kernel density estimation.
    This uses Scott's rule to estimate the bandwidth parameter.

    Scott, D.W. (1992) Multivariate Density Estimation. Theory, Practice and Visualization. New York: Wiley.

    Args:
        p (ndarray): Samples from the first distribution.
        q (ndarray): Samples from the second distribution.
        bandwidth (float, optional): The bandwidth parameter for the kernel density estimation.

    Returns:
        float: The Kullback-Leibler divergence between the first and second distributions.
        float: The bandwidth parameter for the kernel density estimation used in the Kullback-Leibler divergence computation.
    """

    sample_number, dimension = p.shape

    if bandwidth is None:
        bandwidth = sample_number ** (-1 / (dimension + 4))

    p_kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(p)
    q_kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(q)

    p_scores = p_kde.score_samples(p)
    q_scores = q_kde.score_samples(p)

    kl_div = np.sum(p_scores - q_scores) / sample_number

    return kl_div, bandwidth


def js_divergence(p, q, bandwidth=None):
    """
    Computes the Jensen-Shannon divergence between two distributions using `kl_divergence`.

    Args:
        p (ndarray): Samples from the first distribution.
        q (ndarray): Samples from the second distribution.
        bandwidth (float, optional): The bandwidth parameter for the kernel density estimation.

    Returns:
        float: The Jensen-Shannon divergence between the first and second distributions.
    """

    # Calculate the KL divergences and ignore bandwidths
    kl_pq, bandwidth = kl_divergence(p, q, bandwidth)
    kl_qp, _ = kl_divergence(q, p, bandwidth)

    # Calculate the JS divergence
    js_div = 0.5 * kl_pq + 0.5 * kl_qp

    return js_div, bandwidth


def mmd(p, q, gamma=None):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two multidimensional distributions using the RBF kernel.

    Args:
        p (ndarray): Samples from the first distribution.
        q (ndarray): Samples from the second distribution.
        gamma (float, optional): The gamma parameter for the RBF kernel. If not provided, it will be set to the inverse
                                 of the median of the pairwise distances in the combined data.

    Returns:
        float: The Maximum Mean Discrepancy between the first and second distributions.
    """
    combined = np.vstack((p, q))

    if gamma is None:
        pairwise_dists = euclidean_distances(combined, combined)

        gamma = 1.0 / np.median(pairwise_dists)

    p_kernel = rbf_kernel(p, p, gamma)
    q_kernel = rbf_kernel(q, q, gamma)
    cross_kernel = rbf_kernel(p, q, gamma)

    p_mean = np.mean(p_kernel)
    q_mean = np.mean(q_kernel)
    cross_mean = np.mean(cross_kernel)

    mmd = p_mean - 2 * cross_mean + q_mean

    return mmd, gamma


def plot_samples_and_kde(
    ax, samples_vmf, samples_uniform, kappa, dim=3, bandwidth=0.1
):
    if dim not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3.")

    if dim == 3:
        ax.scatter(
            samples_vmf[:, 0],
            samples_vmf[:, 1],
            samples_vmf[:, 2],
            color="darkgreen",
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
        x = np.linspace(
            min(samples_vmf[:, 0]), max(samples_vmf[:, 0]), num_points
        )
        y = np.linspace(
            min(samples_vmf[:, 1]), max(samples_vmf[:, 1]), num_points
        )
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


def sample_and_plot_vMF(num_samples, kappas, dim):
    """
    Demonstrates sampling from an N-dimensional von Mises-Fisher
    distribution with different scale parameters, and compares them to a
    uniform distribution on the unit hypersphere using the Kullback-Leibler divergence.
    This quantifies the extent to which samples of N-dimensional vectors are
    pointing in a coherent direction versus uniformly random directions.

    Args:
        num_samples (int): The number of samples to draw from each distribution.
        kappas (list of float): The scale parameters for the von Mises-Fisher distributions.
        dim (int): The number of dimensions for the distributions.
    """

    mu = np.ones(dim) / np.sqrt(dim)
    print(mu)
    vmf = VMF(mu, kappas[0])
    uniform_samples = sample_uniform(dim, num_samples)

    kl_divs = []

    if dim in [2, 3]:
        fig = plt.figure(figsize=(18, 18))

    # Sample from von Mises-Fisher distributions and
    # compute KL divergence from uniform samples
    for idx, kappa in enumerate(kappas):
        vmf.kappa = kappa  # Update kappa
        vmf_samples = vmf.sample(num_samples)
        kl_div, bandwidth = mmd(vmf_samples, uniform_samples)
        print(f"KL divergence for kappa = {kappa}: {kl_div:.3f}")

        kl_divs.append(kl_div)

        if dim == 2:
            ax = fig.add_subplot(4, 4, idx + 1)
            plot_samples_and_kde(
                ax, vmf_samples, uniform_samples, kappa, dim, bandwidth
            )
        elif dim == 3:
            ax = fig.add_subplot(4, 4, idx + 1, projection="3d")
            plot_samples_and_kde(
                ax, vmf_samples, uniform_samples, kappa, dim, bandwidth
            )

    if dim in [2, 3]:
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        plt.savefig(f"vmf_samples_{dim}D_kde_subplots.pdf", format="pdf")
        plt.savefig(f"vmf_samples_{dim}D_kde_subplots.png", format="png")
        plt.show()

    # Plot KL divergence vs scale parameter
    plt.figure(figsize=(6, 6))
    plt.plot(kappas, kl_divs, "o", color="darkgreen", markersize=10)
    plt.xlabel(r"$\kappa$")
    plt.ylabel("KL Divergence")
    plt.title(f"{dim}D KL Divergence vs Scale parameter of vMF distribution")
    plt.savefig(f"kl_div_vs_kappa_{dim}D.pdf", format="pdf")
    plt.savefig(f"kl_div_vs_kappa_{dim}D.png", format="png")
    plt.show()

    return bandwidth


if __name__ == "__main__":
    """
    Compute examples over a range of dimensions and scale parameters.

    > montage -geometry 800x kl_div_vs_kappa_2D.png kl_div_vs_kappa_3D.png \
        kl_div_vs_kappa_4D.png kl_div_vs_kappa_10D.png kl_div_vs_kappa_100D.png \
        kl_div_vs_kappa_1000D.png ND_KLD_vs_vMF_kappa.png
    """
    num_samples = 30
    kappas = [1, 3, 4, 6, 8, 10, 15, 20, 35, 50, 75, 100, 150, 200, 250, 300]
    dims = [2, 3, 4, 10, 20, 40, 80, 100, 1000]

    for dim in dims:
        print(f"\n==Dimension: {dim}==\n")
        bandwidth = sample_and_plot_vMF(num_samples, kappas, dim)
        print(f"KD bandwidth: {bandwidth}")
