import numpy as np
from beartype import beartype
from numpy.typing import ArrayLike
from scipy.optimize import fsolve

from pyrovelocity.logging import configure_logging

__all__ = [
    "lognormal_tail_probability",
    "solve_for_lognormal_sigma_given_threshold_and_tail_mass",
    "solve_for_lognormal_mu_given_threshold_and_tail_mass",
]

logger = configure_logging(__name__)


@beartype
def lognormal_tail_probability(
    mu: float | ArrayLike,
    sigma: float | ArrayLike,
    threshold: float,
    upper: bool = True,
) -> ArrayLike:
    """
    Calculate the tail probability for a lognormal distribution.

    Args:
        mu (float): mean of the underlying normal distribution.
        sigma (float): standard deviation of the underlying normal distribution.
        threshold (float): threshold value to calculate the tail probability.
        tail_mass (float): desired mass in the tail.
        upper (bool): if True, calculate upper tail, else lower.

    Returns:
        float: tail probability.

    Examples:
        Calculate the upper tail probability beyond a threshold of 0.5 for a
        lognormal distribution with mu=0 and sigma=1:

        >>> lognormal_tail_probability(0.0, 1.0, 0.5, upper=True)
        0.7558914...

        Calculate the lower tail probability below a threshold of 0.5 for a
        lognormal distribution with mu=0 and sigma=1:

        >>> lognormal_tail_probability(0.0, 1.0, 0.5, upper=False)
        0.2441085...
    """
    from scipy.stats import lognorm

    if upper:
        return 1.0 - lognorm.cdf(threshold, s=sigma, scale=np.exp(mu))
    else:
        return lognorm.cdf(threshold, s=sigma, scale=np.exp(mu))


@beartype
def solve_for_lognormal_sigma_given_threshold_and_tail_mass(
    mu: float,
    threshold: float,
    tail_mass: float = 0.05,
    upper: bool = True,
    sigma_guess: float = 0.1,
) -> float:
    """
    Solve for sigma of a lognormal distribution to achieve a specified tail
    probability.

    Args:
        mu (float): mean of the underlying normal distribution.
        threshold (float): threshold value for the tail probability calculation.
        tail_mass (float): desired mass in the tail.
        upper (bool): if True, solve for upper tail, else lower.

    Returns:
        float: sigma that meets the tail probability requirement.

    Examples:
        Solve for sigma to achieve an upper tail probability of 0.05 beyond a
        threshold of 1.5 with mu=0.0:

        >>> solve_for_lognormal_sigma_given_threshold_and_tail_mass(
        ...     0.0, 1.5, tail_mass=0.05, upper=True
        >>> )
        0.2465052...

        Solve for sigma to achieve a lower tail probability of 0.05 below a
        threshold of 1.5 with mu=3.0:

        >>> solve_for_lognormal_sigma_given_threshold_and_tail_mass(
        ...     3.0, 1.5, tail_mass=0.05, upper=False
        >>> )
        1.5773652...
    """

    func_to_solve = lambda sigma: (
        lognormal_tail_probability(
            mu,
            sigma,
            threshold,
            upper,
        )
        - tail_mass
    )
    sigma_solution = fsolve(
        func_to_solve,
        x0=sigma_guess,
    )
    return sigma_solution[0]


@beartype
def solve_for_lognormal_mu_given_threshold_and_tail_mass(
    sigma: float,
    threshold: float,
    tail_mass: float = 0.05,
    upper: bool = True,
) -> float:
    """
    Solve for mu of a lognormal distribution to achieve a specified tail
    probability.

    Args:
        sigma (float): standard deviation of the underlying normal distribution.
        threshold (float): threshold value for the tail probability calculation.
        tail_mass (float): desired mass in the tail.
        upper (bool): if True, solve for upper tail, else lower.

    Returns:
        float: mu that meets the tail probability requirement.

    Examples:
        Solve for mu to achieve an upper tail probability of 0.05 beyond a
        threshold of 1.5 with sigma=0.25:

        >>> solve_for_lognormal_mu_given_threshold_and_tail_mass(
        ...     0.25, 1.5, tail_mass=0.05, upper=True
        >>> )
        -0.0057482...

        Solve for mu to achieve a lower tail probability of 0.05 below a
        threshold of 1.5 with sigma=0.25:

        >>> solve_for_lognormal_mu_given_threshold_and_tail_mass(
        ...     0.25, 1.5, tail_mass=0.05, upper=False
        >>> )
        0.8166785...
    """
    mu_guess = np.log(1.0)

    func_to_solve = lambda mu: (
        lognormal_tail_probability(
            mu,
            sigma,
            threshold,
            upper,
        )
        - tail_mass
    )
    mu_solution = fsolve(
        func_to_solve,
        x0=mu_guess,
    )
    return mu_solution[0]
