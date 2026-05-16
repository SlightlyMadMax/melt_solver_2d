import numpy as np


def nusselt_correlation(tau, Ra, c1=0.27, c2=0.0275, n=-2):
    """
    Compute the average Nusselt number using the Jany & Bejan (1988) correlation.

    Nu(τ) = 1/√(2τ) + [c₁·Ra^(1/4) - 1/√(2τ)]·[1 + (c₂·Ra^(3/4)·τ^(3/2))^n]^(1/n)

    Parameters
    ----------
    tau : float or np.ndarray
        Dimensionless time τ = Fo·Ste (Fourier·Stefan)
    Ra : float
        Rayleigh number
    c1 : float, optional (default 0.27)
        Constant c₁ from the correlation
    c2 : float, optional (default 0.0275)
        Constant c₂ from the correlation
    n : float, optional (default -2)
        Exponent n from the correlation

    Returns
    -------
    Nu : float or np.ndarray
        Average Nusselt number at the hot wall
    """
    # Convert to array if not scalar
    tau = np.asarray(tau) if not np.isscalar(tau) else tau

    # Check for non-positive tau
    if np.any(tau <= 0):
        raise ValueError("tau must be positive")

    # Pure conduction term (Neumann solution)
    nu_conduction = 1.0 / np.sqrt(2.0 * tau)

    # Pure convection limit
    nu_convection = c1 * np.power(Ra, 1.0 / 4.0)

    # Correlation factor for transition regime
    correlation_factor = np.power(
        1.0 + np.power(c2 * np.power(Ra, 3.0 / 4.0) * np.power(tau, 3.0 / 2.0), n),
        1.0 / n,
    )

    # Full formula
    Nu = nu_conduction + (nu_convection - nu_conduction) * correlation_factor

    return Nu
