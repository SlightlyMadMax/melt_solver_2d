from numba import njit


@njit
def get_indicator_function(
    u: float, u_pt_ref: float, delta_u: float, eps: float
) -> float:
    """
    Indicator function for the fictitious domain method.
    Is equal to 0 for liquid phase and 1 / eps^2 for solid phase.

    :param u: The temperature value (deviation from the reference temperature).
    :param u_pt_ref: The phase transition temperature (deviation from the reference temperature).
    :param delta_u: The characteristic temperature difference.
    :param eps: A big parameter.
    :return: The value of the indicator function at u.
    """
    if u * delta_u - u_pt_ref > 0.0:
        return 0.0
    return 1.0 / (eps * eps)
