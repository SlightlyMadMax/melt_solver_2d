import time


def get_remaining_time(n: int, n_t: int, start_time: float) -> float:
    """
    Calculate the approximate remaining calculation time in seconds.
    :param n: current step
    :param n_t: total amount of steps
    :param start_time: value of time.perf_counter() right before the calculation
    :return: estimated remaining time in seconds
    """
    return (time.perf_counter() - start_time) * (n_t - n) / n
