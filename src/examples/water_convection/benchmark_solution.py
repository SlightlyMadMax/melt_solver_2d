import numpy as np


def calculate_U_profile_X05(Y):
    """
    Calculate horizontal velocity U at X = 0.5 for given Y coordinate(s)

    Parameters:
    Y : float or array-like
        Y coordinate(s) where to evaluate U (should be between 0 and 1)

    Returns:
    float or array
        U velocity component at X = 0.5
    """
    # Coefficients from Table A1 for U at X = 0.5
    coefficients = [
        0.653255375988277,  # a0
        -236.702203764653,  # a1
        1443.71621734046,  # a2
        -13999.9971459506,  # a3
        -48978.2873061909,  # a4
        769502.177696391,  # a5
        -2826411.42861687,  # a6
        5049355.25968998,  # a7
        -4889309.49455426,  # a8
        2473294.32955038,  # a9
        -514661.642022168,  # a10
    ]

    Y = np.asarray(Y)
    result = np.zeros_like(Y, dtype=float)

    for i, coeff in enumerate(coefficients):
        result += coeff * (Y**i)

    return result


def calculate_W_profile_X05(Y):
    """
    Calculate vertical velocity W at X = 0.5 for given Y coordinate(s)

    Parameters:
    Y : float or array-like
        Y coordinate(s) where to evaluate W (should be between 0 and 1)

    Returns:
    float or array
        W velocity component at X = 0.5
    """
    # Coefficients from Table A1 for W at X = 0.5
    coefficients = [
        -0.0182133390825522,  # a0
        -0.534506952806084,  # a1
        -4649.62374660758,  # a2
        9166.34090898581,  # a3
        184756.318840003,  # a4
        -2267214.57474188,  # a5
        13921830.6389979,  # a6
        -50905496.7836152,  # a7
        117326421.048108,  # a8
        -175454949.94745,  # a9
        170542299.447756,  # a10
        -104264882.357183,  # a11
        36505160.4951902,  # a12
        -5592440.58260601,  # a13
    ]

    Y = np.asarray(Y)
    result = np.zeros_like(Y, dtype=float)

    for i, coeff in enumerate(coefficients):
        result += coeff * (Y**i)

    return result


def calculate_T_profile_X05(Y):
    """
    Calculate temperature T at X = 0.5 for given Y coordinate(s)

    Parameters:
    Y : float or array-like
        Y coordinate(s) where to evaluate T (should be between 0 and 1)

    Returns:
    float or array
        Temperature at X = 0.5
    """
    # Coefficients from Table A1 for Temperature at X = 0.5
    coefficients = [
        0.375731268271168,  # a0
        0.0646566206852292,  # a1
        -3.44261930694882,  # a2
        80.5716617494023,  # a3
        -849.389178138508,  # a4
        5426.31856180659,  # a5
        -20619.6870300723,  # a6
        47584.9389176856,  # a7
        -66982.5680747791,  # a8
        54146.8042661755,  # a9
        -18312.94874948,  # a10
        -5638.00828334596,  # a11
        6840.33395345218,  # a12
        -1672.49783568399,  # a13
    ]

    Y = np.asarray(Y)
    result = np.zeros_like(Y, dtype=float)

    for i, coeff in enumerate(coefficients):
        result += coeff * (Y**i)

    return result


def calculate_T_profile_Y05(X):
    """
    Calculate temperature T at Y = 0.5 for given X coordinate(s)

    Parameters:
    X : float or array-like
        X coordinate(s) where to evaluate T (should be between 0 and 1)

    Returns:
    float or array
        Temperature at Y = 0.5
    """
    # Coefficients from Table A1 for Temperature at Y = 0.5
    coefficients = [
        0.999467521831559,  # a0
        -6.23069515529224,  # a1
        -18.9999577130502,  # a2
        433.527770212382,  # a3
        7318.66314332766,  # a4
        -180265.707163689,  # a5
        1714792.57838228,  # a6
        -9862261.15685317,  # a7
        38472869.7475459,  # a8
        -106665059.51901,  # a9
        214297962.730994,  # a10
        -313025047.586471,  # a11
        328987879.084903,  # a12
        -242193161.156869,  # a13
        118435075.266555,
        -34530031.2923873,
        4539519.05438302,
    ]

    X = np.asarray(X)
    result = np.zeros_like(X, dtype=float)

    for i, coeff in enumerate(coefficients):
        result += coeff * (X**i)

    return result
