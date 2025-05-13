from numba import njit


@njit
def dilate_mask(mask, dilated, R):
    n_y, n_x = mask.shape
    for j in range(n_y):
        for i in range(n_x):
            included = False
            for dj in range(-R, R + 1):
                for di in range(-R, R + 1):
                    jj = j + dj
                    ii = i + di
                    if 0 <= jj < n_y and 0 <= ii < n_x:
                        if mask[jj, ii]:
                            included = True
                            # break out early
                            dj = di = R + 1
            dilated[j, i] = included
