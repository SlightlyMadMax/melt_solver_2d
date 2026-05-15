import os

os.environ["NUMBA_THREADING_LAYER"] = "omp"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["NUMBA_NUM_THREADS"] = "16"
import numba
import numpy as np
import time

from numba import njit, prange


@njit(parallel=True)
def parallel_init(a):
    s = 0.0
    for i in prange(a.shape[0]):
        s += a[i]
    return s


parallel_init(np.zeros(1, dtype=np.float64))

print("Threading layer:", numba.threading_layer())
print("Numba threads: ", numba.get_num_threads())


@njit
def sequential_sum(a):
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i]
    return s


@njit(parallel=True)
def parallel_sum(a):
    s = 0.0
    for i in prange(a.shape[0]):
        s += a[i]
    return s


def benchmark(fn, a, label):
    _ = fn(a)
    t0 = time.perf_counter()
    res = fn(a)
    print(f"{label:12s} result={res:.6f} time={time.perf_counter()-t0:.4f}s")


def main():
    N = 20_000_000
    a = np.random.rand(N).astype(np.float64)
    print(f"\nArray size: {N:,}\n")
    benchmark(sequential_sum, a, "sequential")
    benchmark(parallel_sum, a, "parallel")


if __name__ == "__main__":
    main()
