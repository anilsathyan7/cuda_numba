from __future__ import print_function

import numpy as np

from numba import cuda

from timeit import default_timer as time


@cuda.jit("(float64[:], float64[:], float64[:])")
def cuda_sum(a, b, c):
    i = cuda.grid(1)
    c[i] = a[i] + b[i]

griddim = 146432, 1
blockdim = 1024,1,1
N = griddim[0]*griddim[1] * blockdim[0] * blockdim[1] * blockdim[2]
print("N", N)
cuda_sum_configured = cuda_sum.configure(griddim, blockdim)
a = np.array(np.random.random(N), dtype=np.float64)
b = np.array(np.random.random(N), dtype=np.float64)
c = np.empty_like(a)

ts = time()
cuda_sum_configured(a, b, c)
te = time()
print(te - ts)
assert (a + b == c).all()
