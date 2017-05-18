from __future__ import print_function

import numpy as np


from numba import cuda
from numba import jit

from timeit import default_timer as time



def sum(a, b, c):
    for i in range(N):
        c[i] = a[i] + b[i]

griddim = 146432, 1
blockdim = 1024, 1, 1
N = griddim[0] * blockdim[0]
print("N", N)
#cuda_sum_configured = cuda_sum.configure(griddim, blockdim)
a = np.array(np.random.random(N), dtype=np.float64)
b = np.array(np.random.random(N), dtype=np.float64)
c = np.empty_like(a)

ts = time()
sum(a, b, c)
te = time()
print(te - ts)
