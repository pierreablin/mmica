import cython
cimport cython
import numpy as np
from scipy.linalg.cython_blas cimport ddot, dgemv, dscal, dcopy, daxpy, dnrm2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:] cg_c(double [:, :] B, int i, int max_iter, double tol, int N):
    '''
    Conjugate gradient with preconditioning to find K^{-1}_i.
    '''
    cdef int inc = 1
    cdef int j
    cdef int n
    cdef double one = 1.
    cdef double zero = 0.
    cdef double prod_old
    cdef double alpha
    cdef double minus_alpha
    cdef double prod_new
    cdef double beta
    cdef double inv_bii
    cdef double norm
    cdef double [:] x = np.zeros(N)
    cdef double [:] B_diag = np.empty(N)
    cdef double [:] r = np.empty(N)
    cdef double [:] y = np.empty(N)
    cdef double [:] p = np.empty(N)
    cdef double[:] Ap = np.empty(N)
    for j in range(N):
        B_diag[j] = B[j, j]
    dcopy(&N, &B[i, 0], &inc, &r[0], &inc)
    inv_bii = 1. / B[i, i]
    dscal(&N, &inv_bii, &r[0], &inc)
    x[i] = inv_bii
    r[i] = 0.
    for j in range(N):
        y[j] = r[j] / B_diag[j]
    dcopy(&N, &y[0], &inc, &p[0], &inc)
    prod_old = ddot(&N, &r[0], &inc, &y[0], &inc)
    for n in range(max_iter):
        dgemv('n', &N, &N, &one, &B[0, 0], &N, &p[0], &inc, &zero, &Ap[0], &inc)
        alpha = prod_old / ddot(&N, &p[0], &inc, &Ap[0], &inc)
        minus_alpha = - alpha
        daxpy(&N, &minus_alpha, &p[0], &inc, &x[0], &inc)
        daxpy(&N, &minus_alpha, &Ap[0], &inc, &r[0], &inc)
        norm = dnrm2(&N, &r[0], &inc)
        if norm < tol:
            break
        for j in range(N):
            y[j] = r[j] / B_diag[j]
        prod_new = ddot(&N, &r[0], &inc, &y[0], &inc)
        beta = prod_new / prod_old
        prod_old = prod_new
        for j in range(N):
            p[j] = y[j] + beta * p[j]
    return x
