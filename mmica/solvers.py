import numpy as np
import numba as nb
from numba import njit

from ._utils import cg
from ._densities import Huber


def solver_incremental(X, max_iter=100, batch_size=100, W_init=None,
                       density=Huber(), maxiter_cg=10, greedy=0):
    np.seterr(invalid='raise')
    N, T = X.shape
    if W_init is None:
        W = np.eye(N)
    else:
        W = W_init.copy()
    U = np.ones_like(X, dtype=float)
    Y = np.zeros_like(X, dtype=float)
    C = X.dot(X.T) / T
    A = np.stack([C, ] * N)
    n_batch = T // batch_size
    for n in range(max_iter):
        idx_int = n % n_batch
        idx = slice(idx_int * batch_size, (idx_int + 1) * batch_size)
        x = X[:, idx]
        u_old = U[:, idx]
        y = np.dot(W, x)
        if greedy:
            y_old = Y[:, idx]
        u_new = density.ustar(y)
        if greedy:
            gaps = duality_gap(y, y_old, u_old, density)
            update_idx = np.argpartition(gaps, -greedy, axis=0)[-greedy:, :]
            A += compute_A_idx((u_new - u_old) * batch_size / T, x, update_idx)

            replace(U, u_new, update_idx, idx_int * batch_size)
            replace(Y, y, update_idx, idx_int * batch_size)
        else:
            A += compute_A((u_new - u_old) * batch_size / T, x)
            U[:, idx] = u_new
        W = min_W(W, A, maxiter_cg)
    return W


def solver_online(sample_generator, p, batch_size, W_init=None, maxiter_cg=10,
                  density=Huber(), alpha=0.7):
    if W_init is None:
        W = np.eye(p)
    else:
        W = W_init.copy()
    A = np.zeros((p, p, p))
    for n, x in enumerate(sample_generator):
        y = np.dot(W, x)
        u = density.ustar(y)
        update_idx = gen_idx(p, batch_size)
        step = 1. / (n + 1) ** alpha
        A *= (1 - step)
        u *= step * p
        A += compute_A_idx(u, x, update_idx)
        W = min_W(W, A, maxiter_cg)
    return W


def gen_idx(N, T):
    b = np.arange(N)
    n_tiles = int(T / N)
    tmp = np.tile(b, n_tiles + 1)
    return tmp[:T].reshape(1, T, order='F')


def duality_gap(y_new, y_old, u_old, density):
    tmp = u_old * (y_new ** 2 - y_old ** 2) / 2.
    return tmp + density.logp(y_old) - density.logp(y_new)


def min_W(W, A, maxiter_cg):
    N, _ = W.shape
    for i in range(N):
        K = W @ A[i] @ W.T
        s = cg(K, i, max_iter=maxiter_cg)
        s /= np.sqrt(s[i])
        W[i] = s @ W
    return W


@njit(fastmath=True, parallel=True)
def replace(A, B, update_idx, beginning):
    q, n = update_idx.shape
    for i in range(n):
        for j in range(q):
            x = update_idx[j, i]
            A[x, i + beginning] = B[x, i]


@njit(fastmath=True, parallel=True)
def compute_A(U, X):
    N, T = U.shape
    A = np.zeros((N, N, N))
    for i in range(N):
        u = U[i]
        for j in range(N):
            x = X[j]
            for k in range(j+1):
                y = X[k]
                tmp = 0.
                for t in range(T):
                    tmp += u[t] * x[t] * y[t]
                tmp /= T
                A[i, j, k] = tmp
                A[i, k, j] = tmp
    return A


@njit(nb.float64[:, :, :](nb.float64[:, :], nb.float64[:, :], nb.int64[:, :]),
      fastmath=True, cache=True)
def compute_A_idx(U, X, update_idx):
    """
    Params:
    U : N x T array
    X : N x T array
    update_idx : n_greedy x T array
    Output:
    A :  N x N x N array
    """
    N, T = X.shape
    A = np.zeros((N, N, N))
    for t in range(T):
        x = X[:, t]
        u = U[:, t]
        idx = update_idx[:, t]
        for i in idx:
            ui = u[i]
            Ai = A[i]
            for j in range(N):
                xj = x[j]
                for k in range(j + 1):
                    tmp = ui * xj * x[k]
                    Ai[j, k] += tmp
    for i in range(N):
        for j in range(N):
            for k in range(j):
                A[i, k, j] = A[i, j, k]
    return A / T
