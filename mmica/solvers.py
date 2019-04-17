# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#
# License: MIT
import numpy as np
import numba as nb
from numba import njit

from ._utils import cg
from ._densities import Huber, Sigmoid


def solver_incremental(X, max_iter=100, batch_size=100, W_init=None,
                       density='huber', maxiter_cg=10, greedy=0):
    """
    Incremental algorithm for ICA
    Parameters
    ----------
    X : array_like, shape (p, n)
        The input data to be unmixed.
    max_iter : int, optional
        Maximum number of iterations
    batch_size : int, optional
        Mini-batch size
    W_init : array_like, shape (p, p), optional
        Initial guess for the unmixing matrix.
        Defaults to identity
    density : 'huber' or 'tanh', optional
        The density to use
    maxiter_cg : int, optional
        The number of iterations of conjuagate gradient to perform
    greedy : int, optional
        The number of sources to update for each sample, chosen greedily.
        If 0, each source is updated.
    Returns
    -------
    W : array_like, shape (p, p)
        The estimated unmixing matrix
    """
    density = {'huber': Huber(),
               'tanh': Sigmoid()}.get(density)
    if density is None:
        raise ValueError('Density should either be tanh or huber')
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


def solver_online(sample_generator, p, W_init=None, density='huber',
                  maxiter_cg=10, greedy=0, alpha=0.7):
    """
    Online algorithm for ICA
    Parameters
    ----------
    sample_generator: generator
        The sample stream generator. The x in `for x in sample_generator:`
        should be a minibatch of size (p, batch_size)
    p : int
        Number of sources
    W_init : array_like, shape (p, p), optional
        Initial guess for the unmixing matrix.
        Defaults to identity
    density : 'huber' or 'tanh', optional
        The density to use
    maxiter_cg : int, optional
        The number of iterations of conjugate gradient to perform
    greedy : int, optional
        The number of sources to update for each sample, chosen randomly.
        If 0, each source is updated.
    Returns
    -------
    W : array_like, shape (p, p)
        The estimated unmixing matrix
    """
    density = {'huber': Huber(),
               'tanh': Sigmoid()}.get(density)
    if density is None:
        raise ValueError('Density should either be tanh or huber')
    if W_init is None:
        W = np.eye(p)
    else:
        W = W_init.copy()
    A = np.zeros((p, p, p))
    for n, x in enumerate(sample_generator):
        _, batch_size = x.shape
        y = np.dot(W, x)
        u = density.ustar(y)
        step = 1. / (n + 1) ** alpha
        A *= (1 - step)
        if greedy:
            u *= step * p / greedy
            update_idx = gen_idx(p, greedy, batch_size)
            A += compute_A_idx(u, x, update_idx)
        else:
            u *= step
            A += compute_A(u, x)
        W = min_W(W, A, maxiter_cg)
    return W


def gen_idx(N, g, T):
    b = np.arange(N)
    n_tiles = int(g * T / N)
    tile = np.tile(b, n_tiles + 1)
    return tile[: g * T].reshape(g, T, order='F')


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
