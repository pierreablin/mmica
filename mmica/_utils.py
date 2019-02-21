import os

import numpy as np
from numba import jit

from ._conjugate_gradient import cg_c


@jit
def _argthc(y, n_iter):
    '''
    Computes the reciprocal of tanh(x) / x
    '''
    x = 1. / y
    for i in range(n_iter):
        thx = np.tanh(x)
        xinv = 1. / x
        thxdx = thx * xinv
        fx = thxdx - y
        fpx = xinv - thxdx * thx - thxdx * xinv
        x -= fx / fpx
    return x


def amari_d(W, A):
    P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)
    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])


def whitening(Y, mode='sph'):
    '''
    Whitens the data Y using sphering or pca
    '''
    R = np.dot(Y, Y.T) / Y.shape[1]
    U, D, _ = np.linalg.svd(R)
    if mode == 'pca':
        W = U.T / np.sqrt(D)[:, None]
        Z = np.dot(W, Y)
    elif mode == 'sph':
        W = np.dot(U, U.T / np.sqrt(D)[:, None])
        Z = np.dot(W, Y)
    return Z, W


@jit
def argthc(Y, n_iter=4):
    out = np.zeros_like(Y)
    shape = Y.shape
    n_dim = len(shape)
    if n_dim == 1:
        N = shape[0]
        for i in range(N):
            out[i] = _argthc(Y[i], n_iter)
        return out
    else:
        N, T = shape
        for j in range(T):
            for i in range(N):
                out[i, j] = _argthc(Y[i, j], n_iter)
        return out


def tanh_star(Z):
    athc = argthc(Z)
    return np.log(np.cosh(athc)) - 0.5 * athc ** 2 * Z


def cg(B, i, max_iter=10, tol=1e-10):
    return cg_c(B, i, max_iter, tol, B.shape[0])


def grad_norm(W, X, density):
    N, T = X.shape
    Y = W @ X
    return np.linalg.norm(density.score(Y) @ Y.T / T - np.eye(N))


def loss(W, X, density):
    N, _ = W.shape
    return - np.linalg.slogdet(W)[1] + np.mean(density.logp(W @ X)) * N
