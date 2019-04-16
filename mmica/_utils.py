# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#
# License: MIT
import os

import numpy as np

from ._conjugate_gradient import cg_c


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


def cg(B, i, max_iter=10, tol=1e-10):
    '''
    Wrapper to call Cython
    '''
    return cg_c(B, i, max_iter, tol, B.shape[0])
