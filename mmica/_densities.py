import numpy as np
from numba import jit


class Sigmoid(object):
    def __init__(self):
        pass

    def score(self, Y):
        return np.tanh(Y)

    def logp(self, Y):
        A = np.abs(Y)
        return A + np.log1p(np.exp(- 2. * A))

    def ustar(self, Y):
        return np.tanh(Y) / Y

    def score_der(self, Y):
        return 1 - np.tanh(Y) ** 2


class Huber(object):
    def __init__(self):
        pass

    def logp(self, Y):
        return logp_u(Y)

    def score(self, Y):
        return score_u(Y)

    def ustar(self, Y):
        return ustar_u(Y)

    def score_der(self, Y):
        return scored_u(Y)


@jit(parallel=True, fastmath=True)
def logp_u(Y):
    N, T = Y.shape
    output = np.empty((N, T))
    for i in range(N):
        for j in range(T):
            y = abs(Y[i, j])
            if y < 1:
                output[i, j] = 0.5 * y ** 2
            else:
                output[i, j] = y - 0.5
    return output


@jit(parallel=True, fastmath=True)
def score_u(Y):
    N, T = Y.shape
    output = np.empty((N, T))
    for i in range(N):
        for j in range(T):
            y = Y[i, j]
            if y > 1:
                output[i, j] = 1.
            elif y < -1:
                output[i, j] = -1.
            else:
                output[i, j] = y
    return output


@jit(parallel=True, fastmath=True)
def scored_u(Y):
    N, T = Y.shape
    output = np.zeros((N, T))
    for i in range(N):
        for j in range(T):
            if abs(Y[i, j]) < 1:
                output[i, j] = 1.
    return output


@jit(parallel=True, fastmath=True)
def ustar_u(Y):
    N, T = Y.shape
    output = np.empty((N, T))
    for i in range(N):
        for j in range(T):
            y = np.abs(Y[i, j])
            if y > 1:
                output[i, j] = 1. / y
            else:
                output[i, j] = 1.
    return output
