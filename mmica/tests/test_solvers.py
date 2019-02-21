import numpy as np

from numpy.testing import assert_array_almost_equal

from mmica import solver_incremental, solver_online
from picard import permute


def test_solver_incremental():
    p, n = 2, 1000
    rng = np.random.RandomState(0)

    S = rng.laplace(size=(p, n))
    A = rng.randn(p, p)
    X = A.dot(S)

    W = solver_incremental(X)
    op = W.dot(A)
    assert_array_almost_equal(permute(op), np.eye(p), decimal=2)


def test_solver_online():
    p = 2
    batch_size = 100
    rng = np.random.RandomState(0)

    A = rng.randn(p, p)
    gen = (A.dot(rng.laplace(size=(p, batch_size)))
           for _ in range(20))
    W = solver_online(gen, p, batch_size, alpha=0.7)
    op = W.dot(A)
    assert_array_almost_equal(permute(op), np.eye(p), decimal=2)
