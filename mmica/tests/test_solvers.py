# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#
# License: MIT
import pytest
import numpy as np

from numpy.testing import assert_array_almost_equal

from mmica import solver_incremental, solver_online
from picard import permute


@pytest.mark.parametrize('greedy', [0, 1, 2])
@pytest.mark.parametrize('density', ['huber', 'tanh'])
def test_solver_incremental(greedy, density):
    p, n = 3, 1000
    rng = np.random.RandomState(0)

    S = rng.laplace(size=(p, n))
    A = rng.randn(p, p)
    X = A.dot(S)

    W = solver_incremental(X, greedy=greedy, density=density)
    op = W.dot(A)
    assert_array_almost_equal(permute(op), np.eye(p), decimal=1)


@pytest.mark.parametrize('greedy', [0, 1, 2])
@pytest.mark.parametrize('density', ['huber', 'tanh'])
def test_solver_online(greedy, density):
    p = 3
    batch_size = 100
    rng = np.random.RandomState(0)

    A = rng.randn(p, p)
    gen = (A.dot(rng.laplace(size=(p, batch_size)))
           for _ in range(100))
    W = solver_online(gen, p, alpha=0.5, greedy=greedy, density=density)
    op = W.dot(A)
    assert_array_almost_equal(permute(op), np.eye(p), decimal=1)
