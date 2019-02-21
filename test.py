import numpy as np
from mmica import solver_incremental, solver_online

p, n = 2, 100000


S = np.random.laplace(size=(p, n))
A = np.random.randn(p, p)
X = A.dot(S)

W = solver_incremental(X, max_iter=100)


batch_size = 100
gen = (A.dot(np.random.laplace(size=(p, batch_size)))
       for _ in range(n // batch_size))
W = solver_online(gen, p, batch_size, alpha=0.7)
print(W.dot(A))
