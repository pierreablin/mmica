# Majorization-minimization ICA
This repository contains the code for the AISTATS 2019 paper "Stochastic algorithms with descent guarantees for ICA":

> Ablin, P., Gramfort, A., Cardoso, J.F. & Bach, F. (2019). Stochastic algorithms with descent guarantees for ICA. Proceedings of Machine Learning Research, in PMLR 89:1564-1573

### Installation
To get started, clone the repository and run `python setup.py install`.

### API

There are two solvers in the package:

* `solver_incremental` takes a `(p, n)` array as input
* `solver_online` takes a generator as input

### Examples
Incremental solver:
```python
import numpy as np
from mmica import solver_incremental

p, n = 2, 1000

S = np.random.laplace(size=(p, n))
A = np.random.randn(p, p)
X = A.dot(S)

W = solver_incremental(X)
print(np.dot(W, A))  # close from a permutation + scale matrix
```

Online solver:

```python
import numpy as np
from mmica import solver_online

p = 2
batch_size = 100

A = np.random.randn(p, p)
S = (np.random.laplace(size=(p, batch_size)) for _ in range(20))
X = (A.dot(s) for s in S)
W = solver_online(X, p)

print(np.dot(W, A))  # close from a permutation + scale matrix
```
