import numpy as np

from setuptools import setup, Extension
from Cython.Build import cythonize


extensions = [
    Extension(name="mmica._conjugate_gradient", sources=["mmica/_conjugate_gradient.pyx"])
]

setup(ext_modules=cythonize(extensions, include_path=[np.get_include()]),)

