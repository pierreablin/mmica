import os

import numpy as np

from setuptools import setup
from Cython.Build import cythonize


descr = 'Stochastic algorithms for ICA'

version = None
with open(os.path.join('mmica', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

DISTNAME = 'mmica'
DESCRIPTION = descr
MAINTAINER = 'Pierre Ablin'
MAINTAINER_EMAIL = 'pierreablin@gmail.com'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/pierreablin/mmica.git'
VERSION = version
URL = 'https://github.com/pierreablin/mmica'

setup(name='mmica',
      version=VERSION,
      description=DESCRIPTION,
      long_description=open('README.md').read(),
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      packages=['mmica'],
      ext_modules=cythonize("mmica/*.pyx", include_path=[np.get_include()]),
      )
