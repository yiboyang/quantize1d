# Yibo Yang, 06/17/2017
from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

import numpy

extensions = [
    Extension('quantizer',
              sources=['quantizer.pyx'],
              language='c++',
              include_dirs=[numpy.get_include()],   # for numpy.math.INFINITY
              extra_compile_args=["-std=c++11", "-fopenmp"],
              extra_link_args=["-std=c++11", "-fopenmp"]
              ),
]

setup(ext_modules=cythonize(extensions))
