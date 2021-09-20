# // Code Written by Minghao Gou

from distutils.core import setup

from Cython.Build import cythonize
import numpy

setup(
    name = 'suction_nms',
    ext_modules=cythonize("suction_nms.pyx"),
    include_dirs=[numpy.get_include()]
)
