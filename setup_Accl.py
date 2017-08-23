# -*- coding: utf-8 -*-


#from distutils.core import setup
#from Cython.Build import cythonize


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

# So I don't have to repeat myself...
main_name = "Accl"

ext_modules = [
    Extension(
        main_name, [main_name+".pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'], # OMP: '/openmp'
        extra_link_args=[], # OMP: '/openmp'
    )
]

setup(
    name=main_name,
    ext_modules=cythonize(ext_modules),
)