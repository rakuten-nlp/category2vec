from distutils.core import setup, Extension
import distutils.ccompiler
from Cython.Build import cythonize
import numpy as np
import os
import sys
import settings

def readSettings():
    macros = [('MAX_SENTENCE_LEN', settings.max_sentence_len),
              ('EXP_TABLE_SIZE',settings.exp_table_size),
              ('MAX_EXP',settings.max_exp),
              ('MAX_DIMENSION',settings.max_dimension),
              ('EPSILON',settings.epsilon),
              ('EPSILON_D',settings.epsilon_d),
              ('RHO',settings.rho),
              ('ADAM_L',settings.adam_l),
              ('ADAM_B1',settings.adam_b1),
              ('ADAM_B2',settings.adam_b2),
              ('ADAM_A',settings.adam_a)
    ]
    ext_inc = []
    ext_libs = []
    ext_c_comp_args = []
    ext_cpp_comp_args = []
    if settings.use_double:
        macros.append(('USE_DOUBLE', 1))
    if settings.use_blas:
        macros.append(('USE_BLAS', 1))
        ext_inc.extend(settings.blas_include)
        ext_libs.extend(settings.blas_libs)
    if settings.fast_sqrt:
        macros.append(('FAST_SQRT', 1))
    if settings.use_avx:
        macros.append(('USE_AVX', 1))
        ext_c_comp_args.append("-mavx")
        ext_cpp_comp_args.append("-mavx")
    if not settings.use_clang:
        ext_cpp_comp_args.append("-std=c++0x")
    if settings.force_gcc:
        os.environ["CC"] = "gcc"
        os.environ["CXX"] = "g++"

    return macros, ext_inc, ext_libs, ext_c_comp_args, ext_cpp_comp_args

model_dir = os.path.dirname(__file__) or os.getcwd()
includes = [model_dir, np.get_include()]
c_comp_args = []
cpp_comp_args = []

macros, ext_inc, ext_libs, ext_c_comp_args, ext_cpp_comp_args = readSettings()
includes.extend(ext_inc)
libs = ext_libs
c_comp_args.extend(ext_c_comp_args)
cpp_comp_args.extend(ext_cpp_comp_args)

extensions = [
    Extension(
        "word2vec_inner",
        define_macros = macros,
        sources=["word2vec_inner.pyx", "options.c"],
        include_dirs = includes,
        libraries=libs,
        extra_compile_args=c_comp_args,
    ),
    Extension(
        "cat2vec_bind",
        define_macros = macros,
        sources=["cat2vec_bind.pyx", "cat2vec_calc.cpp", "options.c"],
        include_dirs = includes,
        libraries=libs,
        language="c++",
        extra_compile_args=cpp_comp_args,
    ),
    Extension(
        "sent2vec_bind",
        define_macros = macros,
        sources=["sent2vec_bind.pyx", "sent2vec_calc.cpp", "options.c"],
        include_dirs = includes,
        libraries=libs,
        language="c++",
        extra_compile_args=cpp_comp_args,
    ),
]

setup(
    ext_modules = cythonize(extensions),
 )
