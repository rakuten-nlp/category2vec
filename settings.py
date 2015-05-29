#!/usr/bin/env python

#######################
### global settings ###
#######################
# use double for distributed representations
use_double = False

# small value to avoid zero division
epsilon = 0.00000001 #AdaGrad
epsilon_d = 0.00000001 #AdaDelta
# used in AdaDelta to update expected value for variance
rho = 0.995

# ADAM parameters
adam_l = .99999999
adam_b1 = 0.9
adam_b2 = 0.999
adam_a = 0.001

###################################
### settings for Cython modules ###
###################################
# Compiler settings
use_clang = False # specify True if you use clang (e.g. Mac OS X)
force_gcc = False # specify True if you force to use gcc

# BLAS settings
use_blas = False # use BLAS directly
blas_include = [] # path to cblas.h
blas_libs = [] # BLAS library
## ATLAS Example
#blas_include = ["/usr/include/atlas"]
#blas_libs = ["cblas", "atlas"]
## OpenBLAS Example
#blas_include = ["/usr/local/include"]
#blas_libs = ["openblas"]

# use AVX
use_avx = True # Only for Intel CPUs (>= Sandy Bridge)

# use fast sqrt and rsqrt (less accurate)
fast_sqrt = False

# max word count in a paragraph/document
max_sentence_len = 10000
# size of exp table
exp_table_size = 1000
# max power for exp table
max_exp = 6
# max dimension
max_dimension = 500
