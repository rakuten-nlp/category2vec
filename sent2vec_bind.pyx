#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2015 Rakuten U.S.A.
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import cython
import numpy as np
cimport numpy as np

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

cdef extern from "options.h":
    ctypedef np.float32_t real
    ctypedef void (*copy_ptr) (const int*, const real*, const int*, real*, const int*)
    ctypedef void (*axpy_ptr) (const int*, const real*, const real*, const int*, real*, const int*)
    ctypedef real (*dot_ptr) (const int*, const real*, const int*, const real*, const int*)
    ctypedef double (*ddot_ptr) (const int*, const real*, const int*, real*, const int*)
    ctypedef double (*nrm2_ptr) (const int*, const real*, const int*)
    ctypedef void (*scal_ptr) (const int*, const real*, const real*, const int*)
    cdef enum: MAX_SENTENCE_LEN
    cdef bint USING_BLAS
    cdef copy_ptr blas_copy
    cdef axpy_ptr blas_axpy
    cdef dot_ptr blas_dot
    cdef ddot_ptr blas_ddot
    cdef scal_ptr blas_scal

cdef extern from "sent2vec_calc.h" namespace "sentence2vec":
    cdef cppclass Sentence2Vec:
        Sentence2Vec() nogil except +
        Sentence2Vec(int, int, int, int, int) nogil except +
        Sentence2Vec(int, int, int, int, int, int) nogil except +
        int sg, hs, negative, size, window
        real *syn0
        real *syn1
        real *syn1neg
        real *syn0_grad
        real *syn1_grad
        real *syn1neg_grad
        np.uint32_t *table
        unsigned long long table_len
        int word_learn
        unsigned long long next_random
        real *sents
        int sents_len
        void set_update_mode(int)
        void train_vec(real*, const real, const int, const np.uint32_t*, np.uint32_t* const*, np.uint8_t* const*, const int*, const np.uint32_t*, real*, real*, real*) nogil
        void calc_sim_sent_vec(const real*, real*) nogil

cdef extern from "sent2vec_calc.h" namespace "sentence2vec::Sentence2Vec":
    void calcExpTable() nogil

IS_DOUBLE = (cython.sizeof(real) == cython.sizeof(np.float64_t))
try:
    from scipy.linalg.blas import fblas
    if not IS_DOUBLE:
        blas_copy=<copy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
        blas_axpy=<axpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
        blas_dot=<dot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
        blas_ddot=<ddot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
        blas_scal=<scal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x
    else:
        blas_copy=<copy_ptr>PyCObject_AsVoidPtr(fblas.dcopy._cpointer)  # y = x
        blas_axpy=<axpy_ptr>PyCObject_AsVoidPtr(fblas.daxpy._cpointer)  # y += alpha * x
        blas_dot=<dot_ptr>PyCObject_AsVoidPtr(fblas.ddot._cpointer)  # double = dot(x, y)
        blas_ddot=<ddot_ptr>PyCObject_AsVoidPtr(fblas.ddot._cpointer)  # double = dot(x, y)
        blas_scal=<scal_ptr>PyCObject_AsVoidPtr(fblas.dscal._cpointer) # x = alpha * x
except ImportError:
    if not USING_BLAS:
        print "The module couldn't load BLAS functions from this version of scipy"
        print "Install the BLAS library and edit settings.py (`use_blas = True`)"
        raise


def train_sent_vec(model, _sent_vec, sentence, alpha, _work, _neu1, _sent_vec_grad):
    cdef Sentence2Vec *s2v = new Sentence2Vec(model.sg, model.hs, model.negative, model.layer1_size, model.window, model.cbow_mean)
    cdef real _alpha = alpha
    cdef real *sent_vec = <real *>(np.PyArray_DATA(_sent_vec))
    cdef real *sent_vec_grad = <real *>(np.PyArray_DATA(_sent_vec_grad))
    cdef real *work
    cdef real *neu1
    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len

    cdef int i
    cdef long result = 0

    # For hierarchical softmax
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    s2v.word_learn = model.word_learn
    s2v.set_update_mode(model.update_mode)
    s2v.syn0 = <real *>(np.PyArray_DATA(model.w2v.syn0))
    s2v.syn0_grad = <real *>(np.PyArray_DATA(model.syn0_grad))

    if s2v.hs:
        s2v.syn1 = <real *>(np.PyArray_DATA(model.w2v.syn1))
        s2v.syn1_grad = <real *>(np.PyArray_DATA(model.syn1_grad))

    if s2v.negative:
        s2v.syn1neg = <real *>(np.PyArray_DATA(model.w2v.syn1neg))
        s2v.syn1neg_grad = <real *>(np.PyArray_DATA(model.syn1neg_grad))
        s2v.table = <np.uint32_t *>(np.PyArray_DATA(model.w2v.table))
        s2v.table_len = len(model.w2v.table)
        s2v.next_random = (2**24)*np.random.randint(0,2**24) + np.random.randint(0,2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <real *>np.PyArray_DATA(_work) ##to assign memory
    neu1 = <real *>np.PyArray_DATA(_neu1) ##to assign memory
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence)) ##length of sentence

    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index ##stores index of a word in a sentence
            reduced_windows[i] = np.random.randint(s2v.window) ##rand int beforehand
            if s2v.hs:
                codelens[i] = <int>len(word.code)
                codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            else:
                codelens[i] = 1
            result += 1

    # release GIL & train on the sentence
    with nogil:
        s2v.train_vec(sent_vec, _alpha, sentence_len, reduced_windows, points, codes, codelens, indexes, work, neu1, sent_vec_grad)
        del s2v
    return result

def sentvec_sim(model, _vec, num, _sims):
    cdef Sentence2Vec *s2v = new Sentence2Vec(model.sg, model.hs, model.negative, model.layer1_size, model.window)
    cdef real *vec = <real *>np.PyArray_DATA(_vec)
    cdef real *sims = <real *>np.PyArray_DATA(_sims)
    s2v.sents = <real *>np.PyArray_DATA(model.sents)
    s2v.sents_len = model.sents_len
    with nogil:
        s2v.calc_sim_sent_vec(vec, sims)
        del s2v
    return

def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.

    """
    calcExpTable()
    return 1

FAST_VERSION = init()  # initialize the module
