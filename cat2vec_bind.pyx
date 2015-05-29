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
from libcpp cimport bool

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

cdef extern from "options.h":
    ctypedef np.float32_t real
    ctypedef void (*copy_ptr) (const int*, const real*, const int*, real*, const int*)
    ctypedef void (*axpy_ptr) (const int*, const real*, const real*, const int*, real*, const int*)
    ctypedef real (*dot_ptr) (const int*, const real*, const int*, const real*, const int*)
    ctypedef double (*ddot_ptr) (const int*, const real*, const int*, real*, const int*)
    ctypedef void (*scal_ptr) (const int*, const real*, real*, const int*)
    cdef enum: MAX_SENTENCE_LEN
    cdef bint USING_BLAS
    cdef double ADAM_B1
    cdef double ADAM_B2
    cdef copy_ptr blas_copy
    cdef axpy_ptr blas_axpy
    cdef dot_ptr blas_dot
    cdef ddot_ptr blas_ddot
    cdef scal_ptr blas_scal

cdef extern from "cat2vec_calc.h" namespace "category2vec":
    cdef cppclass Category2Vec:
        Category2Vec() nogil except +
        Category2Vec(int, int, int, int, int) nogil except +
        Category2Vec(int, int, int, int, int, int) nogil except +
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
        int cat_learn
        unsigned long long next_random
        real *sents
        real *cats
        real *pairtable
        np.uint32_t *pair_sc
        int sents_len
        int cats_len
        int pair_sc_len
        void set_update_mode(int)
        void train_vec(real*, real*, const real, const int, const np.uint32_t*, np.uint32_t* const*, np.uint8_t* const*, const int*, const np.uint32_t*, real*, real*, real*, real*) nogil
        void calc_sim_sent_vec(const real*, real*) nogil
        void calc_sim_cat_vec(const real*, real*) nogil
        void init_pairtable() nogil
        void calc_sim_catsent_concat(const real*, const real*, real*) nogil
        void calc_sim_catsent_sum(const real*, const real*, real*) nogil

cdef extern from "cat2vec_calc.h" namespace "category2vec::Category2Vec":
    void calcExpTable() nogil
    void calc_joint_pairtable(const Category2Vec*, const Category2Vec*, real* table) nogil
    void joint_calc_sim_catsent_sum(const int pair_sc_len, const int size, const real *table, const real *svec1, const real *cvec1, const real *svec2, const real *cvec2, real *sim_ary) nogil

IS_DOUBLE = (cython.sizeof(real) == cython.sizeof(np.float64_t))
ADAM_BETA1 = ADAM_B1
ADAM_BETA2 = ADAM_B2

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

cdef Category2Vec* convert_model(model):
    cdef Category2Vec *c2v = new Category2Vec(model.sg, model.hs, model.negative, model.layer1_size, model.window, model.cbow_mean)
    c2v.word_learn = model.word_learn
    c2v.cat_learn = model.cat_learn
    c2v.set_update_mode(model.update_mode)
    c2v.syn0 = <real *>(np.PyArray_DATA(model.w2v.syn0))
    c2v.syn0_grad = <real *>(np.PyArray_DATA(model.syn0_grad))

    if c2v.hs:
        c2v.syn1 = <real *>(np.PyArray_DATA(model.w2v.syn1))
        c2v.syn1_grad = <real *>(np.PyArray_DATA(model.syn1_grad))

    if c2v.negative:
        c2v.syn1neg = <real *>(np.PyArray_DATA(model.w2v.syn1neg))
        c2v.syn1neg_grad = <real *>(np.PyArray_DATA(model.syn1neg_grad))
        c2v.table = <np.uint32_t *>(np.PyArray_DATA(model.w2v.table))
        c2v.table_len = len(model.w2v.table)
        c2v.next_random = (2**24)*np.random.randint(0,2**24) + np.random.randint(0,2**24)
    return c2v

cdef inline long store_sentence_in_ctypes(sentence, int sentence_len, int hs, int window_size, int* codelens, np.uint32_t* indexes, np.uint32_t* reduced_windows, np.uint32_t** points, np.uint8_t** codes):
    cdef int i
    cdef long job_words = 0
    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index ##stores index of a word in a sentence
            reduced_windows[i] = np.random.randint(window_size) ##rand int beforehand
            if hs:
                codelens[i] = <int>len(word.code)
                codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            else:
                codelens[i] = 1
            job_words += 1
    return job_words

def train_from_job(model, job, alpha, _work, _neu1):
    cdef int job_len = len(job)
    cdef int sent_no
    cdef int cat_no
    cdef long job_words = 0
    cdef Category2Vec *c2v = convert_model(model)
    cdef real _alpha = alpha
    cdef real *sent_vec
    cdef real *cat_vec
    cdef real *sent_vec_grad
    cdef real *cat_vec_grad
    cdef real *work
    cdef real *neu1
    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len

    cdef int i
    cdef int j

    # For hierarchical softmax
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # convert Python structures to primitive types, so we can release the GIL
    work = <real *>np.PyArray_DATA(_work) ##to assign memory
    neu1 = <real *>np.PyArray_DATA(_neu1) ##to assign memory

    for j in range(job_len):
        j_tpl = job[j]
        cat_no = j_tpl[0]
        sent_no = j_tpl[1]
        sentence = j_tpl[2]
        sent_vec = <real *>(np.PyArray_DATA(model.sents[sent_no]))
        cat_vec = <real *>(np.PyArray_DATA(model.cats[cat_no]))
        sent_vec_grad = <real *>(np.PyArray_DATA(model.sents_grad[sent_no]))
        cat_vec_grad = <real *>(np.PyArray_DATA(model.cats_grad[cat_no]))
        sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence)) ##length of sentence
        job_words += store_sentence_in_ctypes(sentence, sentence_len, c2v.hs, c2v.window, codelens, indexes, reduced_windows, points, codes)
        # release GIL & train on the sentence
        with nogil:
            c2v.train_vec(sent_vec, cat_vec, _alpha, sentence_len, reduced_windows, points, codes, codelens, indexes, work, neu1, sent_vec_grad, cat_vec_grad)
    del c2v
    return job_words

def train_cat_vec(model, _sent_vec, _cat_vec, sentence, alpha, _work, _neu1, _sent_vec_grad, _cat_vec_grad):
    cdef Category2Vec *c2v = convert_model(model)
    cdef real _alpha = alpha
    cdef real *sent_vec = <real *>(np.PyArray_DATA(_sent_vec))
    cdef real *cat_vec = <real *>(np.PyArray_DATA(_cat_vec))
    cdef real *sent_vec_grad = <real *>(np.PyArray_DATA(_sent_vec_grad))
    cdef real *cat_vec_grad = <real *>(np.PyArray_DATA(_cat_vec_grad))
    cdef real *work
    cdef real *neu1
    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len

    cdef long result = 0

    # For hierarchical softmax
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # convert Python structures to primitive types, so we can release the GIL
    work = <real *>np.PyArray_DATA(_work) ##to assign memory
    neu1 = <real *>np.PyArray_DATA(_neu1) ##to assign memory
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence)) ##length of sentence
    result += store_sentence_in_ctypes(sentence, sentence_len, c2v.hs, c2v.window, codelens, indexes, reduced_windows, points, codes)
    # release GIL & train on the sentence
    with nogil:
        c2v.train_vec(sent_vec, cat_vec, _alpha, sentence_len, reduced_windows, points, codes, codelens, indexes, work, neu1, sent_vec_grad, cat_vec_grad)
        del c2v
    return result

def sentvec_sim(model, _vec, num, _sims):
    cdef Category2Vec *c2v = new Category2Vec(model.sg, model.hs, model.negative, model.layer1_size, model.window)
    cdef real *vec = <real *>np.PyArray_DATA(_vec)
    cdef real *sims = <real *>np.PyArray_DATA(_sims)
    c2v.sents = <real *>np.PyArray_DATA(model.sents)
    c2v.sents_len = model.sents_len
    with nogil:
        c2v.calc_sim_sent_vec(vec, sims)
        del c2v
    return

def catvec_sim(model, _vec, num,_sims):
    cdef Category2Vec *c2v = new Category2Vec(model.sg, model.hs, model.negative, model.layer1_size, model.window)
    cdef real *vec = <real *>np.PyArray_DATA(_vec)
    cdef real *sims = <real *>np.PyArray_DATA(_sims)
    c2v.cats = <real *>np.PyArray_DATA(model.cats)
    c2v.cats_len = model.cat_len
    with nogil:
        c2v.calc_sim_cat_vec(vec, sims)
        del c2v
    return

def init_pairtable(model):
    cdef Category2Vec *c2v = new Category2Vec(model.sg, model.hs, model.negative, model.layer1_size, model.window)
    c2v.pairtable = <real *>np.PyArray_DATA(model.pairnorm)
    c2v.sents = <real *>np.PyArray_DATA(model.sents)
    c2v.cats = <real *>np.PyArray_DATA(model.cats)
    c2v.pair_sc = <np.uint32_t*>np.PyArray_DATA(model.sent_cat_pair)
    c2v.pair_sc_len = model.pair_len
    with nogil:
        c2v.init_pairtable()
        del c2v

def catsentvec_sim_concat(model, _svec, _cvec, _sims):
    cdef Category2Vec *c2v = new Category2Vec(model.sg, model.hs, model.negative, model.layer1_size, model.window)
    cdef real *svec = <real *>np.PyArray_DATA(_svec)
    cdef real *cvec = <real *>np.PyArray_DATA(_cvec)
    cdef real *sims = <real *>np.PyArray_DATA(_sims)
    c2v.sents = <real *>np.PyArray_DATA(model.sents)
    c2v.cats = <real *>np.PyArray_DATA(model.cats)
    c2v.pair_sc = <np.uint32_t*>np.PyArray_DATA(model.sent_cat_pair)
    c2v.pair_sc_len = model.pair_len
    with nogil:
        c2v.calc_sim_catsent_concat(svec, cvec, sims)
        del c2v
    return

def catsentvec_sim_sum(model, _svec, _cvec, _sims):
    cdef Category2Vec *c2v = new Category2Vec(model.sg, model.hs, model.negative, model.layer1_size, model.window)
    cdef real *svec = <real *>np.PyArray_DATA(_svec)
    cdef real *cvec = <real *>np.PyArray_DATA(_cvec)
    cdef real *sims = <real *>np.PyArray_DATA(_sims)
    c2v.pairtable = <real *>np.PyArray_DATA(model.pairnorm)
    c2v.pair_sc_len = model.pair_len
    with nogil:
        c2v.calc_sim_catsent_sum(svec, cvec, sims)
        del c2v
    return

def init_joint_pairtable(model1, model2, _pairtable):
    cdef Category2Vec *c2v1 = new Category2Vec(model1.sg, model1.hs, model1.negative, model1.layer1_size, model1.window)
    cdef Category2Vec *c2v2 = new Category2Vec(model2.sg, model2.hs, model2.negative, model2.layer1_size, model2.window)
    cdef real *pairtable = <real *>np.PyArray_DATA(_pairtable)
    c2v1.sents = <real *>np.PyArray_DATA(model1.sents)
    c2v1.cats = <real *>np.PyArray_DATA(model1.cats)
    c2v2.sents = <real *>np.PyArray_DATA(model2.sents)
    c2v2.cats = <real *>np.PyArray_DATA(model2.cats)
    c2v1.pair_sc = <np.uint32_t*>np.PyArray_DATA(model1.sent_cat_pair)
    c2v1.pair_sc_len = model1.pair_len
    c2v2.pair_sc = <np.uint32_t*>np.PyArray_DATA(model2.sent_cat_pair)
    c2v2.pair_sc_len = model2.pair_len
    with nogil:
        calc_joint_pairtable(c2v1, c2v2, pairtable)

def joint_catsentvec_sim_sum(joint_pairtable, _svec1, _cvec1, _svec2, _cvec2, _sims):
    cdef real *svec1 = <real *>np.PyArray_DATA(_svec1)
    cdef real *cvec1 = <real *>np.PyArray_DATA(_cvec1)
    cdef real *svec2 = <real *>np.PyArray_DATA(_svec2)
    cdef real *cvec2 = <real *>np.PyArray_DATA(_cvec2)
    cdef real *sims = <real *>np.PyArray_DATA(_sims)
    cdef real *table = <real *>np.PyArray_DATA(joint_pairtable)
    cdef int size = _svec1.shape[0]
    cdef int pair_sc_len = _sims.shape[0]
    with nogil:
        joint_calc_sim_catsent_sum(pair_sc_len, size, table, svec1, cvec1, svec2, cvec2, sims)

def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.

    """
    calcExpTable()
    return 1

FAST_VERSION = init()  # initialize the module

