#ifndef __SENT2VEC_CALC_H
#define __SENT2VEC_CALC_H
#include <cmath>
#include <cstring>
#include <cstdint>
extern "C" {
#include "options.h"
}

namespace sentence2vec {

  class Sentence2Vec {
  public:
    typedef void (Sentence2Vec::*train_ptr) (real*, const real, const int, const uint32_t*, uint32_t* const*, uint8_t* const*, const int*, const uint32_t*, real*, real*, real*);
    typedef void (Sentence2Vec::*sg_hs_ptr) (const uint32_t*, const uint8_t*, const int, real*, const real, real*, real*, real*);
    typedef void (Sentence2Vec::*sg_neg_ptr) (real*, const uint32_t, const real, real*, real*, real*);
    typedef void (Sentence2Vec::*cbow_hs_ptr) (const uint32_t*, const uint32_t*, const uint8_t*, const int*, real*, const real, int, int, int, real*, real*, real*);
    typedef void (Sentence2Vec::*cbow_neg_ptr) (const uint32_t*, const int*, real*, const real, int, int, int, real*, real*, real*);
    static real EXP_TABLE[EXP_TABLE_SIZE];
    static void calcExpTable();
    int sg, hs, negative, size, window, cbow_mean;
    real *syn0, *syn1, *syn1neg, *syn0_grad, *syn1_grad, *syn1neg_grad;
    uint32_t *table;
    uint64_t table_len;
    int word_learn;
    uint64_t next_random;
    real *sents;
    int sents_len;
    train_ptr train_func;
    Sentence2Vec() {}
    Sentence2Vec(int sg, int hs, int neg, int size, int w): sg(sg), hs(hs), negative(neg), size(size), window(w) {
      if (sg) train_func = &Sentence2Vec::train_sg;
      else train_func = &Sentence2Vec::train_cbow;
    }
    Sentence2Vec(int sg, int hs, int neg, int size, int w, int cbow_mean): sg(sg), hs(hs), negative(neg),
								      size(size), window(w), cbow_mean(cbow_mean){
      if (sg) train_func = &Sentence2Vec::train_sg;
      else train_func = &Sentence2Vec::train_cbow;
    }
    void set_update_mode(int update_mode);
    void train_vec(real *sent_vec, const real alpha, const int sentence_len, 
		   const uint32_t *reduced_windows, uint32_t* const *points, uint8_t* const *codes, 
		   const int *codelens, const uint32_t* indexes, real *work, real *neu1, real *sent_vec_grad);
    void train_sg(real *sent_vec, const real alpha, const int sentence_len, 
		  const uint32_t *reduced_windows, uint32_t* const *points, uint8_t* const *codes, 
		  const int *codelens, const uint32_t* indexes, real *work, real *neu1, real *sent_vec_grad);
    void train_cbow(real *sent_vec, const real alpha,const int sentence_len, 
		    const uint32_t *reduced_windows, uint32_t* const *points, uint8_t* const *codes, 
		    const int *codelens, const uint32_t*indexes, real *work, real *neu1, real *sent_vec_grad);
    void calc_sim_sent_vec(const real *sent_vec, real *sim_array);
  private:
    sg_hs_ptr sg_hs;
    sg_neg_ptr sg_neg;
    cbow_hs_ptr cbow_hs;
    cbow_neg_ptr cbow_neg;
    void sg_hs_sgd (const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec, const real alpha, real* work, real* neu1, real *sent_vec_grad);
    void sg_neg_sgd (real *sent_vec,  const uint32_t word_index, const real alpha, real* work, real* neu1, real *sent_vec_grad);
    void cbow_hs_sgd (const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec,  const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad);
    void cbow_neg_sgd (const uint32_t* indexes, const int *codelens, real *sent_vec,  const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad);
    void sg_hs_adagrad (const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec,  const real alpha, real* work, real* neu1, real *sent_vec_grad);
    void sg_neg_adagrad (real *sent_vec,  const uint32_t word_index, const real alpha, real* work, real* neu1, real *sent_vec_grad);
    void cbow_hs_adagrad (const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec,  const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad);
    void cbow_neg_adagrad (const uint32_t* indexes, const int *codelens, real *sent_vec,  const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad);
    void sg_hs_adadelta (const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec,  const real alpha, real* work, real* neu1, real *sent_vec_grad);
    void sg_neg_adadelta (real *sent_vec,  const uint32_t word_index, const real alpha, real* work, real* neu1, real *sent_vec_grad);
    void cbow_hs_adadelta (const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec,  const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad);
    void cbow_neg_adadelta (const uint32_t* indexes, const int *codelens, real *sent_vec,  const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad);
    void sg_hs_adam (const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec,  const real alpha, real* work, real* neu1, real *sent_vec_grad);
    void sg_neg_adam (real *sent_vec,  const uint32_t word_index, const real alpha, real* work, real* neu1, real *sent_vec_grad);
    void cbow_hs_adam (const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec,  const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad);
    void cbow_neg_adam (const uint32_t* indexes, const int *codelens, real *sent_vec,  const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad);
  };
}

#endif
