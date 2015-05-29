#ifndef __CAT2VEC_CALC_H
#define __CAT2VEC_CALC_H
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
extern "C" {
#include "options.h"
}

namespace category2vec {

  class Category2Vec {
  public:
    typedef void (Category2Vec::*train_ptr) (real*, real*, const real, const int, const uint32_t*, uint32_t* const*, uint8_t* const*, const int*, const uint32_t*, real*, real*, real*, real*);
    typedef void (Category2Vec::*sg_hs_ptr) (const uint32_t*, const uint8_t*, const int, real*, real*, const real, real*, real*, real*, real*);
    typedef void (Category2Vec::*sg_neg_ptr) (real*, real*, const uint32_t, const real, real*, real*, real*, real*);
    typedef void (Category2Vec::*cbow_hs_ptr) (const uint32_t*, const uint32_t*, const uint8_t*, const int*, real*, real*, const real, int, int, int, real*, real*, real*, real*);
    typedef void (Category2Vec::*cbow_neg_ptr) (const uint32_t*, const int*, real*, real*, const real, int, int, int, real*, real*, real*, real*);
    static real EXP_TABLE[EXP_TABLE_SIZE];
    static void calcExpTable();
    static void calc_joint_pairtable(Category2Vec* model1, Category2Vec* model2, real* table);
    static void joint_calc_sim_catsent_sum(const int pair_sc_len, const int size, const real *table, const real *svec1, const real *cvec1, const real *svec2, const real *cvec2, real *sim_ary);
    int sg, hs, negative, size, window, cbow_mean;
    real *syn0, *syn1, *syn1neg, *syn0_grad, *syn1_grad, *syn1neg_grad;
    uint32_t *table;
    uint64_t table_len;
    int word_learn;
    int cat_learn;
    uint64_t next_random;
    real *sents, *cats, *pairtable;
    uint32_t *pair_sc;
    int sents_len, cats_len, pair_sc_len;
    train_ptr train_func;
    Category2Vec() {}
    Category2Vec(int sg, int hs, int neg, int size, int w): sg(sg), hs(hs), negative(neg), size(size), window(w) {
      if (sg) train_func = &Category2Vec::train_sg;
      else train_func = &Category2Vec::train_cbow;
    }
    Category2Vec(int sg, int hs, int neg, int size, int w, int cbow_mean): sg(sg), hs(hs), negative(neg),
								      size(size), window(w), cbow_mean(cbow_mean){
      if (sg) train_func = &Category2Vec::train_sg;
      else train_func = &Category2Vec::train_cbow;
    }
    void set_update_mode(int update_mode);
    void train_vec(real *sent_vec, real *cat_vec, const real alpha, const int sentence_len, 
		   const uint32_t *reduced_windows, uint32_t* const *points, uint8_t* const *codes, 
		   const int *codelens, const uint32_t* indexes, real *work, real *neu1, real *sent_vec_grad, real *cat_vec_grad);
    void train_sg(real *sent_vec, real *cat_vec, const real alpha, const int sentence_len, 
		  const uint32_t *reduced_windows, uint32_t* const *points, uint8_t* const *codes, 
		  const int *codelens, const uint32_t* indexes, real *work, real *neu1, real *sent_vec_grad, real *cat_vec_grad);
    void train_cbow(real *sent_vec, real *cat_vec, const real alpha,const int sentence_len, 
		    const uint32_t *reduced_windows, uint32_t* const *points, uint8_t* const *codes, 
		    const int *codelens, const uint32_t*indexes, real *work, real *neu1, real *sent_vec_grad, real *cat_vec_grad);
    void calc_sim_sent_vec(const real *sent_vec, real *sim_array);
    void calc_sim_cat_vec(const real *cat_vec, real *sim_array);
    void init_pairtable();
    void calc_sim_catsent_concat(const real *sent_vec, const real *cat_vec, real *sim_array);
    void calc_sim_catsent_sum(const real *sent_vec, const real *cat_vec, real *sim_array);
  private:
    sg_hs_ptr sg_hs;
    sg_neg_ptr sg_neg;
    cbow_hs_ptr cbow_hs;
    cbow_neg_ptr cbow_neg;
    void sg_hs_sgd (const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec, real *cat_vec, const real alpha, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
    void sg_neg_sgd (real *sent_vec, real *cat_vec, const uint32_t word_index, const real alpha, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
    void cbow_hs_sgd (const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec, real *cat_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
    void cbow_neg_sgd (const uint32_t* indexes, const int *codelens, real *sent_vec, real *cat_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
    void sg_hs_adagrad (const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec, real *cat_vec, const real alpha, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
    void sg_neg_adagrad (real *sent_vec, real *cat_vec, const uint32_t word_index, const real alpha, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
    void cbow_hs_adagrad (const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec, real *cat_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
    void cbow_neg_adagrad (const uint32_t* indexes, const int *codelens, real *sent_vec, real *cat_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
    void sg_hs_adadelta (const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec, real *cat_vec, const real alpha, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
    void sg_neg_adadelta (real *sent_vec, real *cat_vec, const uint32_t word_index, const real alpha, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
    void cbow_hs_adadelta (const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec, real *cat_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
    void cbow_neg_adadelta (const uint32_t* indexes, const int *codelens, real *sent_vec, real *cat_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
    void sg_hs_adam (const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec, real *cat_vec, const real alpha, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
    void sg_neg_adam (real *sent_vec, real *cat_vec, const uint32_t word_index, const real alpha, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
    void cbow_hs_adam (const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec, real *cat_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
    void cbow_neg_adam (const uint32_t* indexes, const int *codelens, real *sent_vec, real *cat_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad);
  };
}

#endif
