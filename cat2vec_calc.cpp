#include "cat2vec_calc.h"
using namespace category2vec;

real Category2Vec::EXP_TABLE[EXP_TABLE_SIZE];

/* precompute function sigmoid(x) = 1 / (1 + exp(-x)) */
void Category2Vec::calcExpTable() {
  for (int i = 0; i < EXP_TABLE_SIZE; ++i) {
    EXP_TABLE[i] = (real) exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
    EXP_TABLE[i] = EXP_TABLE[i] / (EXP_TABLE[i] + 1);
  }
}

void Category2Vec::set_update_mode(int update_mode) {
  switch(update_mode) {
  case 0:
    sg_hs = &Category2Vec::sg_hs_sgd;
    sg_neg = &Category2Vec::sg_neg_sgd;
    cbow_hs = &Category2Vec::cbow_hs_sgd;
    cbow_neg = &Category2Vec::cbow_neg_sgd;
    break;
  case 1:
    sg_hs = &Category2Vec::sg_hs_adagrad;
    sg_neg = &Category2Vec::sg_neg_adagrad;
    cbow_hs = &Category2Vec::cbow_hs_adagrad;
    cbow_neg = &Category2Vec::cbow_neg_adagrad;
    break;
  case 2:
    sg_hs = &Category2Vec::sg_hs_adadelta;
    sg_neg = &Category2Vec::sg_neg_adadelta;
    cbow_hs = &Category2Vec::cbow_hs_adadelta;
    cbow_neg = &Category2Vec::cbow_neg_adadelta;
    break;
  case 3:
    sg_hs = &Category2Vec::sg_hs_adam;
    sg_neg = &Category2Vec::sg_neg_adam;
    cbow_hs = &Category2Vec::cbow_hs_adam;
    cbow_neg = &Category2Vec::cbow_neg_adam;
    break;
  }
}

void Category2Vec::train_vec(real *sent_vec, real *cat_vec, const real alpha, const int sentence_len, const uint32_t *reduced_windows, uint32_t* const *points, uint8_t* const *codes, const int *codelens, const uint32_t* indexes, real *work, real *neu1, real *sent_vec_grad, real *cat_vec_grad) {
  (this->*train_func)(sent_vec, cat_vec, alpha, sentence_len, reduced_windows, points, codes, codelens, indexes, work, neu1, sent_vec_grad, cat_vec_grad);
}

void Category2Vec::train_sg(real *sent_vec, real *cat_vec, const real alpha, const int sentence_len, const uint32_t *reduced_windows, uint32_t* const *points, uint8_t* const *codes, const int *codelens, const uint32_t* indexes, real *work, real *neu1, real *sent_vec_grad, real *cat_vec_grad) {
  for (int i = 0; i < sentence_len; ++i) {
    if (codelens[i] == 0) continue;
    int j = i - window + reduced_windows[i];
    if (j < 0) j = 0;
    int k = i + window + 1 - reduced_windows[i];
    if (k > sentence_len) k = sentence_len;
    if (hs) {
      for (; j < k; ++j) {
	if (codelens[j] == 0) continue;
	(this->*sg_hs)(points[j], codes[j], codelens[j], sent_vec, cat_vec, alpha, work, neu1, sent_vec_grad, cat_vec_grad);
      }
    }
    if (negative) {
      for (; j < k; ++j) {
	if (codelens[j] == 0) continue;
	(this->*sg_neg)(sent_vec, cat_vec, indexes[j], alpha, work, neu1, sent_vec_grad, cat_vec_grad);
      }
    }
  }
}

void Category2Vec::train_cbow(real *sent_vec, real *cat_vec, const real alpha,const int sentence_len, const uint32_t *reduced_windows, uint32_t* const *points, uint8_t* const *codes, const int *codelens, const uint32_t *indexes, real *work, real *neu1, real *sent_vec_grad, real *cat_vec_grad) {
  for (int i = 0; i < sentence_len; ++i) {
    if (codelens[i] == 0) continue;
    int j = i - window + reduced_windows[i];
    if (j < 0) j = 0;
    int k = i + window + 1 - reduced_windows[i];
    if (k > sentence_len) k = sentence_len;
    if (hs)
      (this->*cbow_hs)(indexes, points[i], codes[i], codelens, sent_vec, cat_vec, alpha, i, j, k, work, neu1, sent_vec_grad, cat_vec_grad);
    if (negative)
      (this->*cbow_neg)(indexes, codelens, sent_vec, cat_vec, alpha, i, j, k, work, neu1, sent_vec_grad, cat_vec_grad);
  }
}

void Category2Vec::sg_hs_sgd(const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec, real *cat_vec, const real alpha, real* work, real* l1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t b, row2;
  real f, g;
  memset(l1, 0, size * sizeof(real));
  lib_axpy(size, ONEF, sent_vec, ONE, l1, ONE);
  lib_axpy(size, ONEF, cat_vec, ONE, l1, ONE);
  memset(work, 0, size * sizeof(real));
  for (b = 0; b < codelen; ++b) {
    row2 = word_point[b] * size;
    f = lib_dot(size, l1, ONE, &syn1[row2], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP)
      continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = (1 - word_code[b] - f) * alpha;
    lib_axpy(size, g, &syn1[row2], ONE, work, ONE);
    if (word_learn) lib_axpy(size, g, l1, ONE, &syn1[row2], ONE);
  }
  lib_axpy(size, ONEF, work, ONE, sent_vec, ONE);
  if (cat_learn) lib_axpy(size, ONEF, work, ONE, cat_vec, ONE);
}

void Category2Vec::sg_neg_sgd(real *sent_vec, real *cat_vec, const uint32_t word_index, const real alpha, real* work, real* l1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t row;
  static uint64_t modulo = 281474976710655ULL;
  real f, g, label;
  uint32_t target_index;
  int d;
  memset(l1, 0, size * sizeof(real));
  lib_axpy(size, ONEF, sent_vec, ONE, l1, ONE);
  lib_axpy(size, ONEF, cat_vec, ONE, l1, ONE);
  memset(work, 0, size * sizeof(real));
  for (d = 0; d <= negative; ++d) {
    if (d == 0) {
      target_index = word_index;
      label = ONEF;
    }
    else {
      target_index = table[(next_random >> 16) % table_len];
      next_random = (next_random * (uint64_t)25214903917ULL + 11) & modulo;
      if (target_index == word_index) continue;
      label = 0.0;
    }
    row = target_index * size;
    f = lib_dot(size, l1, ONE, &syn1neg[row], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP) continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = (label - f) * alpha;
    lib_axpy(size, g, &syn1neg[row], ONE, work, ONE);
    if (word_learn) lib_axpy(size, g, l1, ONE, &syn1neg[row], ONE);
  }
  lib_axpy(size, ONEF, work, ONE, sent_vec, ONE);
  if (cat_learn) lib_axpy(size, ONEF, work, ONE, cat_vec, ONE);
}

void Category2Vec::cbow_hs_sgd(const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec, real *cat_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t b, row2;
  real f, g, count, inv_count;
  int m;
  memset(neu1, 0, size * sizeof(real));
  count = 0.0;
  for (m = j; m < k; ++m) {
    if (m == i || codelens[m] == 0) continue;
    count += ONEF;
    lib_axpy(size, ONEF, &syn0[indexes[m] * size], ONE, neu1, ONE);
  }
  lib_axpy(size, ONEF, sent_vec, ONE, neu1, ONE);
  count += ONEF;
  lib_axpy(size, ONEF, cat_vec, ONE, neu1, ONE);
  count += ONEF;
  if (cbow_mean && count > (real)0.5) {
    inv_count = ONEF / count;
    lib_scal(size, inv_count, neu1, ONE);
  }
  memset(work, 0, size * sizeof(real));
  for (b = 0; b < codelens[i]; ++b) {
    row2 = word_point[b] * size;
    f = lib_dot(size, neu1, ONE, &syn1[row2], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP) continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = (1 - word_code[b] - f) * alpha;
    lib_axpy(size, g, &syn1[row2], ONE, work, ONE);
    if (word_learn) lib_axpy(size, g, neu1, ONE, &syn1[row2], ONE);
  }
  if (word_learn) {
    for (m = j; m < k; ++m) {
      if (m == i || codelens[m] == 0) continue;
      lib_axpy(size, ONEF, work, ONE, &syn0[indexes[m] * size], ONE);
    }
  }
  lib_axpy(size, ONEF, work, ONE, sent_vec, ONE);
  if (cat_learn) lib_axpy(size, ONEF, work, ONE, cat_vec, ONE);
}

void Category2Vec::cbow_neg_sgd(const uint32_t* indexes, const int *codelens, real *sent_vec, real *cat_vec, 
		       const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t row2;
  static uint64_t modulo = 281474976710655ULL;
  real f, g, count, inv_count, label;
  uint32_t target_index, word_index;
  int d, m;
  word_index = indexes[i];
  memset(neu1, 0, size * sizeof(real));
  count = 0.0;
  for (m = j; m < k; ++m) {
    if (m == i || codelens[m] == 0) continue;
    count += ONEF;
    lib_axpy(size, ONEF, &syn0[indexes[m] * size], ONE, neu1, ONE);
  }
  lib_axpy(size, ONEF, sent_vec, ONE, neu1, ONE);
  count += ONEF;
  lib_axpy(size, ONEF, cat_vec, ONE, neu1, ONE);
  count += ONEF;
  if (cbow_mean && count > (real)0.5) {
    inv_count = ONEF / count;
    lib_scal(size, inv_count, neu1, ONE);
  }
  memset(work, 0, size * sizeof(real));
  for (d = 0; d <= negative; ++d) {
    if (d == 0) {
      target_index = word_index;
      label = ONEF;
    }
    else {
      target_index = table[(next_random >> 16) % table_len];
      next_random = (next_random * (uint64_t)25214903917ULL + 11) & modulo;
      if (target_index == word_index) continue;
      label = 0.0;
    }
    row2 = target_index * size;
    f = lib_dot(size, neu1, ONE, &syn1neg[row2], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP) continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = (label - f) * alpha;
    lib_axpy(size, g, &syn1neg[row2], ONE, work, ONE);
    if (word_learn) lib_axpy(size, g, neu1, ONE, &syn1neg[row2], ONE);
  }
  if (word_learn) {
    for (m = j; m < k; ++m) {
      if (m == i || codelens[m] == 0) continue;
      lib_axpy(size, ONEF, work, ONE, &syn0[indexes[m] * size], ONE);
    }
  }
  lib_axpy(size, ONEF, work, ONE, sent_vec, ONE);
  if (cat_learn) lib_axpy(size, ONEF, work, ONE, cat_vec, ONE);
}

void Category2Vec::sg_hs_adagrad(const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec, real *cat_vec, const real alpha, real* work, real* l1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t b, row2;
  real f, g;
  lib_copy(size, sent_vec, ONE, l1, ONE);
  lib_axpy(size, ONEF, cat_vec, ONE, l1, ONE);
  
  memset(work, 0, size * sizeof(real));
  for (b = 0; b < codelen; ++b) {
    row2 = word_point[b] * size;
    f = lib_dot(size, l1, ONE, &syn1[row2], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP)
      continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = 1 - word_code[b] - f;
    lib_axpy(size, g, &syn1[row2], ONE, work, ONE);
    if (word_learn) {
      lib_adagrad(size, alpha, g, l1, &syn1_grad[row2], &syn1[row2]);
    }
  }

  lib_adagrad(size, alpha, ONEF, work, sent_vec_grad, sent_vec);
  if (cat_learn) lib_adagrad(size, alpha, ONEF, work, cat_vec_grad, cat_vec);
}

void Category2Vec::sg_neg_adagrad(real *sent_vec, real *cat_vec, const uint32_t word_index, const real alpha, real* work, real* l1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t row;
  static uint64_t modulo = 281474976710655ULL;
  real f, g, label;
  uint32_t target_index;
  int d;
  lib_copy(size, sent_vec, ONE, l1, ONE);
  lib_axpy(size, ONEF, cat_vec, ONE, l1, ONE);

  memset(work, 0, size * sizeof(real));
  for (d = 0; d <= negative; ++d) {
    if (d == 0) {
      target_index = word_index;
      label = ONEF;
    }
    else {
      target_index = table[(next_random >> 16) % table_len];
      next_random = (next_random * (uint64_t)25214903917ULL + 11) & modulo;
      if (target_index == word_index) continue;
      label = 0.0;
    }
    row = target_index * size;
    f = lib_dot(size, l1, ONE, &syn1neg[row], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP) continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = label - f;
    lib_axpy(size, g, &syn1neg[row], ONE, work, ONE);
    if (word_learn) {
      lib_adagrad(size, alpha, g, l1, &syn1neg_grad[row], &syn1neg[row]);
    }
  }
  
  lib_adagrad(size, alpha, ONEF, work, sent_vec_grad, sent_vec);
  if (cat_learn) lib_adagrad(size, alpha, ONEF, work, cat_vec_grad, cat_vec);
}

void Category2Vec::cbow_hs_adagrad(const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec, real *cat_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t b, row2;
  real f, g, count, inv_count;
  int m;
  memset(neu1, 0, size * sizeof(real));
  count = 0.0;
  for (m = j; m < k; ++m) {
    if (m == i || codelens[m] == 0) continue;
    count += ONEF;
    lib_axpy(size, ONEF, &syn0[indexes[m] * size], ONE, neu1, ONE);
  }
  lib_axpy(size, ONEF, sent_vec, ONE, neu1, ONE);
  count += ONEF;
  lib_axpy(size, ONEF, cat_vec, ONE, neu1, ONE);
  count += ONEF;
  if (cbow_mean && count > (real)0.5) {
    inv_count = ONEF / count;
    lib_scal(size, inv_count, neu1, ONE);
  }

  memset(work, 0, size * sizeof(real));
  for (b = 0; b < codelens[i]; ++b) {
    row2 = word_point[b] * size;
    f = lib_dot(size, neu1, ONE, &syn1[row2], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP) continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = 1 - word_code[b] - f;
    lib_axpy(size, g, &syn1[row2], ONE, work, ONE);
    if (word_learn) {
      lib_adagrad(size, alpha, g, neu1, &syn1_grad[row2], &syn1[row2]);
    }
  }

  if (word_learn) {
    for (m = j; m < k; ++m) {
      if (m == i || codelens[m] == 0) continue;
      lib_adagrad(size, alpha, ONEF, work, &syn0_grad[indexes[m] * size], &syn0[indexes[m] * size]);
    }
  }
  lib_adagrad(size, alpha, ONEF, work, sent_vec_grad, sent_vec);
  if (cat_learn) lib_adagrad(size, alpha, ONEF, work, cat_vec_grad, cat_vec);
}

void Category2Vec::cbow_neg_adagrad(const uint32_t* indexes, const int *codelens, real *sent_vec, real *cat_vec, 
		       const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t row2;
  static uint64_t modulo = 281474976710655ULL;
  real f, g, count, inv_count, label;
  uint32_t target_index, word_index;
  int d, m;
  word_index = indexes[i];
  memset(neu1, 0, size * sizeof(real));
  count = 0.0;
  for (m = j; m < k; ++m) {
    if (m == i || codelens[m] == 0) continue;
    count += ONEF;
    lib_axpy(size, ONEF, &syn0[indexes[m] * size], ONE, neu1, ONE);
  }
  lib_axpy(size, ONEF, sent_vec, ONE, neu1, ONE);
  count += ONEF;
  lib_axpy(size, ONEF, cat_vec, ONE, neu1, ONE);
  count += ONEF;
  if (cbow_mean && count > (real)0.5) {
    inv_count = ONEF / count;
    lib_scal(size, inv_count, neu1, ONE);
  }

  memset(work, 0, size * sizeof(real));
  for (d = 0; d <= negative; ++d) {
    if (d == 0) {
      target_index = word_index;
      label = ONEF;
    }
    else {
      target_index = table[(next_random >> 16) % table_len];
      next_random = (next_random * (uint64_t)25214903917ULL + 11) & modulo;
      if (target_index == word_index) continue;
      label = 0.0;
    }
    row2 = target_index * size;
    f = lib_dot(size, neu1, ONE, &syn1neg[row2], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP) continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = label - f;
    lib_axpy(size, g, &syn1neg[row2], ONE, work, ONE);
    if (word_learn) {
      lib_adagrad(size, alpha, g, neu1, &syn1neg_grad[row2], &syn1neg[row2]);
    }
  }

  if (word_learn) {
    for (m = j; m < k; ++m) {
      if (m == i || codelens[m] == 0) continue;
      lib_adagrad(size, alpha, ONEF, work, &syn0_grad[indexes[m] * size], &syn0[indexes[m] * size]);
    }
  }
  lib_adagrad(size, alpha, ONEF, work, sent_vec_grad, sent_vec);
  if (cat_learn) lib_adagrad(size, alpha, ONEF, work, cat_vec_grad, cat_vec);
}

void Category2Vec::sg_hs_adadelta(const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec, real *cat_vec, const real alpha, real* work, real* l1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t b, row2;
  real f, g;
  lib_copy(size, sent_vec, ONE, l1, ONE);
  lib_axpy(size, ONEF, cat_vec, ONE, l1, ONE);
  
  memset(work, 0, size * sizeof(real));
  for (b = 0; b < codelen; ++b) {
    row2 = word_point[b] * size;
    f = lib_dot(size, l1, ONE, &syn1[row2], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP)
      continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = 1 - word_code[b] - f;
    lib_axpy(size, g, &syn1[row2], ONE, work, ONE);
    if (word_learn) {
      lib_adadelta(size, g, l1, &syn1_grad[2 * row2], &syn1[row2]);
    }
  }
  lib_adadelta(size, ONEF, work, sent_vec_grad, sent_vec);
  if (cat_learn) lib_adadelta(size, ONEF, work, cat_vec_grad, cat_vec);
}

void Category2Vec::sg_neg_adadelta(real *sent_vec, real *cat_vec, const uint32_t word_index, const real alpha, real* work, real* l1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t row;
  static uint64_t modulo = 281474976710655ULL;
  real f, g, label;
  uint32_t target_index;
  int d;
  lib_copy(size, sent_vec, ONE, l1, ONE);
  lib_axpy(size, ONEF, cat_vec, ONE, l1, ONE);

  memset(work, 0, size * sizeof(real));
  for (d = 0; d <= negative; ++d) {
    if (d == 0) {
      target_index = word_index;
      label = ONEF;
    }
    else {
      target_index = table[(next_random >> 16) % table_len];
      next_random = (next_random * (uint64_t)25214903917ULL + 11) & modulo;
      if (target_index == word_index) continue;
      label = 0.0;
    }
    row = target_index * size;
    f = lib_dot(size, l1, ONE, &syn1neg[row], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP) continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = label - f;
    lib_axpy(size, g, &syn1neg[row], ONE, work, ONE);
    if (word_learn) {
      lib_adadelta(size, g, l1, &syn1neg_grad[2 * row], &syn1neg[row]);
    }
  }
  lib_adadelta(size, ONEF, work, sent_vec_grad, sent_vec);
  if (cat_learn) lib_adadelta(size, ONEF, work, cat_vec_grad, cat_vec);
}

void Category2Vec::cbow_hs_adadelta(const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec, real *cat_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t b, row, row2;
  real f, g, count, inv_count;
  int m;
  memset(neu1, 0, size * sizeof(real));
  count = 0.0;
  for (m = j; m < k; ++m) {
    if (m == i || codelens[m] == 0) continue;
    count += ONEF;
    lib_axpy(size, ONEF, &syn0[indexes[m] * size], ONE, neu1, ONE);
  }
  lib_axpy(size, ONEF, sent_vec, ONE, neu1, ONE);
  count += ONEF;
  lib_axpy(size, ONEF, cat_vec, ONE, neu1, ONE);
  count += ONEF;
  if (cbow_mean && count > (real)0.5) {
    inv_count = ONEF / count;
    lib_scal(size, inv_count, neu1, ONE);
  }

  memset(work, 0, size * sizeof(real));
  for (b = 0; b < codelens[i]; ++b) {
    row2 = word_point[b] * size;
    f = lib_dot(size, neu1, ONE, &syn1[row2], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP) continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = 1 - word_code[b] - f;
    lib_axpy(size, g, &syn1[row2], ONE, work, ONE);
    if (word_learn) {
      lib_adadelta(size, g, neu1, &syn1_grad[2 * row2], &syn1[row2]);
    }
  }

  if (word_learn) {
    for (m = j; m < k; ++m) {
      if (m == i || codelens[m] == 0) continue;
      row = indexes[m] * size;
      lib_adadelta(size, ONEF, work, &syn0_grad[2 * row], &syn0[row]);
    }
  }

  lib_adadelta(size, ONEF, work, sent_vec_grad, sent_vec);
  if (cat_learn) lib_adadelta(size, ONEF, work, cat_vec_grad, cat_vec);
}

void Category2Vec::cbow_neg_adadelta(const uint32_t* indexes, const int *codelens, real *sent_vec, real *cat_vec, 
		       const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t row, row2;
  static uint64_t modulo = 281474976710655ULL;
  real f, g, count, inv_count, label;
  uint32_t target_index, word_index;
  int d, m;
  word_index = indexes[i];
  memset(neu1, 0, size * sizeof(real));
  count = 0.0;
  for (m = j; m < k; ++m) {
    if (m == i || codelens[m] == 0) continue;
    count += ONEF;
    lib_axpy(size, ONEF, &syn0[indexes[m] * size], ONE, neu1, ONE);
  }
  lib_axpy(size, ONEF, sent_vec, ONE, neu1, ONE);
  count += ONEF;
  lib_axpy(size, ONEF, cat_vec, ONE, neu1, ONE);
  count += ONEF;
  if (cbow_mean && count > (real)0.5) {
    inv_count = ONEF / count;
    lib_scal(size, inv_count, neu1, ONE);
  }

  memset(work, 0, size * sizeof(real));
  for (d = 0; d <= negative; ++d) {
    if (d == 0) {
      target_index = word_index;
      label = ONEF;
    }
    else {
      target_index = table[(next_random >> 16) % table_len];
      next_random = (next_random * (uint64_t)25214903917ULL + 11) & modulo;
      if (target_index == word_index) continue;
      label = 0.0;
    }
    row2 = target_index * size;
    f = lib_dot(size, neu1, ONE, &syn1neg[row2], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP) continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = label - f;
    lib_axpy(size, g, &syn1neg[row2], ONE, work, ONE);
    if (word_learn) {
      lib_adadelta(size, g, neu1, &syn1neg_grad[2 * row2], &syn1neg[row2]);
    }
  }

  if (word_learn) {
    for (m = j; m < k; ++m) {
      if (m == i || codelens[m] == 0) continue;
      row = indexes[m] * size;
      lib_adadelta(size, ONEF, work, &syn0_grad[2 * row], &syn0[row]);
    }
  }

  lib_adadelta(size, ONEF, work, sent_vec_grad, sent_vec);
  if (cat_learn) lib_adadelta(size, ONEF, work, cat_vec_grad, cat_vec);
}

void Category2Vec::sg_hs_adam(const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec, real *cat_vec, const real alpha, real* work, real* l1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t b, row2, row3;
  real f, g;
  lib_copy(size, sent_vec, ONE, l1, ONE);
  lib_axpy(size, ONEF, cat_vec, ONE, l1, ONE);
  memset(work, 0, size * sizeof(real));

  for (b = 0; b < codelen; ++b) {
    row2 = word_point[b] * size;
    row3 = word_point[b] * (2 * size + 3);
    f = lib_dot(size, l1, ONE, &syn1[row2], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP)
      continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = 1 - word_code[b] - f;
    lib_axpy(size, g, &syn1[row2], ONE, work, ONE);
    if (word_learn) {
      lib_adam(size, g, l1, &syn1_grad[row3], &syn1[row2]);
    }
  }
  lib_adam(size, ONEF, work, sent_vec_grad, sent_vec);
  if (cat_learn) lib_adam(size, ONEF, work, cat_vec_grad, cat_vec);
}

void Category2Vec::sg_neg_adam(real *sent_vec, real *cat_vec, const uint32_t word_index, const real alpha, real* work, real* l1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t row, row3;
  static uint64_t modulo = 281474976710655ULL;
  real f, g, label;
  uint32_t target_index;
  int d;
  lib_copy(size, sent_vec, ONE, l1, ONE);
  lib_axpy(size, ONEF, cat_vec, ONE, l1, ONE);
  memset(work, 0, size * sizeof(real));

  for (d = 0; d <= negative; ++d) {
    if (d == 0) {
      target_index = word_index;
      label = ONEF;
    }
    else {
      target_index = table[(next_random >> 16) % table_len];
      next_random = (next_random * (uint64_t)25214903917ULL + 11) & modulo;
      if (target_index == word_index) continue;
      label = 0.0;
    }
    row = target_index * size;
    row3 = target_index * (2 * size + 3);
    f = lib_dot(size, l1, ONE, &syn1neg[row], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP) continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = label - f;
    lib_axpy(size, g, &syn1neg[row], ONE, work, ONE);
    if (word_learn) {
      lib_adam(size, g, l1, &syn1neg_grad[row3], &syn1neg[row]);
    }
  }
  lib_adam(size, ONEF, work, sent_vec_grad, sent_vec);
  if (cat_learn) lib_adam(size, ONEF, work, cat_vec_grad, cat_vec);
}

void Category2Vec::cbow_hs_adam(const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec, real *cat_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t b, row, row2, row3;
  real f, g, count, inv_count;
  int m;
  memset(neu1, 0, size * sizeof(real));
  count = 0.0;
  for (m = j; m < k; ++m) {
    if (m == i || codelens[m] == 0) continue;
    count += ONEF;
    lib_axpy(size, ONEF, &syn0[indexes[m] * size], ONE, neu1, ONE);
  }
  lib_axpy(size, ONEF, sent_vec, ONE, neu1, ONE);
  lib_axpy(size, ONEF, cat_vec, ONE, neu1, ONE);
  count += 2.0 * ONEF;
  if (cbow_mean && count > (real)0.5) {
    inv_count = ONEF / count;
    lib_scal(size, inv_count, neu1, ONE);
  }

  memset(work, 0, size * sizeof(real));
  for (b = 0; b < codelens[i]; ++b) {
    row2 = word_point[b] * size;
    row3 = word_point[b] * (2 * size + 3);
    f = lib_dot(size, neu1, ONE, &syn1[row2], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP) continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = 1 - word_code[b] - f;
    lib_axpy(size, g, &syn1[row2], ONE, work, ONE);
    if (word_learn) {
      lib_adam(size, g, neu1, &syn1_grad[row3], &syn1[row2]);
    }
  }
  if (word_learn) {
    for (m = j; m < k; ++m) {
      if (m == i || codelens[m] == 0) continue;
      row = indexes[m] * size;
      row3 = indexes[m] * (2 * size + 3);
      lib_adam(size, ONEF, work, &syn0_grad[row3], &syn0[row]);
    }
  }
  lib_adam(size, ONEF, work, sent_vec_grad, sent_vec);
  if (cat_learn) lib_adam(size, ONEF, work, cat_vec_grad, cat_vec);
}

void Category2Vec::cbow_neg_adam(const uint32_t* indexes, const int *codelens, real *sent_vec, real *cat_vec, 
		       const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad, real *cat_vec_grad) {
  int64_t row, row2, row3;
  static uint64_t modulo = 281474976710655ULL;
  real f, g, count, inv_count, label;
  uint32_t target_index, word_index;
  int d, m;
  word_index = indexes[i];
  memset(neu1, 0, size * sizeof(real));
  count = 0.0;
  for (m = j; m < k; ++m) {
    if (m == i || codelens[m] == 0) continue;
    count += ONEF;
    lib_axpy(size, ONEF, &syn0[indexes[m] * size], ONE, neu1, ONE);
  }
  lib_axpy(size, ONEF, sent_vec, ONE, neu1, ONE);
  lib_axpy(size, ONEF, cat_vec, ONE, neu1, ONE);
  count += 2.0 * ONEF;
  if (cbow_mean && count > (real)0.5) {
    inv_count = ONEF / count;
    lib_scal(size, inv_count, neu1, ONE);
  }

  memset(work, 0, size * sizeof(real));
  for (d = 0; d <= negative; ++d) {
    if (d == 0) {
      target_index = word_index;
      label = ONEF;
    }
    else {
      target_index = table[(next_random >> 16) % table_len];
      next_random = (next_random * (uint64_t)25214903917ULL + 11) & modulo;
      if (target_index == word_index) continue;
      label = 0.0;
    }
    row2 = target_index * size;
    row3 = target_index * (2 * size + 3);
    f = lib_dot(size, neu1, ONE, &syn1neg[row2], ONE);
    if (f <= -MAX_EXP || f >= MAX_EXP) continue;
    f = EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = label - f;
    lib_axpy(size, g, &syn1neg[row2], ONE, work, ONE);
    if (word_learn) {
      lib_adam(size, g, neu1, &syn1neg_grad[row3], &syn1neg[row2]);
    }
  }
  if (word_learn) {
    for (m = j; m < k; ++m) {
      if (m == i || codelens[m] == 0) continue;
      row = indexes[m] * size;
      row3 = indexes[m] * (2 * size + 3);
      lib_adam(size, ONEF, work, &syn0_grad[row3], &syn0[row]);
    }
  }
  lib_adam(size, ONEF, work, sent_vec_grad, sent_vec);
  if (cat_learn) lib_adam(size, ONEF, work, cat_vec_grad, cat_vec);
}

void Category2Vec::calc_sim_sent_vec(const real *vec, real *sim_ary) {
  real vec_len_r = rsqrt_f(lib_dot(size, vec, ONE, vec, ONE));
  for (int i = 0; i < sents_len; ++i) {
    real *vec2 = &sents[i * size];
    real vec2_len_r = rsqrt_f(lib_dot(size, vec2, ONE, vec2, ONE));
    sim_ary[i] = lib_dot(size, vec, ONE, vec2, ONE) * vec_len_r * vec2_len_r;
  }
}

void Category2Vec::calc_sim_cat_vec(const real *vec, real *sim_ary) {
  real vec_len_r = rsqrt_f(lib_dot(size, vec, ONE, vec, ONE));
  for (int i = 0; i < cats_len; ++i) {
    real *vec2 = &cats[i * size];
    real vec2_len_r = rsqrt_f(lib_dot(size, vec2, ONE, vec2, ONE));
    sim_ary[i] = lib_dot(size, vec, ONE, vec2, ONE) * vec_len_r * vec2_len_r;
  }
}

void Category2Vec::init_pairtable() {
  for (int i = 0; i < pair_sc_len; ++i) {
    real *vec = &pairtable[i * size];
    real *svec = &sents[pair_sc[i * 2] * size];
    real *cvec = &cats[pair_sc[i * 2 + 1] * size];
    lib_copy(size, svec, ONE, vec, ONE);
    lib_axpy(size, ONEF, cvec, ONE, vec, ONE);
    real vec_len_r = rsqrt_f(lib_dot(size, vec, ONE, vec, ONE));
    lib_scal(size, vec_len_r, vec, ONE);
  }
}

void Category2Vec::calc_joint_pairtable(Category2Vec* model1, Category2Vec* model2, real* table) {
  int size = model1->size;
  int size2 = size * 2;
  for (int i = 0; i < model1->pair_sc_len; ++i) {
    real *vec1 = &table[i * size * 2];
    real *svec1 = &model1->sents[model1->pair_sc[i * 2] * size];
    real *cvec1 = &model1->cats[model1->pair_sc[i * 2 + 1] * size];
    lib_copy(size, svec1, ONE, vec1, ONE);
    lib_axpy(size, ONEF, cvec1, ONE, vec1, ONE);
    real *vec2 = &table[(2 * i + 1) * size];
    real *svec2 = &model2->sents[model2->pair_sc[i * 2] * size];
    real *cvec2 = &model2->cats[model2->pair_sc[i * 2 + 1] * size];
    lib_copy(size, svec2, ONE, vec2, ONE);
    lib_axpy(size, ONEF, cvec2, ONE, vec2, ONE);
    real vec_len_r = rsqrt_f(lib_dot(size2, vec1, ONE, vec1, ONE));
    lib_scal(size2, vec_len_r, vec1, ONE);
  }
}

void Category2Vec::calc_sim_catsent_concat(const real *svec, const real *cvec, real *sim_ary) {
  real svec_len_r = rsqrt_f(lib_dot(size, svec, ONE, svec, ONE));
  real cvec_len_r = rsqrt_f(lib_dot(size, cvec, ONE, cvec, ONE));
  for (int i = 0; i < pair_sc_len; ++i) {
    real *svec2 = &sents[pair_sc[i * 2] * size];
    real *cvec2 = &cats[pair_sc[i * 2 + 1] * size];
    real svec2_len_r = rsqrt_f(lib_dot(size, svec2, ONE, svec2, ONE));
    real cvec2_len_r = rsqrt_f(lib_dot(size, cvec2, ONE, cvec2, ONE));
    sim_ary[i] = lib_dot(size, svec, ONE, svec2, ONE) * svec_len_r * svec2_len_r * 0.5
      + lib_dot(size, cvec, ONE, cvec2, ONE) * cvec_len_r * cvec2_len_r * 0.5;
  }
}

void Category2Vec::calc_sim_catsent_sum(const real *svec, const real *cvec, real *sim_ary) {
  real *vec = new real[size];
  lib_copy(size, svec, ONE, vec, ONE);
  lib_axpy(size, ONEF, cvec, ONE, vec, ONE);
  real vec_len_r = rsqrt_f(lib_dot(size, vec, ONE, vec, ONE));
  lib_scal(size, vec_len_r, vec, ONE);
  #ifdef USE_BLAS
  lib_gemv(pair_sc_len, size, ONEF, pairtable, size, vec, ONE, ZEROF, sim_ary, ONE);
  #else
  for (int i = 0; i < pair_sc_len; ++i) {
    real *vec2 = &pairtable[i * size];
    sim_ary[i] = lib_dot(size, vec, ONE, vec2, ONE);
  }
  #endif
  delete[] vec;
}

void Category2Vec::joint_calc_sim_catsent_sum(const int pair_sc_len, const int size, const real *table, const real *svec1, const real *cvec1, const real *svec2, const real *cvec2, real *sim_ary) {
  real *vec = new real[size * 2];
  const int size2 = size * 2;
  lib_copy(size, svec1, ONE, vec, ONE);
  lib_axpy(size, ONEF, cvec1, ONE, vec, ONE);
  real *vec2 = &vec[size];
  lib_copy(size, svec2, ONE, vec2, ONE);
  lib_axpy(size, ONEF, cvec2, ONE, vec2, ONE);
  real vec_len_r = rsqrt_f(lib_dot(size2, vec, ONE, vec, ONE));
  lib_scal(size2, vec_len_r, vec, ONE);
  #ifdef USE_BLAS
  lib_gemv(pair_sc_len, size2, ONEF, table, size2, vec, ONE, ZEROF, sim_ary, ONE);
  #else
  for (int i = 0; i < pair_sc_len; ++i) {
    const real *vec_t = &table[i * size];
    sim_ary[i] = lib_dot(size2, vec_t, ONE, vec, ONE);
  }
  #endif
  delete[] vec;
}
