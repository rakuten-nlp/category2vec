#include "sent2vec_calc.h"
using namespace sentence2vec;

real Sentence2Vec::EXP_TABLE[EXP_TABLE_SIZE];

/* precompute function sigmoid(x) = 1 / (1 + exp(-x)) */
void Sentence2Vec::calcExpTable() {
  for (int i = 0; i < EXP_TABLE_SIZE; ++i) {
    EXP_TABLE[i] = (real) exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
    EXP_TABLE[i] = EXP_TABLE[i] / (EXP_TABLE[i] + 1);
  }
}

void Sentence2Vec::set_update_mode(int update_mode) {
  switch(update_mode) {
  case 0:
    sg_hs = &Sentence2Vec::sg_hs_sgd;
    sg_neg = &Sentence2Vec::sg_neg_sgd;
    cbow_hs = &Sentence2Vec::cbow_hs_sgd;
    cbow_neg = &Sentence2Vec::cbow_neg_sgd;
    break;
  case 1:
    sg_hs = &Sentence2Vec::sg_hs_adagrad;
    sg_neg = &Sentence2Vec::sg_neg_adagrad;
    cbow_hs = &Sentence2Vec::cbow_hs_adagrad;
    cbow_neg = &Sentence2Vec::cbow_neg_adagrad;
    break;
  case 2:
    sg_hs = &Sentence2Vec::sg_hs_adadelta;
    sg_neg = &Sentence2Vec::sg_neg_adadelta;
    cbow_hs = &Sentence2Vec::cbow_hs_adadelta;
    cbow_neg = &Sentence2Vec::cbow_neg_adadelta;
    break;
  case 3:
    sg_hs = &Sentence2Vec::sg_hs_adam;
    sg_neg = &Sentence2Vec::sg_neg_adam;
    cbow_hs = &Sentence2Vec::cbow_hs_adam;
    cbow_neg = &Sentence2Vec::cbow_neg_adam;
    break;
  }
}

void Sentence2Vec::train_vec(real *sent_vec, const real alpha, const int sentence_len, const uint32_t *reduced_windows, uint32_t* const *points, uint8_t* const *codes, const int *codelens, const uint32_t* indexes, real *work, real *neu1, real *sent_vec_grad) {
  (this->*train_func)(sent_vec, alpha, sentence_len, reduced_windows, points, codes, codelens, indexes, work, neu1, sent_vec_grad);
}

void Sentence2Vec::train_sg(real *sent_vec, const real alpha, const int sentence_len, const uint32_t *reduced_windows, uint32_t* const *points, uint8_t* const *codes, const int *codelens, const uint32_t* indexes, real *work, real *neu1, real *sent_vec_grad) {
  for (int i = 0; i < sentence_len; ++i) {
    if (codelens[i] == 0) continue;
    int j = i - window + reduced_windows[i];
    if (j < 0) j = 0;
    int k = i + window + 1 - reduced_windows[i];
    if (k > sentence_len) k = sentence_len;
    if (hs) {
      for (; j < k; ++j) {
	if (codelens[j] == 0) continue;
	(this->*sg_hs)(points[j], codes[j], codelens[j], sent_vec, alpha, work, neu1, sent_vec_grad);
      }
    }
    if (negative) {
      for (; j < k; ++j) {
	if (codelens[j] == 0) continue;
	(this->*sg_neg)(sent_vec, indexes[j], alpha, work, neu1, sent_vec_grad);
      }
    }
  }
}

void Sentence2Vec::train_cbow(real *sent_vec, const real alpha, const int sentence_len, const uint32_t *reduced_windows, uint32_t* const *points, uint8_t* const *codes, const int *codelens, const uint32_t* indexes, real *work, real *neu1, real *sent_vec_grad) {
  for (int i = 0; i < sentence_len; ++i) {
    if (codelens[i] == 0) continue;
    int j = i - window + reduced_windows[i];
    if (j < 0) j = 0;
    int k = i + window + 1 - reduced_windows[i];
    if (k > sentence_len) k = sentence_len;
    if (hs)
      (this->*cbow_hs)(indexes, points[i], codes[i], codelens, sent_vec, alpha, i, j, k, work, neu1, sent_vec_grad);
    if (negative)
      (this->*cbow_neg)(indexes, codelens, sent_vec, alpha, i, j, k, work, neu1, sent_vec_grad);
  }
}

void Sentence2Vec::sg_hs_sgd(const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec, const real alpha, real* work, real* neu1, real *sent_vec_grad) {
  int64_t b, row2;
  real f, g, *l1 = neu1;
  lib_copy(size, sent_vec, ONE, neu1, ONE);
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
}

void Sentence2Vec::sg_neg_sgd(real *sent_vec, const uint32_t word_index, const real alpha, real* work, real* neu1, real *sent_vec_grad) {
  int64_t row;
  static uint64_t modulo = 281474976710655ULL;
  real f, g, label, *l1 = neu1;
  uint32_t target_index;
  int d;
  lib_copy(size, sent_vec, ONE, neu1, ONE);
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
}

void Sentence2Vec::cbow_hs_sgd(const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad) {
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
}

void Sentence2Vec::cbow_neg_sgd(const uint32_t* indexes, const int *codelens, real *sent_vec, 
		       const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad) {
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
}

void Sentence2Vec::sg_hs_adagrad(const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec, const real alpha, real* work, real* neu1, real *sent_vec_grad) {
  int64_t b, row2;
  real f, g, *l1 = neu1;
  lib_copy(size, sent_vec, ONE, neu1, ONE);
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
}

void Sentence2Vec::sg_neg_adagrad(real *sent_vec, const uint32_t word_index, const real alpha, real* work, real* neu1, real *sent_vec_grad) {
  int64_t row;
  static uint64_t modulo = 281474976710655ULL;
  real f, g, label, *l1 = neu1;
  uint32_t target_index;
  int d;
  lib_copy(size, sent_vec, ONE, neu1, ONE);
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
}

void Sentence2Vec::cbow_hs_adagrad(const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad) {
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
}

void Sentence2Vec::cbow_neg_adagrad(const uint32_t* indexes, const int *codelens, real *sent_vec,
		       const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad) {
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
}

void Sentence2Vec::sg_hs_adadelta(const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec, const real alpha, real* work, real* neu1, real *sent_vec_grad) {
  int64_t b, row2;
  real f, g, *l1 = neu1;
  lib_copy(size, sent_vec, ONE, neu1, ONE);
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
}

void Sentence2Vec::sg_neg_adadelta(real *sent_vec, const uint32_t word_index, const real alpha, real* work, real* neu1, real *sent_vec_grad) {
  int64_t row;
  static uint64_t modulo = 281474976710655ULL;
  real f, g, label, *l1 = neu1;
  uint32_t target_index;
  int d;
  lib_copy(size, sent_vec, ONE, neu1, ONE);
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
}

void Sentence2Vec::cbow_hs_adadelta(const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad) {
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
}

void Sentence2Vec::cbow_neg_adadelta(const uint32_t* indexes, const int *codelens, real *sent_vec, 
		       const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad) {
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
}

void Sentence2Vec::sg_hs_adam(const uint32_t *word_point, const uint8_t *word_code, const int codelen, real *sent_vec, const real alpha, real* work, real* l1, real *sent_vec_grad) {
  int64_t b, row2, row3;
  real f, g;
  lib_copy(size, sent_vec, ONE, l1, ONE);
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
}

void Sentence2Vec::sg_neg_adam(real *sent_vec, const uint32_t word_index, const real alpha, real* work, real* l1, real *sent_vec_grad) {
  int64_t row, row3;
  static uint64_t modulo = 281474976710655ULL;
  real f, g, label;
  uint32_t target_index;
  int d;
  lib_copy(size, sent_vec, ONE, l1, ONE);
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
}

void Sentence2Vec::cbow_hs_adam(const uint32_t* indexes, const uint32_t *word_point, const uint8_t *word_code, const int *codelens, real *sent_vec, const real alpha, int i, int j, int k, real* work, real* neu1, real *sent_vec_grad) {
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
  count += ONEF;
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
}

void Sentence2Vec::cbow_neg_adam(const uint32_t* indexes, const int *codelens, real *sent_vec, const real alpha, 
				 int i, int j, int k, real* work, real* neu1, real *sent_vec_grad) {
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
  count +=  ONEF;
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
}

void Sentence2Vec::calc_sim_sent_vec(const real *vec, real *sim_ary) {
  real vec_len_r = rsqrt_f(lib_dot(size, vec, ONE, vec, ONE));
  real vec2_len_r;
  for (int i = 0; i < sents_len; ++i) {
    real *vec2 = &sents[i * size];
    real vec2_len_sq = lib_dot(size, vec2, ONE, vec2, ONE);
    if (vec2_len_sq == 0)
      vec2_len_r = 0;
    else
      vec2_len_r = rsqrt_f(vec2_len_sq);
    sim_ary[i] = lib_dot(size, vec, ONE, vec2, ONE) * vec_len_r * vec2_len_r;
  }
}
