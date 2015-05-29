#include <stdlib.h>
#include <string.h>
#include "options.h"

const real ONEF = 1.0;
const int ONE = 1;
const real ZEROF = 0.0;
const char TRANS_N = 'N';

copy_ptr blas_copy;
axpy_ptr blas_axpy;
dot_ptr blas_dot;
ddot_ptr blas_ddot;
scal_ptr blas_scal;
gemv_ptr blas_gemv;

/* y = x^2 */
void lib_xsq_seq(const int size, real* x, real* y) {
  int i;
  for (i = 0; i < size; ++i)
    y[i] = x[i] * x[i];
}

#ifdef USE_AVX
/* y = x^2 using AVX; assumes x and y is 32byte-aligned */
void lib_xsq_avx(const int size, real* x, real* y) {
  int i;
  int avxloop = size / AVXVECSIZE;
  int rest_idx = avxloop * AVXVECSIZE;
  /* no need to use offsets */
  vreal* vx = (vreal*)x;
  vreal* vy = (vreal*)y;
  /* use AVX */
  for (i = 0; i < avxloop; ++i)
    vy[i] = mm256_mul(vx[i], vx[i]);
  /* compute sequencially for the rest */
  for (i = rest_idx; i < size; ++i)
    y[i] = x[i] * x[i];
}
#endif

#ifdef USE_AVX
/* SGD with AVX; assumes grad is 32byte-aligned */
void lib_sgd_avx(const int size, const real alpha, real* grad, real* vec) {
  const vreal valpha = mm256_load(alpha);
  vreal vvec[MAX_DIMENSION/AVXVECSIZE]__attribute__((aligned(32)));
  vreal *vgrad = (vreal*) grad;
  int avxloop = size / AVXVECSIZE;
  int rest_idx = size - (size % AVXVECSIZE);
  memcpy(vvec, vec, avxloop * sizeof(vreal) / sizeof(char));
  int i;
  /* use AVX */
  for (i = 0; i < avxloop; ++i) {
    vvec[i] = mm256_add(vvec[i], mm256_mul(valpha, vgrad[i]));
  }
  /* compute sequencially for the rest */
  for (i = rest_idx; i < size; ++i) {
    vec[i] += alpha * grad[i];
  }
  /* write back to the original pointer */
  memcpy(vec, vvec, avxloop * sizeof(vreal) / sizeof(char));
}
#endif

/* adagrad */
void lib_adagrad_seq(const int size, const real alpha, const real g, real* grad, real* vecg, real* vec) {
  int a;
  for (a = 0; a < size; ++a) {
    vecg[a] += g * g * grad[a] * grad[a];
    vec[a] += alpha * rsqrt_f(EPSILON + vecg[a]) * g * grad[a];
  }
}

#ifdef USE_AVX
/* adagrad using AVX with stack; assumes grad is 32byte-aligned */
void lib_adagrad_avx(const int size, const real alpha, const real g, real* grad, real* vecg, real* vec) {
  const vreal vepsilon = mm256_load(EPSILON);
  const vreal valpha = mm256_load(alpha);
  const vreal vg = mm256_load(g);
  const vreal vg2 = mm256_load(g * g);
  vreal vvec[MAX_DIMENSION/AVXVECSIZE]__attribute__((aligned(32)));
  vreal vvecg[MAX_DIMENSION/AVXVECSIZE]__attribute__((aligned(32)));
  vreal *vgrad = (vreal*)grad;
  int avxloop = size / AVXVECSIZE;
  int rest_idx = size - (size % AVXVECSIZE);
  memcpy(vvec, vec, avxloop * sizeof(vreal) / sizeof(char));
  memcpy(vvecg, vecg, avxloop * sizeof(vreal) / sizeof(char));
  int i; 
  /* use AVX */
  for (i = 0; i < avxloop; ++i) {
    vvecg[i] = mm256_add(vvecg[i], mm256_mul(vg2, mm256_sq(vgrad[i]))); 
    vvec[i] = mm256_add(vvec[i], mm256_mul(mm256_mul(valpha, vg), mm256_mul(mm256_rsqrt(mm256_add(vepsilon, vvecg[i])), vgrad[i])));
  }
  /* compute sequencially for the rest */
  for (i = rest_idx; i < size; ++i) {
    vecg[i] += g * g * grad[i] * grad[i];
    vec[i] += alpha * g * rsqrt_f(EPSILON + vecg[i]) * grad[i];
  }
  /* write back to the original pointer */
  memcpy(vec, vvec, avxloop * sizeof(vreal) / sizeof(char));
  memcpy(vecg, vvecg, avxloop * sizeof(vreal) / sizeof(char));
}
/* adagrad using AVX without stack; assumes _grad is 32byte-aligned, and vec and vecg have the same offset from 32byte-aligned */
void lib_adagrad_avx_nostack(const int size, const real alpha, const real g, real* _grad, real* vecg, real* vec) {
  const vreal vepsilon = mm256_load(EPSILON);
  const vreal valpha = mm256_load(alpha);
  const vreal vg = mm256_load(g);
  const vreal vg2 = mm256_load(g * g);
  int i;
  vreal *vvec, *vvecg, *vgrad;
  uintptr_t ptr = (uintptr_t)vec;
  //uintptr_t ptrg = (uintptr_t)vecg;
  int shift = (ptr % 32) / sizeof(real);
  //int shift2 = (ptrg % 32) / sizeof(real);
  /* assumption: offsets of vec and vecg are the same */
  //assert(shift == shift2);
  /* from this offset we can use AVX */
  int offset = AVXVECSIZE - shift;
  int avxloop = (size - offset) / AVXVECSIZE;
  int rest_idx = size - (size - offset) % AVXVECSIZE;
  /* set grad to the same offset as vec and vecg */
  real* grad = &_grad[shift];
  memmove(grad, _grad, size * sizeof(real) / sizeof(char));
  vvec = (vreal*)&vec[offset];
  vvecg = (vreal*)&vecg[offset];
  vgrad = (vreal*)&grad[offset];
  /* compute sequencially until offset */
  for (i = 0; i < offset; ++i) {
    vecg[i] += g * g * grad[i] * grad[i];
    vec[i] += alpha * g * rsqrt_f(EPSILON + vecg[i]) * grad[i];
  }
  /* use AVX */
  for (i = 0; i < avxloop; ++i) {
    vvecg[i] = mm256_add(vvecg[i], mm256_mul(vg2, mm256_sq(vgrad[i])));
    vvec[i] = mm256_add(vvec[i], mm256_mul(mm256_mul(valpha, vg), mm256_mul(mm256_rsqrt(mm256_add(vepsilon, vvecg[i])), vgrad[i])));
  }
  /* compute sequencially for the rest */
  for (i = rest_idx; i < size; ++i) {
    vecg[i] += g * g * grad[i] * grad[i];
    vec[i] += alpha * g * rsqrt_f(EPSILON + vecg[i]) * grad[i];
  }
  /* write back to the original pointer */
  memmove(_grad, grad, size * sizeof(real) / sizeof(char));
}
#endif

/* adadelta */
void lib_adadelta_seq(const int size, const real g, real* grad, real* vecg, real* vec) {
  const real rho = RHO;
  int i;
  real d;
  for (i = 0; i < size; ++i) {
    vecg[i] = rho * vecg[i] + (1-rho) * g * g * grad[i] * grad[i];
    d = sqrt_f(EPSILON_D + vecg[size + i]) * rsqrt_f(EPSILON_D + vecg[i]) * g * grad[i];
    vecg[size + i] = rho * vecg[size + i] + (1-rho) * d * d;
    vec[i] += d;
  }
}

#ifdef USE_AVX
/* adadelta using AVX; grad is 32byte-aligned */
void lib_adadelta_avx(const int size, const real g, real* grad, real* vecg, real* vec) {
  const real rho = RHO;
  const vreal vepsilon = mm256_load(EPSILON_D);
  const vreal vrho = mm256_load(rho);
  const vreal v1mrho = mm256_load(1-rho);
  const vreal vg = mm256_load(g);
  const vreal vg2 = mm256_load(g * g);
  const vreal* vgrad = (vreal*)grad;
  /* copy vectors to memory-aligned space (GCC style) */
  vreal vvec[MAX_DIMENSION/AVXVECSIZE]__attribute__((aligned(32)));
  vreal vvecg[MAX_DIMENSION/AVXVECSIZE]__attribute__((aligned(32)));
  vreal vvecd[MAX_DIMENSION/AVXVECSIZE]__attribute__((aligned(32)));
  const int avxloop = size / AVXVECSIZE;
  const int rest_idx = size - (size % AVXVECSIZE);
  memcpy(vvec, vec, avxloop * sizeof(vreal) / sizeof(char));
  memcpy(vvecg, vecg, avxloop * sizeof(vreal) / sizeof(char));
  memcpy(vvecd, &vecg[size], avxloop * sizeof(vreal) / sizeof(char));
  /* use AVX */
  int i;
  real d;
  vreal vd;
  for (i = 0; i < avxloop; ++i) {
    /* E[g^2] = rho * E[g^2] + (1-rho) * g^2 */
    vvecg[i] = mm256_add(mm256_mul(vrho, vvecg[i]), mm256_mul(v1mrho, mm256_mul(vg2, mm256_sq(vgrad[i]))));
    /* d = sqrt(eps + E[d^2]) / sqrt(eps + E[g^2]) * g */
    vd = mm256_mul(mm256_mul(mm256_sqrt(mm256_add(vepsilon, vvecd[i])),mm256_rsqrt(mm256_add(vepsilon, vvecg[i]))), mm256_mul(vg, vgrad[i]));
    /* E[d^2] = rho * E[d^2] + (1-rho) * d^2 */
    vvecd[i] = mm256_add(mm256_mul(vrho, vvecd[i]), mm256_mul(v1mrho, mm256_sq(vd)));
    /* v = v + d */
    vvec[i] = mm256_add(vvec[i], vd);
  }
  /* calc sequencially */
  for (i = rest_idx; i < size; ++i) {
    vecg[i] = rho * vecg[i] + (1-rho) * g * g * grad[i] * grad[i];
    d = sqrt_f(EPSILON_D + vecg[size + i]) * rsqrt_f(EPSILON_D + vecg[i]) * g * grad[i];
    vecg[size + i] = rho * vecg[size + i] + (1-rho) * d * d;
    vec[i] += d;
  }
  /* write back */
  memcpy(vec, vvec, avxloop * sizeof(vreal) / sizeof(char));
  memcpy(vecg, vvecg, avxloop * sizeof(vreal) / sizeof(char));
  memcpy(&vecg[size], vvecd, avxloop * sizeof(vreal) / sizeof(char));
}
#endif

/* adam */
void lib_adam_seq(const int size, const real g, real* grad, real* vecg, real* vec) {
  int i;
  real b1_t = vecg[2 * size];
  real b2 = (1.0 - ADAM_B2) * g * g;
  real bc1 = 1.0 / (1.0 - vecg[2 * size + 1]);
  real bc2 = 1.0 / (1.0 - vecg[2 * size + 2]);
  for (i = 0; i < size; ++i) {
    //update first moment estimate
    vecg[i] = b1_t * vecg[i] + (1 - b1_t) * g * grad[i];
    //update second moment estimate
    vecg[size + i] = ADAM_B2 * vecg[size + i] + b2 * grad[i] * grad[i];
    //update params
    vec[i] += ADAM_A * vecg[i] * bc1 * rsqrt_f(EPSILON + vecg[size + i] * bc2);
  }
  vecg[2 * size] *= ADAM_L;
  vecg[2 * size + 1] *= ADAM_B1;
  vecg[2 * size + 2] *= ADAM_B2;
}

#ifdef USE_AVX
/* adam using AVX; assumes grad is 32byte-aligned */
void lib_adam_avx(const int size, const real g, real* grad, real* vecg, real* vec) {
  const real b1_t = vecg[2 * size];
  const real b2 = (1.0 - ADAM_B2) * g * g;
  const real bc1 = 1.0 / (1.0 - vecg[2 * size + 1]);
  const real bc2 = 1.0 / (1.0 - vecg[2 * size + 2]);
  const vreal vepsilon = mm256_load(EPSILON);
  const vreal valpha = mm256_load(ADAM_A);
  const vreal vbeta2 = mm256_load(ADAM_B2);
  const vreal vb1_t = mm256_load(b1_t);
  const vreal v1mb1_t = mm256_load(1.0 - b1_t);
  const vreal vb2 = mm256_load(b2);
  const vreal vbc1 = mm256_load(bc1);
  const vreal vbc2 = mm256_load(bc2);
  const vreal vg = mm256_load(g);
  const vreal* vgrad = (vreal*)grad;
  int i;
  /* copy vectors to memory-aligned space (GCC style) */
  vreal vvec[MAX_DIMENSION/AVXVECSIZE]__attribute__((aligned(32)));
  vreal vm[MAX_DIMENSION/AVXVECSIZE]__attribute__((aligned(32)));
  vreal vv[MAX_DIMENSION/AVXVECSIZE]__attribute__((aligned(32)));
  const int avxloop = size / AVXVECSIZE;
  const int rest_idx = size - (size % AVXVECSIZE);
  memcpy(vvec, vec, avxloop * sizeof(vreal) / sizeof(char));
  memcpy(vm, vecg, avxloop * sizeof(vreal) / sizeof(char));
  memcpy(vv, &vecg[size], avxloop * sizeof(vreal) / sizeof(char));
  for (i = 0; i < avxloop; ++i) {
    //update first moment estimate
    vm[i] = vb1_t * vm[i] + v1mb1_t * vg * vgrad[i];
    //update second moment estimate
    vv[i] = vbeta2 * vv[i] + vb2 * vgrad[i] * vgrad[i];
    //update params
    vvec[i] = vvec[i] + valpha * vm[i] * vbc1 * mm256_rsqrt(vepsilon + vv[i] * vbc2);
  }
  /* write back */
  memcpy(vec, vvec, avxloop * sizeof(vreal) / sizeof(char));
  memcpy(vecg, vm, avxloop * sizeof(vreal) / sizeof(char));
  memcpy(&vecg[size], vv, avxloop * sizeof(vreal) / sizeof(char));
  for (i = rest_idx; i < size; ++i) {
    //update first moment estimate
    vecg[i] = b1_t * vecg[i] + (1 - b1_t) * g * grad[i];
    //update second moment estimate
    vecg[size + i] = ADAM_B2 * vecg[size + i] + b2 * grad[i] * grad[i];
    //update params
    vec[i] += ADAM_A * vecg[i] * bc1 * rsqrt_f(EPSILON + vecg[size + i] * bc2);
  }
  vecg[2 * size] *= ADAM_L;
  vecg[2 * size + 1] *= ADAM_B1;
  vecg[2 * size + 2] *= ADAM_B2;
}
#endif
