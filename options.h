#ifndef __OPTIONS_H
#define __OPTIONS_H

/* macro */
#ifndef USE_DOUBLE
typedef float real;
#else
typedef double real;
#endif //USE_DOUBLE

extern const real ONEF;
extern const int ONE;
extern const real ZEROF;
extern const char TRANS_N;
typedef void (*copy_ptr) (const int*, const real*, const int*, real*, const int*);
typedef void (*axpy_ptr) (const int*, const real*, const real*, const int*, real*, const int*);
typedef real (*dot_ptr) (const int*, const real*, const int*, const real*, const int*);
typedef double (*ddot_ptr) (const int*, const real*, const int*, real*, const int*);
typedef void (*scal_ptr) (const int*, const real*, real*, const int*);
typedef void (*gemv_ptr) (const char*, const int*, const int*, const real*, const real*, const int*, const real*, const int*, const real*, real*, const int*);
extern copy_ptr blas_copy;
extern axpy_ptr blas_axpy;
extern dot_ptr blas_dot;
extern ddot_ptr blas_ddot;
extern scal_ptr blas_scal;
extern gemv_ptr blas_gemv;

#ifdef USE_BLAS
#include <cblas.h>
#define USING_BLAS 1
#ifndef USE_DOUBLE
#define lib_copy(N, x, dx, y, dy) cblas_scopy(N, x, dx, y, dy)
#define lib_axpy(N, alpha, X, incX, Y, incY) cblas_saxpy(N, alpha, X, incX, Y, incY)
#define lib_dot(N, x, dx, y, dy) cblas_sdot(N, x, dx, y, dy)
#define lib_ddot(N, x, dx, y, dy) cblas_dsdot(N, x, dx, y, dy)
#define lib_scal(N, alpha, x, dx) cblas_sscal(N, alpha, x, dx)
#define lib_gemv(M, N, alpha, A, ldA, x, dx, beta, y, dy) cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, A, ldA, x, dx, beta, y, dy)
#else
#define lib_copy(N, x, dx, y, dy) cblas_dcopy(N, x, dx, y, dy)
#define lib_axpy(N, alpha, X, incX, Y, incY) cblas_daxpy(N, alpha, X, incX, Y, incY)
#define lib_dot(N, x, dx, y, dy) cblas_ddot(N, x, dx, y, dy)
#define lib_ddot(N, x, dx, y, dy) cblas_ddot(N, x, dx, y, dy)
#define lib_scal(N, alpha, x, dx) cblas_dscal(N, alpha, x, dx)
#define lib_gemv(M, N, alpha, A, ldA, x, dx, beta, y, dy) cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, A, ldA, x, dx, beta, y, dy)
#endif //USE_DOUBLE

#else
#define USING_BLAS 0
#define lib_copy(N, x, dx, y, dy) blas_copy(&(N), x, &(dx), y, &(dy))
#define lib_axpy(N, alpha, X, incX, Y, incY) blas_axpy(&(N), &(alpha), X, &(incX), Y, &(incY))
#define lib_dot(N, x, dx, y, dy) blas_dot(&(N), x, &(dx), y, &(dy))
#define lib_ddot(N, x, dx, y, dy) blas_ddot(&(N), x, &(dx), y, &(dy))
#define lib_scal(N, alpha, x, dx) blas_scal(&(N), &(alpha), x, &(dx))
#define lib_gemv(M, N, alpha, A, ldA, x, dx, beta, y, dy) blas_gemv(&TRANS_N, &M, &N, &alpha, A, &ldA, x, &dx, &beta, y, &dy)
#endif //USE_BLAS

#ifdef FAST_SQRT
#ifndef USE_DOUBLE
#define rsqrt_f(x) t_srsqrt(x)
#define sqrt_f(x) x*t_srsqrt(x)
#else
#define rsqrt_f(x) t_drsqrt(x)
#define sqrt_f(x) x*t_drsqrt(x)
#endif //USE_DOUBLE
static inline float t_srsqrt(const float x) {
  float x_half = 0.5f * x;
  int tmp = 0x5F3759DF - (*(int*)&x >> 1); //initial guess
  float x_res = *(float*)&tmp;
  x_res *= (1.5f - (x_half * x_res * x_res));
  x_res *= (1.5f - (x_half * x_res * x_res));
  return x_res;
}
static inline double t_drsqrt(const double x) {
  double x_half = 0.5 * x;
  long long int tmp = 0x5FE6EB50C7B537AAl - (*(long long int*)&x >> 1); //initial guess
  double x_res = *(double*)&tmp;
  x_res *= (1.5 - (x_half * x_res * x_res));
  x_res *= (1.5 - (x_half * x_res * x_res));
  return x_res;
}
#else
#include <math.h>
#ifndef USE_DOUBLE
#define rsqrt_f(x) (1.0f/sqrt(x))
#define sqrt_f(x) sqrt(x)
#else
#define rsqrt_f(x) (1.0/sqrt(x))
#define sqrt_f(x) sqrt(x)
#endif //USE_DOUBLE
#endif //FAST_SQRT

void lib_xsq_seq(const int size, real* x, real* y);
void lib_adagrad_seq(const int size, const real alpha, const real g, real* grad, real* vecg, real* vec);
void lib_adadelta_seq(const int size, const real g, real* grad, real* vecg, real* vec);
void lib_adam_seq(const int size, const real g, real* grad, real* vecg, real* vec);

#ifdef USE_AVX
#include <xmmintrin.h>
#include <immintrin.h>
#include <stdint.h>
#include <assert.h>
void lib_xsq_avx(const int size, real* x, real* y);
void lib_sgd_avx(const int size, const real alpha, real* grad, real* vec);
void lib_adagrad_avx(const int size, const real alpha, const real g, real* grad, real* vecg, real* vec);
void lib_adagrad_avx_nostack(const int size, const real alpha, const real g, real* grad, real* vecg, real* vec);
void lib_adadelta_avx(const int size, const real g, real* grad, real* vecg, real* vec);
void lib_adam_avx(const int size, const real g, real* grad, real* vecg, real* vec);
#ifndef USE_DOUBLE
#define AVXVECSIZE 8
#define mm256_load(x) _mm256_set_ps(x, x, x, x, x, x, x, x)
#define mm256_add(x,y) _mm256_add_ps(x, y)
#define mm256_sub(x,y) _mm256_sub_ps(x, y)
#define mm256_mul(x,y) _mm256_mul_ps(x, y)
#define mm256_sq(x) _mm256_mul_ps(x, x)
#define mm256_rsqrt(x) _mm256_rsqrt_ps(x)
#define mm256_sqrt(x) _mm256_sqrt_ps(x)
typedef __m256 vreal;
#else
#define AVXVECSIZE 4
#define mm256_load(x) _mm256_set_pd(x, x, x, x)
#define mm256_add(x,y) _mm256_add_pd(x, y)
#define mm256_sub(x,y) _mm256_sub_pd(x, y)
#define mm256_mul(x,y) _mm256_mul_pd(x, y)
#define mm256_sq(x) _mm256_mul_pd(x, x)
#define mm256_rsqrt(x) _mm256_div_pd(_mm256_set_pd(1.0,1.0,1.0,1.0),_mm256_sqrt_pd(x))
#define mm256_sqrt(x) _mm256_sqrt_pd(x)
typedef __m256d vreal;
#endif //USE DOUBLE
#define lib_aligned_malloc(size) reinterpret_cast<real*>(_mm_malloc(sizeof(real) * (size + AVXVECSIZE), 32));
#define lib_free(ptr) _mm_free(ptr)
#define lib_xsq(size,x,y) lib_xsq_avx(size, x, y)
#define lib_sgd(size,alpha,grad,vec) lib_sgd_avx(size, alpha, grad, vec)
#define lib_adagrad(size,alpha,g,grad,vecg,vec) lib_adagrad_avx(size, alpha, g, grad, vecg, vec)
/* if you want to use less stack memory, use the following instead */
//#define lib_adagrad(size,alpha,g,_grad,vecg,vec) lib_adagrad_avx_nostack(size, alpha, g, _grad, vecg, vec)
#define lib_adadelta(size,g,grad,vecg,vec) lib_adadelta_avx(size, g, grad, vecg, vec)
#define lib_adam(size,g,grad,vecg,vec) lib_adam_avx(size, g, grad, vecg, vec)
#else
#define lib_aligned_malloc(size) reinterpret_cast<real*>(malloc(sizeof(real) * size))
#define lib_free(ptr) free(ptr)
#define lib_xsq(size,x,y) lib_xsq_seq(size, x, y)
#define lib_sgd(size,alpha,grad,vec) lib_axpy(size, alpha, grad, ONE, vec, ONE)
#define lib_adagrad(size,alpha,g,grad,vecg,vec) lib_adagrad_seq(size, alpha, g, grad, vecg, vec)
#define lib_adadelta(size,g,grad,vecg,vec) lib_adadelta_seq(size, g, grad, vecg, vec)
#define lib_adam(size,g,grad,vecg,vec) lib_adam_seq(size, g, grad, vecg, vec)
#endif //USE_AVX

#endif //OPTIONS_H
