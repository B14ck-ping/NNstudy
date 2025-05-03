#pragma once

#include "cuda_runtime.h"
#include "cuda.h"

#define TILE_SIZE 32

__global__  void increase( float *a, const float *b, size_t rows, size_t cols);

__global__  void add(const float *a, const float *b, float *c, size_t rows, size_t cols);

__global__  void decrease( float *a, const float *b, size_t rows, size_t cols);

__global__  void multiple_two_mtx(const float *a, const float *b, float *c, size_t rows, size_t cols);

__global__  void multiple( float *a, const float *b, size_t rows, size_t cols);

__global__  void multiple_to_val( float *a,  float val, size_t rows, size_t cols);

__global__  void multiple_to_val_new( float *a,  float val, float *c, size_t rows, size_t cols);

__global__  void subtract(const float *a, const float *b, float *c, size_t rows, size_t cols);

__global__  void sygmoid(float *a, size_t rows, size_t cols);

__global__  void transpose_kernel(float *in, float *out, unsigned int nx, unsigned int ny);

__global__  void dot_kernel(float* A, float* B, float* C, int M, int N, int K);

__global__  void fill_mtx(float* A, int M, int N, float value);

__global__  void inversion(float* A, int N, int M);