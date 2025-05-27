#pragma once

// Определения должны приходить из компилятора через -DUSE_CUDA или -DUSE_OPENCL и т.п.

#if defined(USE_CUDA)
    #include "cuda/MatrixKernels.cuh"
    #include "cuda/VectorKernels.cuh"
    #include "cuda/TensorKernels.cuh"
#elif defined(USE_OPENCL)
    #include "opencl/MatrixKernels.hpp"
    // аналогично: VectorKernels.hpp, TensorKernels.hpp
#else
    #error "No parallel backend selected. Define one of USE_CUDA or USE_OPENCL."
#endif

// Общий API (если нужно обобщить вызовы вручную)
// Например, можно определить универсальные inline-функции, вызывающие конкретную реализацию:
namespace parallel {

inline void matrix_multiply(const float* A, const float* B, float* C, int M, int N, int K) {
#if defined(USE_CUDA)
    cuda_matrix_multiply(A, B, C, M, N, K);
#elif defined(USE_OPENCL)
    opencl_matrix_multiply(A, B, C, M, N, K);
#endif
}

// Аналогично vector_add(), tensor_convolve() и т.д.

}
