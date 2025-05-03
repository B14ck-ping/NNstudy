#include "kernels.cuh"
#include <iostream>


__global__  void increase( float *a, const float *b, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    size_t idx = cols*row + col;
    if (row < rows && col < cols) {
        a[idx] += b[idx];
    }  
}

__global__  void add(const float *a, const float *b, float *c, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    size_t idx = cols*row + col;
    if (row < rows && col < cols) {
        c[idx] = a[idx] + b[idx];
    }  
}


__global__  void decrease( float *a, const float *b, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    size_t idx = cols*row + col;
    if (row < rows && col < cols) {
        a[idx] -= b[idx];
    }  
}

__global__  void inversion(float* A, int N, int M)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    size_t idx = N*row + col;
    if (row < M && col < N) {
        A[idx] = 1.0 - A[idx];
    }  
}

__global__  void multiple_two_mtx(const float *a, const float *b, float *c, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    size_t idx = cols*row + col;
    if (row < rows && col < cols) {
        c[idx] = a[idx] * b[idx];
    }  
}


__global__  void multiple( float *a, const float *b, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    size_t idx = cols*row + col;
    if (row < rows && col < cols) {
        a[idx] *= b[idx];
    }  
}

__global__  void multiple_to_val( float *a,  float val, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    size_t idx = cols*row + col;
    if (row < rows && col < cols) {
        a[idx] *= val;
    }  
}

__global__  void multiple_to_val_new( float *a,  float val, float *c, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    size_t idx = cols*row + col;
    if (row < rows && col < cols) {
        c[idx] = a[idx] * val;
    }  
}

__global__  void subtract(const float *a, const float *b, float *c, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    size_t idx = cols*row + col;
    if (row < rows && col < cols) {
        c[idx] = a[idx] - b[idx];
    }  
}

__global__  void sygmoid(float *a, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    size_t idx = cols*row + col;
    if (row < rows && col < cols) {
        a[idx] = 1.0/(1.0 + expf(-a[idx]));
    }  
}

__global__  void transpose_kernel(float *in, float *out, unsigned int nx, unsigned int ny){
    __shared__ float tile[TILE_SIZE][TILE_SIZE+1];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < nx && y < ny) {
        tile[threadIdx.y][threadIdx.x] = in[y * nx + x];
    }

    __syncthreads();

    // Координаты для записи в output (транспонированные)
    int x_out = blockIdx.y * TILE_SIZE + threadIdx.x;  // колонка в транспонированной матрице
    int y_out = blockIdx.x * TILE_SIZE + threadIdx.y;  // строка в транспонированной матрице

    if (x_out < ny && y_out < nx) {
        out[y_out * ny + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}


__global__  void dot_kernel(float* A, float* B, float* C, int M, int N, int K){
    __shared__ float tileA[TILE_SIZE][TILE_SIZE+1];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE+1];

    int row = blockIdx.y * blockDim.y + threadIdx.y; // индекс строки в C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // индекс столбца в C

    float value = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        int tiledRow = t * TILE_SIZE + threadIdx.y;

        // Загружаем A с coalesced доступом
        if (row < M && tiledCol < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + tiledCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Загружаем B с coalesced доступом 
        if (tiledRow < N && col < K)
            tileB[threadIdx.y][threadIdx.x] = B[tiledRow * K + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Умножение плиток
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = value;
    }
}

__global__  void fill_mtx(float* A, int M, int N, float value)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // индекс строки
    int col = blockIdx.x * blockDim.x + threadIdx.x; // индекс столбца

    if (row < M && col < N) {
        A[row * N + col] = value;
    }
}