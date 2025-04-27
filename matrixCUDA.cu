#include <stdio.h>
#include "matrixCUDA.h"
// #include "cuda.h"
// #include "cuda_runtime.h"


#define TILE_SIZE 32

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
	unsigned int ix=blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy=blockDim.y * blockIdx.y + threadIdx.y;
	if (ix < nx && iy < ny) {
        out[ix*ny + iy]=in[iy*nx + ix];
    }	
}


__global__  void dot_kernel(float* A, float* B, float* C, int M, int N, int K){
// __global__  void dot_kernel(float *in1, float *in2, float *out, unsigned int K, unsigned int M, unsigned int N){    

// 	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
// 	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;


// 	if (col < K && row < M) {
//         float result = 0.0f;
//         for (int i = 0; i < N; i++) {
//             result += in1[row*N + i] * in2[i*K + col];
//         }
//         __syncthreads();
//         out[row*K + col] = result;
//     }


    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y; // индекс строки в C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // индекс столбца в C

    float value = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Загрузка A
        if (row < M && (t * TILE_SIZE + threadIdx.x) < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Загрузка B (правильная!)
        if (col < K && (t * TILE_SIZE + threadIdx.y) < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Умножение плиток
        for (int i = 0; i < TILE_SIZE; ++i) {
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = value;
    }
}


MatrixCUDA::MatrixCUDA(unsigned int a_rows, unsigned int a_columns) : rows(a_rows), columns(a_columns)
{
    cudaMalloc( (void**)&matrix, rows*columns*sizeof(float) );
}

MatrixCUDA::MatrixCUDA(unsigned int a_rows, unsigned int a_columns, float value) : rows(a_rows), columns(a_columns)
{
    size_t size = rows*columns*sizeof(float);
    float *b_matrix = (float*)malloc(size);
    std::fill(b_matrix, b_matrix + (size / sizeof(float)), value);
    cudaMalloc( (void**)&matrix, size);
    cudaMemcpy( matrix, b_matrix, size, cudaMemcpyHostToDevice );
    free(b_matrix);
}

MatrixCUDA::MatrixCUDA(MatrixCUDA &mtx)
{
    rows = mtx.get_rows();
    columns = mtx.get_columns();
    size_t size = rows*columns*sizeof(float);
    cudaMalloc( (void**)&matrix, size);
    cudaMemcpy( matrix, mtx.matrix, size, cudaMemcpyDeviceToDevice ); 
}

MatrixCUDA::MatrixCUDA(const MatrixCUDA &mtx)
{
    rows = mtx.get_rows();
    columns = mtx.get_columns();
    size_t size = rows*columns*sizeof(float);
    cudaMalloc( (void**)&matrix, size);
    cudaMemcpy( matrix, mtx.matrix, size, cudaMemcpyDeviceToDevice ); 
}

MatrixCUDA::MatrixCUDA(MatrixCUDA&& move_mtx)
{
    rows = move_mtx.get_rows();
    columns = move_mtx.get_columns();
    size_t size = rows*columns*sizeof(float);
    cudaMalloc( (void**)&matrix, size);
    cudaMemcpy( matrix, move_mtx.matrix, size, cudaMemcpyDeviceToDevice ); 
    cudaFree(move_mtx.matrix);
    move_mtx.matrix = nullptr;
}

// MatrixCUDA::MatrixCUDA(const MatrixCUDA&& move_mtx)
// {
//     rows = move_mtx.get_rows();
//     columns = move_mtx.get_columns();
//     size_t size = rows*columns*sizeof(float);
//     cudaMalloc( (void**)&matrix, size);
//     cudaMemcpy( matrix, move_mtx.matrix, size, cudaMemcpyDeviceToDevice ); 
//     cudaFree(move_mtx.matrix);
//     move_mtx.matrix = nullptr;
// }

MatrixCUDA::MatrixCUDA(float *mtx_arr, unsigned int rows, unsigned int columns): rows(rows), columns(columns)
{
    size_t size = rows*columns*sizeof(float);
    cudaMalloc( (void**)&matrix, size);
    cudaMemcpy( matrix, mtx_arr, size, cudaMemcpyHostToDevice ); 
}


MatrixCUDA::~MatrixCUDA()
{
    cudaFree(matrix);
}

MatrixCUDA MatrixCUDA::dot(MatrixCUDA &mtx1, MatrixCUDA &mtx2)
{
    if (mtx1.get_columns() != mtx2.get_rows())
        throw std::invalid_argument("Matrix multiplication error: columns of first matrix != rows of second matrix");

    unsigned common_size = mtx1.get_columns();
    unsigned output_rows = mtx1.get_rows();
    unsigned int output_columns = mtx2.get_columns();
    MatrixCUDA l_matrix(output_rows, output_columns);
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 BlockDim((output_columns + TILE_SIZE - 1) / TILE_SIZE, (output_rows + TILE_SIZE - 1) / TILE_SIZE);
    dot_kernel<<<BlockDim, threadsPerBlock>>>(mtx1.matrix, mtx2.matrix, l_matrix.matrix, output_rows, common_size, output_columns);
    // dot_kernel<<<BlockDim, threadsPerBlock>>>(mtx1.matrix, mtx2.matrix, l_matrix.matrix, output_columns, output_rows, common_size);
    cudaDeviceSynchronize();

    return l_matrix;
}

MatrixCUDA MatrixCUDA::operator* (float scalar)
{
    MatrixCUDA out_mtx(*this);
    out_mtx *= scalar;
    return out_mtx;
}

void MatrixCUDA::operator*= (float scalar)
{
    // запускаем add() kernel на GPU, передавая параметры
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 BlockDim((columns + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    multiple_to_val<<<BlockDim, threadsPerBlock>>>(matrix, scalar, rows, columns);
}

MatrixCUDA MatrixCUDA::operator* (MatrixCUDA &m2)
{
    if (rows != m2.rows || columns != m2.columns) {
        std::cerr << "Matrix dimensions must match for addition!" << std::endl;
        return MatrixCUDA(0,0);
    }
    MatrixCUDA out_mtx(*this);
    out_mtx *= m2;
    return out_mtx;
}

void MatrixCUDA::operator*= (MatrixCUDA &m2)
{
    if (rows != m2.rows || columns != m2.columns) {
        std::cerr << "Matrix dimensions must match for addition!" << std::endl;
        return;
    }
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 BlockDim((columns + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    multiple<<<BlockDim, threadsPerBlock>>>(matrix, m2.matrix, rows, columns);
    cudaDeviceSynchronize();
}

MatrixCUDA MatrixCUDA::operator+ (MatrixCUDA &m2)
{
    if (rows != m2.rows || columns != m2.columns) {
        std::cerr << "Matrix dimensions must match for addition!" << std::endl;
        return MatrixCUDA(0,0);
    }
    MatrixCUDA out_mtx(*this);
    out_mtx += m2;
    return out_mtx;
}

void MatrixCUDA::operator+= (MatrixCUDA &m2)
{
    if (rows != m2.rows || columns != m2.columns) {
        std::cerr << "Matrix dimensions must match for addition!" << std::endl;
        return;
    }
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 BlockDim((columns + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    increase<<<BlockDim, threadsPerBlock>>>(matrix, m2.matrix, rows, columns);
    cudaDeviceSynchronize();
}

MatrixCUDA MatrixCUDA::operator- (MatrixCUDA& m2)
{
     if (rows != m2.rows || columns != m2.columns) {
        std::cerr << "Matrix dimensions must match for addition!" << std::endl;
        return MatrixCUDA(0,0);
    }
    MatrixCUDA out_mtx(*this);
    out_mtx -= m2;
    return out_mtx;
}

void MatrixCUDA::operator-= (MatrixCUDA &m2)
{
    if (rows != m2.rows || columns != m2.columns) {
        std::cerr << "Matrix dimensions must match for addition!" << std::endl;
        return;
    }
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 BlockDim((columns + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    decrease<<<BlockDim, threadsPerBlock>>>(matrix, m2.matrix, rows, columns);
    cudaDeviceSynchronize();
}

MatrixCUDA MatrixCUDA::operator= (MatrixCUDA &m2)
{
    this->rows = m2.rows;
    this->columns = m2.columns;

    size_t size = rows*columns*sizeof(float);
    cudaMalloc( (void**)&matrix, size);
    cudaMemcpy( matrix, m2.matrix, size, cudaMemcpyDeviceToDevice ); 

    return *this;
}

MatrixCUDA MatrixCUDA::operator= (const MatrixCUDA &m2) 
{
    this->rows = m2.get_rows();
    this->columns = m2.get_columns();

    size_t size = rows*columns*sizeof(float);
    cudaMalloc( (void**)&matrix, size);
    cudaMemcpy( matrix, m2.matrix, size, cudaMemcpyDeviceToDevice ); 

    return *this;
}

MatrixCUDA MatrixCUDA::operator= (MatrixCUDA&& move_mtx)
{
    this->rows = move_mtx.get_rows();
    this->columns = move_mtx.get_columns();

    size_t size = rows*columns*sizeof(float);
    cudaMalloc( (void**)&matrix, size);
    cudaMemcpy( matrix, move_mtx.matrix, size, cudaMemcpyDeviceToDevice ); 
    cudaFree(move_mtx.matrix);
    move_mtx.matrix = nullptr;
    return *this;
}

// MatrixCUDA MatrixCUDA::operator= (const MatrixCUDA&&)
// {
//     this->rows = move_mtx.get_rows();
//     this->columns = move_mtx.get_columns();

//     size_t size = rows*columns*sizeof(float);
//     cudaMalloc( (void**)&matrix, size);
//     cudaMemcpy( matrix, move_mtx.matrix, size, cudaMemcpyDeviceToDevice ); 
//     cudaFree(move_mtx.matrix);
//     move_mtx.matrix = nullptr;
//     return *this;
// }

float MatrixCUDA::getDeterminant() const
{
    return 0;
}

MatrixCUDA MatrixCUDA::getTranspose() const
{
    MatrixCUDA new_mtx(this->matrix, rows, columns);
    new_mtx.transpose();
    return new_mtx;
}

void MatrixCUDA::transpose()
{
    float *t_matrix;

    size_t size = rows*columns*sizeof(float);
    cudaMalloc( (void**)&t_matrix, size);
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 BlockDim((columns + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    transpose_kernel<<<BlockDim, threadsPerBlock>>>(matrix, t_matrix, columns, rows);
    cudaDeviceSynchronize();

    cudaFree(matrix);
    matrix = t_matrix;
    
    unsigned int temp = rows;
    rows = columns;
    columns = temp;
}

MatrixCUDA MatrixCUDA::getTranspose()
{
    MatrixCUDA outputMatrixCUDA(*this);
    outputMatrixCUDA.transpose();
    return outputMatrixCUDA;
}


void MatrixCUDA::applySigmoid()
{
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 BlockDim((columns + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    sygmoid<<<BlockDim, threadsPerBlock>>>(matrix, rows, columns);
}

float* MatrixCUDA::getHost_matrix()
{
    size_t size = rows*columns*sizeof(float);
    float *l_buf = (float*)malloc(size);
    cudaMemcpy( l_buf, matrix, size, cudaMemcpyDeviceToHost); 

    return l_buf;
}