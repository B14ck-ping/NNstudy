#include <stdio.h>
#include "matrixCUDA.h"
#include "kernels.cuh"
#include "cuda_runtime.h"
#include "cuda.h"


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
    if (this->matrix != nullptr)
        cudaFree(matrix);

    matrix = move_mtx.matrix;
    move_mtx.rows = 0;
    move_mtx.columns = 0;
    move_mtx.matrix = nullptr;
}

MatrixCUDA::MatrixCUDA(float *mtx_arr, unsigned int rows, unsigned int columns): rows(rows), columns(columns)
{
    size_t size = rows*columns*sizeof(float);
    cudaMalloc( (void**)&matrix, size);
    cudaMemcpy( matrix, mtx_arr, size, cudaMemcpyHostToDevice ); 
}


MatrixCUDA::~MatrixCUDA()
{
    if(matrix != nullptr)
        cudaFree(matrix);
}

MatrixCUDA MatrixCUDA::dot(const MatrixCUDA &mtx1, const MatrixCUDA &mtx2)
{
    if (mtx1.get_columns() != mtx2.get_rows())
        throw std::invalid_argument("Matrix multiplication error: columns of first matrix != rows of second matrix");

    unsigned common_size = mtx1.get_columns();
    unsigned output_rows = mtx1.get_rows();
    unsigned int output_columns = mtx2.get_columns();
    MatrixCUDA l_matrix(output_rows, output_columns);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 BlockDim((output_columns + TILE_SIZE - 1) / TILE_SIZE, (output_rows + TILE_SIZE - 1) / TILE_SIZE);
    dot_kernel<<<BlockDim, threadsPerBlock>>>(mtx1.matrix,mtx2.matrix, l_matrix.matrix, output_rows, common_size, output_columns);
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
    cudaDeviceSynchronize();
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

    if (this->matrix != nullptr)
        cudaFree(matrix);

    size_t size = rows*columns*sizeof(float);
    cudaMalloc( (void**)&matrix, size);
    cudaMemcpy( matrix, m2.matrix, size, cudaMemcpyDeviceToDevice ); 

    return *this;
}

MatrixCUDA MatrixCUDA::operator= (const MatrixCUDA &m2) 
{
    this->rows = m2.get_rows();
    this->columns = m2.get_columns();

    if (this->matrix != nullptr)
        cudaFree(matrix);

    size_t size = rows*columns*sizeof(float);
    cudaMalloc( (void**)&matrix, size);
    cudaMemcpy( matrix, m2.matrix, size, cudaMemcpyDeviceToDevice ); 

    return *this;
}

MatrixCUDA MatrixCUDA::operator= (MatrixCUDA&& move_mtx)
{
    this->rows = move_mtx.get_rows();
    this->columns = move_mtx.get_columns();

    if (this->matrix != nullptr)
        cudaFree(matrix);

    this->matrix = move_mtx.matrix;
    move_mtx.rows = 0;
    move_mtx.columns = 0;
    move_mtx.matrix = nullptr;
    return *this;
}

float MatrixCUDA::getDeterminant() const
{
    return 0;
}

MatrixCUDA MatrixCUDA::getTranspose() const
{
    MatrixCUDA new_mtx(*this);
    new_mtx.transpose();
    return new_mtx;
}

void MatrixCUDA::transpose()
{
    float *t_matrix;

    size_t size = rows*columns*sizeof(float);
    cudaMalloc( (void**)&t_matrix, size);
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((columns + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    transpose_kernel<<<gridSize, blockSize>>>(matrix, t_matrix, columns, rows);
    cudaDeviceSynchronize();

    cudaFree(matrix);
    matrix = t_matrix;
    
    unsigned int temp = rows;
    rows = columns;
    columns = temp;
}


void MatrixCUDA::applySigmoid()
{
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 BlockDim((columns + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    sygmoid<<<BlockDim, threadsPerBlock>>>(matrix, rows, columns);
    cudaDeviceSynchronize();
}

float* MatrixCUDA::getHost_matrix() const
{
    size_t size = rows*columns*sizeof(float);
    float *l_buf = (float*)malloc(size);
    cudaMemcpy( l_buf, matrix, size, cudaMemcpyDeviceToHost); 

    return l_buf;
}

float* MatrixCUDA::getDeePCopyOnDevice() const
{
    float *l_buf = nullptr;
    cudaMalloc( (void**)&l_buf, rows*columns*sizeof(float) );
    cudaMemcpy( l_buf, matrix, rows*columns*sizeof(float), cudaMemcpyDeviceToDevice ); 
    return l_buf;
}