#include <stdio.h>
#include "matrixCUDA.h"
// #include "cuda.h"
// #include "cuda_runtime.h"


__global__  void increase( float *a, const float *b, size_t size ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] += b[idx];
    }  
}

__global__  void add(const float *a, const float *b, float *c, size_t size ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }  
}


__global__  void decrease( float *a, const float *b, size_t size ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] -= b[idx];
    }  
}

__global__  void multiple_two_mtx(const float *a, const float *b, float *c, size_t size ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }  
}


__global__  void multiple( float *a, const float *b, size_t size ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] *= b[idx];
    }  
}

__global__  void multiple_to_val( float *a,  float val, size_t size ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] *= val;
    }  
}

__global__  void multiple_to_val_new( float *a,  float val, float *c, size_t size ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * val;
    }  
}

__global__  void subtract(const float *a, const float *b, float *c, size_t size ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] - b[idx];
    }  
}

__global__  void sigmoid(float *a, size_t size ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
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

__global__  void dot_kernel(float *in1, float *in2, float *out, unsigned int nx, unsigned int ny, unsigned int size){
	unsigned int ix=blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy=blockDim.y * blockIdx.y + threadIdx.y;
	if (ix < nx && iy < ny) {
        float result = 0;
        for (int i = 0; i < size; i++) {
            result += in1[iy*size + i] * in2[i*nx + ix];
        }

        out[iy*nx + ix] = result;
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
        return mtx1;

    unsigned common_size = mtx1.get_columns();
    unsigned output_rows = mtx1.get_rows();
    unsigned int output_columns = mtx2.get_columns();
    MatrixCUDA l_matrix(output_rows, output_columns);
    // MatrixCUDA mtx_2_t = mtx2.getTranspose();

    dim3 threadsPerBlock(output_columns > 16 ? 16 : output_columns, output_rows > 16 ? 16 : output_rows, 1);
    dim3 BlockDim(output_columns/threadsPerBlock.x, output_rows/threadsPerBlock.y, 1);
    dot_kernel<<<BlockDim, threadsPerBlock>>>(mtx1.matrix, mtx2.matrix, l_matrix.matrix, output_columns, output_rows, common_size);
    cudaDeviceSynchronize();

    return l_matrix;
}

MatrixCUDA MatrixCUDA::operator* (float scalar)
{
    MatrixCUDA out_mtx(rows,columns);
    // запускаем add() kernel на GPU, передавая параметры
    dim3 threadsPerBlock(columns > 16 ? 16 : columns, rows > 16 ? 16 : rows, 1);
    dim3 BlockDim(columns/threadsPerBlock.x, rows/threadsPerBlock.y, 1);
    multiple_to_val_new<<<BlockDim, threadsPerBlock>>>(matrix, scalar, out_mtx.matrix, rows * columns);
    return out_mtx;
}

void MatrixCUDA::operator*= (float scalar)
{
    // запускаем add() kernel на GPU, передавая параметры
    dim3 threadsPerBlock(columns > 16 ? 16 : columns, rows > 16 ? 16 : rows, 1);
    dim3 BlockDim(columns/threadsPerBlock.x, rows/threadsPerBlock.y, 1);
    multiple_to_val<<<BlockDim, threadsPerBlock>>>(matrix, scalar, rows * columns);
}

MatrixCUDA MatrixCUDA::operator* (MatrixCUDA &m2)
{
    if (rows != m2.rows || columns != m2.columns) {
        std::cerr << "Matrix dimensions must match for addition!" << std::endl;
        return MatrixCUDA(0,0);
    }
    MatrixCUDA out_mtx(rows,columns);
    // запускаем add() kernel на GPU, передавая параметры
    dim3 threadsPerBlock(columns > 16 ? 16 : columns, rows > 16 ? 16 : rows, 1);
    dim3 BlockDim(columns/threadsPerBlock.x, rows/threadsPerBlock.y, 1);
    multiple_two_mtx<<<BlockDim, threadsPerBlock>>>(matrix, m2.matrix, out_mtx.matrix, rows * columns);
    return out_mtx;
}

void MatrixCUDA::operator*= (MatrixCUDA &m2)
{
    if (rows != m2.rows || columns != m2.columns) {
        std::cerr << "Matrix dimensions must match for addition!" << std::endl;
        return;
    }
    // запускаем add() kernel на GPU, передавая параметры
    dim3 threadsPerBlock(columns > 16 ? 16 : columns, rows > 16 ? 16 : rows, 1);
    dim3 BlockDim(columns/threadsPerBlock.x, rows/threadsPerBlock.y, 1);
    multiple<<<BlockDim, threadsPerBlock>>>(matrix, m2.matrix, rows * columns);
}

MatrixCUDA MatrixCUDA::operator+ (MatrixCUDA &m2)
{
    if (rows != m2.rows || columns != m2.columns) {
        std::cerr << "Matrix dimensions must match for addition!" << std::endl;
        return MatrixCUDA(0,0);
    }
    MatrixCUDA out_mtx(rows,columns);
    // запускаем add() kernel на GPU, передавая параметры
    dim3 threadsPerBlock(columns > 16 ? 16 : columns, rows > 16 ? 16 : rows, 1);
    dim3 BlockDim(columns/threadsPerBlock.x, rows/threadsPerBlock.y, 1);
    add<<<BlockDim, threadsPerBlock>>>(matrix, m2.matrix, out_mtx.matrix, rows * columns);
    return out_mtx;
}

void MatrixCUDA::operator+= (MatrixCUDA &m2)
{
    if (rows != m2.rows || columns != m2.columns) {
        std::cerr << "Matrix dimensions must match for addition!" << std::endl;
        return;
    }
    // запускаем increase() kernel на GPU, передавая параметры
    dim3 threadsPerBlock(columns > 16 ? 16 : columns, rows > 16 ? 16 : rows, 1);
    dim3 BlockDim(columns/threadsPerBlock.x, rows/threadsPerBlock.y, 1);
    increase<<<BlockDim, threadsPerBlock>>>(matrix, m2.matrix, rows * columns);
}

MatrixCUDA MatrixCUDA::operator- (MatrixCUDA& m2)
{
     if (rows != m2.rows || columns != m2.columns) {
        std::cerr << "Matrix dimensions must match for addition!" << std::endl;
        return MatrixCUDA(0,0);
    }
    MatrixCUDA out_mtx(rows,columns);
    // запускаем add() kernel на GPU, передавая параметры
    dim3 threadsPerBlock(columns > 16 ? 16 : columns, rows > 16 ? 16 : rows, 1);
    dim3 BlockDim(columns/threadsPerBlock.x, rows/threadsPerBlock.y, 1);
    subtract<<<BlockDim, threadsPerBlock>>>(matrix, m2.matrix, out_mtx.matrix, rows * columns);
    return out_mtx;
}

void MatrixCUDA::operator-= (MatrixCUDA &m2)
{
    if (rows != m2.rows || columns != m2.columns) {
        std::cerr << "Matrix dimensions must match for addition!" << std::endl;
        return;
    }
    // запускаем increase() kernel на GPU, передавая параметры
    dim3 threadsPerBlock(columns > 16 ? 16 : columns, rows > 16 ? 16 : rows, 1);
    dim3 BlockDim(columns/threadsPerBlock.x, rows/threadsPerBlock.y, 1);
    decrease<<<BlockDim, threadsPerBlock>>>(matrix, m2.matrix, rows * columns);
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

float MatrixCUDA::getDeterminant() const
{
    return 0;
}

MatrixCUDA MatrixCUDA::getTranpose() const
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
    dim3 threadsPerBlock(columns > 16 ? columns/16 : columns, rows > 16 ? rows/16 : rows, 1);
    dim3 BlockDim(columns/threadsPerBlock.x, rows/threadsPerBlock.y, 1);
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
    sigmoid<<<((rows * columns) + 255) / 256, 256>>>(matrix, rows * columns);
}

float* MatrixCUDA::getHost_matrix()
{
    size_t size = rows*columns*sizeof(float);
    float *l_buf = (float*)malloc(size);
    cudaMemcpy( l_buf, matrix, size, cudaMemcpyDeviceToHost); 

    return l_buf;
}