#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"

using namespace std;

class MatrixCUDA
{
private:
    unsigned int rows;
    unsigned int columns;
    float *matrix;

public:
    explicit MatrixCUDA(unsigned int rows, unsigned int columns);
    explicit MatrixCUDA(unsigned int rows, unsigned int columns, float value);
    MatrixCUDA(MatrixCUDA &);
    MatrixCUDA(float *mtx_arr, unsigned int rows, unsigned int columns);
    ~MatrixCUDA();
    static MatrixCUDA dot(MatrixCUDA&, MatrixCUDA&);
    MatrixCUDA operator* (float);
    void operator*= (float);
    MatrixCUDA operator* (MatrixCUDA&);
    void operator*= (MatrixCUDA&);
    MatrixCUDA operator+ (MatrixCUDA&);
    void operator+= (MatrixCUDA&);
    MatrixCUDA operator- (MatrixCUDA&);
    void operator-= (MatrixCUDA&);
    MatrixCUDA operator= (MatrixCUDA&);
    // float* operator[] (unsigned int);
    float getDeterminant() const;
    unsigned int get_rows()
    {return rows;}
    unsigned int get_columns()
    {return columns;}
    MatrixCUDA getTranpose() const;
    void transpose();
    MatrixCUDA getTranspose();
    void insertLine();
    void insertColumn();

    void applySigmoid();
    float* getHost_matrix();
};
