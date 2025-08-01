#pragma once
#include <iostream>

#include "matrix.hpp"

using namespace std;

class MatrixCUDA : public Matrix
{
public:
    explicit MatrixCUDA(unsigned int rows, unsigned int columns);
    explicit MatrixCUDA(unsigned int rows, unsigned int columns, float value);
    MatrixCUDA(Matrix &);
    MatrixCUDA(const Matrix &);
    MatrixCUDA(Matrix &&);
    MatrixCUDA(float *mtx_arr, unsigned int rows, unsigned int columns);
    virtual ~MatrixCUDA();
    virtual Matrix* dot(const Matrix&, const Matrix&);
    virtual Matrix operator* (float);
    virtual void operator*= (float);
    virtual Matrix operator* (Matrix&);
    virtual void operator*= (Matrix&);
    virtual Matrix operator+ (Matrix&);
    virtual void operator+= (Matrix&);
    virtual Matrix operator- (Matrix&);
    virtual void operator-= (Matrix&);
    virtual Matrix operator= (Matrix&);
    virtual Matrix operator= (const Matrix&);
    virtual Matrix operator= (Matrix&&);
    virtual Matrix operator= (const Matrix&&);
    virtual float getDeterminant() const;
    virtual Matrix getTranspose() const;
    virtual void transpose();
    virtual void insertLine();
    virtual void insertColumn();

    virtual float* getDeePCopyOnDevice() const;
};
