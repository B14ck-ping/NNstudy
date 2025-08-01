#pragma once
#include <iostream>
#include "matrix.hpp"

using namespace std;

class MatrixCPU : public Matrix
{
public:
    explicit MatrixCPU(unsigned int rows, unsigned int columns);
    explicit MatrixCPU(unsigned int rows, unsigned int columns, float value);
    MatrixCPU(Matrix &);
    MatrixCPU(const Matrix &);
    MatrixCPU(Matrix &&);
    MatrixCPU(const Matrix &&);
    MatrixCPU(float *mtx_arr, unsigned int rows, unsigned int columns);
    virtual ~MatrixCPU();
    virtual Matrix* dot(const Matrix&, const Matrix&);
    virtual Matrix* operator* (float);
    virtual void operator*= (float);
    virtual Matrix* operator* (Matrix&);
    virtual void operator*= (Matrix&);
    virtual Matrix* operator+ (Matrix&);
    virtual void operator+= (Matrix&);
    virtual Matrix* operator- (Matrix&);
    virtual void operator-= (Matrix&);
    virtual Matrix* operator= (Matrix&);
    virtual Matrix* operator= (const Matrix&);
    virtual Matrix* operator= (Matrix&&);
    virtual float getDeterminant() const;
    virtual Matrix* getTranspose() const;
    virtual void transpose();
    virtual void insertLine();
    virtual void insertColumn();

    virtual float* getDeePCopyOnDevice() const;

private:
    float* operator[] (unsigned int);
    const float* operator[] (unsigned int) const;
};

