#pragma once
#include <iostream>

using namespace std;

class Matrix
{
protected:
    unsigned int rows;
    unsigned int columns;
    float *matrix;
public:
    Matrix() = delete;
    explicit Matrix(unsigned int rows, unsigned int columns);
    explicit Matrix(unsigned int rows, unsigned int columns, float value);
    Matrix(const Matrix &);
    Matrix(float *mtx_arr, unsigned int rows, unsigned int columns);
    virtual ~Matrix();
    virtual Matrix dot(const Matrix&, const Matrix&) = 0;
    virtual Matrix operator* (float) = 0;
    virtual void operator*= (float) = 0;
    virtual Matrix operator* (Matrix&) = 0;
    virtual void operator*= (Matrix&) = 0;
    virtual Matrix operator+ (Matrix&) = 0;
    virtual void operator+= (Matrix&) = 0;
    virtual Matrix operator- (Matrix&) = 0;
    virtual void operator-= (Matrix&) = 0;
    virtual Matrix operator= (Matrix&) = 0;
    virtual float* operator[] (unsigned int) = 0;
    virtual const float* operator[] (unsigned int) const = 0;
    virtual float getDeterminant() const = 0;
    unsigned int get_rows() const
    {return rows;}
    unsigned int get_columns() const
    {return columns;}
    virtual Matrix getTranpose() const = 0;
    virtual void transpose() = 0;
    virtual void insertLine() = 0;
    virtual void insertColumn() = 0;
    virtual float* getMatrixDeepCopyArr() const = 0;
};

