#pragma once
#include <iostream>

using namespace std;

class Matrix
{
private:
    unsigned int rows;
    unsigned int columns;
    float *matrix;
public:
    explicit Matrix(unsigned int rows, unsigned int columns);
    explicit Matrix(unsigned int rows, unsigned int columns, float value);
    Matrix(const Matrix &);
    Matrix(float *mtx_arr, unsigned int rows, unsigned int columns);
    ~Matrix();
    static Matrix dot(const Matrix&, const Matrix&);
    Matrix operator* (float);
    void operator*= (float);
    Matrix operator* (Matrix&);
    void operator*= (Matrix&);
    Matrix operator+ (Matrix&);
    void operator+= (Matrix&);
    Matrix operator- (Matrix&);
    void operator-= (Matrix&);
    Matrix operator= (Matrix&);
    float* operator[] (unsigned int);
    const float* operator[] (unsigned int) const;
    float getDeterminant() const;
    unsigned int get_rows() const
    {return rows;}
    unsigned int get_columns() const
    {return columns;}
    Matrix getTranpose() const;
    void transpose();
    void insertLine();
    void insertColumn();
    float* getMatrixDeepCopyArr() const;
};

