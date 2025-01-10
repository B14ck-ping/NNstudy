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
    static Matrix dot(Matrix&, Matrix&);
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
    float getDeterminant() const;
    unsigned int get_rows()
    {return rows;}
    unsigned int get_columns()
    {return columns;}
    Matrix getTranpose() const;
    void transpose();
    Matrix getTranspose();
    void insertLine();
    void insertColumn();
    float* getMatrixDeepCopyArr();
};

