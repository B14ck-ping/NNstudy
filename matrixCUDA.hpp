#pragma once
#include <iostream>

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
    MatrixCUDA(const MatrixCUDA &);
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
    float* operator[] (unsigned int);
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
};
