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
    MatrixCUDA(MatrixCUDA &);
    MatrixCUDA(const MatrixCUDA &);
    MatrixCUDA(MatrixCUDA &&);
    MatrixCUDA(const MatrixCUDA &&);
    MatrixCUDA(float *mtx_arr, unsigned int rows, unsigned int columns);
    ~MatrixCUDA();
    static MatrixCUDA dot(const MatrixCUDA&, const MatrixCUDA&);
    MatrixCUDA operator* (float);
    void operator*= (float);
    MatrixCUDA operator* (MatrixCUDA&);
    void operator*= (MatrixCUDA&);
    MatrixCUDA operator+ (MatrixCUDA&);
    void operator+= (MatrixCUDA&);
    MatrixCUDA operator- (MatrixCUDA&);
    void operator-= (MatrixCUDA&);
    MatrixCUDA operator= (MatrixCUDA&);
    MatrixCUDA operator= (const MatrixCUDA&);
    MatrixCUDA operator= (MatrixCUDA&&);
    MatrixCUDA operator= (const MatrixCUDA&&);
    float getDeterminant() const;
    unsigned int get_rows() const
    {return rows;}
    unsigned int get_columns() const
    {return columns;}
    MatrixCUDA getTranspose() const;
    void transpose();
    void insertLine();
    void insertColumn();

    void applySigmoid();
    float* getHost_matrix() const;

    float* getDeePCopyOnDevice() const;
};
