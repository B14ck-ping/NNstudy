#include <iostream>

using namespace std;

class Matrix
{
private:
    unsigned int rows;
    unsigned int columns;
    double **matrix;
public:
    explicit Matrix(unsigned int rows, unsigned int columns);
    Matrix(const Matrix &);
    ~Matrix();
    Matrix operator* (double);
    void operator*= (double);
    Matrix operator* (Matrix&);
    Matrix operator+ (Matrix);
    void operator+= (Matrix);
    Matrix operator- (Matrix);
    void operator-= (Matrix);
    Matrix& operator= (Matrix&);
    Matrix operator= (Matrix);
    double* operator[] (unsigned int);
    double getDeterminant() const;
    unsigned int get_rows()
    {return rows;}
    unsigned int get_columns()
    {return columns;}
};

