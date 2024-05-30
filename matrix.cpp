#include "matrix.hpp"

Matrix::Matrix(unsigned int a_rows, unsigned int a_columns) : rows(a_rows), columns(a_columns)
{
    matrix = new double*[rows];

    for (unsigned int i = 0; i < rows; i++){
        matrix[i] = new double[columns];
    }
}

Matrix::Matrix(const Matrix &mtx) : rows(mtx.rows), columns(mtx.columns)
{
    matrix = new double*[rows];

    for (unsigned int i = 0; i < rows; i++){
        matrix[i] = new double[columns];
    }

    for (unsigned int i = 0; i < rows; i++){
        for(unsigned int j = 0; j < columns; j++){
            matrix[i][j] = mtx.matrix[i][j];
        }
    }
}

Matrix::~Matrix()
{
    for (unsigned int i = 0; i < rows; i++){
        delete [] matrix[i];
    }
    delete [] matrix;
}

Matrix Matrix::operator* (double scalar)
{
    Matrix temp(*this);

    for (unsigned int i = 0; i < temp.rows; i++){
        for(unsigned int j = 0; j < temp.columns; j++){
            temp.matrix[i][j] *= scalar;
        }
    }

    return temp;
}

void Matrix::operator*= (double scalar)
{
    for (unsigned int i = 0; i < rows; i++){
        for(unsigned int j = 0; j < columns; j++){
            matrix[i][j] *= scalar;
        }
    }
}

Matrix Matrix::operator* (Matrix& mtrx)
{

    if (mtrx.get_rows() != columns)
        return *this;

    Matrix l_matrix(rows, mtrx.get_columns());

    unsigned int l_output_columns = mtrx.get_columns();

    for (unsigned int k = 0; k < l_output_columns; k++){
        for (unsigned int i = 0; i < rows; i++){
            double l_temp = 0.0;
            for(unsigned int j = 0; j < columns; j++){
                l_temp += matrix[i][j] * mtrx[j][k];
            }
            l_matrix.matrix[i][k] = l_temp;
        }
    }
    
    return l_matrix;
}

Matrix Matrix::operator+ (Matrix mtrx)
{

}

void Matrix::operator+= (Matrix m2)
{
    for (unsigned int i = 0; i < rows; i++){
        for(unsigned int j = 0; j < columns; j++){
            matrix[i][j] += m2.matrix[i][j];
        }
    }
}

Matrix Matrix::operator- (Matrix)
{

}

void Matrix::operator-= (Matrix m2)
{
    for (unsigned int i = 0; i < rows; i++){
        for(unsigned int j = 0; j < columns; j++){
            matrix[i][j] -= m2.matrix[i][j];
        }
    }
}

Matrix& Matrix::operator= (Matrix &m2)
{
    this->rows = m2.rows;
    this->columns = m2.columns;

    this->matrix = new double*[this->rows];

    for (unsigned int i = 0; i < this->rows; i++){
        this->matrix[i] = new double[this->columns];
    }

    for (unsigned int i = 0; i < this->rows; i++){
        for(unsigned int j = 0; j < this->columns; j++){
            this->matrix[i][j] = m2.matrix[i][j];
        }
    }

    return *this;
}

Matrix Matrix::operator= (Matrix m2)
{
    this->rows = m2.rows;
    this->columns = m2.columns;

    this->matrix = new double*[this->rows];

    for (unsigned int i = 0; i < this->rows; i++){
        this->matrix[i] = new double[this->columns];
    }

    for (unsigned int i = 0; i < this->rows; i++){
        for(unsigned int j = 0; j < this->columns; j++){
            this->matrix[i][j] = m2.matrix[i][j];
        }
    }

    return *this;
}

double* Matrix::operator[] (unsigned int idx)
{
    return matrix[idx];
}

double Matrix::getDeterminant() const
{

}
