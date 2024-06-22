#include "matrix.hpp"

Matrix::Matrix(unsigned int a_rows, unsigned int a_columns) : rows(a_rows), columns(a_columns)
{
    this->matrix = new double*[this->rows];

    for (unsigned int i = 0; i < this->rows; i++){
        this->matrix[i] = new double[this->columns];
    }
}

Matrix::Matrix(unsigned int a_rows, unsigned int a_columns, double value) : rows(a_rows), columns(a_columns)
{
    matrix = new double*[rows];

    for (unsigned int i = 0; i < rows; i++){
        matrix[i] = new double[columns];
    }

    for (unsigned int i = 0; i < rows; i++){
        for(unsigned int j = 0; j < columns; j++){
            matrix[i][j] = value;
        }
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

Matrix Matrix::dot(Matrix &mtx1, Matrix &mtx2)
{
    if (mtx1.get_columns() != mtx2.get_rows())
        return mtx1;

    Matrix l_matrix(mtx1.get_rows(), mtx2.get_columns());

    unsigned int l_output_columns = mtx2.get_columns();

    for (unsigned int k = 0; k < l_output_columns; k++){
        for (unsigned int i = 0; i < mtx1.get_rows(); i++){
            double l_temp = 0.0;
            for(unsigned int j = 0; j < mtx1.get_columns(); j++){
                l_temp += mtx1[i][j] * mtx2[j][k];
            }
            l_matrix.matrix[i][k] = l_temp;
        }
    }
    
    return l_matrix;
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

Matrix Matrix::operator* (Matrix &mtrx)
{
    Matrix out_matrix(mtrx.get_rows(), mtrx.get_columns());

    for (unsigned int i = 0; i < rows; i++){
        for(unsigned int j = 0; j < columns; j++){

            matrix[i][j] *= mtrx.matrix[i][j];
        }
    }

    return out_matrix;      
}

void Matrix::operator*= (Matrix &mtrx)
{
    Matrix out_matrix(mtrx.get_rows(), mtrx.get_columns());

    for (unsigned int i = 0; i < rows; i++){
        for(unsigned int j = 0; j < columns; j++){
            matrix[i][j] *= mtrx.matrix[i][j];
        }
    }  
}

Matrix Matrix::operator+ (Matrix &mtrx)
{

}

void Matrix::operator+= (Matrix &m2)
{
    for (unsigned int i = 0; i < rows; i++){
        for(unsigned int j = 0; j < columns; j++){
            matrix[i][j] += m2.matrix[i][j];
        }
    }
}

Matrix Matrix::operator- (Matrix&)
{

}

void Matrix::operator-= (Matrix &m2)
{
    for (unsigned int i = 0; i < rows; i++){
        for(unsigned int j = 0; j < columns; j++){
            matrix[i][j] -= m2.matrix[i][j];
        }
    }
}

Matrix Matrix::operator= (Matrix &m2)
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

Matrix Matrix::getTranpose() const
{
    Matrix new_mtx(columns, rows);

    for (unsigned int i = 0; i < columns; i++){
        for(unsigned int j = 0; j < rows; j++){
            new_mtx[i][j] = matrix[j][i];
        }
    }

    return new_mtx;
}

void Matrix::transpose()
{
    double **t_matrix = matrix;

    matrix = new double*[columns];

    for (unsigned int i = 0; i < columns; i++){
        matrix[i] = new double[rows];
    }

    for (unsigned int i = 0; i < columns; i++){
        for(unsigned int j = 0; j < rows; j++){
            matrix[i][j] = t_matrix[j][i];
        }
    }

    for (unsigned int i = 0; i < rows; i++){
        delete [] t_matrix[i];
    }
    delete [] t_matrix;

    unsigned int temp = rows;
    rows = columns;
    columns = temp;
}

Matrix Matrix::getTranspose()
{
    Matrix outputMatrix(*this);
    outputMatrix.transpose();
    return outputMatrix;
}