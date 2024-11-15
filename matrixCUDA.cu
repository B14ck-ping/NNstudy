#include "matrixCUDA.cuh"


MatrixCUDA::MatrixCUDA(unsigned int a_rows, unsigned int a_columns) : rows(a_rows), columns(a_columns)
{
    // matrix = (float*)malloc(rows*columns*sizeof(float));
}

MatrixCUDA::MatrixCUDA(unsigned int a_rows, unsigned int a_columns, float value) : rows(a_rows), columns(a_columns)
{
    // matrix = (float*)malloc(rows*columns*sizeof(float));
    // std::fill(matrix, matrix + rows*columns, value);
}

MatrixCUDA::MatrixCUDA(Matrix &mtx)
{
    rows = mtx.get_rows();
    columns = mtx.get_columns();
    // matrix = (float*)malloc(rows*columns*sizeof(float));
    // std::memcpy( matrix, mtx.matrix, sizeof(float)*columns*rows);
}

MatrixCUDA::~MatrixCUDA()
{
    // free(matrix);
}

MatrixCUDA MatrixCUDA::dot(MatrixCUDA &mtx1, MatrixCUDA &mtx2)
{
    // if (mtx1.get_columns() != mtx2.get_rows())
    //     return mtx1;

    unsigned mtx1_rows = mtx1.get_rows();
    // unsigned mtx1_col = mtx1.get_columns();
    unsigned int l_output_columns = mtx2.get_columns();
    MatrixCUDA l_matrix(mtx1_rows, l_output_columns);
    // MatrixCUDA mtx_2_t = mtx2.getTranspose();

    // for (unsigned int k = 0; k < l_output_columns; k++){
    //     for (unsigned int i = 0; i < mtx1_rows; i++){
    //         float l_temp = 0.0;
    //         for(unsigned int j = 0; j < mtx1_col; j++){
    //             l_temp += mtx1[i][j] * mtx2[j][k];
    //         }
    //         *(*(l_matrix.matrix + i) + k) = l_temp;
    //     }
    // }

    // unsigned int mtx1_col_even_idx = mtx1_col - (mtx1_col - 1)%4;
    // if(mtx1_col_even_idx >= 4){
    //     for (unsigned int k = 0; k < l_output_columns; k++){
    //         for (unsigned int i = 0; i < mtx1_rows; i++){
    //             float l_temp = 0.0;
    //             __m128 buffer_vector = _mm_broadcast_ss(&l_temp);
    //             for(unsigned int j = 0; j < mtx1_col_even_idx-4; j+=4){
    //                 auto first_vector = _mm_loadu_ps(&mtx1[i][j]);
    //                 auto second_vector = _mm_loadu_ps(&mtx_2_t[k][j]);

    //                 auto output_vector_1 = _mm_mul_ps(first_vector, second_vector);//_mm_dp_ps(first_vector,second_vector,0xF1);

    //                 buffer_vector = _mm_add_ps(output_vector_1, buffer_vector);
    //             }
    //             float a[4] = {};
    //             _mm_storeu_ps(&a[0], buffer_vector);
    //             l_temp +=  a[0] + a[1] + a[2] + a[3];
    //             if (mtx1_col_even_idx != mtx1_col){
    //                 for (unsigned int j = mtx1_col_even_idx; j < mtx1_col; j++)
    //                     l_temp += mtx1[i][j] * mtx2[j][k];
    //             }

    //             l_matrix[i][k] = l_temp;
    //         }
    //     }
    // } else if(mtx1_col_even_idx == 1){
    //     unsigned out_mtx_cols_even_idx = l_output_columns - l_output_columns%4;
    //     for (unsigned int k = 0; k < mtx1_rows; k++){
    //         auto second_vector = _mm_broadcast_ss(&mtx1[k][0]);
    //         for (unsigned int i = 0; i < out_mtx_cols_even_idx ; i+=4){
    //             auto first_vector = _mm_loadu_ps(&mtx2[0][i]);
    //             auto output_vector_2 = _mm_mul_ps(first_vector, second_vector);
    //             _mm_storeu_ps(&l_matrix[k][i], output_vector_2);
    //             // l_matrix[k][i] = mtx1[k][0]*mtx2[0][i];
    //         }
    //         if (out_mtx_cols_even_idx != l_output_columns)
    //             l_matrix[k][l_output_columns-1] = mtx1[k][0]*mtx2[0][out_mtx_cols_even_idx-1];
    //     }
    // } else {
    //     for (unsigned int k = 0; k < l_output_columns; k++){
    //         for (unsigned int i = 0; i < mtx1_rows; i++){
    //             float l_temp = 0.0;
    //             for(unsigned int j = 0; j < mtx1_col; j++){
    //                 l_temp += mtx1[i][j] * mtx2[j][k];
    //             }
    //             l_matrix[i][k] = l_temp;
    //         }
    //     }

    // }
    return l_matrix;
}

MatrixCUDA MatrixCUDA::operator* (float scalar)
{
    MatrixCUDA temp(*this);

    // unsigned int col_even_idx = temp.columns - (temp.columns - 1)%4;

    // for (unsigned int i = 0; i < temp.rows; i++){
    //     for(unsigned int j = 0; j < col_even_idx; j+=4){
    //         auto first_vector = _mm_loadu_ps(&temp[i][j]);
    //         auto second_vector = _mm_broadcast_ss(&scalar);

    //         auto sum_vector = _mm_mul_ps(first_vector, second_vector);

    //         _mm_storeu_ps(&temp[i][j], sum_vector);
    //         // temp.matrix[i][j] *= scalar;
    //     }
    //     if(col_even_idx != columns)
    //         temp[i][columns-1] *= scalar;
    // }

    return temp;
}

void MatrixCUDA::operator*= (float scalar)
{

    // unsigned int col_even_idx = columns - columns%4;

    // for (unsigned int i = 0; i < rows; i++){
    //     for(unsigned int j = 0; j < col_even_idx; j+=4){
    //         auto first_vector = _mm_loadu_ps(&matrix[i*columns+j]);
    //         auto second_vector = _mm_broadcast_ss(&scalar);

    //         auto sum_vector = _mm_mul_ps(first_vector, second_vector);

    //         _mm_storeu_ps(&matrix[i*columns+j], sum_vector);
    //         // matrix[i*columns+j] *= scalar;
    //     }
    //     if(col_even_idx != columns)
    //         for (unsigned int k = col_even_idx; k < columns; k++)
    //         matrix[i*columns + k] *= scalar;
    // }
}

MatrixCUDA MatrixCUDA::operator* (MatrixCUDA &mtrx)
{
    MatrixCUDA out_matrix(mtrx.get_rows(), mtrx.get_columns());

    // for (unsigned int i = 0; i < rows; i++){
    //     for(unsigned int j = 0; j < columns; j++){

    //         matrix[i*columns + j] *= mtrx[i][j];
    //     }
    // }

    return out_matrix;      
}

void MatrixCUDA::operator*= (MatrixCUDA &mtrx)
{
    // unsigned int col_even_idx = columns - columns%4;

    // for (unsigned int i = 0; i < rows; i++){
    //     for(unsigned int j = 0; j < col_even_idx; j+=4){
    //         auto first_vector = _mm_loadu_ps(&matrix[i*columns+j]);
    //         auto second_vector = _mm_loadu_ps(&mtrx[i][j]);

    //         auto sum_vector = _mm_mul_ps(first_vector, second_vector);

    //         _mm_storeu_ps(&matrix[i*columns+j], sum_vector);
    //         // matrix[i][j] *= mtrx.matrix[i][j];
    //     }
    //     if(col_even_idx != columns)
    //         for (unsigned int k = col_even_idx; k < columns; k++)
    //             matrix[i*columns+k] *= mtrx[i][k];
    // }  
}

MatrixCUDA MatrixCUDA::operator+ (MatrixCUDA &mtrx)
{
    return MatrixCUDA(0,0);
}

void MatrixCUDA::operator+= (MatrixCUDA &m2)
{
    // unsigned int col_even_idx = columns - columns%4;

    // for (unsigned int i = 0; i < rows; i++){
    //     for(unsigned int j = 0; j < col_even_idx; j+=4){
    //         auto first_vector = _mm_loadu_ps(&matrix[i*columns+j]);
    //         auto second_vector = _mm_loadu_ps(&m2.matrix[i*columns+j]);

    //         auto sum_vector = _mm_add_ps(first_vector, second_vector);

    //         _mm_storeu_ps(&matrix[i*columns + j], sum_vector);

    //         // matrix[i*columns + j] += m2.matrix[i*columns + j];
    //     }
    //     if(col_even_idx != columns)
    //         for (unsigned int k = col_even_idx; k < columns; k++)
    //             matrix[i*columns+k] += m2[i][k];

    // }
}

MatrixCUDA MatrixCUDA::operator- (MatrixCUDA&)
{
    return MatrixCUDA(0,0);
}

void MatrixCUDA::operator-= (MatrixCUDA &m2)
{
    // for (unsigned int i = 0; i < rows; i++){
    //     for(unsigned int j = 0; j < columns; j++){
    //         matrix[i*columns+j] -= m2[i][j];
    //     }
    // }
}

MatrixCUDA MatrixCUDA::operator= (MatrixCUDA &m2)
{
    // this->rows = m2.rows;
    // this->columns = m2.columns;

    // matrix = (float*)malloc(rows*columns*sizeof(float));
    // std::memcpy( matrix, m2.matrix, sizeof(float)*columns*rows);

    return *this;
}

float* MatrixCUDA::operator[] (unsigned int idx)
{
    return matrix + idx*columns;
}

float MatrixCUDA::getDeterminant() const
{
    return 0;
}

MatrixCUDA MatrixCUDA::getTranpose() const
{
    MatrixCUDA new_mtx(columns, rows);

    // for (unsigned int i = 0; i < columns; i++){
    //     for(unsigned int j = 0; j < rows; j++){
    //         new_mtx[i][j] = matrix[j*columns+i];
    //     }
    // }

    return new_mtx;
}

void MatrixCUDA::transpose()
{
    // float *t_matrix = matrix;

    // matrix = (float*)malloc(columns*rows*sizeof(float));

    // for (unsigned int i = 0; i < columns; i++){
    //     for(unsigned int j = 0; j < rows; j++){
    //         matrix[i*rows+j] = t_matrix[j*columns+i];
    //     }
    // }

    // free(t_matrix);

    // unsigned int temp = rows;
    // rows = columns;
    // columns = temp;
}

MatrixCUDA MatrixCUDA::getTranspose()
{
    MatrixCUDA outputMatrixCUDA(*this);
    outputMatrixCUDA.transpose();
    return outputMatrixCUDA;
}


MatrixCUDA MatrixCUDA::applySigmoid(MatrixCUDA &mtx)
{
    return mtx;
}

Matrix MatrixCUDA::getHost_matrix()
{
    Matrix out_mtx(this->get_rows(), this->get_columns());



    return out_mtx;
}