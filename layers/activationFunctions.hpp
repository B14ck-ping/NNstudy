#pragma once

#include <iostream>
#include <cmath>
#include "matrix.hpp"

using namespace std;

class actFunc 
{
    public:
    static  float applySigmoid(float in)
    {
        return 1.0f/(1.0f + expf(-in));
    }

    static Matrix applySigmoid(Matrix &in)
    {
        Matrix output_matrix = Matrix(in.get_rows(), in.get_columns());

        for (unsigned int i = 0; i < in.get_rows(); i++){
            for(unsigned int j = 0; j < in.get_columns(); j++){
                output_matrix[i][j] = 1.0f/(1.0f + expf((-1.0f)*in[i][j]));
            }
        }  
        return output_matrix;
    }

    static void applySigmoid_mtx(Matrix &in)
    {
        for (unsigned int i = 0; i < in.get_rows(); i++){
            for(unsigned int j = 0; j < in.get_columns(); j++){
                in[i][j] = 1.0f/(1.0f + expf((-1.0f)*in[i][j]));
            }
        }  
    }
};