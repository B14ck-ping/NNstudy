#pragma once

#include <iostream>
#include <cmath>
#include "matrix.hpp"

using namespace std;

class actFunc 
{
    public:
    static  double applySigmoid(double in)
    {
        return 1.0/(1.0 + exp(-in));
    }

    static Matrix applySigmoid(Matrix &in)
    {
        Matrix output_matrix = Matrix(in.get_rows(), in.get_columns());

        for (unsigned int i = 0; i < in.get_rows(); i++){
            for(unsigned int j = 0; j < in.get_columns(); j++){
                output_matrix[i][j] = 1.0/(1.0 + exp(-in[i][j]));
            }
        }  
        return output_matrix;
    }
};