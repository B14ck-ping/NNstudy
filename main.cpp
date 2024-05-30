#include <iostream>
#include "matrix.hpp"
#include "activationFunctions.hpp"

using namespace std;

int main()
{
    Matrix mtx1(3,3);

    mtx1[0][0] = 0.9;
    mtx1[0][1] = 0.3;
    mtx1[0][2] = 0.4;
    mtx1[1][0] = 0.2;
    mtx1[1][1] = 0.8;
    mtx1[1][2] = 0.2;
    mtx1[2][0] = 0.1;
    mtx1[2][1] = 0.5;
    mtx1[2][2] = 0.6;

    Matrix mtx2(3,1);

    mtx2[0][0] = 0.9;
    mtx2[1][0] = 0.1;
    mtx2[2][0] = 0.8;

    cout << "Input values :" << endl;
    for (unsigned int i = 0; i < mtx1.get_rows(); i++){
        for(unsigned int j = 0; j < mtx1.get_columns(); j++){
            cout << mtx1[i][j] << " ";
        }
        cout << endl;
    }

    Matrix mtx3 = mtx1 * mtx2;

    for (unsigned int i = 0; i < mtx3.get_rows(); i++){
        for(unsigned int j = 0; j < mtx3.get_columns(); j++){
           mtx3[i][j] = actFunc::applySigmoid(mtx3[i][j]);
        }
    }

    mtx1[0][0] = 0.3;
    mtx1[0][1] = 0.7;
    mtx1[0][2] = 0.5;
    mtx1[1][0] = 0.6;
    mtx1[1][1] = 0.5;
    mtx1[1][2] = 0.2;
    mtx1[2][0] = 0.8;
    mtx1[2][1] = 0.1;
    mtx1[2][2] = 0.9;

    mtx3 = mtx1 * mtx3;

    for (unsigned int i = 0; i < mtx3.get_rows(); i++){
        for(unsigned int j = 0; j < mtx3.get_columns(); j++){
           mtx3[i][j] = actFunc::applySigmoid(mtx3[i][j]);
        }
    }
    cout << endl << "Output values :" << endl;
    for (unsigned int i = 0; i < mtx3.get_rows(); i++){
        for(unsigned int j = 0; j < mtx3.get_columns(); j++){
            cout << mtx3[i][j] << " ";
        }
        cout << endl;
    }

    // back propagation

    return 0;
}