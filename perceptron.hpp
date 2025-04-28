#pragma once

#include <iostream>
#include "matrix.hpp"
#include "matrixCUDA.h"
#include "activationFunctions.hpp"


using namespace std;

class perceptron {
private:
    unsigned inputNodes;
    unsigned hiddenNodes;
    unsigned outputNodes;

    MatrixCUDA *wih = NULL, *who = NULL/*,  *inputVector = NULL, *hiddenVector = NULL, *outputVector = NULL, *out1Vector = NULL, *hidden1Vector = NULL */;

public:
    explicit perceptron(unsigned inNodes_cnt, unsigned hidNodes_cnt, unsigned outNodes_cnt);
    ~perceptron();

    int train(const MatrixCUDA &input, const MatrixCUDA &targetValue);
    Matrix queue(Matrix &input);
};