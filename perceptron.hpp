#pragma once

#include <iostream>
#include "matrix.hpp"
#include "activationFunctions.hpp"


using namespace std;

class perceptron {
private:
    unsigned inputNodes;
    unsigned hiddenNodes;
    unsigned outputNodes;

    Matrix *wih = NULL, *who = NULL, *inputVector = NULL, *hiddenVector = NULL, *outputVector = NULL;

public:
    explicit perceptron(unsigned inNodes_cnt, unsigned hidNodes_cnt, unsigned outNodes_cnt);
    ~perceptron();

    int train(Matrix &input, Matrix &targetValue);
    Matrix queue(Matrix &input);
};