#include "perceptron.hpp"
#include "activationFunctions.hpp"


perceptron::perceptron(unsigned inNodes_cnt, unsigned hidNodes_cnt, unsigned outNodes_cnt)
{
    this->inputNodes = inNodes_cnt;
    this->hiddenNodes = hidNodes_cnt;
    this->outputNodes = outNodes_cnt;

    this->wih = new Matrix(hidNodes_cnt, inNodes_cnt);
    this->who = new Matrix(outNodes_cnt, hidNodes_cnt);

    this->inputVector = new Matrix(inNodes_cnt, 1);
    this->hiddenVector = new Matrix(hidNodes_cnt, 1);
    this->outputVector = new Matrix(outNodes_cnt, 1);

    // Set random weights to wih and who
    for (unsigned int i = 0; i < this->wih->get_rows(); i++){
        for(unsigned int j = 0; j < this->wih->get_columns(); j++){
            (*this->wih)[i][j] = (double)(rand()%100)/100 - 0.5;
        }
    }   

    for (unsigned int i = 0; i < this->who->get_rows(); i++){
        for(unsigned int j = 0; j < this->who->get_columns(); j++){
            (*this->who)[i][j] = (double)(rand()%100)/100.0 -0.5;
        }
    }  
}

perceptron::~perceptron()
{
    delete this->wih;
    delete this->who;
    delete this->inputVector;
    delete this->hiddenVector;
    delete this->outputVector;
}

int perceptron::train(Matrix &input, Matrix &targetValue)
{
    Matrix hiddenInputs = Matrix::dot(*this->wih, input); 
    Matrix hiddenOutputs = actFunc::applySigmoid(hiddenInputs);

    Matrix outputInputs = Matrix::dot(*this->who, hiddenOutputs); 
    Matrix outputOutputs = actFunc::applySigmoid(outputInputs);

    Matrix outputErrors = targetValue;
    outputErrors -= outputOutputs;
    Matrix whoT = this->who->getTranspose();
    Matrix hiddenErrors = Matrix::dot(whoT, outputErrors);

    Matrix oneMatrix_10(this->outputNodes, 1, 1.0);

    Matrix oneMatrix_100(this->hiddenNodes, 1, 1.0);
    
    {
        Matrix temp = outputErrors;
        temp *= outputOutputs;
        Matrix temp2 = oneMatrix_10;
        temp2 -= outputOutputs;
        Matrix hiddenOutputsT = hiddenOutputs.getTranspose();
        temp *= temp2;
        Matrix w_delta = Matrix::dot(temp, hiddenOutputsT);
        w_delta *= 0.1;

        *this->who += w_delta;
    }

    Matrix temp = hiddenErrors;
    temp *= hiddenOutputs;
    Matrix temp2 = oneMatrix_100;
    temp2 -= hiddenOutputs;
    Matrix inputsT = input.getTranspose();
    temp *= temp2;
    Matrix w_delta = Matrix::dot(temp, inputsT);
    w_delta *= 0.1;

    *this->wih += w_delta;

    return 0;
}

Matrix perceptron::queue(Matrix &input)
{
    Matrix hiddenInputs = Matrix::dot(*this->wih, input); 
    Matrix hiddenOutputs = actFunc::applySigmoid(hiddenInputs);

    Matrix outputInputs = Matrix::dot(*this->who, hiddenOutputs); 
    Matrix outputOutputs = actFunc::applySigmoid(outputInputs);

    return outputOutputs;
}