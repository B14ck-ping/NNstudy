#include "perceptron.hpp"
#include "activationFunctions.hpp"


static bool using_cuda = true;


perceptron::perceptron(unsigned inNodes_cnt, unsigned hidNodes_cnt, unsigned outNodes_cnt)
{
    this->inputNodes = inNodes_cnt;
        this->hiddenNodes = hidNodes_cnt;
        this->outputNodes = outNodes_cnt;
    if (!using_cuda){
        Matrix b_wih(hidNodes_cnt, inNodes_cnt), b_who(outNodes_cnt, hidNodes_cnt);
        // Set random weights to wih and who
        for (unsigned int i = 0; i < b_wih.get_rows(); i++){
            for(unsigned int j = 0; j < b_wih.get_columns(); j++){
                b_wih[i][j] = (double)(rand()%100)/100 - 0.5;
            }
        }   

        for (unsigned int i = 0; i < b_who.get_rows(); i++){
            for(unsigned int j = 0; j < b_who.get_columns(); j++){
                b_who[i][j] = (double)(rand()%100)/100.0 -0.5;
            }
        }

        this->wih = new MatrixCUDA(b_wih);
        this->who = new MatrixCUDA(b_who);

        this->inputVector = new MatrixCUDA(inNodes_cnt, 1);
        this->hiddenVector = new MatrixCUDA(hidNodes_cnt, 1);
        this->outputVector = new MatrixCUDA(outNodes_cnt, 1);

    } else {

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
    MatrixCUDA _input(input);
    MatrixCUDA hiddenInputs = MatrixCUDA::dot(*this->wih, _input); 
    MatrixCUDA hiddenOutputs = MatrixCUDA::applySigmoid(hiddenInputs);

    MatrixCUDA outputInputs = MatrixCUDA::dot(*this->who, hiddenOutputs); 
    MatrixCUDA outputOutputs = MatrixCUDA::applySigmoid(outputInputs);

    MatrixCUDA outputErrors(targetValue);
    outputErrors -= outputOutputs;
    MatrixCUDA whoT = this->who->getTranspose();
    MatrixCUDA hiddenErrors = MatrixCUDA::dot(whoT, outputErrors);

    MatrixCUDA oneMatrix_10(this->outputNodes, 1, 1.0);

    MatrixCUDA oneMatrix_100(this->hiddenNodes, 1, 1.0);
    
    {
        MatrixCUDA temp = outputErrors;
        temp *= outputOutputs;
        MatrixCUDA temp2 = oneMatrix_10;
        temp2 -= outputOutputs;
        MatrixCUDA hiddenOutputsT = hiddenOutputs.getTranspose();
        temp *= temp2;
        MatrixCUDA w_delta = MatrixCUDA::dot(temp, hiddenOutputsT);
        w_delta *= 0.1;

        *this->who += w_delta;
    }

    MatrixCUDA temp = hiddenErrors;
    temp *= hiddenOutputs;
    MatrixCUDA temp2 = oneMatrix_100;
    temp2 -= hiddenOutputs;
    MatrixCUDA inputsT = _input.getTranspose();
    temp *= temp2;
    MatrixCUDA w_delta = MatrixCUDA::dot(temp, inputsT);
    w_delta *= 0.1;

    *this->wih += w_delta;

    return 0;
}

Matrix perceptron::queue(Matrix &input)
{
    MatrixCUDA _input(input);
    MatrixCUDA hiddenInputs = MatrixCUDA::dot(*this->wih, _input); 
    MatrixCUDA hiddenOutputs = MatrixCUDA::applySigmoid(hiddenInputs);

    MatrixCUDA outputInputs = MatrixCUDA::dot(*this->who, hiddenOutputs); 
    MatrixCUDA outputOutputs = MatrixCUDA::applySigmoid(outputInputs);

    Matrix _output_outputs = outputOutputs.getHost_matrix();
    return _output_outputs;
}