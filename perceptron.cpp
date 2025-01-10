#include "perceptron.hpp"
#include "activationFunctions.hpp"


static bool using_cuda = true;


perceptron::perceptron(unsigned inNodes_cnt, unsigned hidNodes_cnt, unsigned outNodes_cnt)
{
    this->inputNodes = inNodes_cnt;
    this->hiddenNodes = hidNodes_cnt;
    this->outputNodes = outNodes_cnt;
    if (using_cuda){
        Matrix b_wih(hidNodes_cnt, inNodes_cnt), b_who(outNodes_cnt, hidNodes_cnt);
        // Set random weights to wih and who
        for (unsigned int i = 0; i < hidNodes_cnt; i++){
            for(unsigned int j = 0; j < inNodes_cnt; j++){
                b_wih[i][j] = (float)(rand()%100)/100 - 0.5f;
            }
        }   

        for (unsigned int i = 0; i < outNodes_cnt; i++){
            for(unsigned int j = 0; j < hidNodes_cnt; j++){
                b_who[i][j] = (float)(rand()%100)/100.0f -0.5f;
            }
        }


        float *wih_arr = b_wih.getMatrixDeepCopyArr();
        float *who_arr = b_who.getMatrixDeepCopyArr();

        this->wih = new MatrixCUDA(wih_arr, b_wih.get_rows(), b_wih.get_columns());
        this->who = new MatrixCUDA(who_arr, b_who.get_rows(), b_who.get_columns());

        free(wih_arr);
        free(who_arr);
        
        this->inputVector = new MatrixCUDA(inNodes_cnt, 1);
        this->hiddenVector = new MatrixCUDA(hidNodes_cnt, 1);
        this->outputVector = new MatrixCUDA(outNodes_cnt, 1);

    } else {
        // this->wih = new Matrix(hidNodes_cnt, inNodes_cnt);
        // this->who = new Matrix(outNodes_cnt, hidNodes_cnt);

        // this->inputVector = new Matrix(inNodes_cnt, 1);
        // this->hiddenVector = new Matrix(hidNodes_cnt, 1);
        // this->outputVector = new Matrix(outNodes_cnt, 1);

        // // Set random weights to wih and who
        // for (unsigned int i = 0; i < this->wih->get_rows(); i++){
        //     for(unsigned int j = 0; j < this->wih->get_columns(); j++){
        //         (*this->wih)[i][j] = (double)(rand()%100)/100 - 0.5;
        //     }
        // }   

        // for (unsigned int i = 0; i < this->who->get_rows(); i++){
        //     for(unsigned int j = 0; j < this->who->get_columns(); j++){
        //         (*this->who)[i][j] = (double)(rand()%100)/100.0 -0.5;
        //     }
        // }  
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
    if (using_cuda){
        float *input_arr = input.getMatrixDeepCopyArr();
        MatrixCUDA _input(input_arr, input.get_rows(), input.get_columns());
        free(input_arr);
        MatrixCUDA hiddenInputs = MatrixCUDA::dot(*this->wih, _input); 
        hiddenInputs.applySigmoid();
        MatrixCUDA outputInputs = MatrixCUDA::dot(*this->who, hiddenInputs); 
        outputInputs.applySigmoid();

        float *outErrArr = targetValue.getMatrixDeepCopyArr();
        MatrixCUDA outputErrors(outErrArr, targetValue.get_rows(), targetValue.get_columns());
        free(outErrArr);
        outputErrors -= outputInputs;
        this->who->transpose();
        MatrixCUDA hiddenErrors = MatrixCUDA::dot(*this->who, outputErrors);
        this->who->transpose();
    
        outputErrors *= outputInputs;
        MatrixCUDA temp1(this->outputNodes, 1, 1.0);// = oneMatrix_10;
        temp1 -= outputInputs;
        hiddenInputs.transpose();
        outputErrors *= temp1;
        MatrixCUDA w_delta = MatrixCUDA::dot(outputErrors, hiddenInputs);
        hiddenInputs.transpose();
        w_delta *= 0.1f;

        *this->who += w_delta;

        hiddenErrors *= hiddenInputs;
        MatrixCUDA temp2(this->hiddenNodes, 1, 1.0);// = oneMatrix_100;
        temp2 -= hiddenInputs;
        _input.transpose();
        hiddenErrors *= temp2;
        w_delta = MatrixCUDA::dot(hiddenErrors, _input);
        w_delta *= 0.1f;

        *this->wih += w_delta;          
    } else {
        // Matrix hiddenInputs = Matrix::dot(*this->wih, input); 
        // actFunc::applySigmoid_mtx(hiddenInputs);
        // Matrix outputInputs = Matrix::dot(*this->who, hiddenInputs); 
        // actFunc::applySigmoid_mtx(outputInputs);

        // targetValue -= outputInputs;
        // this->who->transpose();
        // Matrix hiddenErrors = Matrix::dot(*this->who, targetValue);
        // this->who->transpose();

        // targetValue *= outputInputs;
        // Matrix temp1(this->outputNodes, 1, 1.0);
        // temp1 -= outputInputs;
        // hiddenInputs.transpose();
        // targetValue *= temp1;
        // Matrix w_delta = Matrix::dot(targetValue, hiddenInputs);
        // hiddenInputs.transpose();
        // w_delta *= 0.1;

        // *this->who += w_delta;

        // hiddenErrors *= hiddenInputs;
        // Matrix temp2(this->hiddenNodes, 1, 1.0);
        // temp2 -= hiddenInputs;
        // input.transpose();
        // hiddenErrors *= temp2;
        // w_delta = Matrix::dot(hiddenErrors, input);
        // w_delta *= 0.1;

        // *this->wih += w_delta;
    }
    return 0;
}

Matrix perceptron::queue(Matrix &input)
{
    if (using_cuda){
        float *input_arr = input.getMatrixDeepCopyArr();
        MatrixCUDA _input(input_arr, input.get_rows(), input.get_columns());
        free(input_arr);
        MatrixCUDA hiddenInputs = MatrixCUDA::dot(*this->wih, _input); 
        hiddenInputs.applySigmoid();

        MatrixCUDA outputInputs = MatrixCUDA::dot(*this->who, hiddenInputs); 
        outputInputs.applySigmoid();

        float *output_arr = outputInputs.getHost_matrix();
        Matrix _output_outputs(output_arr, outputInputs.get_rows(), outputInputs.get_columns());
        free(output_arr);
        return _output_outputs;
    } else {
        // Matrix hiddenInputs = Matrix::dot(*this->wih, input); 
        // Matrix hiddenOutputs = actFunc::applySigmoid(hiddenInputs);

        // Matrix outputInputs = Matrix::dot(*this->who, hiddenOutputs); 
        // Matrix outputOutputs = actFunc::applySigmoid(outputInputs);

        // return outputOutputs;
    }
}