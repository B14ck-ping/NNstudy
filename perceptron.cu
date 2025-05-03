#include "perceptron.hpp"
#include "activationFunctions.hpp"

#include "kernels.cuh"

static bool using_cuda = true;
// static bool native_cuda_nn = true;


perceptron::perceptron(unsigned inNodes_cnt, unsigned hidNodes_cnt, unsigned outNodes_cnt):
    inputNodes(inNodes_cnt), hiddenNodes(hidNodes_cnt), outputNodes(outNodes_cnt)
{
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

#if 0
        cudaMalloc( (void**)&this->wih, hidNodes_cnt*inNodes_cnt*sizeof(float));
        cudaMalloc( (void**)&this->who, hidNodes_cnt*outNodes_cnt*sizeof(float));

        cudaMalloc( (void**)&this->wih_t, hidNodes_cnt*inNodes_cnt*sizeof(float));
        cudaMalloc( (void**)&this->who_t, hidNodes_cnt*outNodes_cnt*sizeof(float));

        cudaMemcpy(this->wih, wih_arr, hidNodes_cnt*inNodes_cnt*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->who, who_arr, outNodes_cnt*hidNodes_cnt*sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc( (void**)&this->inputVector, inNodes_cnt*sizeof(float));
        cudaMalloc( (void**)&this->hiddenVector, hidNodes_cnt*sizeof(float));
        cudaMalloc( (void**)&this->outputVector, outNodes_cnt*sizeof(float));

        cudaMalloc( (void**)&this->hiddenVectorErr, hidNodes_cnt*sizeof(float));
        cudaMalloc( (void**)&this->outputVectorErr, outNodes_cnt*sizeof(float));
#endif

        free(wih_arr);
        free(who_arr);
        
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

#if 0
    cudaFree(this->wih);
    cudaFree(this->who);
    cudaFree(this->wih_t);
    cudaFree(this->who_t);

    cudaFree(this->inputVector);
    cudaFree(this->hiddenVector);
    cudaFree(this->outputVector);
    cudaFree(this->hiddenVectorErr);
    cudaFree(this->outputVectorErr);
#endif
}

int perceptron::train(const MatrixCUDA &input, const MatrixCUDA &targetValue)
{
#if 0
    if (native_cuda_nn){
        // Copy input data to device
        float *input_arr = input.getDeePCopyOnDevice();

        // Compute hidden layer inputs
        // MatrixCUDA hiddenVector(std::move(MatrixCUDA::dot(*this->wih, input)));
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 BlockDim((1 + TILE_SIZE - 1) / TILE_SIZE, (hiddenNodes + TILE_SIZE - 1) / TILE_SIZE);
        dot_kernel<<<BlockDim, threadsPerBlock>>>(wih, inputVector, hiddenVector, hiddenNodes, inputNodes, 1);
        cudaDeviceSynchronize();

        // Apply sigmoid activation function to hidden layer outputs
        // hiddenVector.applySigmoid();
        sygmoid<<<BlockDim, threadsPerBlock>>>(hiddenVector, hiddenNodes, 1);
        cudaDeviceSynchronize();

        // Compute output layer inputs
        // MatrixCUDA outputVector(std::move(MatrixCUDA::dot(*this->who, hiddenVector))); 
        BlockDim.x = (1 + TILE_SIZE - 1) / TILE_SIZE;
        BlockDim.y = (outputNodes + TILE_SIZE - 1) / TILE_SIZE;
        dot_kernel<<<BlockDim, threadsPerBlock>>>(who, hiddenVector, outputVector, outputNodes, hiddenNodes, 1);
        cudaDeviceSynchronize();

        // Apply sigmoid activation function to output layer outputs
        // outputVector.applySigmoid();
        sygmoid<<<BlockDim, threadsPerBlock>>>(outputVector, outputNodes, 1);
        cudaDeviceSynchronize();

        // Compute output errors
        // MatrixCUDA outputErrors(targetValue);
        float *target_arr = targetValue.getDeePCopyOnDevice(); 
        // outputErrors -= outputVector;
        decrease<<<BlockDim, threadsPerBlock>>>(target_arr, outputVector, outputNodes, 1);
        cudaDeviceSynchronize();

        // Compute hidden layer errors
        // MatrixCUDA hiddenErrors = std::move(MatrixCUDA::dot(this->who->getTranspose(), outputErrors));
        BlockDim.x = (hiddenNodes + TILE_SIZE - 1) / TILE_SIZE;
        BlockDim.y = (outputNodes + TILE_SIZE - 1) / TILE_SIZE;
        transpose_kernel<<<BlockDim, threadsPerBlock>>>(who, who_t, hiddenNodes, outputNodes);
        cudaDeviceSynchronize();
        
        BlockDim.x = (1 + TILE_SIZE - 1) / TILE_SIZE;
        BlockDim.y = (hiddenNodes + TILE_SIZE - 1) / TILE_SIZE;
        dot_kernel<<<BlockDim, threadsPerBlock>>>(who_t, target_arr, hiddenVectorErr, hiddenNodes, outputNodes, 1);
        cudaDeviceSynchronize();

        // Multiply output errors to output vector
        // outputErrors *= outputVector;
        BlockDim.x = (1 + TILE_SIZE - 1) / TILE_SIZE;
        BlockDim.y = (outputNodes + TILE_SIZE - 1) / TILE_SIZE;
        multiple<<<BlockDim, threadsPerBlock>>>(target_arr, outputVector, outputNodes, 1);
        cudaDeviceSynchronize();

        // MatrixCUDA temp1(this->outputNodes, 1, 1.0);// = oneMatrix_10;
        // temp1 -= outputVector;
        inversion<<<BlockDim, threadsPerBlock>>>(outputVector, 1, outputNodes);
        cudaDeviceSynchronize();

        // outputErrors *= temp1;
        multiple<<<BlockDim, threadsPerBlock>>>(target_arr, outputVector, outputNodes, 1);

        // MatrixCUDA w_delta(std::move(MatrixCUDA::dot(outputErrors, hiddenVector.getTranspose())));
        float *hiddenVector_tr = nullptr;
        cudaMalloc( (void**)&hiddenVector_tr, hiddenNodes*sizeof(float));
        BlockDim.x = (1 + TILE_SIZE - 1) / TILE_SIZE;
        BlockDim.y = (hiddenNodes + TILE_SIZE - 1) / TILE_SIZE;
        transpose_kernel<<<BlockDim, threadsPerBlock>>>(hiddenVector, hiddenVector_tr, 1, hiddenNodes);
        cudaDeviceSynchronize();

        float *w_delta = nullptr;
        cudaMalloc( (void**)&w_delta, outputNodes*hiddenNodes*sizeof(float));
        BlockDim.x = (hiddenNodes + TILE_SIZE - 1) / TILE_SIZE;
        BlockDim.y = (outputNodes + TILE_SIZE - 1) / TILE_SIZE;
        dot_kernel<<<BlockDim, threadsPerBlock>>>(target_arr, hiddenVector_tr, w_delta, outputNodes, 1, hiddenNodes);
        cudaDeviceSynchronize();
        cudaFree(hiddenVector_tr);

        // w_delta *= 0.1f;
        multiple_to_val<<<BlockDim, threadsPerBlock>>>(w_delta,  0.1f, outputNodes, hiddenNodes);
        cudaDeviceSynchronize();

        // *this->who += w_delta;
        increase<<<BlockDim, threadsPerBlock>>>(who, w_delta, outputNodes, hiddenNodes);
        cudaDeviceSynchronize();

        // hiddenErrors *= hiddenVector;
        BlockDim.x = (1 + TILE_SIZE - 1) / TILE_SIZE;
        BlockDim.y = (hiddenNodes + TILE_SIZE - 1) / TILE_SIZE;
        multiple<<<BlockDim, threadsPerBlock>>>(hiddenVectorErr, hiddenVector, hiddenNodes, 1);
        cudaDeviceSynchronize();

        // MatrixCUDA temp2(this->hiddenNodes, 1, 1.0);// = oneMatrix_100;
        // temp2 -= hiddenVector;
        inversion<<<BlockDim, threadsPerBlock>>>(hiddenVector, 1, hiddenNodes);
        cudaDeviceSynchronize();

        // hiddenErrors *= temp2;
        multiple<<<BlockDim, threadsPerBlock>>>(hiddenVectorErr, hiddenVector, hiddenNodes, 1);
        cudaDeviceSynchronize();

        // w_delta = std::move(MatrixCUDA::dot(hiddenErrors, input.getTranspose()));
        float *inputVector_tr = nullptr;
        cudaMalloc( (void**)&inputVector_tr, inputNodes*sizeof(float));
        BlockDim.x = (1 + TILE_SIZE - 1) / TILE_SIZE;
        BlockDim.y = (inputNodes + TILE_SIZE - 1) / TILE_SIZE;
        transpose_kernel<<<BlockDim, threadsPerBlock>>>(inputVector, inputVector_tr, 1, inputNodes);
        cudaDeviceSynchronize();

        float *w_delta2 = nullptr;
        cudaMalloc( (void**)&w_delta2, inputNodes*hiddenNodes*sizeof(float));
        BlockDim.x = (hiddenNodes + TILE_SIZE - 1) / TILE_SIZE;
        BlockDim.y = (inputNodes + TILE_SIZE - 1) / TILE_SIZE;
        dot_kernel<<<BlockDim, threadsPerBlock>>>(hiddenVectorErr, inputVector_tr, w_delta2, hiddenNodes, 1, inputNodes);
        cudaDeviceSynchronize();
        cudaFree(inputVector_tr);
        // w_delta *= 0.1f;
        multiple_to_val<<<BlockDim, threadsPerBlock>>>(w_delta2, 0.1f, hiddenNodes, inputNodes);
        cudaDeviceSynchronize();

        // *this->wih += w_delta; 
        add<<<BlockDim, threadsPerBlock>>>(wih, w_delta2, wih, hiddenNodes, inputNodes);
        cudaDeviceSynchronize();
        
        cudaFree(target_arr);
        cudaFree(input_arr);
    } else 
#endif
    if (using_cuda){
        MatrixCUDA hiddenVector(std::move(MatrixCUDA::dot(*this->wih, input))); 
        hiddenVector.applySigmoid();
        MatrixCUDA outputVector(std::move(MatrixCUDA::dot(*this->who, hiddenVector))); 
        outputVector.applySigmoid();

        MatrixCUDA outputErrors(targetValue);
        outputErrors -= outputVector;
        MatrixCUDA hiddenErrors = std::move(MatrixCUDA::dot(this->who->getTranspose(), outputErrors));

        outputErrors *= outputVector;
        MatrixCUDA temp1(this->outputNodes, 1, 1.0);// = oneMatrix_10;
        temp1 -= outputVector;
        outputErrors *= temp1;
        MatrixCUDA w_delta(std::move(MatrixCUDA::dot(outputErrors, hiddenVector.getTranspose())));
        w_delta *= 0.1f;

        *this->who += w_delta;

        hiddenErrors *= hiddenVector;
        MatrixCUDA temp2(this->hiddenNodes, 1, 1.0);// = oneMatrix_100;
        temp2 -= hiddenVector;
        hiddenErrors *= temp2;
        w_delta = std::move(MatrixCUDA::dot(hiddenErrors, input.getTranspose()));
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
#if 0
    float *input_arr = input.getMatrixDeepCopyArr();

    // Compute hidden layer inputs
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 BlockDim((1 + TILE_SIZE - 1) / TILE_SIZE, (hiddenNodes + TILE_SIZE - 1) / TILE_SIZE);
    dot_kernel<<<BlockDim, threadsPerBlock>>>(wih, inputVector, hiddenVector, hiddenNodes, inputNodes, 1);
    cudaDeviceSynchronize();

    // Apply sigmoid activation function to hidden layer outputs
    sygmoid<<<BlockDim, threadsPerBlock>>>(hiddenVector, hiddenNodes, 1);
    cudaDeviceSynchronize();

    // Compute output layer inputs
    BlockDim.x = (1 + TILE_SIZE - 1) / TILE_SIZE;
    BlockDim.y = (outputNodes + TILE_SIZE - 1) / TILE_SIZE;
    dot_kernel<<<BlockDim, threadsPerBlock>>>(who, hiddenVector, outputVector, outputNodes, hiddenNodes, 1);
    cudaDeviceSynchronize();

    // Apply sigmoid activation function to output layer outputs
    sygmoid<<<BlockDim, threadsPerBlock>>>(outputVector, outputNodes, 1);
    cudaDeviceSynchronize();

    float *output_arr = (float*)malloc(outputNodes*sizeof(float));;
    cudaMemcpy(output_arr, outputVector, outputNodes*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input_arr);

    Matrix _output_outputs(output_arr, outputNodes, 1);
    free(output_arr);
    return _output_outputs;
#endif
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
        return Matrix(0,0);
    }
}