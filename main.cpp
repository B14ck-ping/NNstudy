#include <iostream>
#include <fstream>

#include <string>
#include "perceptron.hpp"
// #include <unistd.h>      
#include <iomanip>
#include <chrono>
// #include "processor.hpp"
#include "matrixCUDA.h"
#include "matrix.hpp"

using namespace std;
typedef struct{
    MatrixCUDA *target;
    MatrixCUDA *input;
}   input_data_t;

static void s_print_duration(chrono::microseconds duration)
{
    long long hours, minutes, seconds;

    if (duration.count() < 1000){
        cout << duration.count() << " microseconds" << endl;
        return;
    } else if (duration.count() > 1000 && duration.count() < 1000000){
        cout << duration.count()/1000 << " milliseconds" << endl;
        return;
    } else if (duration.count() > 1000000 && duration.count() < 60000000){
        cout << duration.count()/1000000 << " seconds" << endl;
        return;
    } else if (duration.count() > 60000000 && duration.count() < 3600000000){
        seconds = duration.count()/1000000LL;
        minutes = seconds/60;
        seconds = seconds%60;
        cout << minutes << " minutes " << seconds << " seconds" << endl;
        return;
    } else if (duration.count() > 3600000000){
        seconds = duration.count()/1000000LL;
        minutes = seconds/60;
        seconds = seconds%60;
        hours = minutes/60;
        minutes = minutes%60;
        cout << hours << " hours " << minutes << " minutes " << seconds << " seconds" << endl;
        return;
    }
   
}

void printProgressBar(size_t itemCnt, size_t curr_item)
{
    int curr_percent = (int)(((float)(curr_item+1)/(float)itemCnt)*100.0);
    // cout << "\x1b[2K" << "\x1b[1A" << "\x1b[2K" << "\r";
    cout << "\r";
    cout << "Progress: " << curr_percent << "% ";
    cout << "[";
    for (int j = 0; j < curr_percent; j++){
        cout << "=";
    }

    for (int j = 0; j < 100 - curr_percent; j++){
        cout << "-";
    }
    
    cout << "]" << curr_item+1 << "/" << itemCnt;// << endl;
}

void printMatrix(long r, long c, float *arr)
{
    for (long i = 0; i < r; i++){
        for (long j = 0; j < c; j++){
            std::cout << arr[i*c+j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


int main()
{
    cudaSetDevice(0);

#if 0
    // Only test

    const unsigned long testRow = 100;
    const unsigned long testColumn = 145;

    try {
        float inputTestArr[testRow*testColumn] = {};
        for  (long i = 0; i < testRow*testColumn; i++)
            inputTestArr[i] = (float)(std::rand()) / (float)(std::rand());

        std::cout << "Original 1st matrix:" << std::endl;
        printMatrix(testRow, testColumn, inputTestArr);

        Matrix test1mtx(inputTestArr, testRow, testColumn);
        MatrixCUDA test1mtx_cuda(inputTestArr, testRow, testColumn);

        for  (long i = 0; i < testRow*testColumn; i++)
            inputTestArr[i] = (float)(std::rand()) / (float)(std::rand());

        Matrix test2mtx(inputTestArr, testRow, testColumn);
        MatrixCUDA test2mtx_cuda(inputTestArr, testRow, testColumn);

        std::cout << "Original 2nd matrix:" << std::endl;
        printMatrix(testRow, testColumn, inputTestArr);

        // Sum test
        {
            Matrix res = test1mtx + test2mtx;
            MatrixCUDA res_c = test1mtx_cuda + test2mtx_cuda;

            float *res_arr = res.getMatrixDeepCopyArr();

            float *res_arr_c = res_c.getHost_matrix();

            if (memcmp(res_arr, res_arr_c, res.get_columns()*res.get_rows()) == 0){
                std::cout << "Sum test: PASSED" << std::endl;
            } else {
                std::cout << "Sum test: FAILED" << std::endl;
                std::cout << "Expected result of 2 matrix sum (CPU):" << std::endl;
                printMatrix(res.get_columns(), res.get_rows(), res_arr);
                std::cout << "Real result of 2 matrix sum (GPU)" << std::endl;
                printMatrix(res_c.get_columns(), res_c.get_rows(), res_arr_c);
                // for(unsigned long i = 0; i < res_c.get_columns()*res_c.get_rows(); i++){
                //     if (std::abs((res_arr_c[i] - res_arr[i])/res_arr[i]) > 0.01f){
                //         std::cout << "i: " << i << " CPU: " << res_arr[i] << " GPU: " << res_arr_c[i] << std::endl;
                //     }
                // }
                free(res_arr);
                free(res_arr_c);

                return -1;
            }

            free(res_arr);
            free(res_arr_c);
        }
        
        // multiple test
        {
            Matrix res = test1mtx * test2mtx;
            MatrixCUDA res_c = test1mtx_cuda * test2mtx_cuda;

            float *res_arr = res.getMatrixDeepCopyArr();

            float *res_arr_c = res_c.getHost_matrix();

            if (memcmp(res_arr, res_arr_c, res.get_columns()*res.get_rows()) == 0){
                std::cout << "multiplication test: PASSED" << std::endl;
            } else {
                std::cout << "multiplication test: FAILED" << std::endl;
                std::cout << "Expected result of 2 matrix multiplication (CPU):" << std::endl;
                printMatrix(res.get_columns(), res.get_rows(), res_arr);
                std::cout << "Real result of 2 matrix multiplication (GPU)" << std::endl;
                printMatrix(res_c.get_columns(), res_c.get_rows(), res_arr_c);
                // for(unsigned long i = 0; i < res_c.get_columns()*res_c.get_rows(); i++){
                //     if (std::abs((res_arr_c[i] - res_arr[i])/res_arr[i]) > 0.01f){
                //         std::cout << "i: " << i << " CPU: " << res_arr[i] << " GPU: " << res_arr_c[i] << std::endl;
                //     }
                // }
                free(res_arr);
                free(res_arr_c);

                return -1;
            }
            free(res_arr);
            free(res_arr_c);

        }

        // subtruct test
        {
            Matrix res = test1mtx - test2mtx;
            MatrixCUDA res_c = test1mtx_cuda - test2mtx_cuda;

            float *res_arr = res.getMatrixDeepCopyArr();

            float *res_arr_c = res_c.getHost_matrix();

            if (memcmp(res_arr, res_arr_c, res.get_columns()*res.get_rows()) == 0){
                std::cout << "subtruct test: PASSED" << std::endl;
            } else {
                std::cout << "subtruct test: FAILED" << std::endl;
                std::cout << "Expected result of 2 matrix subtruct (CPU):" << std::endl;
                printMatrix(res.get_columns(), res.get_rows(), res_arr);
                std::cout << "Real result of 2 matrix subtruct (GPU)" << std::endl;
                printMatrix(res_c.get_columns(), res_c.get_rows(), res_arr_c);
                // for(unsigned long i = 0; i < res_c.get_columns()*res_c.get_rows(); i++){
                //     if (std::abs((res_arr_c[i] - res_arr[i])/res_arr[i]) > 0.01f){
                //         std::cout << "i: " << i << " CPU: " << res_arr[i] << " GPU: " << res_arr_c[i] << std::endl;
                //     }
                // }
                free(res_arr);
                free(res_arr_c);

                return -1;
            }
            free(res_arr);
            free(res_arr_c);

        }

        // transpose test
        {
            Matrix res(test1mtx);
            res.transpose();
            MatrixCUDA res_c(test1mtx_cuda);
            res_c.transpose();

            float *res_arr = res.getMatrixDeepCopyArr();

            float *res_arr_c = res_c.getHost_matrix();

            if (memcmp(res_arr, res_arr_c, res.get_columns()*res.get_rows()) == 0){
                std::cout << "transpose test: PASSED" << std::endl;
            } else {
                std::cout << "transpose test: FAILED" << std::endl;
                std::cout << "Expected result of 1st matrix transpose (CPU):" << std::endl;
                printMatrix(res.get_columns(), res.get_rows(), res_arr);
                std::cout << "Real result of 1st matrix transpose (GPU)" << std::endl;
                printMatrix(res_c.get_columns(), res_c.get_rows(), res_arr_c);
                // for(unsigned long i = 0; i < res_c.get_columns()*res_c.get_rows(); i++){
                //     if (std::abs((res_arr_c[i] - res_arr[i])/res_arr[i]) > 0.01f){
                //         std::cout << "i: " << i << " CPU: " << res_arr[i] << " GPU: " << res_arr_c[i] << std::endl;
                //     }
                // }
                free(res_arr);
                free(res_arr_c);

                return -1;
            }
            free(res_arr);
            free(res_arr_c);

        }

        // product to scalar test
        {
            float scalar = (float)(std::rand()) / (float)(std::rand());
            Matrix res(test1mtx);
            res *= scalar;
            MatrixCUDA res_c(test1mtx_cuda);
            res_c *= scalar;

            float *res_arr = res.getMatrixDeepCopyArr();

            float *res_arr_c = res_c.getHost_matrix();

            if (memcmp(res_arr, res_arr_c, res.get_columns()*res.get_rows()) == 0){
                std::cout << "product to scalar test: PASSED" << std::endl;
            } else {
                std::cout << "product to scalar test: FAILED" << std::endl;
                std::cout << "Expected result of 1st matrix  product to scalar (CPU):" << std::endl;
                printMatrix(res.get_columns(), res.get_rows(), res_arr);
                std::cout << "Real result of 1st matrix  product to scalar (GPU)" << std::endl;
                printMatrix(res_c.get_columns(), res_c.get_rows(), res_arr_c);

                // for(unsigned long i = 0; i < res_c.get_columns()*res_c.get_rows(); i++){
                //     if (std::abs((res_arr_c[i] - res_arr[i])/res_arr[i]) > 0.001f){
                //         std::cout << "i: " << i << " CPU: " << res_arr[i] << " GPU: " << res_arr_c[i] << std::endl;
                //     }
                // }

                free(res_arr);
                free(res_arr_c);

                return -1;
            }
            free(res_arr);
            free(res_arr_c);

        }

        // dot product test
        {
            Matrix res = Matrix::dot(test1mtx, test2mtx.getTranspose());
            MatrixCUDA res_c = MatrixCUDA::dot(test1mtx_cuda, test2mtx_cuda.getTranspose());

            float *res_arr = res.getMatrixDeepCopyArr();

            float *res_arr_c = res_c.getHost_matrix();

            if (memcmp(res_arr, res_arr_c, res.get_columns()*res.get_rows()) == 0){
                std::cout << "dot product test: PASSED" << std::endl;
            } else {
                std::cout << "dot product test: FAILED" << std::endl;
                std::cout << "Expected result of 2 matrix dot product (CPU):" << std::endl;
                printMatrix(res.get_columns(), res.get_rows(), res_arr);
                std::cout << "Real result of 2 matrix dot product (GPU)" << std::endl;
                printMatrix(res_c.get_columns(), res_c.get_rows(), res_arr_c);
                for(unsigned long i = 0; i < res_c.get_columns()*res_c.get_rows(); i++){
                    if (std::abs((res_arr_c[i] - res_arr[i])/res_arr[i]) > 0.01f){
                        std::cout << "i: " << i << " CPU: " << res_arr[i] << " GPU: " << res_arr_c[i] << std::endl;
                    }
                }
                free(res_arr);
                free(res_arr_c);

                return -1;
            }
            free(res_arr);
            free(res_arr_c);

        }
    } catch (std::exception &e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    // // sygmoid test
    // {
    //     Matrix res(test1mtx);
    //     actFunc::applySigmoid_mtx(res);
    //     MatrixCUDA res_c(test1mtx_cuda);
    //     res_c.applySigmoid();

    //     float *res_arr = res.getMatrixDeepCopyArr();

    //     float *res_arr_c = res_c.getHost_matrix();
    //     int ret_val = 0;
    //     if ((ret_val = memcmp(res_arr, res_arr_c, res.get_columns()*res.get_rows())) == 0){
    //         std::cout << "sygmoid test: PASSED" << std::endl;
    //     } else {
    //         std::cout << "sygmoid test: FAILED" << std::endl;
    //         std::cout << "Expected result of 1st matrix sygmoid applying (CPU):" << std::endl;
    //         printMatrix(res.get_columns(), res.get_rows(), res_arr);
    //         std::cout << "Real result of 1st matrix sygmoid applying (GPU)" << std::endl;
    //         printMatrix(res_c.get_columns(), res_c.get_rows(), res_arr_c);
    //         std::cout << "ret_val: " << ret_val << std::endl;   
    //         std::cout << "ind: " << ret_val/sizeof(float) << std::endl;

    //         for (int i = 0; i < res_c.get_columns()*res_c.get_rows(); i++){
    //             std::cout << "i: " << i << " CPU: " << res_arr[i] << " GPU: " << res_arr_c[i];
    //             if (res_arr[i] != res_arr_c[i]){
    //                 std::cout << " <---------";
    //             }
    //             std::cout << std::endl;
    //         }


    //         free(res_arr);
    //         free(res_arr_c);

    //         return -1;
    //     }
    //     free(res_arr);
    //     free(res_arr_c);

    // }

    return 0;
#endif

    const size_t trainDataSetSize = 60000;
    const size_t testDataSetSize = 1000;
    int epochs = 1;

    // processor::Instance();

    perceptron MNISTperc(784, 300, 10);

    ifstream train_file("mnist_train.csv", ios::in); 
    // Open file         
    // train_file.open("123.txt", ios::in);      
    if (!train_file.is_open())
    {
        cout << "Can't open file!" << std::endl;
        return -1;
    }

    cout << "File opened successfully!" << std::endl;

    // train neural network
    string delimiter = ",";

    auto start = chrono::high_resolution_clock::now();
    input_data_t *inputs = (input_data_t*)malloc(sizeof(input_data_t) * trainDataSetSize);
    for (size_t i = 0; i <  trainDataSetSize; i++){
                    // get line
        string sa;
        // Read data from the file object and put it into a string.
        if (!getline(train_file, sa))
            break;

        // Split it by comma and make input value matrix
        size_t pos = sa.find(delimiter);
        string token = sa.substr(0, pos);
        sa.erase(0, pos + delimiter.length());
        int value = stoi(token);
        
        unsigned int cnt = 0;
        float buffer[784] = {};
        while ((pos = sa.find(delimiter)) != std::string::npos && cnt < 784) {
            token = sa.substr(0, pos);
            sa.erase(0, pos + delimiter.length());
            float buf = stof(token);
            buf = ((buf/255.0f)*0.99f) + 0.01f;
            buffer[cnt] = buf;
            cnt++;
        }

        //  TODO: Copy matrix into CUDA matrix
        inputs[i].input = new MatrixCUDA(buffer, 784U, 1U);

        // Let's make target value matrix
        float resBuff[10] = {0};
        std::fill(resBuff, resBuff + 784, 0.01f);
        if (value >= 10) value = 9; 
        resBuff[value] = 0.99f;
        inputs[i].target = new MatrixCUDA(resBuff, 10U, 1U);
    }
    train_file.close(); 
    auto stopfile = chrono::high_resolution_clock::now();
    auto durationfile = chrono::duration_cast<chrono::microseconds>(stopfile - start);
    cout << "Trainig data is loaded from file in ";
    s_print_duration(durationfile);

    cout << "Training neural net:" << endl;
    for (int ep = 0; ep < epochs; ep++){
        cout << "Start epoch "<< ep+1 << endl << endl;
        auto epoch_start = chrono::high_resolution_clock::now();
        for (size_t i = 0; i < trainDataSetSize; i++){
            printProgressBar(trainDataSetSize, i);
            // Train matrix
            MNISTperc.train(*inputs[i].input, *inputs[i].target);
        }
        cout << endl;
        auto epoch_finish = chrono::high_resolution_clock::now();
        auto epochduration = chrono::duration_cast<chrono::microseconds>(epoch_finish - epoch_start);
        cout << "Epoch complete in ";
        s_print_duration(epochduration);
    }

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    cout << "Network training complete in ";
    s_print_duration(duration);
    start = chrono::high_resolution_clock::now();
    ifstream test_file("mnist_test.csv", ios::in); 
    // Open file         
    // train_file.open("123.txt", ios::in);      
    if (!test_file.is_open())
    {
        cout << "Can't open file!" << std::endl;
        return -1;
    }

    cout << "File opened successfully!" << std::endl;

    // Let's test neural network
   
    cout << "Test neural net:" << endl << endl;
    size_t goodAns = 0, badAns = 0;
    for (size_t i = 0; i < testDataSetSize; i++){
        printProgressBar(testDataSetSize, i);
        
        // get line
        string sa;
        // Read data from the file object and put it into a string.
        if (!getline(test_file, sa))
            break;

        // Split it by comma and make input value matrix
        size_t pos = sa.find(delimiter);
        string token = sa.substr(0, pos);
        sa.erase(0, pos + delimiter.length());
        int value = stoi(token);
        Matrix inputValues(784, 1, 0);
        unsigned cnt = 0;
        while ((pos = sa.find(delimiter)) != std::string::npos) {
            token = sa.substr(0, pos);
            sa.erase(0, pos + delimiter.length());
            inputValues[cnt][0] = stof(token);
            inputValues[cnt][0] = ((inputValues[cnt][0]/255.0f)*0.99f) + 0.01f;
            cnt++;
        }

        // cout << "total : " << cnt + 1 << endl;

       // Test matrix
        Matrix output = MNISTperc.queue(inputValues);

        // Find answer
        double maxVal = output[0][0];
        int maxValIdx = 0;
        for (unsigned i = 1; i < output.get_rows(); i++){
            if (output[i][0] > maxVal){
                maxVal = output[i][0];
                maxValIdx = i;
            }
        }

        if (maxValIdx == value)
            goodAns++;
        else
            badAns++;
    }
    cout << endl;

    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "Network testing complete in ";
    s_print_duration(duration);
    // print statictics
    double accuracy = (((double)goodAns)/(goodAns + badAns))*100.0;
    cout << "Bad answers: " << badAns << endl << "Good answers: " << goodAns << endl;
    cout << "Acuracy: " << std::fixed << std::setw(8) << std::setprecision(2) <<  accuracy << " %" << endl;
    
    // free memory
    for (size_t i = 0; i < trainDataSetSize; i++){
        delete inputs[i].input;
        delete inputs[i].target;
    }
    free(inputs);

    return 0;
}