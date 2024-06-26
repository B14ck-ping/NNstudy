#include <iostream>
#include <fstream>
#include <string>
#include "perceptron.hpp"
#include <unistd.h>      
#include <iomanip>
#include <chrono>


using namespace std;

void printProgressBar(size_t itemCnt, size_t curr_item)
{
    int curr_percent = (int)(((float)(curr_item+1)/(float)itemCnt)*100.0);
    cout << "\x1b[2K" << "\x1b[1A" << "\x1b[2K" << "\r";
    cout << "[";
    for (int j = 0; j < curr_percent; j++){
        cout << "=";
    }

    for (int j = 0; j < 100 - curr_percent; j++){
        cout << "-";
    }
    
    cout << "]" << curr_item+1 << "/" << itemCnt << endl;
}


int main()
{
    size_t trainDataSetSize = 60000;
    size_t testDataSetSize = 1000;

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

    cout << "Training neural net:" << endl;
    int epochs = 6;
    for (int ep = 0; ep < epochs; ep++){
        cout << "Start epoch "<< ep+1 << endl << endl;
        for (int i = 0; i < trainDataSetSize; i++){
            printProgressBar(trainDataSetSize, i);
            
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
            Matrix inputValues(784, 1, 0);
            size_t cnt = 0;
            while ((pos = sa.find(delimiter)) != std::string::npos) {
                token = sa.substr(0, pos);
                sa.erase(0, pos + delimiter.length());
                inputValues[cnt][0] = stod(token);
                inputValues[cnt][0] = ((inputValues[cnt][0]/255.0)*0.99) + 0.01;
                cnt++;
            }

            // cout << "total : " << cnt + 1 << endl;

            // Let's make target value matrix
            Matrix targetValues(10, 1, 0.01);
            targetValues[value][0] = 0.99;

            // Train matrix
            MNISTperc.train(inputValues, targetValues);
        }
        train_file.clear();
        train_file.seekg(0);
    }
    train_file.close(); 

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    cout << "Network training complete in " << duration.count() << " microsec" << endl;
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
    for (int i = 0; i < testDataSetSize; i++){
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
        size_t cnt = 0;
        while ((pos = sa.find(delimiter)) != std::string::npos) {
            token = sa.substr(0, pos);
            sa.erase(0, pos + delimiter.length());
            inputValues[cnt][0] = stod(token);
            inputValues[cnt][0] = ((inputValues[cnt][0]/255.0)*0.99) + 0.01;
            cnt++;
        }

        // cout << "total : " << cnt + 1 << endl;

       // Test matrix
        Matrix output = MNISTperc.queue(inputValues);

        // Find answer
        double maxVal = output[0][0];
        int maxValIdx = 0;
        for (size_t i = 1; i < output.get_rows(); i++){
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

    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "Network testing complete in " << duration.count() << " microsec" << endl;
    // print statictics
    double accuracy = (((double)goodAns)/(goodAns + badAns))*100.0;
    cout << "Bad answers: " << badAns << endl << "Good answers: " << goodAns << endl;
    cout << "Acuracy: " << std::fixed << std::setw(8) << std::setprecision(2) <<  accuracy << " %" << endl;
    
    

    return 0;
}