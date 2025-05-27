#if 0
    // Only test

    const unsigned long testRow = 10000;
    const unsigned long testColumn = 30000;

    try {
        float *inputTestArr = new float[testRow*testColumn];
        for  (long i = 0; i < testRow*testColumn; i++)
            inputTestArr[i] = (float)(std::rand()) / (float)(std::rand());

        // std::cout << "Original 1st matrix:" << std::endl;
        // printMatrix(testRow, testColumn, inputTestArr);

        Matrix test1mtx(inputTestArr, testRow, testColumn);
        MatrixCUDA test1mtx_cuda(inputTestArr, testRow, testColumn);

        for  (long i = 0; i < testRow*testColumn; i++)
            inputTestArr[i] = (float)(std::rand()) / (float)(std::rand());

        Matrix test2mtx(inputTestArr, testRow, testColumn);
        MatrixCUDA test2mtx_cuda(inputTestArr, testRow, testColumn);

        // std::cout << "Original 2nd matrix:" << std::endl;
        // printMatrix(testRow, testColumn, inputTestArr);

        // Sum test
        if (0){
            Matrix res = test1mtx + test2mtx;
            MatrixCUDA res_c = test1mtx_cuda + test2mtx_cuda;

            float *res_arr = res.getMatrixDeepCopyArr();

            float *res_arr_c = res_c.getHost_matrix();

            if (memcmp(res_arr, res_arr_c, res.get_columns()*res.get_rows()*sizeof(float)) == 0){
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
        if (0){
            Matrix res = test1mtx * test2mtx;
            MatrixCUDA res_c = test1mtx_cuda * test2mtx_cuda;

            float *res_arr = res.getMatrixDeepCopyArr();

            float *res_arr_c = res_c.getHost_matrix();

            if (memcmp(res_arr, res_arr_c, res.get_columns()*res.get_rows()*sizeof(float)) == 0){
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
        if (0){
            Matrix res = test1mtx - test2mtx;
            MatrixCUDA res_c = test1mtx_cuda - test2mtx_cuda;

            float *res_arr = res.getMatrixDeepCopyArr();

            float *res_arr_c = res_c.getHost_matrix();

            if (memcmp(res_arr, res_arr_c, res.get_columns()*res.get_rows()*sizeof(float)) == 0){
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
        if (0){
            Matrix res(test1mtx);
            auto start = chrono::high_resolution_clock::now();
            res.transpose();
            auto stop = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "CPU transpose time: ";
            s_print_duration(duration);


            
            MatrixCUDA res_c(test1mtx_cuda);
            start = chrono::high_resolution_clock::now();
            res_c.transpose();
            stop = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "GPU transpose time: ";
            s_print_duration(duration);

            float *res_arr = res.getMatrixDeepCopyArr();

            float *res_arr_c = res_c.getHost_matrix();

            if (memcmp(res_arr, res_arr_c, res.get_columns()*res.get_rows()*sizeof(float)) == 0){
                std::cout << "transpose test: PASSED" << std::endl;
            } else {
                std::cout << "transpose test: FAILED" << std::endl;
                std::cout << "Expected result of 1st matrix transpose (CPU):" << std::endl;
                printMatrix(res.get_rows(), res.get_columns(), res_arr);
                std::cout << "Real result of 1st matrix transpose (GPU)" << std::endl;
                printMatrix(res_c.get_rows(), res_c.get_columns(), res_arr_c);
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
        if (0){
            float scalar = (float)(std::rand()) / (float)(std::rand());
            Matrix res(test1mtx);
            res *= scalar;
            MatrixCUDA res_c(test1mtx_cuda);
            res_c *= scalar;

            float *res_arr = res.getMatrixDeepCopyArr();

            float *res_arr_c = res_c.getHost_matrix();

            if (memcmp(res_arr, res_arr_c, res.get_columns()*res.get_rows()*sizeof(float)) == 0){
                std::cout << "product to scalar test: PASSED" << std::endl;
            } else {
                std::cout << "product to scalar test: FAILED" << std::endl;
                std::cout << "Expected result of 1st matrix  product to scalar (CPU):" << std::endl;
                printMatrix(res.get_rows(), res.get_columns(), res_arr);
                std::cout << "Real result of 1st matrix  product to scalar (GPU)" << std::endl;
                printMatrix(res_c.get_rows(), res_c.get_columns(), res_arr_c);

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
            auto start = chrono::high_resolution_clock::now();
            Matrix res = Matrix::dot(test1mtx, test2mtx.getTranspose());
            auto stop = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "CPU Dot product time: ";
            s_print_duration(duration);
            
            long long average = 0;
            MatrixCUDA mt2trans = test2mtx_cuda.getTranspose();
            MatrixCUDA res_c(0, 0);
            for(int i = 0; i < 10000; ++i){
                start = chrono::high_resolution_clock::now();
                res_c = std::move(MatrixCUDA::dot(test1mtx_cuda, mt2trans));
                stop = chrono::high_resolution_clock::now();
                duration = chrono::duration_cast<chrono::microseconds>(stop - start);
                average += duration.count();
            }
            cout << "GPU Dot product average time: " << average/10000 << " microseconds" << endl;
            
            
            res_c = MatrixCUDA::dot(test1mtx_cuda, test2mtx_cuda.getTranspose());

            float *res_arr = res.getMatrixDeepCopyArr();
            float *res_arr_c = res_c.getHost_matrix();

            bool passed = true;
            for(unsigned long i = 0; i < res_c.get_columns()*res_c.get_rows(); i++){
                if (std::abs((res_arr_c[i] - res_arr[i])/res_arr[i]) > 0.01f){
                    passed = false;
                }
            }

            if (passed){
                std::cout << "dot product test: PASSED" << std::endl;
            } else {
                std::cout << "dot product test: FAILED" << std::endl;
                // std::cout << "Expected result of 2 matrix dot product (CPU):" << std::endl;
                // printMatrix(res.get_rows(), res.get_columns(), res_arr);
                // std::cout << "Real result of 2 matrix dot product (GPU)" << std::endl;
                // printMatrix(res_c.get_rows(), res_c.get_columns(), res_arr_c);
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