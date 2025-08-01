
#include "../MathFactory.hpp"
#include "matrixCUDA.hpp"


class CUDAFactory : public MathFactory {
public:
    std::unique_ptr<Matrix> createMatrix(int rows, int cols) override {
        return std::make_unique<MatrixCUDA>(rows, cols);
    }

    std::unique_ptr<Matrix> createMatrix(int rows, int cols, float fill_value) override {
        return std::make_unique<MatrixCUDA>(rows, cols, fill_value);
    }

    std::unique_ptr<Matrix> createMatrix(const Matrix& other) override {
        return std::make_unique<MatrixCUDA>(other);
    }

    std::unique_ptr<Matrix> createMatrix(Matrix& other) override {
        return std::make_unique<MatrixCUDA>(other);
    }

    std::unique_ptr<Matrix> createMatrix(Matrix&& other) override {
        return std::make_unique<MatrixCUDA>(std::move(other));
    }

    std::unique_ptr<Matrix> createMatrix(float* mtx_arr, int rows, int cols) override {
        return std::make_unique<MatrixCUDA>(mtx_arr, rows, cols);
    }   
    

    // std::unique_ptr<Vector> createVector(int size) override {
    //     return std::make_unique<VectorCUDA>(size);
    // }
};