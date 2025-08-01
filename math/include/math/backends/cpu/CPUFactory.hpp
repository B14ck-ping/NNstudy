
#include "../MathFactory.hpp"
#include "matrixCPU.hpp"

class CPUFactory : public MathFactory {
public:
    std::unique_ptr<Matrix> createMatrix(int rows, int cols) override {
        return std::make_unique<MatrixCPU>(rows, cols);
    }

    std::unique_ptr<Matrix> createMatrix(int rows, int cols, float fill_value) override {
        return std::make_unique<MatrixCPU>(rows, cols, fill_value);
    }

    std::unique_ptr<Matrix> createMatrix(const Matrix& other) override {
        return std::make_unique<MatrixCPU>(other);
    }

    std::unique_ptr<Matrix> createMatrix(Matrix& other) override {
        return std::make_unique<MatrixCPU>(other);
    }

    std::unique_ptr<Matrix> createMatrix(Matrix&& other) override {
        return std::make_unique<MatrixCPU>(std::move(other));
    }

    std::unique_ptr<Matrix> createMatrix(float* mtx_arr, int rows, int cols) override {
        return std::make_unique<MatrixCPU>(mtx_arr, rows, cols);
    }

    // std::unique_ptr<Vector> createVector(int size) override {
    //     return std::make_unique<VectorCPU>(size);
    // }
};