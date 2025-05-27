
#include "../MathFactory.hpp"
#include "matrixCUDA.hpp"


class CUDAFactory : public MathFactory {
public:
    std::unique_ptr<Matrix> createMatrix(int rows, int cols) override {
        return std::make_unique<MatrixCUDA>(rows, cols);
    }

    // std::unique_ptr<Vector> createVector(int size) override {
    //     return std::make_unique<VectorCUDA>(size);
    // }
};