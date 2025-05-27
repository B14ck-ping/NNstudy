
#include "../MathFactory.hpp"
#include "matrixCPU.hpp"

class CPUFactory : public MathFactory {
public:
    std::unique_ptr<Matrix> createMatrix(int rows, int cols) override {
        return std::make_unique<MatrixCPU>(rows, cols);
    }

    // std::unique_ptr<Vector> createVector(int size) override {
    //     return std::make_unique<VectorCPU>(size);
    // }
};