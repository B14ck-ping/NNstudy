#include <memory>
#include <stdexcept>
#include <iostream>


#include "CPUFactory.hpp"   
#include "CUDAFactory.hpp"
#include "Matrix.hpp"
#include "MAthFactory.hpp"

enum class Backend {
    CPU,
    GPU
};

class MathFactoryProvider {
public:
    static std::unique_ptr<Matrix> createMatrix(int rows, int cols, Backend backend) {
        return getFactory(backend)->createMatrix(rows, cols);
    }

    // static std::unique_ptr<Vector> createVector(int size, Backend backend) {
    //     return getFactory(backend)->createVector(size);
    // }

private:
    static std::unique_ptr<MathFactory> getFactory(Backend backend) {
        switch (backend) {
            case Backend::CPU:
                return std::make_unique<CPUFactory>();
            case Backend::GPU:
                return std::make_unique<CUDAFactory>();
            default:
                throw std::runtime_error("Unsupported backend");
        }
    }
};