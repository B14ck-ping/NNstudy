#include <memory>
#include <stdexcept>
#include <iostream>

#include "math/matrix.hpp"


class MathFactory
{
public:
    virtual ~MathFactory() = default;

    virtual std::unique_ptr<Matrix> createMatrix(int rows, int cols) = 0;
    // virtual std::unique_ptr<Vector> createVector(int size) = 0;
};