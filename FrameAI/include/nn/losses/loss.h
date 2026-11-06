#pragma once
#include "../Tensor.h"

class Loss {
public:
    virtual ~Loss() = default;

    virtual float forward(
        const Tensor<float>& prediction,
        const Tensor<float>& target
    ) = 0;

    virtual Tensor<float> backward(
        const Tensor<float>& prediction,
        const Tensor<float>& target
    ) = 0;
};