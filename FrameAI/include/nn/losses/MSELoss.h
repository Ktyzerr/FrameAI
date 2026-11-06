#pragma once
#include "Loss.h"

class MSELoss : public Loss {
public:
    float forward(
        const Tensor<float>& prediction,
        const Tensor<float>& target
    ) override;

    Tensor<float> backward(
        const Tensor<float>& prediction,
        const Tensor<float>& target
    ) override;
};