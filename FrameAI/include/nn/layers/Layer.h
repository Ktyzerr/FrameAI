#pragma once

#include "../Tensor.h"
#include <vector>

class Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor<float> forward(const Tensor<float>& input) = 0;

    virtual std::vector<Tensor<float>*> parameters() {
        return {};
    }

    virtual std::vector<Tensor<float>*> gradients() {
        return {};
    }

    virtual Tensor<float> backward(
        const Tensor<float>& grad_output
    ) = 0;
};