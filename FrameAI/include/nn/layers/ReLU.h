#pragma once

#include "./Layer.h"

class ReLU : public Layer {
public:
    ReLU() : last_input(Shape{ 0 })
    {
    }

    Tensor<float> forward(const Tensor<float>& input) override ;
    Tensor<float> backward(const Tensor<float>& grad_output) override;
private:
    Tensor<float> last_input;
};