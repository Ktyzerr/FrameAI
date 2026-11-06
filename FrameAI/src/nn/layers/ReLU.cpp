#include "../../../include/nn/layers/ReLU.h"
#include "../../../include/nn/Tensor.h"
#include <iostream>

Tensor<float> ReLU::forward(const Tensor<float>& input) {
    Tensor<float> output(input.get_shape());

    last_input = input;

    std::cout << std::endl;


    for (size_t i = 0; i < input.size(); ++i)
    {
        if (input[i] < 0.0f)
            output[i] = 0.0f;
        else
            output[i] = input[i];
    }

    return output;
}

Tensor<float> ReLU::backward(const Tensor<float>& grad_output) {
    Tensor<float> gradient(grad_output.get_shape());

    for (size_t i = 0; i < grad_output.size(); ++i)
    {
        if (last_input[i] > 0.0f)
            gradient[i] = grad_output[i];
        else
            gradient[i] = 0.0f;
    }

    return gradient;
}