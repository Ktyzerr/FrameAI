#include "../../../include/nn/Tensor.h"
#include "../../../include/nn/layers/Linear.h"
#include "../../../include/nn/Shape.h"

Tensor<float> Linear::forward(const Tensor<float>& input) {
    if (input.size() != input_size) {
        throw std::invalid_argument(
            "Linear::forward: input size does not match input_size"
        );
    }

    last_input = input;

    Tensor<float> output(Shape{ output_size });

    for (size_t i = 0; i < output_size; ++i) {
        float sum = 0.0f;

        for (size_t j = 0; j < input_size; ++j) {
            sum += input[j] * weights(i, j);
        }

        output[i] = sum + bias[i];
    }

    return output;
}

Tensor<float> Linear::backward(const Tensor<float>& grad_output)
{
    if (grad_output.size() != output_size) {
        throw std::invalid_argument(
            "Linear::backward: gradient size does not match output_size"
        );
    }

    Tensor<float> grad_input(Shape{ input_size });

    for (size_t i = 0; i < output_size; ++i)
    {
        grad_bias[i] = grad_output[i];

        for (size_t j = 0; j < input_size; ++j)
        {
            grad_weights(i, j) =
                grad_output[i] * last_input[j];

            grad_input[j] +=
                grad_output[i] * weights(i, j);
        }
    }

    return grad_input;
}

std::vector<Tensor<float>*> Linear::parameters() {
    return {
        &weights,
        &bias
    };
}

std::vector<Tensor<float>*> Linear::gradients() {
    return {
        &grad_weights,
        &grad_bias
    };
}