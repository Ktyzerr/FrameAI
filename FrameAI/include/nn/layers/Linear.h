#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include "../Tensor.h"
#include "Layer.h"

class Linear : public Layer {
public:
    Linear(size_t input_size, size_t output_size)
        : weights({ output_size, input_size }),
        bias({ output_size }),
        grad_weights({ output_size, input_size }),
        grad_bias({ output_size }),
        last_input(Shape{ input_size }),
        input_size(input_size),
        output_size(output_size)
    {
        std::random_device rd;
        std::mt19937 generator(rd());

        float stddev = std::sqrt(2.0f / static_cast<float>(input_size));

        std::normal_distribution<float> distribution(0.0f, stddev);

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = distribution(generator);
        }

        for (size_t i = 0; i < bias.size(); ++i) {
            bias[i] = 0.0f;
        }
    }

    std::vector<Tensor<float>*> parameters() override;
    std::vector<Tensor<float>*> gradients() override;

    Tensor<float> forward(const Tensor<float>& input) override;
    Tensor<float> backward(const Tensor<float>& grad_output) override;

private:
	Tensor<float> weights;
	Tensor<float> bias;

    Tensor<float> grad_weights;
    Tensor<float> grad_bias;

    Tensor<float> last_input;

	size_t input_size;
    size_t output_size;
};