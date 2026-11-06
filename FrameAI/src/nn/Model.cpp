#include "../../include/nn/Model.h"

Tensor<float> Model::forward(const Tensor<float>& input) {
	Tensor<float> output = input;
	for (auto& layer : layers) {
		output = layer->forward(output);
	}
	return output;
}

Tensor<float> Model::backward(const Tensor<float>& grad_output) {
	Tensor<float> gradient = grad_output;

	for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
		gradient = (*it)->backward(gradient);
	}

	return gradient;
}

std::vector<Tensor<float>*> Model::parameters() {
    std::vector<Tensor<float>*> result;

    for (auto& layer : layers)
    {
        auto params = layer->parameters();

        result.insert(
            result.end(),
            params.begin(),
            params.end()
        );
    }

    return result;
}

std::vector<Tensor<float>*> Model::gradients() {
    std::vector<Tensor<float>*> result;

    for (auto& layer : layers) {
        auto grads = layer->gradients();

        result.insert(
            result.end(),
            grads.begin(),
            grads.end()
        );
    }

    return result;
}