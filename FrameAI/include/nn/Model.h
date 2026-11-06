#pragma once
#include <vector>
#include <memory>
#include <utility>
#include "layers/Layer.h"

class Model {
public:
	template <typename LayerType, typename... Args>
	void add(Args... args) {
		layers.push_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
	}
	Tensor<float> forward(const Tensor<float>& input);
	Tensor<float> backward(const Tensor<float>& grad_output);

	std::vector<Tensor<float>*> parameters();
	std::vector<Tensor<float>*> gradients();
private:
	std::vector<std::unique_ptr<Layer>> layers;
};