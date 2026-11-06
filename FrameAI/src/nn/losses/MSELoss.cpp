#include "../../../include/nn/losses/MSELoss.h"
#include <stdexcept>

float MSELoss::forward(
    const Tensor<float>& prediction,
    const Tensor<float>& target
)
{
    if (prediction.size() != target.size()) {
        throw std::invalid_argument(
            "MSELoss::forward: prediction and target sizes do not match"
        );
    }

    float sum = 0.0f;

    for (size_t i = 0; i < prediction.size(); ++i) {
        float diff = prediction[i] - target[i];
        sum += diff * diff;
    }

    return sum / static_cast<float>(prediction.size());
}

Tensor<float> MSELoss::backward(
    const Tensor<float>& prediction,
    const Tensor<float>& target
)
{
    if (prediction.size() != target.size()) {
        throw std::invalid_argument(
            "MSELoss::backward: prediction and target sizes do not match"
        );
    }

    Tensor<float> gradient(prediction.get_shape());

    float scale = 2.0f / static_cast<float>(prediction.size());

    for (size_t i = 0; i < prediction.size(); ++i) {
        gradient[i] =
            scale * (prediction[i] - target[i]);
    }

    return gradient;
} 