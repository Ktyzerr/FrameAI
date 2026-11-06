#include "../../../include/nn/optimizers/SGD.h"
#include <stdexcept>

SGD::SGD(float learning_rate)
    : learning_rate(learning_rate)
{
    if (learning_rate <= 0.0f) {
        throw std::invalid_argument(
            "SGD: learning rate must be positive"
        );
    }
}

void SGD::step(Model& model)
{
    auto parameters = model.parameters();
    auto gradients = model.gradients();

    if (parameters.size() != gradients.size()) {
        throw std::runtime_error(
            "SGD: parameters and gradients count do not match"
        );
    }

    for (size_t i = 0; i < parameters.size(); ++i)
    {
        if (parameters[i]->size() != gradients[i]->size()) {
            throw std::runtime_error(
                "SGD: parameter and gradient sizes do not match"
            );
        }

        for (size_t j = 0; j < parameters[i]->size(); ++j)
        {
            (*parameters[i])[j] -=
                learning_rate * (*gradients[i])[j];
        }
    }
}