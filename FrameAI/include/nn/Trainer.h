#pragma once

#include <vector>
#include "./Tensor.h"
#include "./Model.h"
#include "./losses/Loss.h"
#include "./optimizers/Optimizer.h"

class Trainer {
public:
    static void train(
        Model& model,
        Loss& loss,
        Optimizer& optimizer,
        const std::vector<Tensor<float>>& inputs,
        const std::vector<Tensor<float>>& targets,
        size_t epochs
    );
};