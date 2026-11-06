#pragma once

#include "Optimizer.h"

class SGD : public Optimizer {
public:
    explicit SGD(float learning_rate);

    void step(Model& model) override;

private:
    float learning_rate;
};