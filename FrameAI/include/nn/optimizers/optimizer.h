#pragma once

#include "../Model.h"

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void step(Model& model) = 0;
};