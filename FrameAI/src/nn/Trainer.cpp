#include "../../include/nn/Trainer.h"

#include <stdexcept>
#include <iostream>
#include <iomanip>

void Trainer::train(
    Model& model,
    Loss& loss,
    Optimizer& optimizer,
    const std::vector<Tensor<float>>& inputs,
    const std::vector<Tensor<float>>& targets,
    size_t epochs
)
{
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument(
            "Trainer::train: inputs and targets count do not match"
        );
    }

    if (inputs.empty()) {
        throw std::invalid_argument(
            "Trainer::train: dataset is empty"
        );
    }

    if (epochs == 0) {
        throw std::invalid_argument(
            "Trainer::train: epochs must be greater than zero"
        );
    }

    std::cout << "Обучение...\n";

    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        float total_loss = 0.0f;

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            Tensor<float> prediction =
                model.forward(inputs[i]);

            total_loss +=
                loss.forward(prediction, targets[i]);

            Tensor<float> gradient =
                loss.backward(prediction, targets[i]);

            model.backward(gradient);

            optimizer.step(model);
        }

        float progress =
            static_cast<float>(epoch + 1) /
            static_cast<float>(epochs);

        int percent =
            static_cast<int>(progress * 100.0f);

        const int bar_width = 30;
        int filled =
            static_cast<int>(progress * bar_width);

        std::cout << '\r';

        std::cout << '[';

        for (int i = 0; i < bar_width; ++i)
        {
            if (i < filled)
                std::cout << '#';
            else
                std::cout << '-';
        }

        std::cout << "] "
            << std::setw(3)
            << percent
            << "% | "
            << "loss: "
            << std::fixed
            << std::setprecision(6)
            << total_loss;

        std::cout << std::flush;
    }

    std::cout << "\n\n";
    std::cout << "Обучение завершено!\n";
}