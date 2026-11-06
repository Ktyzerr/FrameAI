```cpp
#include <iostream>
#include <vector>
#include <iomanip>

#include "include/nn/Model.h"
#include "include/nn/Tensor.h"
#include "include/nn/Shape.h"

#include "include/nn/layers/Linear.h"
#include "include/nn/layers/ReLU.h"

#include "include/nn/losses/MSELoss.h"
#include "include/nn/optimizers/SGD.h"

#include "include/nn/Trainer.h"


Tensor<float> image(std::initializer_list<float> pixels)
{
    return Tensor<float>(
        pixels,
        Shape{ 25 }
    );
}


void print_image(const Tensor<float>& input)
{
    for (size_t y = 0; y < 5; ++y)
    {
        for (size_t x = 0; x < 5; ++x)
        {
            float pixel = input[y * 5 + x];

            if (pixel > 0.5f)
                std::cout << "##";
            else
                std::cout << "..";
        }

        std::cout << '\n';
    }
}


size_t predicted_class(const Tensor<float>& prediction)
{
    if (prediction[0] > prediction[1])
        return 0;

    return 1;
}


int main()
{
    // MODEL

    Model model;

    model.add<Linear>(25, 16);
    model.add<ReLU>();
    model.add<Linear>(16, 2);


    // LOSS

    MSELoss loss;


    // OPTIMIZER

    SGD optimizer(0.01f);


    // TRAINING DATA

    std::vector<Tensor<float>> train_inputs = {

        // SQUARE

        image({
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0
        }),

        image({
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 0,
            1, 1, 1, 1, 0,
            1, 1, 1, 1, 0,
            0, 0, 0, 0, 0
        }),

        image({
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 1,
            0, 1, 1, 1, 1,
            0, 1, 1, 1, 1,
            0, 0, 0, 0, 0
        }),

        image({
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            0, 0, 0, 0, 0
        }),


            // CROSS

            image({
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                1, 1, 1, 1, 1,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0
            }),

            image({
                0, 1, 0, 0, 0,
                0, 1, 0, 0, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 0, 0,
                0, 1, 0, 0, 0
            }),

            image({
                0, 0, 0, 1, 0,
                0, 0, 0, 1, 0,
                1, 1, 1, 1, 1,
                0, 0, 0, 1, 0,
                0, 0, 0, 1, 0
            }),

            image({
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1
            })
    };


    std::vector<Tensor<float>> train_targets = {

        Tensor<float>({ 1.0f, 0.0f }, Shape{ 2 }),
        Tensor<float>({ 1.0f, 0.0f }, Shape{ 2 }),
        Tensor<float>({ 1.0f, 0.0f }, Shape{ 2 }),
        Tensor<float>({ 1.0f, 0.0f }, Shape{ 2 }),

        Tensor<float>({ 0.0f, 1.0f }, Shape{ 2 }),
        Tensor<float>({ 0.0f, 1.0f }, Shape{ 2 }),
        Tensor<float>({ 0.0f, 1.0f }, Shape{ 2 }),
        Tensor<float>({ 0.0f, 1.0f }, Shape{ 2 })
    };


    // TEST DATA

    std::vector<Tensor<float>> test_inputs = {

        // NEW SQUARE

        image({
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            0, 0, 0, 0, 0
        }),

        image({
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 1,
            0, 1, 1, 1, 1,
            0, 1, 1, 1, 1,
            0, 0, 0, 0, 0
        }),


            // NEW CROSS

            image({
                1, 0, 0, 0, 1,
                1, 1, 0, 1, 1,
                0, 0, 1, 0, 0,
                1, 1, 0, 1, 1,
                1, 0, 0, 0, 1
            }),

            image({
                0, 1, 0, 1, 0,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                0, 1, 0, 1, 0
            })
    };


    std::vector<Tensor<float>> test_targets = {

        Tensor<float>({ 1.0f, 0.0f }, Shape{ 2 }),
        Tensor<float>({ 1.0f, 0.0f }, Shape{ 2 }),

        Tensor<float>({ 0.0f, 1.0f }, Shape{ 2 }),
        Tensor<float>({ 0.0f, 1.0f }, Shape{ 2 })
    };


    // TRAINING

    std::cout << "\n";
    std::cout << "=============================\n";
    std::cout << "          TRAINING\n";
    std::cout << "=============================\n\n";


    Trainer::train(
        model,
        loss,
        optimizer,
        train_inputs,
        train_targets,
        10000
    );


    // TEST

    std::cout << "\n";
    std::cout << "=============================\n";
    std::cout << "        TEST RESULTS\n";
    std::cout << "=============================\n\n";


    size_t correct = 0;


    for (size_t i = 0; i < test_inputs.size(); ++i)
    {
        Tensor<float> prediction =
            model.forward(test_inputs[i]);


        size_t predicted =
            predicted_class(prediction);


        size_t expected =
            predicted_class(test_targets[i]);


        std::cout
            << "Image "
            << i + 1
            << "\n\n";


        print_image(test_inputs[i]);


        std::cout << "\n";


        std::cout << "Expected: ";

        if (expected == 0)
            std::cout << "SQUARE\n";
        else
            std::cout << "CROSS\n";


        std::cout << "Prediction: ";

        if (predicted == 0)
            std::cout << "SQUARE\n";
        else
            std::cout << "CROSS\n";


        std::cout
            << std::fixed
            << std::setprecision(4);


        std::cout
            << "Class 0 score: "
            << prediction[0]
            << '\n';


        std::cout
            << "Class 1 score: "
            << prediction[1]
            << '\n';


        if (predicted == expected)
        {
            std::cout << "Result: CORRECT\n";
            ++correct;
        }
        else
        {
            std::cout << "Result: WRONG\n";
        }


        std::cout
            << "\n-----------------------------\n\n";
    }


    // ACCURACY

    float accuracy =
        static_cast<float>(correct) /
        static_cast<float>(test_inputs.size()) *
        100.0f;


    std::cout
        << "=============================\n";

    std::cout
        << "Test accuracy: "
        << std::fixed
        << std::setprecision(2)
        << accuracy
        << "%\n";

    std::cout
        << "Correct: "
        << correct
        << " / "
        << test_inputs.size()
        << '\n';

    std::cout
        << "=============================\n";


    return 0;
}

