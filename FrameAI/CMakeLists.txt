# CMakeList.txt: проект CMake для FrameAI; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#

# Добавьте источник в исполняемый файл этого проекта.
add_executable (FrameAI "FrameAI.cpp" "FrameAI.h"  "example/main.cpp" "src/nn/Shape.cpp" "src/nn/layers/Linear.cpp" "src/nn/layers/ReLU.cpp" "src/nn/Model.cpp" "include/nn/losses/loss.h" "include/nn/optimizers/optimizer.h" "include/nn/losses/MSELoss.h" "src/nn/losses/MSELoss.cpp" "include/nn/optimizers/SGD.h" "src/nn/optimizer/SGD.cpp" "include/nn/Trainer.h" "src/nn/Trainer.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET FrameAI PROPERTY CXX_STANDARD 20)
endif()

# TODO: Добавьте тесты и целевые объекты, если это необходимо.
