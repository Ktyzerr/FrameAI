#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>
#include "Shape.h"

template <typename T>
class Tensor {
public:
    Tensor(std::initializer_list<T> data,
           Shape shape);

    Tensor(Shape shape);

    template <typename... Indices>
    T& operator()(Indices... indices);

    const T& operator[](size_t index) const;
    T& operator[](size_t index);

    Shape get_shape() const;
    size_t size() const;

    void set_data(std::initializer_list<T> data);

private:
    std::vector<T> data;
    Shape shape;
};

//==============================================================================================================================================================

template <typename T>
Tensor<T>::Tensor(std::initializer_list<T> data,
    Shape shape)
    : data(data), shape(shape)
{
    size_t size = 1;
    for (size_t i = 0; i < shape.rank(); i++) {
        size *= shape[i];
    }
    if (data.size() != size) {
        throw std::invalid_argument(
            "Tensor: data size does not match tensor shape"
        );
    }
}

template <typename T>
Tensor<T>::Tensor(Shape shape)
    : shape(shape)
{
    size_t size = 1;

    for (size_t i = 0; i < shape.rank(); i++) {
        size *= shape[i];
    }

    data.resize(size);
}

template <typename T>
template <typename... Indices>
T& Tensor<T>::operator()(Indices... indices)
{
    std::vector<size_t> indexes{ static_cast<size_t>(indices)... };

    size_t shape_rank = shape.rank();
    if (indexes.size() != shape_rank) {
        throw std::invalid_argument("Tensor::operator(): number of indices does not match tensor rank");
    }
    size_t data_index = 0;
    for (size_t i = 0; i < shape_rank; ++i) {
        if (indexes[i] >= shape[i]) {
            throw std::out_of_range(
                "Tensor::operator(): index is out of range"
            );
        }
        size_t multiplier = 1;
        for (size_t j = i + 1; j < shape_rank; ++j) {
            multiplier *= shape[j];
        }
        data_index += indexes[i] * multiplier;
    }
    return data[data_index];
}

template <typename T>
T& Tensor<T>::operator[](size_t index) {
    return data.at(index);
}

template <typename T>
const T& Tensor<T>::operator[](size_t index) const {
    return data.at(index);
}

template <typename T>
void Tensor<T>::set_data(std::initializer_list<T> data) {
    if (this->data.size() == data.size()) {
        this->data = data;
        return;
    }
    throw std::invalid_argument(
        "Tensor::set_data: data size does not match tensor shape"
    );
}

template <typename T>
size_t Tensor<T>::size() const {
    return data.size();
}

template <typename T>
Shape Tensor<T>::get_shape() const {
    return shape;
}