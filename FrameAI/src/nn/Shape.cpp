#include "../../include/nn/Shape.h"

Shape::Shape(std::initializer_list<size_t> dimensions)
	: dimensions(dimensions)
{
}

size_t Shape::rank() const {
	return this->dimensions.size();
}

size_t Shape::operator[](size_t index) const {
	return this->dimensions.at(index);
}