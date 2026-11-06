#pragma once
#include <vector>
#include <initializer_list>


class Shape {
public:
	Shape(std::initializer_list<size_t> dimensions);

	size_t rank() const;
	size_t operator[](size_t index) const;
private:
	std::vector<size_t> dimensions;
};