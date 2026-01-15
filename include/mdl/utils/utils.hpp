#ifndef MDL_UTILS_UTILS_HPP
#define MDL_UTILS_UTILS_HPP

#include "../parser/datatype.hpp"

#include <string>

namespace std {
	std::string to_string(const mdl::Primitive& prim);
	std::string to_string(const mdl::AggregateType& agg);
	std::string to_string(const mdl::TensorType& tensortype);
	std::string to_string(const mdl::DataType& dtype);
} // namespace std

#endif