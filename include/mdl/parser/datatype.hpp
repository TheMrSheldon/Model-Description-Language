#ifndef MDL_PARSER_DATATYPE_HPP
#define MDL_PARSER_DATATYPE_HPP

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace mdl {
	struct Elipsis {};

	enum class Primitive { floating, integer, uint, boolean };

	struct TensorType final {
		using Shape = std::vector<std::variant<unsigned, Elipsis>>;
		Primitive type;
		Shape shape;
	};

	using DataType = std::variant<Primitive, TensorType, struct AggregateType>;

	struct AggregateType final {
		std::map<std::string, DataType> members;
	};
} // namespace mdl

#endif