#ifndef MDL_PARSER_DATATYPE_HPP
#define MDL_PARSER_DATATYPE_HPP

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace mdl {
	struct Node;
	struct Module;

	struct Elipsis {};

	enum class Primitive {
		error /**< used to mark invalid nodes in the computegraph (e.g., due to semantic errors) **/,
		floating,
		integer,
		uint,
		boolean
	};

	struct TensorType final {
		using Shape = std::vector<std::variant<unsigned, Elipsis, const Node*>>;
		Primitive type;
		Shape shape;
	};

	using DataType = std::variant<Primitive, TensorType, struct AggregateType, std::reference_wrapper<const Module>>;

	struct AggregateType final {
		std::map<std::string, DataType> members;
	};
} // namespace mdl

#endif