#ifndef MDL_COMPUTEGRAPH_DATATYPE_HPP
#define MDL_COMPUTEGRAPH_DATATYPE_HPP

#include "../utils/overload.hpp"
#include "node_decl.hpp"

#include <expected>
#include <map>
#include <string>
#include <vector>

namespace mdl {
	struct Elipsis {};

	enum class PrimitiveType {
		error,	  /**< used to mark invalid nodes in the computegraph (e.g., due to semantic errors) **/
		floating, /**< represents arbitrary floating points. The precise datatype is compile-time configurable. */
		integer,  /**< represents arbitrary signed integers. The precise datatype is compile-time configurable. */
		uint,	  /**< represents arbitrary unsigned integers. The precise datatype is compile-time configurable. */
		boolean	  /**< represents booleans. */
	};

	struct TensorType final {
		using Shape = std::vector<std::variant<unsigned, Elipsis, const Node*>>;
		PrimitiveType type;
		Shape shape;
	};

	struct FunctionDecl;
	struct ModuleType final {
		std::vector<std::pair<std::string, std::reference_wrapper<HyperparameterNode>>> typeargs;
		std::vector<std::pair<std::string, std::reference_wrapper<ParameterNode>>> variables;
		std::map<std::string, FunctionDecl> functions;
	};

	/** \todo handle member functions? **/
	using DataType = std::variant<PrimitiveType, TensorType, struct AggregateType, std::reference_wrapper<ModuleType>>;

	struct AggregateType final {
		std::map<std::string, DataType> members;
	};

	using TypeArg = std::variant<Elipsis, Atomic, const Node*>; /** \todo define TypeArgs */

	// inline std::expected<DataType, std::string> setTypeArgs(DataType dtype, const std::vector<TypeArg>& typeargs) {
	// 	if (auto primitive = std::get_if<PrimitiveType>(&dtype)) {
	// 		TensorType::Shape shape;
	// 		for (auto& arg : typeargs) {
	// 			using RT = std::expected<TensorType::Shape::value_type, std::string>;
	// 			auto val = std::visit(
	// 					overloaded{
	// 							[](Elipsis) -> RT { return Elipsis{}; },
	// 							[](const Atomic& value) -> RT {
	// 								if (auto val = std::get_if<int>(&value)) {
	// 									return static_cast<unsigned>(*val);
	// 								} else {
	// 									return std::unexpected("Invalid type"); /** \todo more descriptive message **/
	// 								}
	// 							},
	// 							[](const Node* node) -> RT { return node; }
	// 					},
	// 					arg
	// 			);
	// 			if (val.has_value())
	// 				shape.emplace_back(*val);
	// 			else
	// 				return std::unexpected(val.error());
	// 		}
	// 		return TensorType{.type = *primitive, .shape = std::move(shape)};
	// 	} else {
	// 		return std::unexpected("Type arguments for non-primitive types not yet implemented");
	// 	}
	// }
} // namespace mdl

#endif