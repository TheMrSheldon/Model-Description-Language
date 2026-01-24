#ifndef MDL_PARSER_DEFINITIONS_HPP
#define MDL_PARSER_DEFINITIONS_HPP

#include <expected>
#include <map>
#include <string>
#include <variant>

#include <mdl/computegraph.hpp>
#include <mdl/namespace.hpp>
#include <mdl/utils/overload.hpp>

using mdl::AggregateType;
using mdl::Computegraph;
using mdl::DataType;
using mdl::Namespace;
using mdl::Primitive;
using mdl::TensorType;

// Not a reference because ANTLR wants to copy construct them :/
using ComputegraphRef = Computegraph*;
using Node = Computegraph::Node;
using Operation = Computegraph::Operation;

using TypeArg = std::variant<mdl::Elipsis, mdl::Computegraph::Value, const Node* /** Nodes must be Arguments **/>;

extern Computegraph empty;

struct Module;
struct Struct;
struct VarDesc {
	std::string name;
	const Computegraph::Node* node;
};
struct FuncDesc {
	std::string name;
	std::vector<VarDesc> arguments;
	Computegraph* graph;
};

struct Program {
	std::map<std::string, std::variant<Module, Struct>> definitions;
};

struct Module {
	std::string name;
	Computegraph graph;
	std::vector<VarDesc> arguments;
	std::vector<VarDesc> variables;
	std::vector<FuncDesc> functions;
};

struct Struct {
	std::string name;
	AggregateType type;
};

inline std::expected<mdl::DataType, std::string>
setTypeArgs(mdl::DataType dtype, const std::vector<TypeArg>& typeargs) {
	if (auto primitive = std::get_if<mdl::Primitive>(&dtype)) {
		TensorType::Shape shape;
		for (auto& arg : typeargs) {
			using RT = std::expected<TensorType::Shape::value_type, std::string>;
			auto val = std::visit(
					mdl::overloaded{
							[](mdl::Elipsis) -> RT { return mdl::Elipsis{}; },
							[](const mdl::Computegraph::Value& value) -> RT {
								if (auto val = std::get_if<int>(&value)) {
									return static_cast<unsigned>(*val);
								} else {
									return std::unexpected("Invalid type"); /** \todo more descriptive message **/
								}
							},
							[](const mdl::Computegraph::Node* node) -> RT {
								/** \todo implement **/
								return std::unexpected("Not yet implemented");
							}
					},
					arg
			);
			if (val.has_value())
				shape.emplace_back(*val);
			else
				return std::unexpected(val.error());
		}
		return TensorType{.type = *primitive, .shape = std::move(shape)};
	} else {
		/** \todo implement */
		return dtype;
	}
}

#endif