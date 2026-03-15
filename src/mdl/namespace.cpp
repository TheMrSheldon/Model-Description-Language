#include <mdl/namespace.hpp>

#include <mdl/computegraphv2/builtins.hpp>
#include <mdl/computegraphv2/function.hpp>
#include <mdl/computegraphv2/graph.hpp>
#include <mdl/computegraphv2/object.hpp>

using namespace std::literals;

using mdl::DataType;
using mdl::Namespace;
using mdl::PrimitiveType;

static std::map<std::string, std::any> initBuiltins() {
	static mdl::Computegraph graph;
	std::map<std::string, std::any> builtins;

	{
		/** \todo template the base type? **/
		auto dtype = mdl::TensorType{.type = mdl::PrimitiveType::floating, .shape = {mdl::Elipsis{}}};
		auto& param = graph.addParameterNode(dtype);
		auto& output = graph.addInvocationNode(mdl::builtins::FunctionIDs::COS, dtype, {std::cref(param)});
		builtins.emplace(
				"__builtin_cos",
				mdl::Object(
						mdl::FunctionDecl{
								.name = "__builtin_cos",
								.templateParameters = {},
								.parameters = {std::pair{"x"s, std::cref(std::get<mdl::ParameterNode>(param))}},
								.output = std::cref(output)
						}
				)
		);
	}
	{
		/** \todo template the base type? **/
		auto dtype = mdl::TensorType{.type = mdl::PrimitiveType::floating, .shape = {mdl::Elipsis{}}};
		auto& param = graph.addParameterNode(dtype);
		auto& output = graph.addInvocationNode(mdl::builtins::FunctionIDs::SIN, dtype, {std::cref(param)});
		builtins.emplace(
				"__builtin_sin",
				mdl::Object(
						mdl::FunctionDecl{
								.name = "__builtin_sin",
								.templateParameters = {},
								.parameters = {std::pair{"x"s, std::cref(std::get<mdl::ParameterNode>(param))}},
								.output = std::cref(output)
						}
				)
		);
	}
	{
		/** \todo template the base type? **/
		auto dtype = mdl::TensorType{.type = mdl::PrimitiveType::floating, .shape = {mdl::Elipsis{}}};
		auto& param = graph.addParameterNode(dtype);
		auto& output = graph.addInvocationNode(mdl::builtins::FunctionIDs::TAN, dtype, {std::cref(param)});
		builtins.emplace(
				"__builtin_tan",
				mdl::Object(
						mdl::FunctionDecl{
								.name = "__builtin_tan",
								.templateParameters = {},
								.parameters = {std::pair{"x"s, std::cref(std::get<mdl::ParameterNode>(param))}},
								.output = std::cref(output)
						}
				)
		);
	}
	{
		/** \todo template the base type? **/
		auto dtype = mdl::TensorType{.type = mdl::PrimitiveType::floating, .shape = {mdl::Elipsis{}}};
		auto& param = graph.addParameterNode(dtype);
		auto& output = graph.addInvocationNode(mdl::builtins::FunctionIDs::EXP, dtype, {std::cref(param)});
		builtins.emplace(
				"__builtin_exp",
				mdl::Object(
						mdl::FunctionDecl{
								.name = "__builtin_exp",
								.templateParameters = {},
								.parameters = {std::pair{"x"s, std::cref(std::get<mdl::ParameterNode>(param))}},
								.output = std::cref(output)
						}
				)
		);
	}
	{
		/** \todo template the base type? **/
		auto dtype = mdl::TensorType{.type = mdl::PrimitiveType::floating, .shape = {mdl::Elipsis{}}};
		auto& param = graph.addParameterNode(dtype);
		auto& output = graph.addInvocationNode(mdl::builtins::FunctionIDs::ERF, dtype, {std::cref(param)});
		builtins.emplace(
				"__builtin_erf",
				mdl::Object(
						mdl::FunctionDecl{
								.name = "__builtin_erf",
								.templateParameters = {},
								.parameters = {std::pair{"x"s, std::cref(std::get<mdl::ParameterNode>(param))}},
								.output = std::cref(output)
						}
				)
		);
	}

	return builtins;
}

const std::map<std::string, std::any> Namespace::builtins = initBuiltins();

const std::map<std::string, DataType> Namespace::builtinTypes{
		{"int", PrimitiveType::integer},
		{"uint", PrimitiveType::uint},
		{"bool", PrimitiveType::boolean},
		{"float", PrimitiveType::floating}
};