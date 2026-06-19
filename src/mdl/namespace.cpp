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
		auto& param = graph.addParameterNode(dtype, "x");
		auto& output = graph.addInvocationNode(mdl::builtins::FunctionIDs::MAX, dtype, {std::cref(param)});
		builtins.emplace(
				"__builtin_max",
				mdl::Object(
						mdl::FunctionDecl{
								.name = "__builtin_max",
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
		auto& param = graph.addParameterNode(dtype, "x");
		auto& output = graph.addInvocationNode(mdl::builtins::FunctionIDs::MIN, dtype, {std::cref(param)});
		builtins.emplace(
				"__builtin_min",
				mdl::Object(
						mdl::FunctionDecl{
								.name = "__builtin_min",
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
		auto& param = graph.addParameterNode(dtype, "x");
		auto& output = graph.addInvocationNode(mdl::builtins::FunctionIDs::MEAN, dtype, {std::cref(param)});
		builtins.emplace(
				"__builtin_mean",
				mdl::Object(
						mdl::FunctionDecl{
								.name = "__builtin_mean",
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
		auto& param = graph.addParameterNode(dtype, "x");
		auto& output = graph.addInvocationNode(mdl::builtins::FunctionIDs::SUM, dtype, {std::cref(param)});
		builtins.emplace(
				"__builtin_sum",
				mdl::Object(
						mdl::FunctionDecl{
								.name = "__builtin_sum",
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
		auto& param = graph.addParameterNode(dtype, "x");
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
		auto& param = graph.addParameterNode(dtype, "x");
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
		auto& param = graph.addParameterNode(dtype, "x");
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
		auto& param = graph.addParameterNode(dtype, "x");
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
		auto& param = graph.addParameterNode(dtype, "x");
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
	{
		/** \todo template the base type? **/
		auto dtype = mdl::TensorType{.type = mdl::PrimitiveType::floating, .shape = {mdl::Elipsis{}}};
		auto& param = graph.addParameterNode(dtype, "x");
		auto& output = graph.addInvocationNode(mdl::builtins::FunctionIDs::SQRT, dtype, {std::cref(param)});
		builtins.emplace(
				"__builtin_sqrt",
				mdl::Object(
						mdl::FunctionDecl{
								.name = "__builtin_sqrt",
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
		auto& param = graph.addParameterNode(dtype, "x");
		auto& output = graph.addInvocationNode(mdl::builtins::FunctionIDs::RSQRT, dtype, {std::cref(param)});
		builtins.emplace(
				"__builtin_rsqrt",
				mdl::Object(
						mdl::FunctionDecl{
								.name = "__builtin_rsqrt",
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
		auto& param = graph.addParameterNode(dtype, "x");
		auto& dofParam = graph.addParameterNode(mdl::PrimitiveType::uint, "dof");
		auto& output = graph.addInvocationNode(mdl::builtins::FunctionIDs::VAR, dtype, {std::cref(param)});
		builtins.emplace(
				"__builtin_var",
				mdl::Object(
						mdl::FunctionDecl{
								.name = "__builtin_var",
								.templateParameters = {},
								.parameters =
										{std::pair{"x"s, std::cref(std::get<mdl::ParameterNode>(param))},
										 std::pair{"dof"s, std::cref(std::get<mdl::ParameterNode>(dofParam))}},
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