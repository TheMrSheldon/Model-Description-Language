#include <optional>
#include <mdl/namespace.hpp>

#include <mdl/computegraphv2/builtins.hpp>
#include <mdl/computegraphv2/function.hpp>
#include <mdl/computegraphv2/graph.hpp>
#include <mdl/computegraphv2/object.hpp>

#include <utility>
#include <vector>

using mdl::DataType;
using mdl::Namespace;
using mdl::PrimitiveType;

/**
 * @brief Creates the `__builtin_*` functions that expose builtin operations (FunctionIDs) to the language.
 * @details Each builtin takes a float tensor `x` (plus optional scalar arguments that are not forwarded to the builtin
 * operation) and returns the result of the operation.
 */
static std::map<std::string, std::any> initBuiltins() {
	using mdl::builtins::FunctionIDs;
	static mdl::Computegraph graph;

	struct Builtin {
		const char* name;
		FunctionIDs id;
		std::vector<std::pair<const char*, DataType>> extraParams = {}; ///< Parameters after `x`
		bool extraParamsAreArguments = false; ///< Whether the extra parameters are arguments of the builtin
		std::optional<DataType> xType = std::nullopt; ///< Type of `x` (default: a float tensor)
	};
	/** \todo template the base type? **/
	const auto tensor = mdl::TensorType{.type = PrimitiveType::floating, .shape = {mdl::Elipsis{}}};
	const auto intTensor = mdl::TensorType{.type = PrimitiveType::integer, .shape = {mdl::Elipsis{}}};
	const Builtin table[] = {
			{"__builtin_max", FunctionIDs::MAX},
			{"__builtin_min", FunctionIDs::MIN},
			{"__builtin_mean", FunctionIDs::MEAN},
			{"__builtin_sum", FunctionIDs::SUM},
			{"__builtin_cos", FunctionIDs::COS},
			{"__builtin_sin", FunctionIDs::SIN},
			{"__builtin_tan", FunctionIDs::TAN},
			{"__builtin_exp", FunctionIDs::EXP},
			{"__builtin_erf", FunctionIDs::ERF},
			{"__builtin_sqrt", FunctionIDs::SQRT},
			{"__builtin_rsqrt", FunctionIDs::RSQRT},
			{"__builtin_log", FunctionIDs::LOG},
			{"__builtin_tanh", FunctionIDs::TANH},
			{"__builtin_var", FunctionIDs::VAR, {{"dof", PrimitiveType::uint}}},
			{"__builtin_sum_like", FunctionIDs::COLLAPSE_SUM_LIKE, {{"like", tensor}}, true},
			{"__builtin_broadcast_like", FunctionIDs::BROADCAST_LIKE, {{"like", tensor}}, true},
			{"__builtin_stop_gradient", FunctionIDs::STOP_GRADIENT},
			{"__builtin_sum_all", FunctionIDs::SUM_ALL},
			{"__builtin_mean_all", FunctionIDs::MEAN_ALL},
			{"__builtin_take", FunctionIDs::TAKE, {{"ids", intTensor}}, true},
			{"__builtin_one_hot_like", FunctionIDs::ONE_HOT_LIKE, {{"like", tensor}}, true, intTensor},
	};

	std::map<std::string, std::any> builtins;
	for (const auto& builtin : table) {
		const auto& x = graph.addInputNode(builtin.xType.value_or(tensor), "x");
		std::vector<std::pair<std::string, std::reference_wrapper<const mdl::InputNode>>> params{
				{"x", std::cref(std::get<mdl::InputNode>(x))}
		};
		std::vector<std::reference_wrapper<const mdl::Node>> arguments{std::cref(x)};
		for (const auto& [pname, ptype] : builtin.extraParams) {
			const auto& param = graph.addInputNode(ptype, pname);
			params.emplace_back(pname, std::cref(std::get<mdl::InputNode>(param)));
			if (builtin.extraParamsAreArguments)
				arguments.push_back(std::cref(param));
		}
		const auto& output = graph.addInvocationNode(builtin.id, tensor, std::move(arguments));
		builtins.emplace(
				builtin.name, mdl::Object(
									  mdl::FunctionDecl{
											  .name = builtin.name,
											  .templateParameters = {},
											  .parameters = std::move(params),
											  .outputs = {std::cref(output)}
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
