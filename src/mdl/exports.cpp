#include <mdl/exports.hpp>

#include "./argvals.hpp"

#include <mdl/compiler.hpp>
#include <mdl/computegraphv2/function.hpp>
#include <mdl/parserv2/compilation_unit.hpp>
#include <mdl/utils/overload.hpp>

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace mdl;

std::optional<int64_t> mdl::evalConstant(const Node& node, const ArgVals& argvals) {
	using FIDs = builtins::FunctionIDs;
	return std::visit(
			overloaded{
					[](const ConstNode& cn) -> std::optional<int64_t> {
						if (const int* v = std::get_if<int>(&cn.value))
							return *v;
						return std::nullopt;
					},
					[&](const HyperparameterNode& hn) -> std::optional<int64_t> {
						if (auto v = argvals.lookup(hn.name))
							return *v;
						return std::nullopt;
					},
					[&](const InvocationNode& inv) -> std::optional<int64_t> {
						if (inv.arguments.size() != 2)
							return std::nullopt;
						auto lhs = evalConstant(inv.arguments[0].get(), argvals);
						auto rhs = evalConstant(inv.arguments[1].get(), argvals);
						if (!lhs || !rhs)
							return std::nullopt;
						switch (inv.id) {
						case FIDs::TENSOR_ADD:
							return *lhs + *rhs;
						case FIDs::TENSOR_SUB:
							return *lhs - *rhs;
						case FIDs::TENSOR_PROD:
							return *lhs * *rhs;
						case FIDs::TENSOR_DIV:
							return *rhs != 0 ? std::optional<int64_t>{*lhs / *rhs} : std::nullopt;
						default:
							return std::nullopt;
						}
					},
					[](const auto&) -> std::optional<int64_t> { return std::nullopt; },
			},
			node
	);
}

static TensorSignature makeSignature(
		std::string name, const DataType& dtype, const Node* node, const ArgVals& argvals, unsigned ellipsisRank
) {
	TensorSignature sig{.name = std::move(name), .dtype = PrimitiveType::error, .shape = {}, .node = node};
	std::visit(
			overloaded{
					[&](const PrimitiveType& p) { sig.dtype = p; },
					[&](const TensorType& t) {
						sig.dtype = t.type;
						for (const auto& dim : t.shape) {
							if (std::holds_alternative<Elipsis>(dim)) {
								sig.shape.insert(sig.shape.end(), ellipsisRank, -1);
								continue;
							}
							sig.shape.push_back(std::visit(
									overloaded{
											[](unsigned u) -> int64_t { return u; },
											[](Elipsis) -> int64_t { return -1; },
											[&](const Node* n) -> int64_t { return evalConstant(*n, argvals).value_or(-1); },
									},
									dim
							));
						}
					},
					[](const auto&) {},
			},
			dtype
	);
	return sig;
}

/**
 * @brief The value of a hyperparameter of type @p type from the argument values at @p path (e.g. `Module.conf`):
 * a ConstNode for scalars and an AggregateInstance for structs; std::nullopt if a value is missing.
 */
static std::optional<Object>
hyperparameterValue(const CompilationUnit& unit, const DataType& type, const std::string& path, const ArgVals& argvals) {
	if (std::holds_alternative<PrimitiveType>(type)) {
		auto value = argvals.lookupScalar(path);
		if (!value)
			return std::nullopt;
		Atomic atomic = std::visit(
				overloaded{
						[](int64_t i) { return Atomic{static_cast<int>(i)}; },
						[](double d) { return Atomic{d}; },
						[](bool b) { return Atomic{b ? 1 : 0}; },
				},
				*value
		);
		return Object{&unit.graph().addConstNode(std::move(atomic))};
	}
	if (const auto* agg = std::get_if<AggregateType>(&type)) {
		std::map<std::string, Object> members;
		for (const auto& [name, memberType] : agg->members) {
			auto member = hyperparameterValue(unit, memberType, path + "." + name, argvals);
			if (!member)
				return std::nullopt;
			members.emplace(name, std::move(*member));
		}
		auto instance = AggregateInstance::create(*agg, std::move(members));
		return instance ? std::optional<Object>{Object{std::move(*instance)}} : std::nullopt;
	}
	return std::nullopt;
}

/** @brief Flattens a (concrete) hyperparameter value into (path, value) pairs. */
static void flattenHyperparameter(
		const std::string& path, const Object& value,
		std::vector<std::pair<std::string, std::variant<int64_t, double>>>& out
) {
	if (const auto* node = std::get_if<const Node*>(&value)) {
		if (const auto* cn = std::get_if<ConstNode>(*node)) {
			if (const auto* i = std::get_if<int>(&cn->value))
				out.emplace_back(path, int64_t{*i});
			else if (const auto* d = std::get_if<double>(&cn->value))
				out.emplace_back(path, *d);
		}
	} else if (const auto* agg = std::get_if<AggregateInstance>(&value)) {
		size_t i = 0;
		for (const auto& [name, _] : agg->getType().members)
			flattenHyperparameter(path + "." + name, agg->getMembers()[i++], out);
	}
}

std::vector<ExportedFunction>
mdl::exportedFunctions(const CompilationUnit& unit, const ArgVals& argvals, unsigned ellipsisRank) {
	std::vector<ExportedFunction> exports;
	for (const auto& [moduleName, member] : unit.definitions) {
		const auto* definition = std::get_if<ModuleType>(&member);
		if (definition == nullptr)
			continue;
		// Modules with hyperparameters are instantiated with the values given in the argument values; modules whose
		// hyperparameters are not (all) given are not compiled.
		const ModuleType* module = definition;
		std::vector<std::pair<std::string, std::variant<int64_t, double>>> hyperparameters;
		if (!definition->instantiated) {
			std::map<std::string, Object> values;
			bool complete = true;
			for (const auto& [pname, hn] : definition->typeargs) {
				auto value = hyperparameterValue(unit, hn.get().type, moduleName + "." + pname, argvals);
				if (!value) {
					complete = false;
					break;
				}
				flattenHyperparameter(pname, *value, hyperparameters);
				values.emplace(pname, std::move(*value));
			}
			if (!complete)
				continue;
			auto result = unit.instantiate(moduleName, values);
			if (!result)
				throw std::runtime_error(result.error());
			module = &result->get();
		}
		const auto order = weightPaths(*module);

		for (const auto& [funcName, decl] : module->functions) {
			if (!decl.templateParameters.empty())
				continue;
			ExportedFunction fn{
					.module = moduleName,
					.function = funcName,
					.symbol = mangledName(moduleName, funcName),
					.decl = &decl,
					.inputs = {},
					.weights = {},
					.outputs = {},
					.moduleWeights = order,
					.hyperparameters = hyperparameters,
			};

			// Collect the graph nodes of the arguments and all weights reachable from the outputs.
			std::unordered_map<const void*, const Node*> inputNodes;
			std::vector<const Node*> weights;
			std::unordered_set<const Node*> visited;
			std::function<void(const Node&)> collect = [&](const Node& node) {
				if (!visited.insert(&node).second)
					return;
				if (const auto* inv = std::get_if<InvocationNode>(&node)) {
					for (const auto& arg : inv->arguments)
						collect(arg.get());
				} else if (const auto* in = std::get_if<InputNode>(&node)) {
					inputNodes[in] = &node;
				} else if (std::holds_alternative<ParameterNode>(node)) {
					weights.push_back(&node);
				}
			};
			for (const auto& output : decl.outputs)
				collect(output.get());

			for (const auto& [pname, input] : decl.parameters) {
				auto it = inputNodes.find(&input.get());
				fn.inputs.push_back(makeSignature(
						pname, input.get().type, it == inputNodes.end() ? nullptr : it->second, argvals, ellipsisRank
				));
			}
			auto rank = [&](const Node* node) {
				const auto& name = std::get<ParameterNode>(*node).name;
				return std::pair{std::find(order.begin(), order.end(), name) - order.begin(), name};
			};
			std::sort(weights.begin(), weights.end(), [&](auto* a, auto* b) { return rank(a) < rank(b); });
			for (const auto* weight : weights) {
				const auto& pn = std::get<ParameterNode>(*weight);
				fn.weights.push_back(makeSignature(pn.name, pn.type, weight, argvals, ellipsisRank));
			}
			for (size_t i = 0; i < decl.outputs.size(); ++i) {
				const auto& out = decl.outputs[i].get();
				auto name = decl.outputs.size() == 1 ? std::string{"output"} : "output" + std::to_string(i);
				// The declared return type defines the ABI (inferred node types may be less precise).
				const DataType type = i < decl.returnTypes.size()
											  ? std::visit([](const auto& t) -> DataType { return t; }, decl.returnTypes[i])
											  : getDataType(out);
				fn.outputs.push_back(makeSignature(std::move(name), type, &out, argvals, ellipsisRank));
			}
			exports.push_back(std::move(fn));
		}
	}
	return exports;
}
