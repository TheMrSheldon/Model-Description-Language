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

static TensorSignature
makeSignature(std::string name, const DataType& dtype, const Node* node, const ArgVals& argvals) {
	TensorSignature sig{.name = std::move(name), .dtype = PrimitiveType::error, .shape = {}, .node = node};
	std::visit(
			overloaded{
					[&](const PrimitiveType& p) { sig.dtype = p; },
					[&](const TensorType& t) {
						sig.dtype = t.type;
						for (const auto& dim : t.shape) {
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

std::vector<ExportedFunction> mdl::exportedFunctions(const CompilationUnit& unit, const ArgVals& argvals) {
	std::vector<ExportedFunction> exports;
	for (const auto& [moduleName, member] : unit.definitions) {
		const auto* module = std::get_if<ModuleType>(&member);
		if (module == nullptr)
			continue;
		// Struct-typed hyperparameters are resolved per field (InvalidDataType means the struct itself is present).
		bool resolvable = std::all_of(module->typeargs.begin(), module->typeargs.end(), [&](const auto& typearg) {
			auto res = argvals.lookup(typearg.second.get().name);
			return res.has_value() || res.error() != ArgVals::ErrorCode::NotFound;
		});
		if (!resolvable)
			continue;
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
						pname, input.get().type, it == inputNodes.end() ? nullptr : it->second, argvals
				));
			}
			auto rank = [&](const Node* node) {
				const auto& name = std::get<ParameterNode>(*node).name;
				return std::pair{std::find(order.begin(), order.end(), name) - order.begin(), name};
			};
			std::sort(weights.begin(), weights.end(), [&](auto* a, auto* b) { return rank(a) < rank(b); });
			for (const auto* weight : weights) {
				const auto& pn = std::get<ParameterNode>(*weight);
				fn.weights.push_back(makeSignature(pn.name, pn.type, weight, argvals));
			}
			for (size_t i = 0; i < decl.outputs.size(); ++i) {
				const auto& out = decl.outputs[i].get();
				auto name = decl.outputs.size() == 1 ? std::string{"output"} : "output" + std::to_string(i);
				// The declared return type defines the ABI (inferred node types may be less precise).
				const DataType type = i < decl.returnTypes.size()
											  ? std::visit([](const auto& t) -> DataType { return t; }, decl.returnTypes[i])
											  : getDataType(out);
				fn.outputs.push_back(makeSignature(std::move(name), type, &out, argvals));
			}
			exports.push_back(std::move(fn));
		}
	}
	return exports;
}
