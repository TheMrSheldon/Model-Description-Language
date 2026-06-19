#include <mdl/computegraphv2/graph.hpp>

#include <mdl/computegraphv2/function.hpp>

#include <cassert>
#include <format>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mdl;
using mdl::builtins::FunctionIDs;

/** \todo this must be refactored so that shape inference is supported based on the Function Signatures */
/**
 * @brief Infers the output DataType of a builtin operation from its actual argument types.
 * @details Used during function inlining so that the cloned InvocationNodes carry the
 * correct, narrowed type instead of the generic float[...] placeholder that the
 * static operator FunctionDecls use.
 */
static DataType inferOutputType(FunctionIDs id, const std::vector<std::reference_wrapper<const Node>>& args) {
	auto getTensor = [](const Node& n) -> const TensorType* { return std::get_if<TensorType>(&getDataType(n)); };
	auto fallback = []() -> DataType { return TensorType{PrimitiveType::floating, {Elipsis{}}}; };

	switch (id) {
	// Element-wise binary ops: result has the same shape as the tensor operand.
	// If lhs is a tensor, return lhs; otherwise return rhs (scalar op tensor broadcast).
	case FunctionIDs::TENSOR_ADD:
	case FunctionIDs::TENSOR_SUB:
	case FunctionIDs::TENSOR_PROD:
	case FunctionIDs::TENSOR_DIV:
		if (args.size() >= 1 && getTensor(args[0].get()))
			return getDataType(args[0].get());
		if (args.size() >= 2 && getTensor(args[1].get()))
			return getDataType(args[1].get());
		return fallback();

	// Matrix multiply: A[..., m, k] @ B[..., k, n] -> [..., m, n]
	// Replace the last dim of lhs with the last dim of rhs.
	case FunctionIDs::TENSOR_DOT:
		if (args.size() >= 2) {
			const auto* lhsT = getTensor(args[0].get());
			const auto* rhsT = getTensor(args[1].get());
			if (lhsT && rhsT && lhsT->shape.size() >= 2 && !rhsT->shape.empty()) {
				TensorType::Shape s = lhsT->shape;
				s.back() = rhsT->shape.back();
				return TensorType{lhsT->type, std::move(s)};
			}
			if (lhsT && !lhsT->shape.empty())
				return *lhsT;
		}
		return fallback();

	// Reduction ops: remove the last shape dimension.
	// If shape has only one element (e.g. a bare Elipsis{}), keep it as-is.
	case FunctionIDs::MAX:
	case FunctionIDs::MIN:
	case FunctionIDs::MEAN:
	case FunctionIDs::SUM:
		if (!args.empty()) {
			if (const auto* t = getTensor(args[0].get())) {
				if (t->shape.size() >= 2) {
					TensorType::Shape s = t->shape;
					s.pop_back();
					return TensorType{t->type, std::move(s)};
				}
			}
		}
		return fallback();

	// Element-wise unary ops: preserve the tensor shape; scalar in → float scalar out.
	case FunctionIDs::COS:
	case FunctionIDs::SIN:
	case FunctionIDs::TAN:
	case FunctionIDs::EXP:
	case FunctionIDs::ERF:
	case FunctionIDs::SQRT:
	case FunctionIDs::RSQRT:
		if (!args.empty()) {
			if (getTensor(args[0].get()))
				return getDataType(args[0].get());
			if (std::holds_alternative<PrimitiveType>(getDataType(args[0].get())))
				return PrimitiveType::floating;
		}
		return fallback();

	// VAR returns a scalar of the same element type.
	case FunctionIDs::VAR:
		if (!args.empty()) {
			if (const auto* t = getTensor(args[0].get()))
				return TensorType{t->type, {}};
		}
		return TensorType{PrimitiveType::floating, {}};

	// PERMUTE: reorder shape dimensions according to the permutation vector.
	// Supports trailing-only permutations (perm.size() < rank): leading dims are kept.
	case FunctionIDs::PERMUTE:
		if (args.size() >= 2) {
			const auto* t = getTensor(args[0].get());
			const auto* cn = std::get_if<ConstNode>(&args[1].get());
			if (t && cn) {
				const auto* pv = std::get_if<std::vector<int>>(&cn->value);
				if (pv) {
					int rank = static_cast<int>(t->shape.size());
					int offset = rank - static_cast<int>(pv->size());
					TensorType::Shape s;
					s.reserve(rank);
					for (int i = 0; i < offset; ++i)
						s.push_back(t->shape[i]);
					for (int idx : *pv) {
						if (idx < 0)
							idx += rank;
						s.push_back(
								(idx >= 0 && idx < rank) ? t->shape[idx] : TensorType::Shape::value_type{Elipsis{}}
						);
					}
					return TensorType{t->type, std::move(s)};
				}
			}
		}
		return fallback();

	// RESHAPE / EXPAND: type is set explicitly by the einops layer; fall through to
	// use the template type (caller keeps invn->type for these).
	default:
		return fallback();
	}
}

Computegraph::Computegraph() = default;
Computegraph::Computegraph(Computegraph&&) = default;
Computegraph& Computegraph::operator=(Computegraph&&) = default;

const Node& Computegraph::addConstNode(Atomic&& value) {
	nodes.emplace_back(ConstNode{.value = std::move(value)});
	return nodes.back();
}

const Node& Computegraph::addInvocationNode(
		FunctionIDs func, DataType dtype, std::vector<std::reference_wrapper<const Node>>&& args
) {
	bool constevalable = std::all_of(args.begin(), args.end(), isCompiletimeEvalable);

	return nodes.emplace_back(
			InvocationNode{
					.type = std::move(dtype),
					.constevalable = constevalable,
					.id = func,
					.arguments = std::move(args),
			}
	);
}

/**
 * @brief Recursively inlines a node from a function template into @p graph, substituting
 * ParameterNodes with the corresponding actual argument nodes.
 */
static const Node& inlineNode(
		Computegraph& graph, const Node& templateNode, const std::unordered_map<const ParameterNode*, const Node*>& subs
) {
	if (const auto* pn = std::get_if<ParameterNode>(&templateNode)) {
		auto it = subs.find(pn);
		if (it == subs.end())
			return templateNode; // concrete parameter from outer scope; keep as-is
		return *it->second;
	} else if (std::get_if<ConstNode>(&templateNode)) {
		return templateNode;
	} else if (const auto* invn = std::get_if<InvocationNode>(&templateNode)) {
		std::vector<std::reference_wrapper<const Node>> newArgs;
		newArgs.reserve(invn->arguments.size());
		for (const auto& arg : invn->arguments)
			newArgs.push_back(std::cref(inlineNode(graph, arg.get(), subs)));
		// RESHAPE and EXPAND carry precise output shapes set by the einops layer.
		// inferOutputType returns a 1-D fallback for these ops, which would lose
		// the split/merge shape information computed during einops processing.
		DataType outType = (invn->id == FunctionIDs::RESHAPE || invn->id == FunctionIDs::EXPAND)
			? invn->type
			: inferOutputType(invn->id, newArgs);
		return graph.addInvocationNode(invn->id, std::move(outType), std::move(newArgs));
	} else if (
			std::holds_alternative<ErrorNode>(templateNode) || std::holds_alternative<ConstNode>(templateNode) ||
			std::holds_alternative<HyperparameterNode>(templateNode)
	)
		return templateNode;
}

std::expected<std::reference_wrapper<const Node>, std::string> Computegraph::addInvocationNode(
		const FunctionDecl& func, std::vector<std::reference_wrapper<const Node>>&& arguments,
		std::vector<std::reference_wrapper<const Node>>&& templateArguments
) {
	if (func.parameters.empty())
		return std::cref(func.output.get());
	if (func.parameters.size() != arguments.size())
		return std::unexpected(
				std::format(
						"'{}' expects {} argument(s) but got {}", func.name, func.parameters.size(), arguments.size()
				)
		);

	std::unordered_map<const ParameterNode*, const Node*> subs;
	for (size_t i = 0; i < func.parameters.size(); ++i)
		subs[&func.parameters[i].second.get()] = &arguments[i].get();

	return std::cref(inlineNode(*this, func.output.get(), subs));
}

std::expected<std::reference_wrapper<const Node>, std::string> Computegraph::addModuleInvocationNode(
		const BoundModuleType& boundModule, const std::string& instanceName, const FunctionDecl& func,
		std::vector<std::reference_wrapper<const Node>>&& arguments
) {
	if (func.parameters.size() != arguments.size())
		return std::unexpected(
				std::format(
						"'{}' expects {} argument(s) but got {}", func.name, func.parameters.size(), arguments.size()
				)
		);

	// Build hyperparameter substitution: module HyperparameterNode* → bound value Node*
	std::unordered_map<const HyperparameterNode*, const Node*> hpSubs;
	const auto& mod = boundModule.definition;
	for (size_t i = 0; i < mod.typeargs.size(); ++i) {
		const auto& [name, hnRef] = mod.typeargs[i];
		auto it = boundModule.hyperparamValues.find(name);
		if (it != boundModule.hyperparamValues.end())
			hpSubs[&hnRef.get()] = it->second;
	}

	// Resolve a TensorType's shape by substituting module HyperparameterNode references
	// with the concrete bound value nodes from the outer scope.
	auto resolveType = [&](const DataType& dtype) -> DataType {
		const auto* tt = std::get_if<TensorType>(&dtype);
		if (!tt)
			return dtype;
		TensorType resolved{tt->type, {}};
		for (const auto& dim : tt->shape) {
			auto newDim = dim;
			if (const auto* nodePtr = std::get_if<const Node*>(&dim)) {
				if (const auto* hn = std::get_if<HyperparameterNode>(*nodePtr)) {
					auto it = hpSubs.find(hn);
					if (it != hpSubs.end())
						newDim = TensorType::Shape::value_type{it->second};
				}
			}
			resolved.shape.push_back(newDim);
		}
		return resolved;
	};

	std::unordered_map<const ParameterNode*, const Node*> subs;

	// Function parameter substitutions.
	for (size_t i = 0; i < func.parameters.size(); ++i)
		subs[&func.parameters[i].second.get()] = &arguments[i].get();

	// Create per-instance ParameterNodes for each module variable and add to subs.
	// This ensures weightQuery.W, weightKey.W, etc. become distinct runtime arguments
	// instead of all sharing the single W ParameterNode from the Linear type definition.
	for (const auto& [varName, varNodeRef] : mod.variables) {
		const DataType resolvedType = resolveType(varNodeRef.get().type);
		const Node& instanceNode = addParameterNode(resolvedType, instanceName + "." + varName);
		subs[&varNodeRef.get()] = &instanceNode;
	}

	return std::cref(inlineNode(*this, func.output.get(), subs));
}

const Node& Computegraph::addErrorNode() {
	nodes.emplace_back(ErrorNode{});
	return nodes.back();
}

const Node& Computegraph::addParameterNode(DataType dtype, std::string name) {
	return nodes.emplace_back(ParameterNode{.type = std::move(dtype), .constevalable = false, .name = std::move(name)});
}

const Node& Computegraph::addHyperparameterNode(DataType dtype, std::string name) {
	return nodes.emplace_back(
			HyperparameterNode{.type = std::move(dtype), .constevalable = true, .name = std::move(name)}
	);
}

size_t Computegraph::getNumNodes() { return nodes.size(); }
