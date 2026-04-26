#include <mdl/computegraphv2/graph.hpp>

#include <mdl/computegraphv2/function.hpp>

#include <cassert>
#include <format>
#include <unordered_map>

using namespace mdl;
using mdl::builtins::FunctionIDs;

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
		return graph.addInvocationNode(invn->id, invn->type, std::move(newArgs));
	} else if (std::holds_alternative<ErrorNode>(templateNode) || std::holds_alternative<ConstNode>(templateNode))
		return templateNode;

	assert(false); /** \todo handle HyperparameterNode inlining **/
	return graph.addErrorNode();
}

std::expected<std::reference_wrapper<const Node>, std::string> Computegraph::addInvocationNode(
		const FunctionDecl& func, std::vector<std::reference_wrapper<const Node>>&& arguments,
		std::vector<std::reference_wrapper<const Node>>&& templateArguments
) {
	if (func.parameters.empty())
		return std::cref(func.output.get());
	if (func.parameters.size() != arguments.size())
		return std::unexpected(std::format(
				"'{}' expects {} argument(s) but got {}", func.name, func.parameters.size(), arguments.size()
		));

	std::unordered_map<const ParameterNode*, const Node*> subs;
	for (size_t i = 0; i < func.parameters.size(); ++i)
		subs[&func.parameters[i].second.get()] = &arguments[i].get();

	return std::cref(inlineNode(*this, func.output.get(), subs));
}

const Node& Computegraph::addErrorNode() {
	nodes.emplace_back(ErrorNode{});
	return nodes.back();
}

const Node& Computegraph::addParameterNode(DataType dtype) {
	return nodes.emplace_back(ParameterNode{.type = std::move(dtype), .constevalable = false, .name = {}});
}

const Node& Computegraph::addHyperparameterNode(DataType dtype) {
	return nodes.emplace_back(HyperparameterNode{.type = std::move(dtype), .constevalable = true, .name = {}});
}

size_t Computegraph::getNumNodes() { return nodes.size(); }
