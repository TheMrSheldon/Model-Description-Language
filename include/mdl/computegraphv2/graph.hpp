#ifndef MDL_COMPUTEGRAPH_GRAPH_HPP
#define MDL_COMPUTEGRAPH_GRAPH_HPP

#include "./builtins.hpp"
#include "./node.hpp"

#include <expected>
#include <list>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace mdl {
	struct FunctionDecl;

	class Computegraph final {
	private:
		Computegraph(Computegraph&) = delete;
		Computegraph& operator=(Computegraph&) = delete;

	public:
		using NodeRef = std::reference_wrapper<const Node>;
		/** The output nodes of an inlined function call (one entry per function output). */
		using Outputs = std::vector<NodeRef>;

		Computegraph();
		Computegraph(Computegraph&&);
		Computegraph& operator=(Computegraph&&);

		const Node& addConstNode(Atomic&& value);

		/**
		 * @brief Inlines a call of @p func with the given arguments.
		 * @details Function arguments (InputNodes) are substituted by @p arguments and template parameters by
		 * @p templateArguments. The inlined subgraph shares all nodes that do not depend on substituted nodes with the
		 * function template, and shared subexpressions within the function stay shared (the inlined graph is a DAG
		 * isomorphic to the template, not a tree). Weights (ParameterNodes) are kept, i.e., the callee must be in the
		 * same module scope as the caller.
		 *
		 * @return the output nodes of the inlined call.
		 */
		std::expected<Outputs, std::string> addInvocationNode(
				const FunctionDecl& func, std::vector<NodeRef>&& arguments, std::vector<NodeRef>&& templateArguments = {}
		);

		/**
		 * @brief Inlines a member function call on the module instance @p instanceName.
		 * @details Besides substituting the arguments, this substitutes the module's hyperparameters by the bound values
		 * and maps every weight `w` referenced by the function to the weight `instanceName.w` of the caller's scope
		 * @p scope. This yields fully qualified weight paths (e.g., `layers.3.weightQuery.W`) and guarantees that
		 * distinct instances have distinct weights while repeated calls of the same instance share them.
		 */
		std::expected<Outputs, std::string> addModuleInvocationNode(
				const BoundModuleType& boundModule, const std::string& scope, const std::string& instanceName,
				const FunctionDecl& func, std::vector<NodeRef>&& arguments
		);

		const Node& addInvocationNode(builtins::FunctionIDs func, DataType dtype, std::vector<NodeRef>&& arguments);

		/**
		 * @brief Adds an invalid node that signifies an uncompilable part of the computegraph.
		 * @details For example, if the expression `a + b` is compiled, where `a` is a valid variable but `b` is not,
		 * then `b` is represented by an error node.
		 *
		 * @return A reference to the error node.
		 */
		const Node& addErrorNode();

		/** @brief Adds a function argument. */
		const Node& addInputNode(DataType dtype, std::string name = "");

		/**
		 * @brief Adds a weight node that is not registered in any scope.
		 * @see getOrAddParameterNode for weights of modules.
		 */
		const Node& addParameterNode(DataType dtype, std::string name = "");

		/**
		 * @brief Returns the weight with the fully qualified @p path within @p scope (typically the name of the module
		 * whose functions reference the weight), creating it with type @p dtype on first use.
		 */
		const Node& getOrAddParameterNode(const std::string& scope, const std::string& path, DataType dtype);

		const Node& addHyperparameterNode(DataType dtype, std::string name = "");

		size_t getNumNodes() const;

	private:
		/**
		 * @brief Stores all nodes in this compute graph.
		 * @details std::list is used to guarantee reference stability: adding new nodes must never invalidate existing
		 * Node references or reference_wrappers held by InvocationNode arguments, FunctionDecl parameters, or any other
		 * part of the graph. std::vector would silently dangle those references on reallocation.
		 */
		std::list<Node> nodes;

		/** @brief Weights by (scope, fully qualified path); see getOrAddParameterNode. */
		std::map<std::pair<std::string, std::string>, const Node*> weights;
	};
} // namespace mdl

#endif
