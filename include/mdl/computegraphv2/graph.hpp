#ifndef MDL_COMPUTEGRAPH_GRAPH_HPP
#define MDL_COMPUTEGRAPH_GRAPH_HPP

#include "./builtins.hpp"
#include "./node.hpp"

#include <list>
#include <string>
#include <vector>

namespace mdl {
	struct FunctionDecl;

	class Computegraph final {
	private:
		Computegraph(Computegraph&) = delete;
		Computegraph& operator=(Computegraph&) = delete;

	public:
		Computegraph();
		Computegraph(Computegraph&&);
		Computegraph& operator=(Computegraph&&);

		const Node& addConstNode(Atomic&& value);

		const Node& addInvocationNode(
				const FunctionDecl& func, std::vector<std::reference_wrapper<const Node>>&& arguments,
				std::vector<std::reference_wrapper<const Node>>&& templateArguments = {}
		);

		const Node& addInvocationNode(
				builtins::FunctionIDs func, DataType dtype, std::vector<std::reference_wrapper<const Node>>&& arguments
		);

		/**
		 * @brief Adds an invalid node that signifies an uncompilable part of the computegraph.
		 * @details For example, if the expression `a + b` is compiled, where `a` is a valid variable but `b` is not,
		 * then `b` is represented by an error node.
		 * 
		 * @return A reference to the error node.
		 */
		const Node& addErrorNode();

		const Node& addParameterNode(DataType dtype /** \todo parameters **/);

		const Node& addHyperparameterNode(DataType dtype /** \todo parameters **/);

		size_t getNumNodes();

	private:
		/**
		 * @brief Stores all nodes in this compute graph.
		 * @details std::list is used to guarantee reference stability: adding new nodes must never invalidate existing
		 * Node references or reference_wrappers held by InvocationNode arguments, FunctionDecl parameters, or any other
		 * part of the graph. std::vector would silently dangle those references on reallocation.
		 */
		std::list<Node> nodes;
	};
} // namespace mdl

#endif