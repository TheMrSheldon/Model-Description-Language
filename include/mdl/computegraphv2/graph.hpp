#ifndef MDL_COMPUTEGRAPH_GRAPH_HPP
#define MDL_COMPUTEGRAPH_GRAPH_HPP

#include "./builtins.hpp"
#include "./node.hpp"
#include "./object.hpp"

#include <string>
#include <vector>

namespace mdl {
	/**
	 * @brief Represents a function definition.
	 * @details Each function is associated with a computegraph from which it references nodes.
	 */
	struct FunctionDecl final {
		std::string name;
		std::vector<std::pair<std::string, std::reference_wrapper<const HyperparameterNode>>> templateParameters;
		std::vector<std::pair<std::string, std::reference_wrapper<const ParameterNode>>> parameters;
		std::reference_wrapper<const Node> output;

		const DataType& returnType() const { return mdl::getDataType(output.get()); }
	};

	class Computegraph final {
	private:
		Computegraph(Computegraph&) = delete;
		Computegraph& operator=(Computegraph&) = delete;

	public:
		Computegraph();
		Computegraph(Computegraph&&);
		Computegraph& operator=(Computegraph&&);

		Node& addConstNode(ObjectRef&& value);

		Node& addConstNode(Atomic&& value);

		Node& addInvocationNode(
				FunctionDecl&& func, std::vector<std::reference_wrapper<Node>>&& arguments,
				std::vector<std::reference_wrapper<const Node>>&& templateArguments
		);

		Node& addInvocationNode(
				builtins::FunctionIDs func, DataType dtype, std::vector<std::reference_wrapper<const Node>>&& arguments
		);

		/**
		 * @brief Adds an invalid node that signifies an uncompilable part of the computegraph.
		 * @details For example, if the expression `a + b` is compiled, where `a` is a valid variable but `b` is not, then `b` is represented by an error node.
		 * 
		 * @return A reference to the error node.
		 */
		Node& addErrorNode();

		Node& addParameterNode(DataType dtype /** \todo parameters **/);

		Node& addHyperparameterNode(DataType dtype /** \todo parameters **/);

		size_t getNumNodes();
	};
} // namespace mdl

#endif