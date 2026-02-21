#ifndef MDL_COMPUTEGRAPH_NODE_HPP
#define MDL_COMPUTEGRAPH_NODE_HPP

#include <variant>

namespace mdl {
	struct DataType;

	/**
     * @brief "Const nodes" represent constant values in the compute graph. They consist of a data type and an atomic value.
     * @see Atomic
     */
	struct ConstNode;
	/**
     * @brief "Invocation nodes" represent function calls in the compute graph. They consist of a function reference and a list of argument nodes.
     * @details Member functions and operators are represented as invocation nodes with a function reference to the corresponding function and the object on which the member function is called as the first argument.
     */
	struct InvocationNode;
	/**
     * @brief "Error nodes" represent invalid nodes in the compute graph (e.g., due to semantic errors).
     * @details Computegraphs with these nodes should not be processed further and should be rejected by the compiler. These are used to simplify error handling within the compiler, as they can be generated at any point during graph construction.
     */
	struct ErrorNode;
	/**
     * @brief "Parameter nodes" represent member variables.
     */
	struct ParameterNode;
	/**
     * @brief "Hyperparameter nodes" repreesent "template parameters", i.e., compile-time constant variables (e.g., module arguments and function template arguments).
     */
	struct HyperparameterNode;

	/**
     * @brief An atomic value.
     */
	using Atomic = std::variant<int, double>;

	/**
     * @brief A node in the compute graph.
     */
	using Node = std::variant<ConstNode, InvocationNode, ErrorNode, ParameterNode, HyperparameterNode>;

	/**
     * @brief Get the data type of a node.
     */
	DataType getDataType(const Node& node);
	/**
	 * @brief Returns true iff the expression represented by \p node can be evaluate at compile time.
	 * @details An expression can be evaluated at compile time if it consists only of arithmetic expressions, constants and arguments.
	 * 
	 * @param node 
	 * @return true iff the expression represented by \p node can be evaluate at compile time.
	 */
	bool isCompiletimeEvalable(const Node& node);
} // namespace mdl

#endif