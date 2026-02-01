#ifndef MDL_COMPUTEGRAPH_HPP
#define MDL_COMPUTEGRAPH_HPP

#include <algorithm>
#include <assert.h>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "parser/datatype.hpp"

namespace mdl {

	class Computegraph final {
	public:
		enum class Operation {
			Input /**< Inputs to functions **/,
			Argument /**< Hyperparameters or trainable weights **/,
			Const,
			ADD,
			SUB,
			MULT,
			DIV,
			MATMULT
		};
		struct Node;
		using Value = std::variant<int, double>;

	private:
		std::vector<std::unique_ptr<Node>> nodes;
		std::vector<Computegraph> subgraphs;
		const Node* outputNode;

		Computegraph(const Computegraph& other) = delete;
		Computegraph& operator=(const Computegraph& other) = delete;

	public:
		Computegraph();
		Computegraph(Computegraph&& other);
		~Computegraph();

		Computegraph& operator=(Computegraph&& other);

		/**
		 * @brief Adds an invalid node that signifies an uncompilable part of the computegraph.
		 * @details For example, if the expression `a + b` is compiled, where `a` is a valid variable but `b` is not, then `b` is represented by an error node.
		 * 
		 * @return A reference to the error node.
		 */
		const Node& addErrorNode();

		const Node& addArgumentNode(std::string_view name, DataType dtype);

		const Node& addInputNode(std::string_view name, DataType dtype);

		const Node& addConstNode(Value value);

		const Node& addOpNode(Operation operation, std::vector<const Node*>&& inputs) noexcept;

		Computegraph& addSubgraph();

		void setOutputNode(const Node* node);

		size_t getNumInputNodes() const;
		const Node& getInputNode(size_t idx) const;
		const Node* getOutputNode() const;

		size_t getNumNodes() const;
		const Node& getNode(size_t idx) const;
	};

	const std::string& getName(const Computegraph::Node& node);
	Computegraph::Operation getOperation(const Computegraph::Node& node);
	DataType getDataType(const Computegraph::Node& node);
	Computegraph::Value getConst(const Computegraph::Node& node);
	const std::vector<const mdl::Computegraph::Node*>& getInputs(const Computegraph::Node& node);
	/**
	 * @brief Returns true iff the expression represented by \p node can be evaluate at compile time.
	 * @details An expression can be evaluated at compile time if it consists only of arithmetic expressions, constants and arguments.
	 * 
	 * @param node 
	 * @return true iff the expression represented by \p node can be evaluate at compile time.
	 */
	bool isCompiletimeEvalable(const Computegraph::Node& node);
} // namespace mdl

#endif