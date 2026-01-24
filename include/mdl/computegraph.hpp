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
} // namespace mdl

#endif