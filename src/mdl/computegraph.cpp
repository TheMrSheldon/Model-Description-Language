#include <mdl/computegraph.hpp>

#include <mdl/utils/overload.hpp>

#include <functional>
#include <vector>

#include <stdexcept> /** \todo remove */

using mdl::Computegraph;
using Value = Computegraph::Value;
using mdl::Node;
using Operation = Computegraph::Operation;

struct mdl::Node {
	using NValue = std::variant<std::vector<const Node*>, Value, std::string>;
	Operation operation;
	NValue value;
	mdl::DataType dtype;
	bool isConsteval;

	const std::vector<const Node*>& inputs() const {
		assert(operation != Operation::Const);
		return std::get<std::vector<const Node*>>(value);
	}

	const std::string& name() const {
		assert(operation == Operation::Input || operation == Operation::Argument);
		return std::get<std::string>(value);
	}

	const Value& constValue() const {
		assert(operation == Operation::Const);
		return std::get<Value>(value);
	}
};

Computegraph::Computegraph() : nodes{}, subgraphs{}, outputNode{nullptr} {}
Computegraph::Computegraph(Computegraph&& other)
		: nodes(std::move(other.nodes)), subgraphs{std::move(other.subgraphs)}, outputNode{other.outputNode} {}
Computegraph::~Computegraph() = default;

Computegraph& Computegraph::operator=(Computegraph&& other) {
	nodes = std::move(other.nodes);
	subgraphs = std::move(other.subgraphs);
	outputNode = std::move(other.outputNode);
	return *this;
}

size_t Computegraph::getNumNodes() const { return nodes.size(); }

const Node& Computegraph::getNode(size_t idx) const { return *nodes[idx]; }

Computegraph& Computegraph::addSubgraph() { return subgraphs.emplace_back(); }

const Node& Computegraph::addErrorNode() {
	return *nodes.emplace_back(
			std::make_unique<Node>(
					Node{.operation = Operation::Const,
						 .value = Value{},
						 .dtype = DataType{Primitive::error},
						 .isConsteval = true}
			)
	);
}

const Node& Computegraph::addArgumentNode(std::string_view name, DataType dtype) {
	return *nodes.emplace_back(
			std::make_unique<Node>(Node{
					.operation = Operation::Argument, .value = std::string{name}, .dtype = dtype, .isConsteval = true
			})
	);
}

const Node& Computegraph::addInputNode(std::string_view name, DataType dtype) {
	return *nodes.emplace_back(
			std::make_unique<Node>(Node{
					.operation = Operation::Input, .value = std::string{name}, .dtype = dtype, .isConsteval = false
			})
	);
}

const Node& Computegraph::addConstNode(Value value) {
	auto dtype = std::visit(
			mdl::overloaded{
					[](int) { return DataType{Primitive::integer}; },
					[](double) { return DataType{Primitive::floating}; }
			},
			value
	);
	return *nodes.emplace_back(
			std::make_unique<Node>(Node{
					.operation = Operation::Const, .value = {std::move(value)}, .dtype = dtype, .isConsteval = true
			})
	);
}

const Node& Computegraph::addOpNode(Operation operation, std::vector<const Node*>&& inputs) noexcept {
	assert(operation != Operation::Const && operation != Operation::Input && operation != Operation::Argument);
	/** \todo infer datatype **/
	const bool isConsteval = std::all_of(inputs.begin(), inputs.end(), [](auto i) { return i->isConsteval; });
	return *nodes.emplace_back(
			std::make_unique<Node>(
					Node{.operation = operation, .value = {std::move(inputs)}, .isConsteval = isConsteval}
			)
	);
}

void Computegraph::setOutputNode(const Node* node) { outputNode = node; }

size_t Computegraph::getNumInputNodes() const {
	/** \todo implement **/
	throw std::runtime_error{"Not implemented"};
}
const Node& Computegraph::getInputNode(size_t idx) const {
	/** \todo implement **/
	throw std::runtime_error{"Not implemented"};
}
const Node* Computegraph::getOutputNode() const { return outputNode; }

const std::string& mdl::getName(const Computegraph::Node& node) { return node.name(); }
mdl::Computegraph::Operation mdl::getOperation(const Computegraph::Node& node) { return node.operation; }
mdl::DataType mdl::getDataType(const Computegraph::Node& node) { return node.dtype; }
mdl::Computegraph::Value mdl::getConst(const Computegraph::Node& node) { return node.constValue(); }
const std::vector<const mdl::Computegraph::Node*>& mdl::getInputs(const Computegraph::Node& node) {
	return node.inputs();
}

bool mdl::isCompiletimeEvalable(const Computegraph::Node& node) { return node.isConsteval; }