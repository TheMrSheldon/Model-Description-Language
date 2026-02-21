#include <mdl/computegraphv2/node.hpp>

#include <mdl/computegraphv2/datatype.hpp>

#include <vector>

using namespace mdl;

struct BaseNode {
private:
	DataType type;
	bool constevalable;

protected:
	BaseNode() = default;

public:
	/**
     * @brief Get the data type of this node.
     */
	DataType datatype() const { return type; }

	/**
     * @brief Whether this node can be evaluated at compile time.
     */
	bool isConsteval() const { return constevalable; }
};

struct mdl::ConstNode : BaseNode {
	Atomic value;
};

struct mdl::InvocationNode : BaseNode {
	std::vector<std::reference_wrapper<Node>> arguments;
};

struct ErrorNode {};

struct ParameterNode {};

struct HyperparameterNode {};

DataType mdl::getDataType(const Node& node) { return std::visit(BaseNode::datatype, node); }

bool mdl::isCompiletimeEvalable(const Node& node) { return std::visit(BaseNode::isConsteval, node); }