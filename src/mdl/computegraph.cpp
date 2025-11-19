#include <mdl/computegraph.hpp>

#include <functional>
#include <vector>

#include <stdexcept> /** \todo remove */

using mdl::Computegraph;
using Value = Computegraph::Value;
using Node = Computegraph::Node;
using Operation = Computegraph::Operation;

/*static const std::function<Value(std::vector<Node>)> operations[]{
		[static_cast<size_t>(Operation::Const)] = [](std::vector<Node>) { assert(false); }, //
		[static_cast<size_t>(Operation::ADD)] = [](std::vector<Node> in) {},				//
		[static_cast<size_t>(Operation::SUB)] = [](std::vector<Node> in) {},
};*/

size_t Computegraph::getNumNodes() const { return nodes.size(); }

const Node& Computegraph::getNode(size_t idx) const { return nodes[idx]; }

Computegraph& Computegraph::addSubgraph() { return subgraphs.emplace_back(); }

void Computegraph::setOutputNode() {
	/** \todo implement **/
	throw std::runtime_error{"Not implemented"};
}

size_t Computegraph::getNumInputNodes() const {
	/** \todo implement **/
	throw std::runtime_error{"Not implemented"};
}
const Node& Computegraph::getInputNode(size_t idx) const {
	/** \todo implement **/
	throw std::runtime_error{"Not implemented"};
}
const Node& Computegraph::getOutputNode() const {
	/** \todo implement **/
	throw std::runtime_error{"Not implemented"};
}