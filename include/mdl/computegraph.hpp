#ifndef MDL_COMPUTEGRAPH_HPP
#define MDL_COMPUTEGRAPH_HPP

#include <algorithm>
#include <assert.h>
#include <functional>
#include <variant>
#include <vector>

namespace mdl {

	class Computegraph final {
	public:
		enum class Operation { Input, Argument, Const, ADD, SUB };
		struct Node {
			Operation operation;
			std::variant<std::vector<Node*>, std::variant<double, int>> value;

			const std::vector<Node*>& inputs() const {
				assert(operation == Operation::Const);
				return std::get<std::vector<Node*>>(value);
			}
		};
		using Value = std::variant<int, double>;

	private:
		std::vector<Node> nodes;
		std::vector<Computegraph> subgraphs;

		Computegraph(const Computegraph& other) = delete;
		Computegraph& operator=(const Computegraph& other) = delete;

	public:
		Computegraph() {}
		Computegraph(Computegraph&& other) : nodes(std::move(other.nodes)) {}
		~Computegraph() = default;

		Computegraph& operator=(Computegraph&& other) {
			nodes = std::move(other.nodes);
			return *this;
		}

		const Node& addArgumentNode() {
			return nodes.emplace_back(Node{.operation = Operation::Argument, .value = {}});
		}

		const Node& addInputNode() { return nodes.emplace_back(Node{.operation = Operation::Input, .value = {}}); }

		const Node& addConstNode(std::variant<double, int> value) {
			return nodes.emplace_back(Node{.operation = Operation::Const, .value = {std::move(value)}});
		}

		const Node& addOpNode(Operation operation, std::vector<Node*>&& inputs) noexcept {
			assert(operation != Operation::Const);
			// TODO: if all inputs are const, evaluate and add const node instead of opnode.
			/*auto allconst = std::all_of(inputs.begin(), inputs.end(), [](const Node* n) {
				return n->operation == Operation::Const;
			});*/
			/*if (allconst) {
				return nodes.emplace_back(
						Node{.operation = Operation::Const, .value = getOp(operation)(std::move(inputs))}
				);
			} else {*/
			return nodes.emplace_back(Node{.operation = operation, .value = {std::move(inputs)}});
			//}
		}

		Computegraph& addSubgraph();

		void setOutputNode();

		size_t getNumInputNodes() const;
		const Node& getInputNode(size_t idx) const;
		const Node& getOutputNode() const;

		size_t getNumNodes() const;
		const Node& getNode(size_t idx) const;
	};

} // namespace mdl

#endif