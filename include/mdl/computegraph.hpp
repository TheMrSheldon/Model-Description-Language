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
		struct Node {
			using ConstValue = std::variant<double, int>;
			using Value = std::variant<std::vector<const Node*>, ConstValue, std::string>;
			Operation operation;
			Value value;
			DataType dtype;

			const std::vector<const Node*>& inputs() const {
				assert(operation != Operation::Const);
				return std::get<std::vector<const Node*>>(value);
			}

			const std::string& name() const {
				assert(operation == Operation::Input || operation == Operation::Argument);
				return std::get<std::string>(value);
			}

			const ConstValue& constValue() const {
				assert(operation == Operation::Const);
				return std::get<ConstValue>(value);
			}
		};
		using Value = std::variant<int, double>;

	private:
		std::vector<std::unique_ptr<Node>> nodes;
		std::vector<Computegraph> subgraphs;
		const Node* outputNode;

		Computegraph(const Computegraph& other) = delete;
		Computegraph& operator=(const Computegraph& other) = delete;

	public:
		Computegraph() : nodes{}, subgraphs{}, outputNode{nullptr} {}
		Computegraph(Computegraph&& other)
				: nodes(std::move(other.nodes)), subgraphs{std::move(other.subgraphs)}, outputNode{other.outputNode} {}
		~Computegraph() = default;

		Computegraph& operator=(Computegraph&& other) {
			nodes = std::move(other.nodes);
			subgraphs = std::move(other.subgraphs);
			outputNode = std::move(other.outputNode);
			return *this;
		}

		const Node& addArgumentNode(std::string_view name, DataType dtype) {
			return *nodes.emplace_back(
					std::make_unique<Node>(
							Node{.operation = Operation::Argument, .value = std::string{name}, .dtype = dtype}
					)
			);
		}

		const Node& addInputNode(std::string_view name, DataType dtype) {
			return *nodes.emplace_back(
					std::make_unique<Node>(
							Node{.operation = Operation::Input, .value = std::string{name}, .dtype = dtype}
					)
			);
		}

		const Node& addConstNode(std::variant<double, int> value) {
			return *nodes.emplace_back(
					std::make_unique<Node>(Node{.operation = Operation::Const, .value = {std::move(value)}})
			);
		}

		const Node& addOpNode(Operation operation, std::vector<const Node*>&& inputs) noexcept {
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
			return *nodes.emplace_back(
					std::make_unique<Node>(Node{.operation = operation, .value = {std::move(inputs)}})
			);
			//}
		}

		Computegraph& addSubgraph();

		void setOutputNode(const Node* node);

		size_t getNumInputNodes() const;
		const Node& getInputNode(size_t idx) const;
		const Node* getOutputNode() const;

		size_t getNumNodes() const;
		const Node& getNode(size_t idx) const;
	};

} // namespace mdl

#endif