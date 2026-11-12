#include <mdl/passes/autodiff.hpp>

#include <mdl/computegraphv2/function.hpp>
#include <mdl/utils/overload.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <functional>
#include <map>
#include <numbers>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace mdl::autodiff {

	namespace {
		using FIDs = builtins::FunctionIDs;
		using NodeRef = Computegraph::NodeRef;

		/** @brief Thrown (internally) when a builtin has no gradient rule. */
		struct NoGradientRule {
			FIDs id;
		};

		/** @brief Helpers to emit nodes of the gradient computation. */
		class Builder {
		public:
			explicit Builder(Computegraph& graph) : graph(graph) {}

			const Node& op(FIDs id, const DataType& type, std::vector<NodeRef> args) {
				return graph.addInvocationNode(id, type, std::move(args));
			}
			const Node& constant(double value) { return graph.addConstNode(Atomic{value}); }

			const Node& add(const Node& a, const Node& b) { return op(FIDs::TENSOR_ADD, getDataType(a), {a, b}); }
			const Node& sub(const Node& a, const Node& b) { return op(FIDs::TENSOR_SUB, getDataType(a), {a, b}); }
			const Node& mul(const Node& a, const Node& b) { return op(FIDs::TENSOR_PROD, typeOfTensor(a, b), {a, b}); }
			const Node& div(const Node& a, const Node& b) { return op(FIDs::TENSOR_DIV, typeOfTensor(a, b), {a, b}); }
			const Node& neg(const Node& a) { return op(FIDs::TENSOR_SUB, getDataType(a), {constant(0.0), a}); }
			const Node& unary(FIDs id, const Node& a) { return op(id, getDataType(a), {a}); }
			const Node& transposed(const Node& a) {
				return op(FIDs::PERMUTE, swapLastTwo(getDataType(a)), {a, graph.addConstNode(Atomic{std::vector<int>{-1, -2}})});
			}
			const Node& matmul(const Node& a, const Node& b, const DataType& type) {
				return op(FIDs::TENSOR_DOT, type, {a, b});
			}

			/** @brief Sums @p g to the shape of @p like (undoes broadcasting); identity if the shapes are equal. */
			const Node& unbroadcast(const Node& g, const Node& like) {
				if (sameShape(getDataType(g), getDataType(like)))
					return g;
				return op(FIDs::COLLAPSE_SUM_LIKE, getDataType(like), {g, like});
			}

			const Node& zerosLike(const Node& like) {
				return op(FIDs::BROADCAST_LIKE, getDataType(like), {constant(0.0), like});
			}

		private:
			Computegraph& graph;

			static DataType typeOfTensor(const Node& a, const Node& b) {
				return std::holds_alternative<TensorType>(getDataType(a)) ? getDataType(a) : getDataType(b);
			}

			static DataType swapLastTwo(const DataType& type) {
				auto tensor = std::get_if<TensorType>(&type);
				if (tensor == nullptr || tensor->shape.size() < 2)
					return type;
				TensorType result = *tensor;
				std::swap(result.shape[result.shape.size() - 1], result.shape[result.shape.size() - 2]);
				return result;
			}

			static bool sameShape(const DataType& a, const DataType& b) {
				const auto* ta = std::get_if<TensorType>(&a);
				const auto* tb = std::get_if<TensorType>(&b);
				if (ta == nullptr || tb == nullptr || ta->shape.size() != tb->shape.size())
					return false;
				for (size_t i = 0; i < ta->shape.size(); ++i) {
					const auto &da = ta->shape[i], &db = tb->shape[i];
					if (std::holds_alternative<Elipsis>(da) && std::holds_alternative<Elipsis>(db))
						continue; // all `...` of a function denote the same dims
					if (const auto *ua = std::get_if<unsigned>(&da), *ub = std::get_if<unsigned>(&db); ua && ub && *ua == *ub)
						continue;
					if (const auto *na = std::get_if<const Node*>(&da), *nb = std::get_if<const Node*>(&db); na && nb && *na == *nb)
						continue;
					return false;
				}
				return true;
			}
		};

		/** @brief The size of the last dimension of @p type as a node (for the gradient of mean), if known. */
		const Node* lastDimSize(Computegraph& graph, const DataType& type) {
			const auto* tensor = std::get_if<TensorType>(&type);
			if (tensor != nullptr && !tensor->shape.empty()) {
				const auto& dim = tensor->shape.back();
				if (const auto* u = std::get_if<unsigned>(&dim))
					return &graph.addConstNode(Atomic{static_cast<double>(*u)});
				if (const auto* n = std::get_if<const Node*>(&dim))
					return *n;
			}
			return nullptr;
		}

		/**
		 * @brief The gradient contribution of the output gradient @p g of @p node to its argument @p arg.
		 */
		const Node& rule(Builder& b, Computegraph& graph, const Node& node, const InvocationNode& inv, size_t arg, const Node& g) {
			const Node& x = inv.arguments[0].get();
			const Node* other = inv.arguments.size() > 1 ? &inv.arguments[1].get() : nullptr;
			switch (inv.id) {
			case FIDs::TENSOR_ADD:
				return b.unbroadcast(g, inv.arguments[arg].get());
			case FIDs::TENSOR_SUB:
				return arg == 0 ? b.unbroadcast(g, x) : b.unbroadcast(b.neg(g), *other);
			case FIDs::TENSOR_PROD:
				return arg == 0 ? b.unbroadcast(b.mul(g, *other), x) : b.unbroadcast(b.mul(g, x), *other);
			case FIDs::TENSOR_DIV:
				// d(a/b)/da = 1/b, d(a/b)/db = -(a/b)/b
				return arg == 0 ? b.unbroadcast(b.div(g, *other), x)
								: b.unbroadcast(b.neg(b.div(b.mul(g, node), *other)), *other);
			case FIDs::TENSOR_DOT: {
				const auto* ta = std::get_if<TensorType>(&getDataType(x));
				const auto* tb = std::get_if<TensorType>(&getDataType(*other));
				if (ta == nullptr || tb == nullptr || ta->shape.size() < 2 || tb->shape.size() < 2)
					throw NoGradientRule{inv.id}; // \todo matrix-vector products
				// y = A @ B: dA = g @ B^T, dB = A^T @ g (summed over broadcast batch dims)
				if (arg == 0)
					return b.unbroadcast(b.matmul(g, b.transposed(*other), getDataType(x)), x);
				return b.unbroadcast(b.matmul(b.transposed(x), g, getDataType(*other)), *other);
			}
			case FIDs::MAX:
			case FIDs::MIN:
				// Reductions over the last axis that keep it (size 1): the gradient flows to the extremal elements.
				return b.mul(g, b.op(FIDs::EQUAL_MASK, getDataType(x), {x, node}));
			case FIDs::SUM:
				return b.op(FIDs::BROADCAST_LIKE, getDataType(x), {g, x});
			case FIDs::MEAN: {
				// The last axis is removed: restore it (size 1), broadcast, and divide by its size.
				const Node& group = graph.addConstNode(Atomic{std::vector<int>{0, 0}}); // all dims (see RESHAPE)
				const Node& keep = b.op(FIDs::RESHAPE, getDataType(x), {g, group, graph.addConstNode(Atomic{1})});
				const Node& spread = b.op(FIDs::BROADCAST_LIKE, getDataType(x), {keep, x});
				if (const Node* n = lastDimSize(graph, getDataType(x)))
					return b.div(spread, *n);
				// Size not known statically: count the elements of the last axis at runtime.
				const Node& ones = b.op(FIDs::BROADCAST_LIKE, getDataType(x), {b.constant(1.0), x});
				return b.div(spread, b.op(FIDs::SUM, getDataType(x), {ones}));
			}
			case FIDs::EXP:
				return b.mul(g, node);
			case FIDs::LOG:
				return b.div(g, x);
			case FIDs::SQRT:
				return b.div(b.mul(g, b.constant(0.5)), node);
			case FIDs::RSQRT: // d x^(-1/2) = -1/2 x^(-3/2) = -1/2 y^3
				return b.mul(b.mul(g, b.constant(-0.5)), b.mul(node, b.mul(node, node)));
			case FIDs::ERF: // 2/sqrt(pi) exp(-x^2)
				return b.mul(b.mul(g, b.constant(2.0 / std::sqrt(std::numbers::pi))), b.unary(FIDs::EXP, b.neg(b.mul(x, x))));
			case FIDs::TANH:
				return b.mul(g, b.sub(b.constant(1.0), b.mul(node, node)));
			case FIDs::SIN:
				return b.mul(g, b.unary(FIDs::COS, x));
			case FIDs::COS:
				return b.neg(b.mul(g, b.unary(FIDs::SIN, x)));
			case FIDs::TAN:
				return b.mul(g, b.add(b.constant(1.0), b.mul(node, node)));
			case FIDs::RESHAPE:
				return b.op(FIDs::RESHAPE_LIKE, getDataType(x), {g, x});
			case FIDs::PERMUTE:
				return b.op(FIDs::PERMUTE_INVERSE, getDataType(x), {g, *other});
			case FIDs::EXPAND:
				return b.op(FIDs::COLLAPSE_SUM_LIKE, getDataType(x), {g, x});
			case FIDs::COLLAPSE_SUM_LIKE:
				return b.op(FIDs::BROADCAST_LIKE, getDataType(x), {g, x});
			case FIDs::BROADCAST_LIKE:
				return b.op(FIDs::COLLAPSE_SUM_LIKE, getDataType(x), {g, x});
			case FIDs::RESHAPE_LIKE:
				return b.op(FIDs::RESHAPE_LIKE, getDataType(x), {g, x});
			case FIDs::PERMUTE_INVERSE:
				return b.op(FIDs::PERMUTE, getDataType(x), {g, *other});
			default:
				throw NoGradientRule{inv.id};
			}
		}

		/** @brief Which arguments of a builtin carry a gradient (e.g. not the reference of a *_LIKE builtin). */
		bool differentiableArgument(FIDs id, size_t arg) {
			switch (id) {
			case FIDs::RESHAPE:
			case FIDs::PERMUTE:
			case FIDs::EXPAND:
			case FIDs::COLLAPSE_SUM_LIKE:
			case FIDs::BROADCAST_LIKE:
			case FIDs::RESHAPE_LIKE:
			case FIDs::PERMUTE_INVERSE:
				return arg == 0;
			case FIDs::EQUAL_MASK:
				return false; // piecewise constant
			case FIDs::STOP_GRADIENT:
				return false;
			default:
				return true;
			}
		}

		/** @brief The name of the custom gradient function of the member function @p function. */
		std::string customGradientName(const std::string& function) {
			return function == "forward" || function == "operator()" ? "gradient" : function + "_gradient";
		}

		/** @brief A module call whose module provides a custom gradient function. */
		struct CustomCall {
			const ModuleCall* call;
			const FunctionDecl* gradient;
		};

		/**
		 * @brief The module call with a custom gradient that produced @p node, if any. If calls are nested (an inner
		 * call's output is also an output of the enclosing call), the outermost one (shortest instance path) wins.
		 */
		std::optional<CustomCall> customCallProducing(const Computegraph& graph, const Node& node) {
			std::optional<CustomCall> best;
			for (const ModuleCall* call : graph.moduleCallsProducing(node)) {
				if (call->module == nullptr)
					continue;
				auto fn = call->module->functions.find(customGradientName(call->function));
				if (fn == call->module->functions.end())
					continue;
				// A call that returns one of its inputs unchanged cannot be treated as a unit.
				if (std::any_of(call->outputs.begin(), call->outputs.end(), [&](const Node* out) {
						return std::find(call->inputs.begin(), call->inputs.end(), out) != call->inputs.end();
					}))
					continue;
				if (!best || call->path.size() < best->call->path.size())
					best = CustomCall{call, &fn->second};
			}
			return best;
		}

		std::string identifier(const std::string& path) {
			std::string id = path;
			std::replace(id.begin(), id.end(), '.', '_');
			return id;
		}
		/**
		 * @brief Inlines the custom gradient function of a module call and accumulates its results onto the call's
		 * inputs and weights.
		 * @details The gradient function takes the call's inputs followed by one cotangent per output and returns one
		 * cotangent per input followed by one per weight of the module (in declaration order, see mdl::weightPaths).
		 */
		template <typename Accumulate>
		std::expected<void, std::string> applyCustomGradient(
				Computegraph& graph, Builder& b, const CustomCall& custom, const std::vector<const Node*>& weights,
				const std::unordered_map<const Node*, const Node*>& adjoint,
				const std::unordered_set<const Node*>& dependsOnTarget, Accumulate&& accumulate
		) {
			const ModuleCall& call = *custom.call;
			const FunctionDecl& gradient = *custom.gradient;
			const auto where = std::format(
					"custom gradient '{}' of '{}'", gradient.name, call.path.empty() ? call.scope : call.path
			);
			if (gradient.parameters.size() != call.inputs.size() + call.outputs.size())
				return std::unexpected(std::format(
						"{} must take {} argument(s) ({} input(s) of '{}' and one cotangent per output) but takes {}",
						where, call.inputs.size() + call.outputs.size(), call.inputs.size(), call.function,
						gradient.parameters.size()
				));
			if (gradient.outputs.size() != call.inputs.size() + weights.size())
				return std::unexpected(std::format(
						"{} must return {} value(s) (one per input of '{}', then one per weight) but returns {}", where,
						call.inputs.size() + weights.size(), call.function, gradient.outputs.size()
				));

			bool needed = false;
			std::vector<NodeRef> arguments;
			for (const Node* input : call.inputs)
				arguments.push_back(std::cref(*input));
			for (const Node* output : call.outputs) {
				auto adj = adjoint.find(output);
				needed |= adj != adjoint.end();
				arguments.push_back(std::cref(adj != adjoint.end() ? *adj->second : b.zerosLike(*output)));
			}
			if (!needed)
				return {};

			BoundModuleType bound{.definition = *call.module, .hyperparamValues = call.hyperparameters};
			auto results = graph.addModuleInvocationNode(bound, call.scope, call.path, gradient, std::move(arguments));
			if (!results)
				return std::unexpected(where + ": " + results.error());
			for (size_t i = 0; i < call.inputs.size(); ++i)
				if (dependsOnTarget.contains(call.inputs[i]))
					accumulate(*call.inputs[i], (*results)[i].get());
			for (size_t i = 0; i < weights.size(); ++i)
				if (weights[i] != nullptr && dependsOnTarget.contains(weights[i]))
					accumulate(*weights[i], (*results)[call.inputs.size() + i].get());
			return {};
		}
	} // namespace

	std::expected<FunctionDecl, std::string> vjp(
			Computegraph& graph, const FunctionDecl& func, const std::vector<std::string>& weightOrder,
			const Options& options
	) {
		// Topological order (post-order) of all nodes reachable from the outputs. A module call with a custom gradient
		// is one step: its outputs are not differentiated through, but its inputs and weights are visited.
		struct Step {
			const Node* node = nullptr;
			std::optional<CustomCall> custom;
		};
		std::vector<Step> steps;
		std::vector<const Node*> order;
		std::unordered_set<const Node*> visited;
		std::unordered_set<const ModuleCall*> visitedCalls;
		std::function<void(const Node&)> visit = [&](const Node& node) {
			if (!visited.insert(&node).second)
				return;
			if (auto custom = customCallProducing(graph, node)) {
				if (!visitedCalls.insert(custom->call).second)
					return;
				for (const Node* input : custom->call->inputs)
					visit(*input);
				for (const auto& path : weightPaths(*custom->call->module))
					if (const Node* w = graph.findParameterNode(custom->call->scope, custom->call->weightPath(path)))
						visit(*w);
				steps.push_back(Step{.node = nullptr, .custom = custom});
				return;
			}
			if (const auto* inv = std::get_if<InvocationNode>(&node))
				for (const auto& arg : inv->arguments)
					visit(arg.get());
			steps.push_back(Step{.node = &node, .custom = std::nullopt});
			order.push_back(&node);
		};
		for (const auto& output : func.outputs)
			visit(output.get());

		// The weights of a module call with custom gradient, in the order of its gradient's results.
		auto callWeights = [&](const ModuleCall& call) {
			std::vector<const Node*> result;
			for (const auto& path : weightPaths(*call.module))
				result.push_back(graph.findParameterNode(call.scope, call.weightPath(path)));
			return result;
		};

		// The differentiation targets: selected weights and (optionally) the inputs.
		std::map<std::string, const Node*> weights;
		std::unordered_map<const void*, const Node*> inputNodes;
		for (const auto* node : order) {
			if (const auto* pn = std::get_if<ParameterNode>(node))
				weights.emplace(pn->name, node);
			else if (const auto* in = std::get_if<InputNode>(node))
				inputNodes.emplace(in, node);
		}
		auto selected = [&](const std::string& path) {
			return options.trainable.empty() || options.trainable.selects(path);
		};
		std::unordered_set<const Node*> dependsOnTarget;
		for (const auto& step : steps) {
			if (step.custom) {
				const ModuleCall& call = *step.custom->call;
				bool target = std::any_of(call.inputs.begin(), call.inputs.end(), [&](const Node* in) {
					return dependsOnTarget.contains(in);
				});
				for (const Node* w : callWeights(call))
					target |= w != nullptr && dependsOnTarget.contains(w);
				if (target)
					dependsOnTarget.insert(call.outputs.begin(), call.outputs.end());
				continue;
			}
			const Node* node = step.node;
			bool target = false;
			if (const auto* pn = std::get_if<ParameterNode>(node))
				target = selected(pn->name);
			else if (std::holds_alternative<InputNode>(*node))
				target = options.inputs;
			else if (const auto* inv = std::get_if<InvocationNode>(node))
				for (size_t i = 0; i < inv->arguments.size(); ++i)
					target |= differentiableArgument(inv->id, i) && dependsOnTarget.contains(&inv->arguments[i].get());
			if (target)
				dependsOnTarget.insert(node);
		}

		Builder b{graph};
		FunctionDecl grad{
				.name = func.name + "_vjp",
				.templateParameters = func.templateParameters,
				.parameters = func.parameters,
				.outputs = func.outputs,
				.returnTypes = func.returnTypes,
				.outputNames = {},
		};
		for (size_t i = 0; i < func.outputs.size(); ++i)
			grad.outputNames.push_back(func.outputs.size() == 1 ? "output" : "output" + std::to_string(i));

		// Seed: one cotangent input per output.
		std::unordered_map<const Node*, const Node*> adjoint;
		auto accumulate = [&](const Node& node, const Node& contribution) {
			auto [it, inserted] = adjoint.try_emplace(&node, &contribution);
			if (!inserted)
				it->second = &b.add(*it->second, contribution);
		};
		for (size_t i = 0; i < func.outputs.size(); ++i) {
			const auto& name = grad.outputNames[i];
			DataType type = i < func.returnTypes.size()
									? std::visit([](const auto& t) -> DataType { return t; }, func.returnTypes[i])
									: getDataType(func.outputs[i].get());
			const Node& cotangent = graph.addInputNode(type, "d_" + name);
			grad.parameters.emplace_back("d_" + name, std::cref(std::get<InputNode>(cotangent)));
			if (dependsOnTarget.contains(&func.outputs[i].get()))
				accumulate(func.outputs[i].get(), cotangent);
		}

		// Backward sweep in reverse topological order.
		try {
			for (auto it = steps.rbegin(); it != steps.rend(); ++it) {
				if (it->custom) {
					if (auto ok = applyCustomGradient(graph, b, *it->custom, callWeights(*it->custom->call), adjoint,
													  dependsOnTarget, accumulate);
						!ok)
						return std::unexpected(func.name + ": " + ok.error());
					continue;
				}
				const auto* inv = std::get_if<InvocationNode>(it->node);
				auto adj = adjoint.find(it->node);
				if (inv == nullptr || adj == adjoint.end())
					continue;
				for (size_t i = 0; i < inv->arguments.size(); ++i) {
					const Node& arg = inv->arguments[i].get();
					if (differentiableArgument(inv->id, i) && dependsOnTarget.contains(&arg))
						accumulate(arg, rule(b, graph, *it->node, *inv, i, *adj->second));
				}
			}
		} catch (const NoGradientRule& e) {
			return std::unexpected(std::format(
					"{}: no gradient rule for builtin #{} (used in the function)", func.name, std::to_underlying(e.id)
			));
		}

		// Outputs: weight gradients (in canonical order), then input gradients.
		auto addOutput = [&](const std::string& name, const Node& value, const DataType& type) {
			grad.outputs.push_back(std::cref(value));
			grad.outputNames.push_back(name);
			if (const auto* t = std::get_if<TensorType>(&type))
				grad.returnTypes.emplace_back(*t);
			else if (const auto* p = std::get_if<PrimitiveType>(&type))
				grad.returnTypes.emplace_back(*p);
		};
		for (const auto& path : weightOrder) {
			if (!selected(path))
				continue;
			auto w = weights.find(path);
			if (w == weights.end())
				continue; // \todo zero gradients of weights the function does not use (needs their type)
			auto adj = adjoint.find(w->second);
			const Node& value = adj != adjoint.end() ? *adj->second : b.zerosLike(*w->second);
			addOutput("d_" + identifier(path), value, getDataType(*w->second));
		}
		if (options.inputs) {
			for (const auto& [pname, input] : func.parameters) {
				auto node = inputNodes.find(&input.get());
				if (node == inputNodes.end())
					continue;
				auto adj = adjoint.find(node->second);
				const Node& value = adj != adjoint.end() ? *adj->second : b.zerosLike(*node->second);
				addOutput("d_" + pname, value, input.get().type);
			}
		}
		return grad;
	}

} // namespace mdl::autodiff
