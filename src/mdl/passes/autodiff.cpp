#include <mdl/passes/autodiff.hpp>

#include <mdl/computegraphv2/function.hpp>
#include <mdl/utils/overload.hpp>

#include "../utils/einops.hpp"

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
			Builder(Computegraph& graph, shapes::Oracle* shapes) : graph(graph), shapes(shapes) {}

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
				// (a^T)^T = a (e.g. the gradient of `x @ W.T()` with respect to x is `g @ W`)
				if (const auto* inv = std::get_if<InvocationNode>(&a); inv && inv->id == FIDs::PERMUTE && isSwapLastTwo(*inv))
					return inv->arguments[0].get();
				return op(FIDs::PERMUTE, swapLastTwo(getDataType(a)), {a, graph.addConstNode(Atomic{std::vector<int>{-1, -2}})});
			}

			/** @brief `x[... k -> (...) k]`: merges all dimensions but the last one. */
			const Node* flattenLeading(const Node& x) {
				// With exact shapes: reshape to [-2 (inferred), k] (the type of x may be imprecise).
				if (const shapes::Shape* s = shapes ? shapes->shapeOf(x) : nullptr; s && !s->empty() && s->back().isConstant()) {
					const int k = static_cast<int>(s->back().factor);
					TensorType type{PrimitiveType::floating, {Elipsis{}, static_cast<unsigned>(k)}};
					return &op(FIDs::RESHAPE, type, {x, graph.addConstNode(Atomic{-2}), graph.addConstNode(Atomic{k})});
				}
				using namespace einops;
				const Spec lhs{{false, {{AtomKind::Ellipsis, "", 0}}}, {false, {{AtomKind::Named, "k", 0}}}};
				const Spec rhs{{true, {{AtomKind::Ellipsis, "", 0}}}, {false, {{AtomKind::Named, "k", 0}}}};
				auto result = apply(graph, x, lhs, rhs, {});
				return result ? &result->get() : nullptr;
			}
			const Node& matmul(const Node& a, const Node& b, const DataType& type) {
				return op(FIDs::TENSOR_DOT, type, {a, b});
			}

			/** @brief Sums @p g to the shape of @p like (undoes broadcasting); identity if the shapes are equal. */
			const Node& unbroadcast(const Node& g, const Node& like) {
				// Exact shapes decide if available; the (possibly imprecise) types otherwise.
				const shapes::Shape* sg = shapes ? shapes->shapeOf(g) : nullptr;
				const shapes::Shape* sl = shapes ? shapes->shapeOf(like) : nullptr;
				if (sg != nullptr && sl != nullptr) {
					if (*sg == *sl)
						return g;
					return op(FIDs::COLLAPSE_SUM_LIKE, getDataType(like), {g, like});
				}
				if (sameShape(getDataType(g), getDataType(like)))
					return g;
				return op(FIDs::COLLAPSE_SUM_LIKE, getDataType(like), {g, like});
			}

			const Node& zerosLike(const Node& like) {
				return op(FIDs::BROADCAST_LIKE, getDataType(like), {constant(0.0), like});
			}

			/** @brief The exact shape of @p node, if shapes are available. */
			const shapes::Shape* shapeOf(const Node& node) { return shapes ? shapes->shapeOf(node) : nullptr; }

		private:
			Computegraph& graph;
			shapes::Oracle* shapes;

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

		public:
			static bool isSwapLastTwo(const InvocationNode& permute) {
				if (permute.arguments.size() != 2)
					return false;
				const auto* c = std::get_if<ConstNode>(&permute.arguments[1].get());
				const auto* perm = c ? std::get_if<std::vector<int>>(&c->value) : nullptr;
				return perm != nullptr && *perm == std::vector<int>{-1, -2};
			}

		private:

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
				// y = A @ B: dA = g @ B^T, dB = A^T @ g (summed over broadcast batch dims). If only the other operand
				// has batch dimensions (`...`, e.g. activations times a weight), the product keeps (some of) them: its
				// type keeps a `...`, so that it is summed to the operand's shape. (The number of dimensions of `...` is
				// only known when compiling, so the graph types cannot tell whether the product already reduced them.)
				auto batched = [](const TensorType* t) {
					return std::any_of(t->shape.begin(), t->shape.end(), [](const auto& d) {
						return std::holds_alternative<Elipsis>(d);
					});
				};
				auto withBatch = [&](const TensorType* t) -> DataType {
					TensorType result{t->type, {Elipsis{}}};
					result.shape.insert(result.shape.end(), t->shape.begin(), t->shape.end());
					return result;
				};
				if (arg == 0) {
					DataType type = batched(tb) && !batched(ta) ? withBatch(ta) : getDataType(x);
					return b.unbroadcast(b.matmul(g, b.transposed(*other), type), x);
				}
				// Activations times a matrix (e.g. a weight): dB = A^T @ g as one GEMM over all batch dimensions,
				// flatten(A)^T @ flatten(g), instead of one product per batch element that is summed afterwards.
				const shapes::Shape* sa = b.shapeOf(x);
				const shapes::Shape* sb = b.shapeOf(*other);
				const bool flatten = sa != nullptr && sb != nullptr ? sa->size() > 2 && sb->size() == 2
																	: batched(ta) && !batched(tb) && tb->shape.size() == 2;
				if (flatten) {
					const Node* a2 = b.flattenLeading(x);
					const Node* g2 = b.flattenLeading(g);
					if (a2 != nullptr && g2 != nullptr) {
						// y = A @ W^T (std.Linear): d(W^T) = (g^T @ A)^T, so that the rule of the transposition yields
						// dW = g^T @ A without a transposed copy.
						const auto* perm = std::get_if<InvocationNode>(other);
						if (perm != nullptr && perm->id == FIDs::PERMUTE && Builder::isSwapLastTwo(*perm))
							return b.transposed(b.matmul(b.transposed(*g2), *a2, getDataType(perm->arguments[0].get())));
						return b.matmul(b.transposed(*a2), *g2, getDataType(*other));
					}
				}
				DataType type = batched(ta) && !batched(tb) ? withBatch(tb) : getDataType(*other);
				return b.unbroadcast(b.matmul(b.transposed(x), g, type), *other);
			}
			case FIDs::MAX:
			case FIDs::MIN:
				// Reductions over the last axis that keep it (size 1): the gradient flows to the extremal elements.
				return b.mul(g, b.op(FIDs::EQUAL_MASK, getDataType(x), {x, node}));
			case FIDs::SUM:
			case FIDs::SUM_ALL:
				return b.op(FIDs::BROADCAST_LIKE, getDataType(x), {g, x});
			case FIDs::MEAN_ALL: {
				// The number of elements (only known at runtime with `...`): the sum of ones of x's shape.
				const Node& count = b.op(FIDs::SUM_ALL, PrimitiveType::floating, {b.op(FIDs::BROADCAST_LIKE, getDataType(x), {b.constant(1.0), x})});
				return b.op(FIDs::BROADCAST_LIKE, getDataType(x), {b.div(g, count), x});
			}
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
				if (Builder::isSwapLastTwo(inv))
					return b.transposed(g); // its own inverse; cancels a transposed gradient
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
			case FIDs::WHERE: { // the gradient goes to the selected operand
				const Node& zero = b.constant(0.0);
				const Node& selected = arg == 1 ? b.op(FIDs::WHERE, getDataType(g), {x, g, zero})
												: b.op(FIDs::WHERE, getDataType(g), {x, zero, g});
				return b.unbroadcast(selected, inv.arguments[arg].get());
			}
			case FIDs::SELECT: // the gradient goes to the selected position
				return b.op(FIDs::UNSELECT_LIKE, getDataType(x), {g, inv.arguments[1], inv.arguments[2], x});
			case FIDs::UNSELECT_LIKE:
				return b.op(FIDs::SELECT, getDataType(x), {g, inv.arguments[1], inv.arguments[2]});
			case FIDs::TAKE: // rows of a table: their gradients are added to the selected rows
				return b.op(FIDs::SCATTER_ADD_ROWS, getDataType(x), {g, *other, x});
			case FIDs::SCATTER_ADD_ROWS:
				return b.op(FIDs::TAKE, getDataType(x), {g, *other});
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
			case FIDs::ONE_HOT_LIKE:
				return false;
			case FIDs::TAKE:
			case FIDs::SCATTER_ADD_ROWS:
				return arg == 0; // not the (integer) indices
			case FIDs::WHERE:
				return arg != 0; // not the condition
			case FIDs::SELECT:
			case FIDs::UNSELECT_LIKE:
				return arg == 0; // not the axis, index, or reference
			default:
				return true;
			}
		}

		/** @brief The name of the custom gradient function of the member function @p function. */
		std::string customGradientName(const std::string& function) {
			return function == "forward" || function == "operator()" ? "gradient" : function + "_gradient";
		}

		/** @brief A call of a module or free function that provides a custom gradient function. */
		struct CustomCall {
			const ModuleCall* call;
			const FunctionDecl* gradient;
		};

		/**
		 * @brief The call with a custom gradient that produced @p node, if any. If calls are nested (an inner call's
		 * output is also an output of the enclosing call), the outermost one wins: module calls (which enclose the free
		 * functions they call) by the shortest instance path, then free functions by the latest record (the enclosing
		 * call is recorded after the calls in its body).
		 */
		std::optional<CustomCall> customCallProducing(const Computegraph& graph, const Node& node) {
			std::optional<CustomCall> best;
			for (const ModuleCall* call : graph.moduleCallsProducing(node)) {
				if (!call->customGradient)
					continue;
				const FunctionDecl* gradient = call->gradient;
				if (call->module != nullptr) {
					auto fn = call->module->functions.find(customGradientName(call->function));
					gradient = fn == call->module->functions.end() ? nullptr : &fn->second;
				}
				if (gradient == nullptr)
					continue;
				// A call that returns one of its inputs unchanged cannot be treated as a unit.
				if (std::any_of(call->outputs.begin(), call->outputs.end(), [&](const Node* out) {
						return std::find(call->inputs.begin(), call->inputs.end(), out) != call->inputs.end();
					}))
					continue;
				const bool better = !best ||
						(call->module != nullptr
								 ? best->call->module == nullptr || call->path.size() < best->call->path.size()
								 : best->call->module == nullptr);
				if (better)
					best = CustomCall{call, gradient};
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
					"custom gradient '{}' of '{}'", gradient.name,
					call.module == nullptr ? call.function : call.path.empty() ? call.scope : call.path
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

			auto results = [&] {
				if (call.module == nullptr)
					return graph.addInvocationNode(gradient, std::move(arguments));
				BoundModuleType bound{.definition = *call.module, .hyperparamValues = call.hyperparameters};
				return graph.addModuleInvocationNode(bound, call.scope, call.path, gradient, std::move(arguments));
			}();
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
		bool isFloating(const DataType& type) {
			if (const auto* p = std::get_if<PrimitiveType>(&type))
				return *p == PrimitiveType::floating;
			const auto* t = std::get_if<TensorType>(&type);
			return t != nullptr && t->type == PrimitiveType::floating;
		}

		bool isScalar(const DataType& type) {
			if (std::holds_alternative<PrimitiveType>(type))
				return true;
			const auto* t = std::get_if<TensorType>(&type);
			return t != nullptr && t->shape.empty();
		}
	} // namespace

	size_t numCotangents(const FunctionDecl& func) {
		for (size_t i = 0; i < func.outputs.size(); ++i) {
			DataType type = i < func.returnTypes.size()
									? std::visit([](const auto& t) -> DataType { return t; }, func.returnTypes[i])
									: getDataType(func.outputs[i].get());
			if (isScalar(type))
				return 0;
		}
		return func.outputs.size();
	}

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
				if (custom->call->module != nullptr)
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
			if (call.module == nullptr)
				return result;
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
			else if (const auto* in = std::get_if<InputNode>(node))
				target = options.inputs && isFloating(in->type); // integer inputs (e.g. token ids) have no gradient
			else if (const auto* inv = std::get_if<InvocationNode>(node))
				for (size_t i = 0; i < inv->arguments.size(); ++i)
					target |= differentiableArgument(inv->id, i) && dependsOnTarget.contains(&inv->arguments[i].get());
			if (target)
				dependsOnTarget.insert(node);
		}

		Builder b{graph, options.shapes};
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
		// A function with scalar outputs is an objective: the scalar outputs are differentiated (seeded with 1) and the
		// other outputs are auxiliary. Otherwise, the caller provides one cotangent per output.
		const bool objective = numCotangents(func) == 0;
		for (size_t i = 0; i < func.outputs.size(); ++i) {
			const auto& name = grad.outputNames[i];
			DataType type = i < func.returnTypes.size()
									? std::visit([](const auto& t) -> DataType { return t; }, func.returnTypes[i])
									: getDataType(func.outputs[i].get());
			const Node* seed = nullptr;
			if (!objective) {
				const Node& cotangent = graph.addInputNode(type, "d_" + name);
				grad.parameters.emplace_back("d_" + name, std::cref(std::get<InputNode>(cotangent)));
				seed = &cotangent;
				// The cotangent has the exact shape of the output (whose `...` may have fewer dimensions).
				if (options.shapes != nullptr)
					if (const shapes::Shape* s = options.shapes->shapeOf(func.outputs[i].get()))
						options.shapes->assume(cotangent, *s);
			} else if (isScalar(type)) {
				seed = &b.constant(1.0);
			}
			if (seed != nullptr && dependsOnTarget.contains(&func.outputs[i].get()))
				accumulate(func.outputs[i].get(), *seed);
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
				if (node == inputNodes.end() || !isFloating(input.get().type))
					continue;
				auto adj = adjoint.find(node->second);
				const Node& value = adj != adjoint.end() ? *adj->second : b.zerosLike(*node->second);
				addOutput("d_" + pname, value, input.get().type);
			}
		}
		return grad;
	}

	std::expected<Split, std::string> split(
			Computegraph& graph, const FunctionDecl& vjp, size_t numOutputs, const PathPatterns* checkpoint
	) {
		if (vjp.parameters.size() < numOutputs || vjp.outputs.size() < numOutputs)
			return std::unexpected(vjp.name + ": not a gradient function with " + std::to_string(numOutputs) + " output(s)");
		const size_t numInputs = vjp.parameters.size() - numOutputs;
		std::unordered_set<const void*> cotangents, inputs;
		for (size_t i = 0; i < vjp.parameters.size(); ++i)
			(i < numInputs ? inputs : cotangents).insert(&vjp.parameters[i].second.get());

		// Memoized dependency analysis: whether a node depends on a cotangent / on a (primal) input.
		std::unordered_map<const Node*, std::pair<bool, bool>> deps;
		std::function<std::pair<bool, bool>(const Node&)> dependencies = [&](const Node& node) {
			if (auto it = deps.find(&node); it != deps.end())
				return it->second;
			std::pair<bool, bool> result{false, false};
			if (const auto* in = std::get_if<InputNode>(&node)) {
				result = {cotangents.contains(in), inputs.contains(in)};
			} else if (const auto* inv = std::get_if<InvocationNode>(&node)) {
				for (const auto& arg : inv->arguments) {
					auto [cot, inp] = dependencies(arg.get());
					result.first |= cot;
					result.second |= inp;
				}
			}
			deps.emplace(&node, result);
			return result;
		};
		auto cheap = [](const Node& node) {
			const auto* inv = std::get_if<InvocationNode>(&node);
			if (inv == nullptr)
				return false;
			switch (inv->id) {
			case FIDs::RESHAPE:
			case FIDs::PERMUTE:
			case FIDs::EXPAND:
			case FIDs::RESHAPE_LIKE:
			case FIDs::PERMUTE_INVERSE:
			case FIDs::BROADCAST_LIKE:
			case FIDs::STOP_GRADIENT:
				return true;
			default:
				return false;
			}
		};

		// Checkpointing: the computations of the selected module calls are recomputed; their inputs are saved.
		std::unordered_set<const Node*> recompute, boundary;
		if (checkpoint != nullptr) {
			std::unordered_set<const Node*> seen;
			std::vector<const Node*> stack;
			for (const auto& out : vjp.outputs)
				stack.push_back(&out.get());
			std::vector<const ModuleCall*> calls;
			while (!stack.empty()) {
				const Node* n = stack.back();
				stack.pop_back();
				if (!seen.insert(n).second)
					continue;
				for (const ModuleCall* call : graph.moduleCallsProducing(*n))
					if (call->module != nullptr && !call->path.empty() && checkpoint->selects(call->path) &&
						std::find(calls.begin(), calls.end(), call) == calls.end())
						calls.push_back(call);
				if (const auto* inv = std::get_if<InvocationNode>(n))
					for (const auto& a : inv->arguments)
						stack.push_back(&a.get());
			}
			// Only the outermost selected calls define the boundaries (the calls nested in them are recomputed with them).
			auto nested = [&](const ModuleCall* call) {
				return std::any_of(calls.begin(), calls.end(), [&](const ModuleCall* other) {
					return other != call && call->path.starts_with(other->path + ".");
				});
			};
			std::erase_if(calls, nested);
			for (const ModuleCall* call : calls) {
				boundary.insert(call->inputs.begin(), call->inputs.end());
				std::unordered_set<const Node*> stop(call->inputs.begin(), call->inputs.end()), internal;
				std::vector<const Node*> work(call->outputs.begin(), call->outputs.end());
				while (!work.empty()) {
					const Node* n = work.back();
					work.pop_back();
					if (stop.contains(n) || !internal.insert(n).second)
						continue;
					recompute.insert(n);
					if (const auto* inv = std::get_if<InvocationNode>(n))
						for (const auto& a : inv->arguments)
							work.push_back(&a.get());
				}
			}
		}

		// The nodes of the forward computation (the others were created by the gradient pass).
		std::unordered_set<const Node*> forward;
		{
			std::vector<const Node*> stack;
			for (size_t i = 0; i < numOutputs; ++i)
				stack.push_back(&vjp.outputs[i].get());
			while (!stack.empty()) {
				const Node* n = stack.back();
				stack.pop_back();
				if (!forward.insert(n).second)
					continue;
				if (const auto* inv = std::get_if<InvocationNode>(n))
					for (const auto& a : inv->arguments)
						stack.push_back(&a.get());
			}
		}

		// The residuals: the frontier of primal, input-dependent values that the backward computation reaches.
		std::vector<const Node*> residuals;
		std::unordered_set<const Node*> visited;
		std::function<void(const Node&)> visit = [&](const Node& node) {
			if (!visited.insert(&node).second)
				return;
			auto [cot, inp] = dependencies(node);
			if (!cot && !inp)
				return; // depends only on weights and constants: recomputed
			// (Only computed values are recomputed; inputs used in the region, e.g. the mask, are saved. With
			// checkpointing, the values the gradient pass computes from forward values are recomputed as well.)
			const bool recomputed = std::holds_alternative<InvocationNode>(node) && !boundary.contains(&node) &&
									(recompute.contains(&node) || (checkpoint != nullptr && !forward.contains(&node)));
			if (cot || cheap(node) || recomputed) {
				if (const auto* inv = std::get_if<InvocationNode>(&node))
					for (const auto& arg : inv->arguments)
						visit(arg.get());
				return;
			}
			residuals.push_back(&node);
		};
		for (size_t i = numOutputs; i < vjp.outputs.size(); ++i)
			visit(vjp.outputs[i].get());

		// Do not save values that elementwise operations recompute from the other residuals (e.g. `1 - h*h` of the
		// tanh rule when `h` is saved anyway): this only removes residuals, so the saved memory never grows.
		auto elementwise = [&](const Node& node) {
			const auto* inv = std::get_if<InvocationNode>(&node);
			if (inv == nullptr)
				return false;
			switch (inv->id) {
			case FIDs::TENSOR_ADD:
			case FIDs::TENSOR_SUB:
			case FIDs::TENSOR_PROD:
			case FIDs::TENSOR_DIV:
			case FIDs::EXP:
			case FIDs::LOG:
			case FIDs::SQRT:
			case FIDs::RSQRT:
			case FIDs::ERF:
			case FIDs::TANH:
			case FIDs::SIN:
			case FIDs::COS:
			case FIDs::TAN:
			case FIDs::EQUAL_MASK:
				return true;
			default:
				return cheap(node);
			}
		};
		// Save the input of an elementwise computation instead of its result if the result depends on one computed value
		// only (e.g. `x` instead of `1 + erf(x / sqrt(2))` in the gradient of GELU): this never needs more memory (an
		// elementwise result is at least as large as its arguments), and the forward pass does not compute values just to
		// save them (which would also keep backends from replacing the forward computation by a fused kernel). The
		// backward pass recomputes the elementwise computation, where it fuses with the operations that use it.
		// Only values that the forward pass would not store anyway are moved: those it does not compute, and those whose
		// forward uses are all elementwise arithmetic (fused away), unlike e.g. `tanh(h)` that a matrix product takes
		// (also through reshapes and permutations, which materialize their argument as well).
		std::unordered_set<const Node*> stored;
		for (size_t i = 0; i < numOutputs; ++i)
			stored.insert(&vjp.outputs[i].get());
		for (const Node* n : forward)
			if (const auto* inv = std::get_if<InvocationNode>(n); inv != nullptr && (!elementwise(*n) || cheap(*n)))
				for (const auto& arg : inv->arguments)
					stored.insert(&arg.get());
		// The nearest value through which the residual depends on all computed values that it depends on, if the
		// residual is computed from it by elementwise operations (and weights and constants).
		auto singleSource = [&](const Node* residual) -> const Node* {
			if (boundary.contains(residual) || stored.contains(residual) || !elementwise(*residual))
				return nullptr;
			auto interior = [&](const Node* n) {
				return std::holds_alternative<InvocationNode>(*n) && elementwise(*n) &&
					   (n == residual || (!boundary.contains(n) && !stored.contains(n)));
			};
			// The elementwise computation (interior nodes) and the computed values it starts from.
			const Node* source = nullptr;
			std::vector<const Node*> nodes; // interior nodes, the residual first
			std::unordered_set<const Node*> seen;
			std::vector<const Node*> stack{residual};
			while (!stack.empty()) {
				const Node* n = stack.back();
				stack.pop_back();
				if (!seen.insert(n).second || !dependencies(*n).second)
					continue; // weights and constants are not saved
				if (interior(n)) {
					nodes.push_back(n);
					for (const auto& arg : std::get<InvocationNode>(*n).arguments)
						stack.push_back(&arg.get());
					continue;
				}
				if (source != nullptr && source != n)
					return nullptr;
				source = n;
			}
			if (source == nullptr)
				return nullptr;
			// Whether the source is reachable from the residual without passing through `avoid`.
			auto reaches = [&](const Node* avoid) {
				std::unordered_set<const Node*> visited;
				std::vector<const Node*> work{residual};
				while (!work.empty()) {
					const Node* n = work.back();
					work.pop_back();
					if (n == avoid || !visited.insert(n).second || !dependencies(*n).second)
						continue;
					if (n == source)
						return true;
					if (interior(n))
						for (const auto& arg : std::get<InvocationNode>(*n).arguments)
							work.push_back(&arg.get());
				}
				return false;
			};
			// The dominators lie on every path to the source; the nearest one reaches all others.
			std::vector<const Node*> dominators;
			for (size_t i = 1; i < nodes.size(); ++i)
				if (!reaches(nodes[i]))
					dominators.push_back(nodes[i]);
			dominators.push_back(source);
			auto below = [&](const Node* from, const Node* to) { // `to` is an argument (transitively) of `from`
				std::unordered_set<const Node*> visited;
				std::vector<const Node*> work{from};
				while (!work.empty()) {
					const Node* n = work.back();
					work.pop_back();
					if (!visited.insert(n).second)
						continue;
					if (n == to)
						return true;
					if (interior(n))
						for (const auto& arg : std::get<InvocationNode>(*n).arguments)
							work.push_back(&arg.get());
				}
				return false;
			};
			for (const Node* d : dominators)
				if (std::all_of(dominators.begin(), dominators.end(), [&](const Node* o) { return below(d, o); }))
					return d;
			return source;
		};
		{
			std::vector<const Node*> moved;
			std::unordered_set<const Node*> unique;
			for (const Node* r : residuals) {
				const Node* source = singleSource(r);
				if (unique.insert(source != nullptr ? source : r).second)
					moved.push_back(source != nullptr ? source : r);
			}
			residuals = std::move(moved);
		}

		std::unordered_set<const Node*> saved(residuals.begin(), residuals.end());
		std::function<bool(const Node&, const Node*)> recomputable = [&](const Node& node, const Node* self) {
			if (&node != self && saved.contains(&node))
				return true;
			if (!dependencies(node).second)
				return true; // weights and constants
			const auto* inv = std::get_if<InvocationNode>(&node);
			if (inv == nullptr || !elementwise(node))
				return false;
			return std::all_of(inv->arguments.begin(), inv->arguments.end(), [&](const auto& arg) {
				return recomputable(arg.get(), self);
			});
		};
		std::erase_if(residuals, [&](const Node* r) {
			if (!recomputable(*r, r))
				return false;
			saved.erase(r);
			return true;
		});

		auto typeOf =[](const Node& node) -> std::variant<PrimitiveType, TensorType> {
			const DataType& type = getDataType(node);
			if (const auto* t = std::get_if<TensorType>(&type))
				return *t;
			if (const auto* p = std::get_if<PrimitiveType>(&type))
				return *p;
			return PrimitiveType::error;
		};
		auto declaredType = [&](size_t i) {
			return i < vjp.returnTypes.size() ? vjp.returnTypes[i] : typeOf(vjp.outputs[i].get());
		};
		auto outputName = [&](size_t i) {
			return i < vjp.outputNames.size() ? vjp.outputNames[i] : "output" + std::to_string(i);
		};

		Split result;
		result.numResiduals = residuals.size();
		const std::string base = vjp.name.ends_with("_vjp") ? vjp.name.substr(0, vjp.name.size() - 4) : vjp.name;

		// Forward pass: the outputs followed by the residuals.
		result.train.name = base + "_train";
		result.train.templateParameters = vjp.templateParameters;
		result.train.parameters.assign(vjp.parameters.begin(), vjp.parameters.begin() + numInputs);
		for (size_t i = 0; i < numOutputs; ++i) {
			result.train.outputs.push_back(vjp.outputs[i]);
			result.train.returnTypes.push_back(declaredType(i));
			result.train.outputNames.push_back(outputName(i));
		}
		for (size_t i = 0; i < residuals.size(); ++i) {
			result.train.outputs.push_back(std::cref(*residuals[i]));
			result.train.returnTypes.push_back(typeOf(*residuals[i]));
			result.train.outputNames.push_back("residual" + std::to_string(i));
		}

		// Backward pass: the residuals become inputs.
		result.backward.name = base + "_backward";
		result.backward.templateParameters = vjp.templateParameters;
		std::unordered_map<const Node*, const Node*> replacements;
		for (size_t i = 0; i < residuals.size(); ++i) {
			const std::string name = "residual" + std::to_string(i);
			const Node& input = graph.addInputNode(getDataType(*residuals[i]), name);
			replacements.emplace(residuals[i], &input);
			result.backward.parameters.emplace_back(name, std::cref(std::get<InputNode>(input)));
		}
		result.backward.parameters.insert(
				result.backward.parameters.end(), vjp.parameters.begin() + numInputs, vjp.parameters.end()
		);
		std::vector<NodeRef> gradients(vjp.outputs.begin() + numOutputs, vjp.outputs.end());
		result.backward.outputs = graph.substitute(gradients, replacements);
		for (size_t i = numOutputs; i < vjp.outputs.size(); ++i) {
			result.backward.returnTypes.push_back(declaredType(i));
			result.backward.outputNames.push_back(outputName(i));
		}

		// Sanity check: the backward pass must not read the primal inputs directly.
		std::unordered_set<const Node*> seen;
		std::function<bool(const Node&)> readsInput = [&](const Node& node) {
			if (!seen.insert(&node).second)
				return false;
			if (const auto* in = std::get_if<InputNode>(&node))
				return inputs.contains(in);
			if (const auto* inv = std::get_if<InvocationNode>(&node))
				return std::any_of(inv->arguments.begin(), inv->arguments.end(), [&](const auto& a) {
					return readsInput(a.get());
				});
			return false;
		};
		for (const auto& out : result.backward.outputs)
			if (readsInput(out.get()))
				return std::unexpected(vjp.name + ": internal error: the backward pass depends on the inputs");
		return result;
	}

} // namespace mdl::autodiff
