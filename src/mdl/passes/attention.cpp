#include <mdl/passes/attention.hpp>

#include <mdl/computegraphv2/node.hpp>

#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace mdl::attention {

	namespace {
		using FIDs = builtins::FunctionIDs;
		using NodeRef = Computegraph::NodeRef;

		const InvocationNode* invocation(const Node& node, FIDs id) {
			const auto* inv = std::get_if<InvocationNode>(&node);
			return inv != nullptr && inv->id == id ? inv : nullptr;
		}

		const TensorType* tensor(const Node& node) { return std::get_if<TensorType>(&getDataType(node)); }

		bool isScalar(const Node& node) { return std::holds_alternative<PrimitiveType>(getDataType(node)); }

		/** @brief `reshape(r(x), ...)` (the reduction over the last axis with the axis restored, e.g. `max(x)[... -> ... 1]`). */
		bool isRestoredReduction(const Node& node, FIDs r, const Node& x) {
			const auto* reshape = invocation(node, FIDs::RESHAPE);
			if (reshape == nullptr)
				return false;
			const auto* reduction = invocation(reshape->arguments[0].get(), r);
			return reduction != nullptr && &reduction->arguments[0].get() == &x;
		}

		/** @brief The probabilities `exp(x - max(x)) / sum(...)` of softmax(x): x, or nullptr. */
		const Node* softmaxInput(const Node& p) {
			const auto* div = invocation(p, FIDs::TENSOR_DIV);
			if (div == nullptr)
				return nullptr;
			const Node& e = div->arguments[0].get();
			if (!isRestoredReduction(div->arguments[1].get(), FIDs::SUM, e))
				return nullptr;
			const auto* exp = invocation(e, FIDs::EXP);
			const auto* sub = exp ? invocation(exp->arguments[0].get(), FIDs::TENSOR_SUB) : nullptr;
			if (sub == nullptr)
				return nullptr;
			const Node& x = sub->arguments[0].get();
			return isRestoredReduction(sub->arguments[1].get(), FIDs::MAX, x) ? &x : nullptr;
		}

		std::optional<double> constant(const Node& node) {
			const auto* c = std::get_if<ConstNode>(&node);
			if (c == nullptr)
				return std::nullopt;
			if (const auto* i = std::get_if<int>(&c->value))
				return *i;
			if (const auto* d = std::get_if<double>(&c->value))
				return *d;
			return std::nullopt;
		}

		bool isConstant(const Node& node, double value) {
			auto c = constant(node);
			return c && *c == value;
		}

		/**
		 * @brief Whether @p bias is the additive mask of std's masked softmax for the condition @p mask:
		 * `(where(mask, 1, 0) - 1) * big` (the indicator possibly converted, e.g. to bfloat16) with a huge `big` (e.g.
		 * the largest float): 0 where mask holds, the most negative value elsewhere.
		 */
		bool isMaskBias(const Node& bias, const Node& mask) {
			const auto* prod = invocation(bias, FIDs::TENSOR_PROD);
			if (prod == nullptr)
				return false;
			for (size_t i = 0; i < 2; ++i) {
				const auto big = constant(prod->arguments[1 - i].get());
				const auto* sub = invocation(prod->arguments[i].get(), FIDs::TENSOR_SUB);
				const Node* indicator = sub ? &sub->arguments[0].get() : nullptr;
				if (const auto* cast = indicator ? invocation(*indicator, FIDs::CAST) : nullptr) // (e.g. to bfloat16)
					indicator = &cast->arguments[0].get();
				const auto* where = indicator ? invocation(*indicator, FIDs::WHERE) : nullptr;
				if (big && *big >= 1e30 && where != nullptr && isConstant(sub->arguments[1].get(), 1.0) &&
					&where->arguments[0].get() == &mask && isConstant(where->arguments[1].get(), 1.0) &&
					isConstant(where->arguments[2].get(), 0.0))
					return true;
			}
			return false;
		}

		/**
		 * @brief The lengths of a suffix mask if @p mask is the conjunction of two dimensions' masks of the same suffix
		 * mask (`where(a, b, a)` with a and b reshapes of SUFFIX_MASK(lengths, ...): e.g. the masks of the queries' and
		 * the keys' positions of self-attention), else nullptr.
		 */
		const Node* suffixLengths(const Node& mask) {
			auto lengthsOf = [](const Node* n) -> const Node* {
				while (const auto* inv = std::get_if<InvocationNode>(n)) {
					if (inv->id == FIDs::SUFFIX_MASK)
						return &inv->arguments[0].get();
					if (inv->id != FIDs::RESHAPE && inv->id != FIDs::EXPAND && inv->id != FIDs::BROADCAST_LIKE)
						return nullptr;
					n = &inv->arguments[0].get();
				}
				return nullptr;
			};
			const auto* both = invocation(mask, FIDs::WHERE);
			if (both == nullptr || &both->arguments[0].get() != &both->arguments[2].get())
				return nullptr;
			const Node* a = lengthsOf(&both->arguments[0].get());
			const Node* b = lengthsOf(&both->arguments[1].get());
			return a != nullptr && a == b ? a : nullptr;
		}

		/** @brief The scores of attention: `s / c`, `s * c`, `c * s`, or `s` with s = q @ k^T (the scale c: a scalar). */
		struct Scaled {
			const Node* scores;
			const Node* scale;
		};
		std::optional<Scaled> scaledScores(Computegraph& graph, const Node& x) {
			const Node* scores = &x;
			const Node* scale = nullptr;
			if (const auto* div = invocation(x, FIDs::TENSOR_DIV); div && isScalar(div->arguments[1].get())) {
				scores = &div->arguments[0].get();
				scale = &graph.addInvocationNode(
						FIDs::TENSOR_DIV, {graph.addConstNode(Atomic{1.0}), div->arguments[1].get()}
				);
			} else if (const auto* prod = invocation(x, FIDs::TENSOR_PROD)) {
				for (size_t i = 0; i < 2 && scale == nullptr; ++i)
					if (isScalar(prod->arguments[1 - i].get())) {
						scores = &prod->arguments[i].get();
						scale = &prod->arguments[1 - i].get();
					}
			}
			if (invocation(*scores, FIDs::TENSOR_DOT) == nullptr)
				return std::nullopt;
			return Scaled{scores, scale != nullptr ? scale : &graph.addConstNode(Atomic{1.0})};
		}

		/** @brief Matches @p node as attention: the ATTENTION node, or nullptr. */
		const Node* match(Computegraph& graph, const Node& node) {
			const auto* out = invocation(node, FIDs::TENSOR_DOT);
			if (out == nullptr)
				return nullptr;
			const Node* x = softmaxInput(out->arguments[0].get());
			if (x == nullptr)
				return nullptr;
			// The softmax's input: the scaled scores, or the scaled scores plus a bias (e.g. a mask: std's masked
			// softmax adds `(mask - 1) * big` to the masked scores, `where(mask, scores, 0)`, which are the scores
			// wherever they count: the bias rounds every other sum to the most negative value).
			auto scaled = scaledScores(graph, *x);
			const Node* bias = nullptr;
			const Node* lengths = nullptr; // (of the mask, if a suffix mask)
			if (const auto* add = scaled ? nullptr : invocation(*x, FIDs::TENSOR_ADD)) {
				for (size_t i = 0; i < 2 && !scaled; ++i) {
					const Node* data = &add->arguments[i].get();
					const Node& b = add->arguments[1 - i].get();
					const Node* mask = nullptr;
					if (const auto* w = invocation(*data, FIDs::WHERE); w != nullptr &&
																		isConstant(w->arguments[2].get(), 0.0) &&
																		isMaskBias(b, w->arguments[0].get())) {
						data = &w->arguments[1].get();
						mask = &w->arguments[0].get();
					}
					if (tensor(b) == nullptr)
						continue;
					scaled = scaledScores(graph, *data);
					if (scaled) {
						bias = &b;
						lengths = mask != nullptr ? suffixLengths(*mask) : nullptr;
					}
				}
			}
			if (!scaled)
				return nullptr;
			const Node* scores = scaled->scores;
			const Node* scale = scaled->scale;
			const auto* product = invocation(*scores, FIDs::TENSOR_DOT);
			if (product == nullptr)
				return nullptr;
			const Node& q = product->arguments[0].get();
			const Node& kt = product->arguments[1].get();
			const Node& v = out->arguments[1].get();
			// k: the transposed operand (k^T = permute(k, {-1, -2}) or else k as the transposition of k^T).
			const Node* k = nullptr;
			if (const auto* perm = invocation(kt, FIDs::PERMUTE); perm && perm->arguments.size() == 2) {
				const auto* c = std::get_if<ConstNode>(&perm->arguments[1].get());
				const auto* axes = c ? std::get_if<std::vector<int>>(&c->value) : nullptr;
				if (axes != nullptr && *axes == std::vector<int>{-1, -2})
					k = &perm->arguments[0].get();
			}
			if (k == nullptr)
				k = &graph.addInvocationNode(FIDs::PERMUTE, {kt, graph.addConstNode(Atomic{std::vector<int>{-1, -2}})});
			// Tensors of the same rank (at least matrices) and element type: [..., seq, dim]. (A masked type is only an
			// annotation here: the masking pass made the masking explicit, e.g. as the bias.)
			const auto *tq = tensor(q), *tk = tensor(*k), *tv = tensor(v);
			if (tq == nullptr || tk == nullptr || tv == nullptr || tq->shape.size() < 2 ||
				tk->shape.size() != tq->shape.size() || tv->shape.size() != tq->shape.size() || tq->type != tk->type ||
				tq->type != tv->type)
				return nullptr;
			if (bias != nullptr && (tensor(*bias)->type != tq->type || tensor(*bias)->shape.size() != tq->shape.size()))
				return nullptr; // (a bias of the scores' type and rank, possibly broadcast, e.g. over the heads)
			std::vector<Computegraph::NodeRef> args{q, *k, v, *scale};
			if (bias != nullptr)
				args.push_back(std::cref(*bias));
			if (lengths != nullptr) // (the bias is this suffix mask: kernels may skip the masked suffix instead)
				args.push_back(std::cref(*lengths));
			return &graph.addInvocationNode(FIDs::ATTENTION, std::move(args));
		}
	} // namespace

	FunctionDecl fuse(Computegraph& graph, const FunctionDecl& func) {
		// Rounds of substitutions: a replacement refers to the original operands, which may contain further attention
		// (e.g. of the previous layer), formed by the next round.
		FunctionDecl result = func;
		for (bool changed = true; changed;) {
			std::unordered_map<const Node*, const Node*> replacements;
			std::unordered_set<const Node*> visited;
			std::function<void(const Node&)> visit = [&](const Node& node) {
				if (!visited.insert(&node).second)
					return;
				if (const Node* fused = match(graph, node))
					replacements.emplace(&node, fused);
				if (const auto* inv = std::get_if<InvocationNode>(&node))
					for (const auto& arg : inv->arguments)
						visit(arg.get());
			};
			for (const auto& output : result.outputs)
				visit(output.get());
			changed = !replacements.empty();
			if (changed)
				result.outputs = graph.substitute(result.outputs, replacements, /*keepCalls=*/true);
		}
		return result;
	}

} // namespace mdl::attention
