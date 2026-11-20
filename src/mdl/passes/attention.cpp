#include <mdl/passes/attention.hpp>

#include <mdl/computegraphv2/node.hpp>

#include <functional>
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

		/** @brief Matches @p node as attention: the ATTENTION node, or nullptr. */
		const Node* match(Computegraph& graph, const Node& node) {
			const auto* out = invocation(node, FIDs::TENSOR_DOT);
			if (out == nullptr)
				return nullptr;
			const Node* x = softmaxInput(out->arguments[0].get());
			if (x == nullptr)
				return nullptr;
			// The scale: x = s / c, s * c, c * s, or s.
			const Node* scores = x;
			const Node* scale = nullptr;
			if (const auto* div = invocation(*x, FIDs::TENSOR_DIV); div && isScalar(div->arguments[1].get())) {
				scores = &div->arguments[0].get();
				scale = &graph.addInvocationNode(
						FIDs::TENSOR_DIV, {graph.addConstNode(Atomic{1.0}), div->arguments[1].get()}
				);
			} else if (const auto* prod = invocation(*x, FIDs::TENSOR_PROD)) {
				for (size_t i = 0; i < 2 && scale == nullptr; ++i)
					if (isScalar(prod->arguments[1 - i].get())) {
						scores = &prod->arguments[i].get();
						scale = &prod->arguments[1 - i].get();
					}
			}
			if (scale == nullptr)
				scale = &graph.addConstNode(Atomic{1.0});
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
			// Tensors of the same rank (at least matrices) and element type: [..., seq, dim].
			const auto *tq = tensor(q), *tk = tensor(*k), *tv = tensor(v);
			if (tq == nullptr || tk == nullptr || tv == nullptr || tq->shape.size() < 2 ||
				tk->shape.size() != tq->shape.size() || tv->shape.size() != tq->shape.size() || tq->type != tk->type ||
				tq->type != tv->type || tq->masked || tk->masked || tv->masked)
				return nullptr;
			return &graph.addInvocationNode(FIDs::ATTENTION, {q, *k, v, *scale});
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
