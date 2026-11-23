#include <mdl/passes/masking.hpp>

#include <mdl/computegraphv2/builtins.hpp>

#include <algorithm>
#include <format>
#include <limits>
#include <map>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace mdl::masking {

	namespace {
		using FIDs = builtins::FunctionIDs;
		using NodeRef = Computegraph::NodeRef;

		struct MaskError {
			std::string message;
		};

		class Masker {
		public:
			Masker(Computegraph& graph, const shapes::Shapes& shapes, const Node& mask)
					: graph(graph), shapes(shapes), mask(mask) {
				if (shapes.ellipsis.empty())
					throw MaskError{"the function has no `...` dimensions to mask"};
				maskSymbol = shapes.ellipsis.back();
				batchSymbols.assign(shapes.ellipsis.begin(), shapes.ellipsis.end() - 1);
				merged = shapes::Dim::constant(1);
				for (const auto& s : shapes.ellipsis)
					merged = merged * shapes::Dim::symbol(s);
			}

			/**
			 * @brief Zeroes the masked positions of a floating-point input: then all values at masked positions are
			 * finite (whatever the caller passed there, e.g. NaN), so that they cannot reach the valid results or the
			 * gradients (where they are multiplied by zero cotangents).
			 */
			void fillInput(const Node& input) {
				const auto* t = std::get_if<TensorType>(&getDataType(input));
				if (t == nullptr || !isFloat(t->type) || shapes.find(input) == nullptr)
					return; // (an unused input has no shape)
				const auto& s = shapeOf(input);
				auto dims = maskedDims(s);
				if (dims.empty())
					return;
				memo.emplace(&input, &fill(input, s, dims, 0.0));
				inserted = false;
			}

			const Node& rebuild(const Node& node) {
				if (auto it = memo.find(&node); it != memo.end())
					return *it->second;
				const bool outer = std::exchange(inserted, false);
				const Node& result = std::holds_alternative<InvocationNode>(node)
											 ? rebuildInvocation(node, std::get<InvocationNode>(node))
											 : node;
				if (inserted)
					maskedNodes.insert(&node); // masking was inserted at this node
				inserted = outer;
				memo.emplace(&node, &result);
				return result;
			}

			/**
			 * @brief Records the module calls of the rebuilt function for the new nodes (see ModuleCall). A call whose
			 * computation got masking loses its custom gradient (which does not know the mask).
			 */
			void propagateModuleCalls() {
				std::vector<const ModuleCall*> calls;
				for (const auto& [original, rebuilt] : memo)
					for (const ModuleCall* call : graph.moduleCallsProducing(*original))
						if (std::find(calls.begin(), calls.end(), call) == calls.end())
							calls.push_back(call);
				for (const ModuleCall* call : calls) {
					ModuleCall copy = *call;
					bool complete = true;
					for (auto* nodes : {&copy.inputs, &copy.outputs})
						for (auto& n : *nodes) {
							auto it = memo.find(n);
							complete &= it != memo.end();
							if (it != memo.end())
								n = it->second;
						}
					if (!complete)
						continue;
					// The call's computation: the nodes between its outputs and its inputs.
					std::unordered_set<const Node*> stop(call->inputs.begin(), call->inputs.end()), seen;
					std::vector<const Node*> stack(call->outputs.begin(), call->outputs.end());
					while (!stack.empty() && copy.customGradient) {
						const Node* n = stack.back();
						stack.pop_back();
						if (stop.contains(n) || !seen.insert(n).second)
							continue;
						if (maskedNodes.contains(n))
							copy.customGradient = false;
						if (const auto* inv = std::get_if<InvocationNode>(n))
							for (const auto& a : inv->arguments)
								stack.push_back(&a.get());
					}
					graph.addModuleCall(std::move(copy));
				}
			}

			/** @brief Zero at the masked positions of an output (if it has masked dimensions). */
			const Node& maskOutput(const Node& original, const Node& rebuilt) {
				const auto& s = shapeOf(original);
				auto dims = maskedDims(s);
				return dims.empty() ? rebuilt : fill(rebuilt, s, dims, 0.0);
			}

		private:
			Computegraph& graph;
			const shapes::Shapes& shapes;
			const Node& mask;
			std::string maskSymbol;
			std::vector<std::string> batchSymbols;
			shapes::Dim merged; ///< all dimensions of `...` flattened into one
			std::unordered_map<const Node*, const Node*> memo;
			std::unordered_set<const Node*> maskedNodes; ///< original nodes at which masking was inserted
			bool inserted = false;						 ///< whether masking was inserted while rebuilding a node
			std::map<std::string, const Node*> maskCache;

			const shapes::Shape& shapeOf(const Node& node) const {
				const auto* s = shapes.find(node);
				if (s == nullptr) {
					const auto* inv = std::get_if<InvocationNode>(&node);
					throw MaskError{std::format(
							"no shape for {} (masking needs the shapes of all nodes)",
							inv ? "builtin #" + std::to_string(std::to_underlying(inv->id)) : std::string{"a leaf"}
					)};
				}
				return *s;
			}

			bool isMasked(const shapes::Dim& d) const {
				return d == shapes::Dim::symbol(maskSymbol) || (batchSymbols.size() > 0 && d == merged);
			}

			std::vector<size_t> maskedDims(const shapes::Shape& s) const {
				std::vector<size_t> dims;
				for (size_t i = 0; i < s.size(); ++i)
					if (isMasked(s[i]))
						dims.push_back(i);
				return dims;
			}

			const Node& intConst(int v) { return graph.addConstNode(Atomic{v}); }


			/** @brief The mask of dimension @p d of a tensor of shape @p s, as a bool tensor broadcastable to @p s. */
			const Node& dimMask(const shapes::Shape& s, size_t d) {
				const std::string key = shapes::str(s) + "#" + std::to_string(d);
				if (auto it = maskCache.find(key); it != maskCache.end())
					return *it->second;
				const Node* result = nullptr;
				const int rank = static_cast<int>(s.size());
				if (batchSymbols.size() > 0 && s[d] == merged) {
					// A flattened [batch..., seq] dimension: the flattened mask (row-major, like the flattening).
					std::vector<NodeRef> args{mask};
					for (int k = 0; k < rank; ++k)
						args.push_back(intConst(k == static_cast<int>(d) ? -2 : 1));
					result = &graph.addInvocationNode(FIDs::RESHAPE, std::move(args));
				} else {
					// The positions of the mask's dimensions (batch..., seq) in s.
					std::vector<int> positions;
					for (const auto& b : batchSymbols) {
						auto it = std::find(s.begin(), s.end(), shapes::Dim::symbol(b));
						if (it == s.end())
							throw MaskError{std::format(
									"cannot align the mask with a tensor of shape {} (no batch dimension {})",
									shapes::str(s), b
							)};
						positions.push_back(static_cast<int>(it - s.begin()));
					}
					positions.push_back(static_cast<int>(d));
					// Permute the mask so that its dimensions are in the order of their positions in s.
					std::vector<int> order(positions.size());
					std::iota(order.begin(), order.end(), 0);
					std::sort(order.begin(), order.end(), [&](int a, int b) { return positions[a] < positions[b]; });
					const Node* m = &mask;
					if (!std::is_sorted(positions.begin(), positions.end()))
						m = &graph.addInvocationNode(
								FIDs::PERMUTE, {*m, graph.addConstNode(Atomic{order})}
						);
					std::vector<int> sorted(positions);
					std::sort(sorted.begin(), sorted.end());
					// Insert size-1 dimensions everywhere else (-1 copies the next dimension of the mask).
					std::vector<NodeRef> args{*m};
					for (int k = 0; k < rank; ++k)
						args.push_back(intConst(std::binary_search(sorted.begin(), sorted.end(), k) ? -1 : 1));
					result = &graph.addInvocationNode(FIDs::RESHAPE, std::move(args));
				}
				maskCache.emplace(key, result);
				return *result;
			}

			/** @brief The conjunction of the masks of the dimensions @p dims of a tensor of shape @p s. */
			const Node& combinedMask(const shapes::Shape& s, const std::vector<size_t>& dims) {
				const Node* m = &dimMask(s, dims.front());
				for (size_t i = 1; i < dims.size(); ++i)
					m = &graph.addInvocationNode(FIDs::WHERE, {*m, dimMask(s, dims[i]), *m}); // m and m_i
				return *m;
			}

			/** @brief 1 where @p cond holds, else 0, in the float type of @p like (e.g. bfloat16). */
			const Node& indicator(const Node& cond, const Node& like) {
				const Node& w = graph.addInvocationNode(
						FIDs::WHERE, {cond, graph.addConstNode(Atomic{1.0}), graph.addConstNode(Atomic{0.0})}
				);
				const auto* target = std::get_if<TensorType>(&getDataType(like));
				const auto* type = std::get_if<TensorType>(&getDataType(w));
				if (target == nullptr || type == nullptr || !isFloat(target->type) || target->type == type->type)
					return w;
				return graph.addInvocationNode(FIDs::CAST, {w, graph.addConstNode(Atomic{static_cast<int>(target->type)})});
			}

			/** @brief @p x with @p value at the masked positions of the dimensions @p dims. */
			const Node& fill(const Node& x, const shapes::Shape& s, const std::vector<size_t>& dims, double value) {
				const Node& cond = combinedMask(s, dims);
				// Already filled with the same mask and value (e.g. the forward part of a gradient function)?
				if (const auto* inv = std::get_if<InvocationNode>(&x); inv && inv->id == FIDs::WHERE &&
																	 &inv->arguments[0].get() == &cond) {
					const auto* c = std::get_if<ConstNode>(&inv->arguments[2].get());
					if (c != nullptr && std::holds_alternative<double>(c->value) && std::get<double>(c->value) == value)
						return x;
				}
				inserted = true;
				return graph.addInvocationNode(FIDs::WHERE, {cond, x, graph.addConstNode(Atomic{value})});
			}

			const Node& rebuildInvocation(const Node& node, const InvocationNode& inv) {
				std::vector<NodeRef> args;
				bool changed = false;
				for (const auto& a : inv.arguments) {
					const Node& r = rebuild(a.get());
					changed |= &r != &a.get();
					args.push_back(std::cref(r));
				}
				auto argShape = [&](size_t i) -> const shapes::Shape& { return shapeOf(inv.arguments[i].get()); };
				// Only values of masked type have masked elements to take care of: values computed from `x.data` and
				// `x.mask` only are plain, their masked elements are what the code made them (e.g. a large negative bias
				// before a softmax).
				auto plain = [&](size_t i) {
					const auto* t = std::get_if<TensorType>(&getDataType(inv.arguments[i].get()));
					return t != nullptr && !t->masked;
				};
				auto make = [&]() -> const Node& {
					if (!changed)
						return node;
					return graph.addInvocationNode(inv.id, std::vector<NodeRef>(args));
				};

				switch (inv.id) {
				case FIDs::MAX:
				case FIDs::MIN:
				case FIDs::SUM:
				case FIDs::MEAN: {
					const auto& s = argShape(0);
					const size_t last = s.size() - 1;
					if (s.empty() || !isMasked(s[last]) || plain(0))
						return make();
					const double value = inv.id == FIDs::MAX ? std::numeric_limits<float>::lowest()
									   : inv.id == FIDs::MIN ? std::numeric_limits<float>::max()
															 : 0.0;
					args[0] = std::cref(fill(args[0].get(), s, {last}, value));
					const Node& reduced = graph.addInvocationNode(inv.id, std::vector<NodeRef>(args));
					if (inv.id != FIDs::MEAN)
						return reduced;
					// mean = sum / valid count = mean(filled) / (fraction of valid elements)
					const Node& indicator = this->indicator(dimMask(s, last), inv.arguments[0].get());
					const Node& fraction = graph.addInvocationNode(FIDs::MEAN, {indicator});
					return graph.addInvocationNode(FIDs::TENSOR_DIV, {reduced, fraction});
				}
				case FIDs::SUM_ALL:
				case FIDs::MEAN_ALL: {
					const auto& s = argShape(0);
					auto dims = maskedDims(s);
					if (dims.empty() || plain(0))
						return make();
					const Node& filled = fill(args[0].get(), s, dims, 0.0);
					const Node& sum = graph.addInvocationNode(FIDs::SUM_ALL, {filled});
					if (inv.id == FIDs::SUM_ALL)
						return sum;
					const Node& indicator = this->indicator(combinedMask(s, dims), inv.arguments[0].get());
					const Node& count = graph.addInvocationNode(
							FIDs::SUM_ALL,
							{graph.addInvocationNode(FIDs::BROADCAST_LIKE, {indicator, args[0]})}
					);
					return graph.addInvocationNode(FIDs::TENSOR_DIV, {sum, count});
				}
				case FIDs::TENSOR_DOT: {
					// Contraction over K: the last dimension of a and the second-to-last (or only) one of b.
					const auto& sa = argShape(0);
					const auto& sb = argShape(1);
					const size_t ka = sa.size() - 1, kb = sb.size() >= 2 ? sb.size() - 2 : 0;
					bool filled = false;
					if (!sa.empty() && isMasked(sa[ka]) && !plain(0)) {
						args[0] = std::cref(fill(args[0].get(), sa, {ka}, 0.0));
						filled = true;
					}
					if (!sb.empty() && isMasked(sb[kb]) && !plain(1)) {
						args[1] = std::cref(fill(args[1].get(), sb, {kb}, 0.0));
						filled = true;
					}
					if (!filled)
						return make();
					return graph.addInvocationNode(inv.id, std::vector<NodeRef>(args));
				}
				case FIDs::COLLAPSE_SUM_LIKE: {
					// Sums over the dimensions in which `like` was broadcast: zero the masked ones that are summed.
					const auto& sg = argShape(0);
					const auto& sl = argShape(1);
					std::vector<size_t> dims;
					for (size_t k = 0; k < sg.size(); ++k) {
						const auto offset = static_cast<std::ptrdiff_t>(sg.size()) - static_cast<std::ptrdiff_t>(sl.size());
						const auto kl = static_cast<std::ptrdiff_t>(k) - offset;
						const bool summed = kl < 0 || (sl[kl] == shapes::Dim::constant(1) && !(sg[k] == shapes::Dim::constant(1)));
						if (summed && isMasked(sg[k]))
							dims.push_back(k);
					}
					if (dims.empty())
						return make();
					args[0] = std::cref(fill(args[0].get(), sg, dims, 0.0));
					return graph.addInvocationNode(inv.id, std::vector<NodeRef>(args));
				}
				case FIDs::SCATTER_ADD_ROWS: {
					// Adds the rows of g at their indices (e.g. embedding gradients): masked rows add nothing.
					const auto& sg = argShape(0);
					auto dims = maskedDims(sg);
					if (dims.empty())
						return make();
					args[0] = std::cref(fill(args[0].get(), sg, dims, 0.0));
					return graph.addInvocationNode(inv.id, std::vector<NodeRef>(args));
				}
				case FIDs::MASK_OF: { // x.mask: 1.0 at the valid elements, 0.0 at the masked ones
					inserted = true;
					const auto& s = argShape(0);
					auto dims = maskedDims(s);
					if (dims.empty())
						return graph.addConstNode(Atomic{1.0});
					return indicator(combinedMask(s, dims), inv.arguments[0].get());
				}
				case FIDs::UNMASK: { // x.data: zeros at the masked elements
					const auto& s = argShape(0);
					auto dims = maskedDims(s);
					return dims.empty() ? args[0].get() : fill(args[0].get(), s, dims, 0.0);
				}
				default:
					return make();
				}
			}
		};
	} // namespace

	std::expected<Masked, std::string> apply(
			Computegraph& graph, const FunctionDecl& func, const shapes::Shapes& shapes
	) {
		FunctionDecl result = func;
		const Node* mask = &graph.addInputNode(TensorType{PrimitiveType::boolean, {Elipsis{}}}, "mask");
		result.parameters.emplace_back("mask", std::cref(std::get<InputNode>(*mask)));
		try {
			Masker masker{graph, shapes, *mask};
			{
				// The graph nodes of the inputs (the parameters refer to the InputNode alternative of the node).
				std::unordered_set<const Node*> seen;
				std::vector<const Node*> stack;
				for (const auto& output : func.outputs)
					stack.push_back(&output.get());
				while (!stack.empty()) {
					const Node* n = stack.back();
					stack.pop_back();
					if (!seen.insert(n).second)
						continue;
					if (const auto* in = std::get_if<InputNode>(n)) {
						if (std::any_of(func.parameters.begin(), func.parameters.end(), [&](const auto& p) {
								return &p.second.get() == in;
							}))
							masker.fillInput(*n);
					} else if (const auto* inv = std::get_if<InvocationNode>(n)) {
						for (const auto& a : inv->arguments)
							stack.push_back(&a.get());
					}
				}
			}
			result.outputs.clear();
			for (const auto& output : func.outputs)
				result.outputs.push_back(std::cref(masker.maskOutput(output.get(), masker.rebuild(output.get()))));
			masker.propagateModuleCalls();
		} catch (const MaskError& e) {
			return std::unexpected(func.name + ": " + e.message);
		}
		return Masked{std::move(result), mask};
	}

} // namespace mdl::masking
