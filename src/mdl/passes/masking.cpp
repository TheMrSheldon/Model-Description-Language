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
						m = &graph.addInvocationNode(FIDs::PERMUTE, {*m, graph.addConstNode(Atomic{order})});
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
				return graph.addInvocationNode(
						FIDs::CAST, {w, graph.addConstNode(Atomic{static_cast<int>(target->type)})}
				);
			}

			/** @brief @p x with @p value at the masked positions of the dimensions @p dims. */
			const Node& fill(const Node& x, const shapes::Shape& s, const std::vector<size_t>& dims, double value) {
				const Node& cond = combinedMask(s, dims);
				// Already filled with the same mask and value (e.g. the forward part of a gradient function)?
				if (const auto* inv = std::get_if<InvocationNode>(&x);
					inv && inv->id == FIDs::WHERE && &inv->arguments[0].get() == &cond) {
					const auto* c = std::get_if<ConstNode>(&inv->arguments[2].get());
					if (c != nullptr && std::holds_alternative<double>(c->value) && std::get<double>(c->value) == value)
						return x;
				}
				inserted = true;
				return graph.addInvocationNode(FIDs::WHERE, {cond, x, graph.addConstNode(Atomic{value})});
			}

			/** @brief Whether @p d merges the masked dimension with others (e.g. `(...)` or `(n ...)`). */
			bool mergesMask(const shapes::Dim& d) const {
				return std::ranges::find(d.symbols, maskSymbol) != d.symbols.end();
			}

			/**
			 * @brief A reduction whose argument flattens the masked dimension together with others (e.g.
			 * `mean(x[... -> (...)])` over all elements): the masked positions are filled with @p value in the shape
			 * before the flattening (so any order of the merged dimensions works); a mean divides by the fraction of
			 * valid elements, flattened the same way. Null if the argument is not such a flattening.
			 */
			const Node* reduceFlattened(const InvocationNode& inv, double value) {
				const auto* flat = std::get_if<InvocationNode>(&inv.arguments[0].get());
				if (flat == nullptr || flat->id != FIDs::RESHAPE)
					return nullptr;
				const Node& x = flat->arguments[0].get();
				const auto& sx = shapeOf(x);
				const auto dims = maskedDims(sx);
				if (dims.empty())
					return nullptr;
				auto reshaped = [&](const Node& input) -> const Node& {
					std::vector<NodeRef> args{std::cref(input)};
					for (size_t i = 1; i < flat->arguments.size(); ++i)
						args.push_back(std::cref(rebuild(flat->arguments[i].get())));
					return graph.addInvocationNode(FIDs::RESHAPE, std::move(args));
				};
				const Node& filled = reshaped(fill(rebuild(x), sx, dims, value));
				std::vector<NodeRef> args{std::cref(filled)};
				for (size_t i = 1; i < inv.arguments.size(); ++i)
					args.push_back(std::cref(rebuild(inv.arguments[i].get())));
				const Node& reduced = graph.addInvocationNode(inv.id, std::move(args));
				if (inv.id != FIDs::MEAN)
					return &reduced;
				// mean = sum / valid count = mean(filled) / (fraction of valid elements)
				const Node& valid = graph.addInvocationNode(
						FIDs::BROADCAST_LIKE, {indicator(combinedMask(sx, dims), x), rebuild(x)}
				);
				const Node& fraction = graph.addInvocationNode(FIDs::MEAN, {reshaped(valid)});
				return &graph.addInvocationNode(FIDs::TENSOR_DIV, {reduced, fraction});
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
					const double value = inv.id == FIDs::MAX   ? std::numeric_limits<float>::lowest()
										 : inv.id == FIDs::MIN ? std::numeric_limits<float>::max()
															   : 0.0;
					if (!s.empty() && !isMasked(s[last]) && !plain(0) && mergesMask(s[last])) {
						if (const Node* r = reduceFlattened(inv, value))
							return *r;
						// (Its elements cannot be told apart once other operations work on the flattened values.)
						throw MaskError{std::format(
								"cannot mask a reduction over the dimension {}, which merges the masked dimension with "
								"others: reduce the flattened tensor directly (e.g. mean_flat(x * 2) instead of "
								"mean(x[... -> (...)] * 2))",
								s[last].str()
						)};
					}
					if (s.empty() || !isMasked(s[last]) || plain(0))
						return make();
					args[0] = std::cref(fill(args[0].get(), s, {last}, value));
					const Node& reduced = graph.addInvocationNode(inv.id, std::vector<NodeRef>(args));
					if (inv.id != FIDs::MEAN)
						return reduced;
					// mean = sum / valid count = mean(filled) / (fraction of valid elements)
					const Node& indicator = this->indicator(dimMask(s, last), inv.arguments[0].get());
					const Node& fraction = graph.addInvocationNode(FIDs::MEAN, {indicator});
					return graph.addInvocationNode(FIDs::TENSOR_DIV, {reduced, fraction});
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
						const auto offset =
								static_cast<std::ptrdiff_t>(sg.size()) - static_cast<std::ptrdiff_t>(sl.size());
						const auto kl = static_cast<std::ptrdiff_t>(k) - offset;
						const bool summed =
								kl < 0 || (sl[kl] == shapes::Dim::constant(1) && !(sg[k] == shapes::Dim::constant(1)));
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

	namespace {
		/** @brief The boolean mask of a suffix mask: SUFFIX_MASK of the lengths and the first parameter of `...`. */
		const Node& suffixMask(Computegraph& graph, const FunctionDecl& func, const Node& lengths) {
			const InputNode* like = nullptr;
			for (const auto& [name, p] : func.parameters) {
				const auto* t = std::get_if<TensorType>(&p.get().type);
				if (t != nullptr && !t->shape.empty() && std::holds_alternative<Elipsis>(t->shape.front())) {
					like = &p.get();
					break;
				}
			}
			// Its graph node (the parameters refer to the InputNode alternative of the node).
			std::unordered_set<const Node*> seen;
			std::vector<const Node*> stack;
			for (const auto& output : func.outputs)
				stack.push_back(&output.get());
			while (like != nullptr && !stack.empty()) {
				const Node* n = stack.back();
				stack.pop_back();
				if (!seen.insert(n).second)
					continue;
				if (std::get_if<InputNode>(n) == like)
					return graph.addInvocationNode(builtins::FunctionIDs::SUFFIX_MASK, {lengths, *n});
				if (const auto* inv = std::get_if<InvocationNode>(n))
					for (const auto& a : inv->arguments)
						stack.push_back(&a.get());
			}
			throw MaskError{"a suffix mask needs a parameter whose shape starts with `...` (and that the function uses)"
			};
		}
	} // namespace

	std::expected<Masked, std::string>
	apply(Computegraph& graph, const FunctionDecl& func, const shapes::Shapes& shapes, MaskKind kind) {
		FunctionDecl result = func;
		const Node* input = nullptr;
		const Node* mask = nullptr;
		if (kind == MaskKind::element) {
			input = mask = &graph.addInputNode(TensorType{PrimitiveType::boolean, {Elipsis{}}}, "mask");
			result.parameters.emplace_back("mask", std::cref(std::get<InputNode>(*input)));
		} else {
			// The number of valid elements per row (the boolean mask over `...`: see suffixMask).
			input = &graph.addInputNode(TensorType{PrimitiveType::integer, {Elipsis{}}}, "lengths");
			result.parameters.emplace_back("lengths", std::cref(std::get<InputNode>(*input)));
		}
		try {
			if (mask == nullptr)
				mask = &suffixMask(graph, func, *input);
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
		return Masked{std::move(result), input};
	}

} // namespace mdl::masking
