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
			Masker(Computegraph& graph, const shapes::Shapes& shapes, const Node& mask, bool sanitize)
					: graph(graph), shapes(shapes), mask(mask), sanitize(sanitize) {
				if (shapes.ellipsis.empty())
					throw MaskError{"the function has no `...` dimensions to mask"};
				maskSymbol = shapes.ellipsis.back();
				batchSymbols.assign(shapes.ellipsis.begin(), shapes.ellipsis.end() - 1);
				merged = shapes::Dim::constant(1);
				for (const auto& s : shapes.ellipsis)
					merged = merged * shapes::Dim::symbol(s);
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
			const bool sanitize;
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

			static DataType boolType() { return TensorType{PrimitiveType::boolean, {Elipsis{}}}; }
			static DataType floatType() { return TensorType{PrimitiveType::floating, {Elipsis{}}}; }

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
					result = &graph.addInvocationNode(FIDs::RESHAPE, boolType(), std::move(args));
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
								FIDs::PERMUTE, boolType(), {*m, graph.addConstNode(Atomic{order})}
						);
					std::vector<int> sorted(positions);
					std::sort(sorted.begin(), sorted.end());
					// Insert size-1 dimensions everywhere else (-1 copies the next dimension of the mask).
					std::vector<NodeRef> args{*m};
					for (int k = 0; k < rank; ++k)
						args.push_back(intConst(std::binary_search(sorted.begin(), sorted.end(), k) ? -1 : 1));
					result = &graph.addInvocationNode(FIDs::RESHAPE, boolType(), std::move(args));
				}
				maskCache.emplace(key, result);
				return *result;
			}

			/** @brief The conjunction of the masks of the dimensions @p dims of a tensor of shape @p s. */
			const Node& combinedMask(const shapes::Shape& s, const std::vector<size_t>& dims) {
				const Node* m = &dimMask(s, dims.front());
				for (size_t i = 1; i < dims.size(); ++i)
					m = &graph.addInvocationNode(FIDs::WHERE, boolType(), {*m, dimMask(s, dims[i]), *m}); // m and m_i
				return *m;
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
				return graph.addInvocationNode(FIDs::WHERE, getDataType(x), {cond, x, graph.addConstNode(Atomic{value})});
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
				auto make = [&]() -> const Node& {
					if (!changed)
						return node;
					return graph.addInvocationNode(inv.id, inv.type, std::vector<NodeRef>(args));
				};

				switch (inv.id) {
				case FIDs::MAX:
				case FIDs::MIN:
				case FIDs::SUM:
				case FIDs::MEAN: {
					const auto& s = argShape(0);
					const size_t last = s.size() - 1;
					if (s.empty() || !isMasked(s[last]))
						return make();
					const double value = inv.id == FIDs::MAX ? std::numeric_limits<float>::lowest()
									   : inv.id == FIDs::MIN ? std::numeric_limits<float>::max()
															 : 0.0;
					if (sanitize && (inv.id == FIDs::MAX || inv.id == FIDs::MIN))
						return make(); // the backward pass creates no extrema; the forward ones are masked already
					args[0] = std::cref(fill(args[0].get(), s, {last}, value));
					const Node& reduced = graph.addInvocationNode(inv.id, inv.type, std::vector<NodeRef>(args));
					if (inv.id != FIDs::MEAN || sanitize)
						return reduced;
					// mean = sum / valid count = mean(filled) / (fraction of valid elements)
					const Node& indicator = graph.addInvocationNode(
							FIDs::WHERE, floatType(),
							{dimMask(s, last), graph.addConstNode(Atomic{1.0}), graph.addConstNode(Atomic{0.0})}
					);
					const Node& fraction = graph.addInvocationNode(FIDs::MEAN, floatType(), {indicator});
					return graph.addInvocationNode(FIDs::TENSOR_DIV, inv.type, {reduced, fraction});
				}
				case FIDs::SUM_ALL:
				case FIDs::MEAN_ALL: {
					const auto& s = argShape(0);
					auto dims = maskedDims(s);
					if (dims.empty())
						return make();
					const Node& filled = fill(args[0].get(), s, dims, 0.0);
					const Node& sum = graph.addInvocationNode(FIDs::SUM_ALL, inv.type, {filled});
					if (inv.id == FIDs::SUM_ALL || sanitize)
						return inv.id == FIDs::SUM_ALL ? sum
													   : graph.addInvocationNode(FIDs::MEAN_ALL, inv.type, {filled});
					const Node& indicator = graph.addInvocationNode(
							FIDs::WHERE, floatType(),
							{combinedMask(s, dims), graph.addConstNode(Atomic{1.0}), graph.addConstNode(Atomic{0.0})}
					);
					const Node& count = graph.addInvocationNode(
							FIDs::SUM_ALL, inv.type,
							{graph.addInvocationNode(FIDs::BROADCAST_LIKE, getDataType(args[0].get()), {indicator, args[0]})}
					);
					return graph.addInvocationNode(FIDs::TENSOR_DIV, inv.type, {sum, count});
				}
				case FIDs::TENSOR_DOT: {
					// Contraction over K: the last dimension of a and the second-to-last (or only) one of b.
					const auto& sa = argShape(0);
					const auto& sb = argShape(1);
					const size_t ka = sa.size() - 1, kb = sb.size() >= 2 ? sb.size() - 2 : 0;
					bool filled = false;
					if (!sa.empty() && isMasked(sa[ka])) {
						args[0] = std::cref(fill(args[0].get(), sa, {ka}, 0.0));
						filled = true;
					}
					if (!sb.empty() && isMasked(sb[kb])) {
						args[1] = std::cref(fill(args[1].get(), sb, {kb}, 0.0));
						filled = true;
					}
					if (!filled)
						return make();
					return graph.addInvocationNode(inv.id, inv.type, std::vector<NodeRef>(args));
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
					return graph.addInvocationNode(inv.id, inv.type, std::vector<NodeRef>(args));
				}
				case FIDs::SCATTER_ADD_ROWS: {
					// Adds the rows of g at their indices (e.g. embedding gradients): masked rows add nothing.
					const auto& sg = argShape(0);
					auto dims = maskedDims(sg);
					if (dims.empty())
						return make();
					args[0] = std::cref(fill(args[0].get(), sg, dims, 0.0));
					return graph.addInvocationNode(inv.id, inv.type, std::vector<NodeRef>(args));
				}
				case FIDs::MASK_OF: { // x.mask: 1.0 at the valid elements, 0.0 at the masked ones
					inserted = true;
					const auto& s = argShape(0);
					auto dims = maskedDims(s);
					if (dims.empty())
						return graph.addConstNode(Atomic{1.0});
					return graph.addInvocationNode(
							FIDs::WHERE, floatType(),
							{combinedMask(s, dims), graph.addConstNode(Atomic{1.0}), graph.addConstNode(Atomic{0.0})}
					);
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
			Computegraph& graph, const FunctionDecl& func, const shapes::Shapes& shapes, const Node* mask
	) {
		const bool sanitize = mask != nullptr;
		FunctionDecl result = func;
		if (mask == nullptr) {
			mask = &graph.addInputNode(TensorType{PrimitiveType::boolean, {Elipsis{}}}, "mask");
			result.parameters.emplace_back("mask", std::cref(std::get<InputNode>(*mask)));
		}
		try {
			Masker masker{graph, shapes, *mask, sanitize};
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
