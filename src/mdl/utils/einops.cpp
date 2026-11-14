#include <mdl/computegraphv2/builtins.hpp>
#include <mdl/computegraphv2/datatype.hpp>
#include <mdl/computegraphv2/node.hpp>
#include <mdl/utils/einops.hpp>

#include <algorithm>
#include <cassert>
#include <format>
#include <numeric>
#include <optional>
#include <set>

using namespace mdl;
using namespace mdl::einops;
using mdl::builtins::FunctionIDs;

namespace {

	// A resolved dimension size: either a compile-time integer or a runtime node.
	using DimSize = std::variant<int64_t, const Node*>;

	// Maps a named dim to its resolved size.
	using DimSizeMap = std::map<std::string, DimSize>;

	/// The size of a dimension if it is statically known: a literal or a constant node (e.g. a hyperparameter value).
	static std::optional<int64_t> staticSize(const TensorType::Shape::value_type& dim) {
		if (const auto* u = std::get_if<unsigned>(&dim))
			return static_cast<int64_t>(*u);
		if (const auto* np = std::get_if<const Node*>(&dim))
			if (const auto* cn = std::get_if<ConstNode>(*np))
				if (const auto* i = std::get_if<int>(&cn->value))
					return *i;
		return std::nullopt;
	}

	// -----------------------------------------------------------------------
	// Helpers
	// -----------------------------------------------------------------------

	static const Node& constNode(Computegraph& graph, int64_t val) {
		return graph.addConstNode(Atomic{static_cast<int>(val)});
	}

	static const Node& permTensorNode(Computegraph& graph, const std::vector<int>& perm) {
		return graph.addConstNode(Atomic{perm});
	}

	static const Node* dimSizeAsNode(Computegraph& graph, const DimSize& s) {
		if (const auto* i = std::get_if<int64_t>(&s))
			return &constNode(graph, *i);
		return std::get<const Node*>(s);
	}

	/// Returns true if the spec contains an ellipsis atom anywhere.
	static bool specHasEllipsis(const Spec& spec) {
		for (auto& item : spec)
			for (auto& atom : item.atoms)
				if (atom.kind == AtomKind::Ellipsis)
					return true;
		return false;
	}

	/// Count items in spec that are NOT the ellipsis item.
	static size_t countNonEllipsisItems(const Spec& spec) {
		size_t n = 0;
		for (auto& item : spec)
			if (!(item.atoms.size() == 1 && item.atoms[0].kind == AtomKind::Ellipsis))
				++n;
		return n;
	}

	/// Flatten a Spec (expanding groups) into a linear sequence of atoms.
	/// Ellipsis is kept as a single entry.
	static std::vector<Atom> flattenSpec(const Spec& spec) {
		std::vector<Atom> atoms;
		for (auto& item : spec)
			for (auto& a : item.atoms)
				atoms.push_back(a);
		return atoms;
	}

	// -----------------------------------------------------------------------
	// Step 1 – Build DimSizeMap from LHS + RHS + input shape + user bindings
	// -----------------------------------------------------------------------

	/// Inject sizes for LHS group atoms that are absent from the RHS but whose
	/// sizes can be deduced from RHS literals.
	///
	/// Algorithm: match RHS literals (positionally) to LHS group atoms that do not
	/// appear by name anywhere in the RHS and have no binding yet.  The first
	/// unmatched literal goes to the first such absent atom, the second to the
	/// second, etc.
	///
	/// Example: (a b) -> 2 b         → a=2  (b is present by name, 2 is matched to a)
	///          (a b) -> a 3         → b=3  (a is present by name, 3 is matched to b)
	///          (a b) c -> a c b     → no literals, no-op
	///          a b  -> a 1 b        → no groups, no absent atoms, 1 becomes a new dim
	static void injectRhsLiteralBindings(const Spec& lhs, const Spec& rhs, DimSizeMap& sizes) {
		// Collect all named dims that appear in the RHS.
		std::set<std::string> rhsNamed;
		for (auto& item : rhs)
			for (auto& atom : item.atoms)
				if (atom.kind == AtomKind::Named)
					rhsNamed.insert(atom.name);

		// Collect LHS group atoms that are absent from the RHS and not yet in sizes,
		// in the order they appear across all LHS groups.
		std::vector<std::string> absentAtoms;
		for (auto& item : lhs)
			if (item.is_group)
				for (auto& atom : item.atoms)
					if (atom.kind == AtomKind::Named && !rhsNamed.count(atom.name) && !sizes.count(atom.name))
						absentAtoms.push_back(atom.name);

		if (absentAtoms.empty())
			return;

		// Collect all RHS literals in order.
		std::vector<int64_t> rhsLiterals;
		for (auto& item : rhs)
			for (auto& atom : item.atoms)
				if (atom.kind == AtomKind::Literal)
					rhsLiterals.push_back(atom.value);

		// Positional match: first literal → first absent atom, etc.
		for (size_t i = 0; i < absentAtoms.size() && i < rhsLiterals.size(); ++i)
			sizes[absentAtoms[i]] = rhsLiterals[i];
	}

	/// For a single LHS group at input shape position i, try to infer the size of
	/// the one unknown atom from the merged dim size and the known atom sizes.
	/// Returns an error if there are 2+ unknowns or if the merged size is dynamic.
	static std::expected<void, std::string>
	inferGroupSizes(const SpecItem& group, const TensorType::Shape::value_type& shapeDim, DimSizeMap& sizes) {
		// Merged size must be statically known for inference.
		int64_t mergedSize = staticSize(shapeDim).value_or(-1);

		std::string unknownName;
		int64_t knownProduct = 1;
		bool dynamicKnown = false; // a known size is a runtime node
		int unknownCount = 0;

		for (auto& atom : group.atoms) {
			if (atom.kind != AtomKind::Named)
				continue;
			auto it = sizes.find(atom.name);
			if (it == sizes.end()) {
				++unknownCount;
				unknownName = atom.name;
			} else if (auto* sz = std::get_if<int64_t>(&it->second)) {
				knownProduct *= *sz;
			} else {
				dynamicKnown = true; // Node* size; can't do static inference
			}
		}

		if (unknownCount == 0)
			return {}; // All known; nothing to do.

		if (unknownCount > 1) {
			// Build a helpful error listing all unknowns.
			std::string unknowns;
			for (auto& atom : group.atoms)
				if (atom.kind == AtomKind::Named && !sizes.count(atom.name))
					unknowns += (unknowns.empty() ? "" : ", ") + atom.name;
			return std::unexpected(std::format(
					"Cannot infer sizes of multiple grouped dimensions ({}); "
					"provide all but one via bindings or as RHS literals",
					unknowns
			));
		}

		// Exactly one unknown.
		if (mergedSize >= 0 && !dynamicKnown) {
			// Merged size and all peers are statically known: infer.
			if (knownProduct == 0 || mergedSize % knownProduct != 0)
				return std::unexpected(std::format(
						"Merged size {} is not divisible by the product of known sizes {} "
						"when inferring grouped dimension '{}'",
						mergedSize, knownProduct, unknownName
				));
			sizes[unknownName] = mergedSize / knownProduct;
		}
		// If the merged size or a peer is dynamic, leave the atom unresolved.
		// applyUnmerge will emit -1 for it and let the backend infer at runtime.
		return {};
	}

	static std::expected<DimSizeMap, std::string> buildDimSizes(
			const Spec& lhs, const Spec& rhs, const TensorType& inputType,
			const std::map<std::string, const Node*>& bindings
	) {
		DimSizeMap sizes;
		for (auto& [name, node] : bindings) {
			// Constant bindings (e.g. `h=2`) are folded so that the sizes of their peers in a group can be inferred.
			if (const auto* cn = std::get_if<ConstNode>(node); cn && std::holds_alternative<int>(cn->value))
				sizes[name] = static_cast<int64_t>(std::get<int>(cn->value));
			else
				sizes[name] = node;
		}

		// Pre-pass: use RHS literals to bind absent LHS group atoms before the
		// main loop, so that the inference step below can see them.
		injectRhsLiteralBindings(lhs, rhs, sizes);

		const auto& shape = inputType.shape;
		bool hasEllipsis = specHasEllipsis(lhs);

		if (!hasEllipsis) {
			if (lhs.size() != shape.size())
				return std::unexpected(std::format(
						"LHS has {} dimension items but input tensor has {} dimensions", lhs.size(), shape.size()
				));

			for (size_t i = 0; i < lhs.size(); ++i) {
				auto& item = lhs[i];
				auto& shapeDim = shape[i];

				auto extractSize = [&]() -> std::optional<DimSize> {
					if (auto size = staticSize(shapeDim))
						return DimSize{*size};
					if (auto* np = std::get_if<const Node*>(&shapeDim))
						return DimSize{*np};
					return std::nullopt; // Elipsis in shape
				};

				if (item.is_group) {
					// Infer the one unknown atom (if any) from the merged dim size.
					if (auto err = inferGroupSizes(item, shapeDim, sizes); !err)
						return std::unexpected(err.error());
				} else {
					assert(item.atoms.size() == 1);
					auto& atom = item.atoms[0];
					if (atom.kind == AtomKind::Named) {
						if (auto sz = extractSize())
							sizes.emplace(atom.name, *sz);
						// If shapeDim is Elipsis (unknown), skip – size stays unknown
					}
					// Anon / Ellipsis / Literal atoms don't contribute named sizes
				}
			}
			return sizes;
		}

		// --- Ellipsis case ---
		// Find ellipsis position in lhs (must be a single-item non-group with Ellipsis atom).
		size_t ellipsisPos = std::numeric_limits<size_t>::max();
		for (size_t i = 0; i < lhs.size(); ++i) {
			if (!lhs[i].is_group && lhs[i].atoms.size() == 1 && lhs[i].atoms[0].kind == AtomKind::Ellipsis) {
				ellipsisPos = i;
				break;
			}
		}
		if (ellipsisPos == std::numeric_limits<size_t>::max())
			return std::unexpected("Internal: ellipsis expected but not found");

		size_t nonEllipsis = countNonEllipsisItems(lhs);
		if (shape.size() < nonEllipsis)
			return std::unexpected(
					std::format("Input rank {} too small for LHS with {} non-ellipsis items", shape.size(), nonEllipsis)
			);

		// Helper: extract size for one non-grouped, non-ellipsis LHS item.
		auto extractNamed = [&](const SpecItem& item, const TensorType::Shape::value_type& dim) {
			if (item.atoms[0].kind == AtomKind::Named) {
				if (auto size = staticSize(dim))
					sizes[item.atoms[0].name] = *size;
				else if (auto* np = std::get_if<const Node*>(&dim))
					sizes[item.atoms[0].name] = *np;
			}
		};

		// Helper: resolve a single LHS item against its corresponding shape dimension.
		// For groups, calls inferGroupSizes (which may infer the one unknown atom).
		auto resolveItem = [&](const SpecItem& item,
							   const TensorType::Shape::value_type& dim) -> std::expected<void, std::string> {
			if (item.is_group)
				return inferGroupSizes(item, dim, sizes);
			extractNamed(item, dim);
			return {};
		};

		// Items before ellipsis match shape[0 .. ellipsisPos-1]
		for (size_t i = 0; i < ellipsisPos; ++i) {
			if (auto err = resolveItem(lhs[i], shape[i]); !err)
				return std::unexpected(err.error());
		}
		// Items after ellipsis match shape[rank - (lhs.size()-1-ellipsisPos) .. rank-1]
		size_t suffixLen = lhs.size() - 1 - ellipsisPos;
		for (size_t k = 0; k < suffixLen; ++k) {
			size_t lhsIdx = ellipsisPos + 1 + k;
			size_t shapeIdx = shape.size() - suffixLen + k;
			if (auto err = resolveItem(lhs[lhsIdx], shape[shapeIdx]); !err)
				return std::unexpected(err.error());
		}
		return sizes;
	}

	// -----------------------------------------------------------------------
	// Step 2 – Un-merge (RESHAPE) if LHS has grouped dims
	// -----------------------------------------------------------------------

	/// If the LHS contains any grouped item, emit a RESHAPE that splits merged
	/// dims into their constituents.  Returns the resulting node (possibly the
	/// original if no reshape is needed) and the flat ordered list of atoms.
	static std::expected<std::pair<const Node*, std::vector<Atom>>, std::string>
	applyUnmerge(Computegraph& graph, const Node& input, const Spec& lhs, const DimSizeMap& sizes, bool hasEllipsis) {
		bool needReshape = false;
		for (auto& item : lhs)
			if (item.is_group) {
				needReshape = true;
				break;
			}

		auto flatAtoms = flattenSpec(lhs); // atoms in the desired flat order

		if (!needReshape)
			return std::make_pair(&input, flatAtoms);

		// Build new shape: for each LHS item expand groups into individual dims.
		// Named dims use sizes; Anon dims use -1 (keep as is); Ellipsis becomes Elipsis{}.
		std::vector<std::reference_wrapper<const Node>> shapeArgs;
		shapeArgs.emplace_back(std::cref(input));

		// The ellipsis group covers the input dims between the LHS items before and after it (one input dim per item).
		auto ellipsisAt = std::find_if(lhs.begin(), lhs.end(), [](const SpecItem& item) {
			return !item.is_group && item.atoms.front().kind == AtomKind::Ellipsis;
		});
		const int ellipsisBefore = static_cast<int>(ellipsisAt - lhs.begin());
		const int ellipsisAfter = ellipsisAt == lhs.end() ? 0 : static_cast<int>(lhs.end() - ellipsisAt - 1);

		for (auto& item : lhs) {
			for (auto& atom : item.atoms) {
				if (atom.kind == AtomKind::Ellipsis) {
					shapeArgs.emplace_back(
							std::cref(graph.addConstNode(Atomic{std::vector<int>{ellipsisBefore, ellipsisAfter}}))
					);
				} else if (atom.kind == AtomKind::Named) {
					auto it = sizes.find(atom.name);
					if (it == sizes.end()) {
						// Size not statically known; use -2 sentinel so the backend passes
						// TVM's infer-dimension marker (-1) to topi::reshape, not passthrough.
						shapeArgs.emplace_back(constNode(graph, -2));
					} else if (auto* sz = std::get_if<int64_t>(&it->second)) {
						shapeArgs.emplace_back(constNode(graph, *sz));
					} else {
						const Node* n = std::get<const Node*>(it->second);
						shapeArgs.emplace_back(std::cref(*n));
					}
				} else if (atom.kind == AtomKind::Anon) {
					shapeArgs.emplace_back(constNode(graph, -1));
				} else if (atom.kind == AtomKind::Literal) {
					shapeArgs.emplace_back(constNode(graph, atom.value));
				}
			}
		}

		const Node& reshaped = graph.addInvocationNode(FunctionIDs::RESHAPE, std::move(shapeArgs));
		return std::make_pair(&reshaped, flatAtoms);
	}

	// -----------------------------------------------------------------------
	// Step 3 – PERMUTE (reorder named dims)
	// -----------------------------------------------------------------------

	/// Compute the permutation that maps flat LHS named-dim order to flat RHS
	/// named-dim order.  Anon and Ellipsis dims are treated as "pass-through" and
	/// assigned indices in relative order.
	///
	/// When \p hasEllipsis is true, dims before the ellipsis use positive indices
	/// and dims after it use negative indices, so the permutation is rank-agnostic.
	static std::vector<int>
	computePermutation(const std::vector<Atom>& flatLhs, const std::vector<Atom>& flatRhs, bool hasEllipsis) {
		// Build an index into flatLhs for each named dim
		std::map<std::string, int> lhsIndex;
		int ellipsisFlatIdx = -1; // index of the Ellipsis atom in flatLhs
		for (int i = 0; i < static_cast<int>(flatLhs.size()); ++i) {
			if (flatLhs[i].kind == AtomKind::Named)
				lhsIndex[flatLhs[i].name] = i;
			else if (flatLhs[i].kind == AtomKind::Ellipsis)
				ellipsisFlatIdx = i;
		}

		// Track anon dim order
		std::vector<int> anonIndices;
		for (int i = 0; i < static_cast<int>(flatLhs.size()); ++i)
			if (flatLhs[i].kind == AtomKind::Anon)
				anonIndices.push_back(i);
		int nextAnon = 0;

		std::vector<int> perm;
		perm.reserve(flatRhs.size());

		int numDimsAfterEllipsis = 0;
		if (hasEllipsis) {
			// Count named dims after ellipsis in flatLhs
			if (ellipsisFlatIdx >= 0)
				for (int i = ellipsisFlatIdx + 1; i < static_cast<int>(flatLhs.size()); ++i)
					if (flatLhs[i].kind != AtomKind::Ellipsis)
						++numDimsAfterEllipsis;
		}

		for (auto& atom : flatRhs) {
			if (atom.kind == AtomKind::Ellipsis) {
				perm.push_back(builtins::kEllipsisPermMarker); // stands for all (batch) dims of the ellipsis
			} else if (atom.kind == AtomKind::Named) {
				auto it = lhsIndex.find(atom.name);
				if (it != lhsIndex.end()) {
					int idx = it->second;
					if (hasEllipsis && ellipsisFlatIdx >= 0 && idx > ellipsisFlatIdx) {
						// Use negative index for post-batch dim
						int negIdx = idx - static_cast<int>(flatLhs.size());
						perm.push_back(negIdx);
					} else {
						perm.push_back(idx);
					}
				}
				// If not in LHS (e.g. new literal dim), skip permutation entry for it
			} else if (atom.kind == AtomKind::Anon) {
				if (nextAnon < static_cast<int>(anonIndices.size()))
					perm.push_back(anonIndices[nextAnon++]);
			}
			// Literals added to RHS don't correspond to an input dim
		}
		return perm;
	}

	static bool isIdentityPermutation(const std::vector<int>& perm) {
		if (perm.size() == 1 && perm[0] == builtins::kEllipsisPermMarker)
			return true; // only the ellipsis (all dims in their order), e.g. `... 1 -> ...`
		for (int i = 0; i < static_cast<int>(perm.size()); ++i)
			if (perm[i] != i)
				return false;
		return true;
	}

	static std::expected<const Node*, std::string>
	applyPermute(Computegraph& graph, const Node& input, const std::vector<int>& perm) {
		if (isIdentityPermutation(perm))
			return &input;

		const Node& permNode = permTensorNode(graph, perm);

		std::vector<std::reference_wrapper<const Node>> args;
		args.emplace_back(std::cref(input));
		args.emplace_back(std::cref(permNode));

		const Node& node = graph.addInvocationNode(FunctionIDs::PERMUTE, std::move(args));
		return &node;
	}

	// -----------------------------------------------------------------------
	// Step 4 – EXPAND (for Anon dims that need broadcasting to a literal size)
	// -----------------------------------------------------------------------

	/// If the RHS contains a literal-size dim corresponding to an Anon (`_`) dim in
	/// LHS, emit an EXPAND to broadcast that dim from 1 (or its current size) to N.
	/// Returns the (possibly unchanged) node.
	static std::expected<const Node*, std::string>
	applyExpand(Computegraph& graph, const Node& input, const Spec& lhs, const Spec& rhs, const DimSizeMap& sizes) {
		// Collect Anon positions in flat LHS (pre-unmerge, but we care about the count)
		std::vector<size_t> anonLhsPositions;
		{
			size_t pos = 0;
			for (auto& item : lhs)
				for (auto& atom : item.atoms) {
					if (atom.kind == AtomKind::Anon)
						anonLhsPositions.push_back(pos);
					if (atom.kind != AtomKind::Ellipsis)
						++pos;
				}
		}

		// Check if any RHS Anon or Literal needs expanding
		// We do a simple pass: if RHS has a literal N and the corresponding LHS dim is Anon
		// (i.e., it could be 1 needing broadcast), emit EXPAND.
		// For simplicity, build the target expand shape from the current node shape, replacing
		// dims that correspond to Anon-with-literal with the literal value.
		const auto& inType = std::get<TensorType>(getDataType(input));
		TensorType::Shape expandShape = inType.shape;
		bool needExpand = false;

		// Map RHS flat position -> literal value for Anon dims
		size_t anonIdx = 0;
		size_t flatPos = 0;
		for (auto& item : rhs) {
			for (auto& atom : item.atoms) {
				if (atom.kind == AtomKind::Anon) {
					// Next anon dim – check if there's a literal following it or if it IS the target
					// (Simple heuristic: if the source dim size differs, expand is needed)
					if (anonIdx < anonLhsPositions.size()) {
						size_t srcPos = anonLhsPositions[anonIdx];
						// Not a literal expansion here; move on
						++anonIdx;
					}
				} else if (atom.kind == AtomKind::Literal && flatPos < expandShape.size()) {
					// Check if the input dim at this position is 1 (broadcastable)
					if (auto* u = std::get_if<unsigned>(&expandShape[flatPos])) {
						if (*u == 1 && atom.value != 1) {
							expandShape[flatPos] = static_cast<unsigned>(atom.value);
							needExpand = true;
						}
					}
				}
				if (atom.kind != AtomKind::Ellipsis)
					++flatPos;
			}
		}

		if (!needExpand)
			return &input;

		std::vector<std::reference_wrapper<const Node>> args;
		args.emplace_back(std::cref(input));
		for (auto& dim : expandShape) {
			if (auto* u = std::get_if<unsigned>(&dim))
				args.emplace_back(constNode(graph, static_cast<int64_t>(*u)));
			else if (auto* np = std::get_if<const Node*>(&dim))
				args.emplace_back(std::cref(**np));
			else
				args.emplace_back(constNode(graph, -1)); // Elipsis -> -1 sentinel
		}
		const Node& node = graph.addInvocationNode(FunctionIDs::EXPAND, std::move(args));
		return &node;
	}

	// -----------------------------------------------------------------------
	// Step 5 – Re-merge (RESHAPE) for RHS groups and literals
	// -----------------------------------------------------------------------

	/// Computes the product of sizes for a group, if all are statically known.
	static std::optional<int64_t> groupProduct(const std::vector<Atom>& atoms, const DimSizeMap& sizes) {
		int64_t prod = 1;
		for (auto& atom : atoms) {
			if (atom.kind == AtomKind::Literal) {
				prod *= atom.value;
			} else if (atom.kind == AtomKind::Named) {
				auto it = sizes.find(atom.name);
				if (it == sizes.end())
					return std::nullopt;
				if (auto* sz = std::get_if<int64_t>(&it->second))
					prod *= *sz;
				else
					return std::nullopt; // dynamic
			} else {
				return std::nullopt;
			}
		}
		return prod;
	}

	static std::expected<const Node*, std::string>
	applyRemerge(Computegraph& graph, const Node& input, const Spec& lhs, const Spec& rhs, const DimSizeMap& sizes) {
		// Determine if we need a reshape: RHS has groups OR new literal dims
		bool needReshape = false;
		for (auto& item : rhs) {
			if (item.is_group) {
				needReshape = true;
				break;
			}
			if (item.atoms.size() == 1 && item.atoms[0].kind == AtomKind::Literal)
				needReshape = true; // literal dim in RHS (e.g. unsqueeze with '1')
		}
		// Also need reshape if the number of output dims differs from current node dims
		const auto& inType = std::get<TensorType>(getDataType(input));
		size_t outDimCount = 0;
		for (auto& item : rhs)
			++outDimCount; // each RHS item is one output dim
		if (outDimCount != inType.shape.size())
			needReshape = true;

		if (!needReshape)
			return &input;

		// The input dims are the flat RHS atoms that exist in the input (all but new literal dims); the ellipsis group
		// covers the input dims between those before and after it.
		int ellipsisBefore = 0, ellipsisAfter = 0;
		{
			bool seen = false;
			for (const auto& atom : flattenSpec(rhs)) {
				if (atom.kind == AtomKind::Ellipsis)
					seen = true;
				else if (atom.kind != AtomKind::Literal)
					++(seen ? ellipsisAfter : ellipsisBefore);
			}
			// Size-1 dims of the input that are squeezed away (literals of the LHS, e.g. `... 1 -> ...`) are not part of the
			// ellipsis either: the permutation leaves them next to the ellipsis dims (before or after them).
			seen = false;
			for (const auto& atom : flattenSpec(lhs)) {
				if (atom.kind == AtomKind::Ellipsis)
					seen = true;
				else if (atom.kind == AtomKind::Literal)
					++(seen ? ellipsisAfter : ellipsisBefore);
			}
		}

		std::vector<std::reference_wrapper<const Node>> shapeArgs;
		shapeArgs.emplace_back(std::cref(input));

		for (auto& item : rhs) {
			if (item.is_group) {
				if (auto prod = groupProduct(item.atoms, sizes)) {
					shapeArgs.emplace_back(constNode(graph, *prod));
				} else {
					// Dynamic group product – use -2 sentinel so backend passes TVM's
					// infer-dimension marker (-1) to topi::reshape, not passthrough.
					shapeArgs.emplace_back(constNode(graph, -2));
				}
			} else {
				assert(item.atoms.size() == 1);
				auto& atom = item.atoms[0];
				if (atom.kind == AtomKind::Named) {
					auto it = sizes.find(atom.name);
					if (it == sizes.end())
						return std::unexpected(std::format("Unknown dim '{}' in RHS", atom.name));
					if (auto* sz = std::get_if<int64_t>(&it->second)) {
						shapeArgs.emplace_back(constNode(graph, *sz));
					} else {
						const Node* n = std::get<const Node*>(it->second);
						shapeArgs.emplace_back(std::cref(*n));
					}
				} else if (atom.kind == AtomKind::Literal) {
					shapeArgs.emplace_back(constNode(graph, atom.value));
				} else if (atom.kind == AtomKind::Ellipsis) {
					shapeArgs.emplace_back(
							std::cref(graph.addConstNode(Atomic{std::vector<int>{ellipsisBefore, ellipsisAfter}}))
					);
				} else if (atom.kind == AtomKind::Anon) {
					shapeArgs.emplace_back(constNode(graph, -1));
				}
			}
		}

		const Node& node = graph.addInvocationNode(FunctionIDs::RESHAPE, std::move(shapeArgs));
		return &node;
	}

} // anonymous namespace

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

std::expected<std::reference_wrapper<const Node>, std::string> mdl::einops::apply(
		Computegraph& graph, const Node& input, const Spec& lhs, const Spec& rhs,
		const std::map<std::string, const Node*>& dimBindings
) {
	// Validate the input is a tensor.
	const DataType& dtype = getDataType(input);
	const auto* inputType = std::get_if<TensorType>(&dtype);
	if (!inputType)
		return std::unexpected("einops: input must be a tensor");

	bool hasEllipsis = specHasEllipsis(lhs) || specHasEllipsis(rhs);

	// Step 1: resolve named dim sizes from LHS + input shape + user bindings.
	auto dimSizes = buildDimSizes(lhs, rhs, *inputType, dimBindings);
	if (!dimSizes)
		return std::unexpected(dimSizes.error());

	// Step 2: un-merge grouped LHS dims via RESHAPE (if any).
	auto unmergeResult = applyUnmerge(graph, input, lhs, *dimSizes, hasEllipsis);
	if (!unmergeResult)
		return std::unexpected(unmergeResult.error());
	auto [afterUnmerge, flatLhsAtoms] = *unmergeResult;

	// Step 3: compute permutation and emit PERMUTE (if order changes).
	auto flatRhsAtoms = flattenSpec(rhs);
	auto perm = computePermutation(flatLhsAtoms, flatRhsAtoms, hasEllipsis);
	auto afterPermute = applyPermute(graph, *afterUnmerge, perm);
	if (!afterPermute)
		return std::unexpected(afterPermute.error());

	// Step 4: EXPAND for Anon dims that need broadcasting to a literal size.
	auto afterExpand = applyExpand(graph, **afterPermute, lhs, rhs, *dimSizes);
	if (!afterExpand)
		return std::unexpected(afterExpand.error());

	// Step 5: merge grouped RHS dims and handle literal dims via RESHAPE.
	auto afterRemerge = applyRemerge(graph, **afterExpand, lhs, rhs, *dimSizes);
	if (!afterRemerge)
		return std::unexpected(afterRemerge.error());

	return std::cref(**afterRemerge);
}
