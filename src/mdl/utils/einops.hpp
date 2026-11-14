#pragma once

#include <mdl/computegraphv2/graph.hpp>
#include <mdl/computegraphv2/node.hpp>

#include <expected>
#include <map>
#include <string>
#include <vector>

namespace mdl::einops {

	enum class AtomKind { Named, Anon, Ellipsis, Literal, Chunk };

	struct Atom {
		AtomKind kind;
		std::string name; ///< for Named
		int64_t value;	  ///< for Literal (size) or Chunk (chunk size N in /N)
	};

	/// A spec item: a single atom (ungrouped) or a merged group (atoms in parens).
	struct SpecItem {
		bool is_group;			 ///< true → enclosed in parentheses
		std::vector<Atom> atoms; ///< atoms inside (exactly 1 for ungrouped items)
	};

	/// Parsed left- or right-hand side of an einops expression.
	using Spec = std::vector<SpecItem>;

	/**
 * @brief Apply an einops rearrangement to a tensor in the compute graph.
 *
 * Emits PERMUTE, EXPAND, and/or RESHAPE InvocationNodes as needed to transform
 * @p input from the layout described by @p lhs to the layout described by @p rhs.
 *
 * Dimension sizes are inferred from the input TensorType and from @p dimBindings.
 * When all non-batch dimensions are statically known, a full integer permutation
 * is emitted.  When an ellipsis is present, post-batch dimensions use negative
 * indices (e.g. -1 = last dim) so the backend can handle unknown batch rank.
 *
 * @param graph       Compute graph where nodes are added.
 * @param input       Input tensor node.
 * @param lhs         Parsed LHS spec (describes the input layout).
 * @param rhs         Parsed RHS spec (describes the desired output layout).
 * @param dimBindings Named dimension → size node (from einops key=value pairs).
 * @return The result node, or an error message describing what went wrong.
 */
	std::expected<std::reference_wrapper<const Node>, std::string>
	apply(Computegraph& graph, const Node& input, const Spec& lhs, const Spec& rhs,
		  const std::map<std::string, const Node*>& dimBindings);

} // namespace mdl::einops
