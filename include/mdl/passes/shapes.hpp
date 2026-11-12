#ifndef MDL_PASSES_SHAPES_HPP
#define MDL_PASSES_SHAPES_HPP

#include "../computegraphv2/datatype.hpp"
#include "../computegraphv2/node.hpp"

#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mdl::shapes {

	/**
	 * @brief The size of one dimension: a constant times a product of symbols (a monomial), e.g. `768`, `B1`, or
	 * `B0*B1` (a flattened batch and sequence dimension).
	 * @details The symbols are the runtime dimensions of `...` in the signature of the analyzed function (`B` or `B0`,
	 * `B1`, ...). Products of symbols arise from reshapes that merge dimensions. A dimension that cannot be expressed
	 * this way is unknown.
	 */
	struct Dim {
		int64_t factor = 1;
		std::vector<std::string> symbols; ///< sorted
		bool known = true;

		static Dim constant(int64_t value) { return Dim{.factor = value, .symbols = {}, .known = true}; }
		static Dim symbol(std::string name) { return Dim{.factor = 1, .symbols = {std::move(name)}, .known = true}; }
		static Dim unknown() { return Dim{.factor = 1, .symbols = {}, .known = false}; }

		bool isConstant() const { return known && symbols.empty(); }
		bool operator==(const Dim&) const = default;
		Dim operator*(const Dim& other) const;
		/** @brief The quotient if this dimension is divisible by @p other (as monomials), else unknown. */
		Dim operator/(const Dim& other) const;
		std::string str() const;
	};

	using Shape = std::vector<Dim>;

	std::string str(const Shape& shape);

	/** @brief The shapes of the nodes of a function, for a fixed number of dimensions of `...`. */
	struct Shapes {
		std::unordered_map<const Node*, Shape> of;
		/** @brief The symbols of the dimensions of `...` (B0, B1, ... or B). */
		std::vector<std::string> ellipsis;

		const Shape* find(const Node& node) const {
			auto it = of.find(&node);
			return it == of.end() ? nullptr : &it->second;
		}
	};

	/** @brief Evaluates compile-time integer expressions (e.g. hyperparameters); std::nullopt if not constant. */
	using ConstantEvaluator = std::function<std::optional<int64_t>(const Node&)>;

	/**
	 * @brief Infers the shape of every node reachable from the outputs of @p func, where every `...` in the signature
	 * (parameters and weights) stands for the same @p ellipsisRank dimensions.
	 * @details The rules match the semantics of the builtins as the backends implement them (e.g. `max` and `sum` keep
	 * the reduced axis with size 1, `mean` removes it). Scalars have the shape `[]`.
	 * @param known shapes of nodes that are known otherwise, e.g. of inputs whose declared types are imprecise (the
	 * residuals of a backward pass have the shapes the forward pass inferred for them).
	 * @return the shapes, or an error naming the first node whose shape could not be inferred.
	 */
	std::expected<Shapes, std::string> infer(
			const FunctionDecl& func, unsigned ellipsisRank, const ConstantEvaluator& evaluate,
			const std::unordered_map<const Node*, Shape>& known = {}
	);

} // namespace mdl::shapes

#endif
