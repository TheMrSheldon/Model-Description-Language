#ifndef MDL_PASSES_SHAPES_HPP
#define MDL_PASSES_SHAPES_HPP

#include "../computegraphv2/builtins.hpp"
#include "../computegraphv2/datatype.hpp"
#include "../computegraphv2/node.hpp"

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
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
	 *
	 * The rank-polymorphic types of the computegraph (see resultType) have two more kinds of entries: a group of any
	 * number of dimensions (`...`, or dimensions of unknown size), and a compile-time dimension that is not a constant
	 * (yet), e.g. a hyperparameter of a template.
	 */
	struct Dim {
		int64_t factor = 1;
		std::vector<std::string> symbols; ///< sorted
		bool known = true;
		bool group = false;			///< `...`: any number of dimensions
		const Node* node = nullptr; ///< a compile-time dimension that is not constant (yet)

		static Dim constant(int64_t value) { return Dim{.factor = value}; }
		static Dim symbol(std::string name) { return Dim{.factor = 1, .symbols = {std::move(name)}}; }
		static Dim unknown() { return Dim{.factor = 1, .symbols = {}, .known = false}; }
		static Dim ellipsis() { return Dim{.group = true}; }
		static Dim of(const Node& node) { return Dim{.node = &node}; }

		bool isConstant() const { return known && !group && node == nullptr && symbols.empty(); }
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
	/**
	 * @brief Infers shapes on demand, also of nodes that are created later (e.g. by the gradient pass); the shapes are
	 * memoized. The rules are those of infer().
	 */
	class Oracle {
	public:
		Oracle(unsigned ellipsisRank, ConstantEvaluator evaluate, std::unordered_map<const Node*, Shape> known = {});
		~Oracle();
		Oracle(const Oracle&) = delete;
		Oracle& operator=(const Oracle&) = delete;

		/** @brief The shape of @p node, or nullptr if it cannot be inferred. */
		const Shape* shapeOf(const Node& node);

		/** @brief Declares the shape of @p node (e.g. of an input whose declared type is imprecise). */
		void assume(const Node& node, Shape shape);

	private:
		struct Impl;
		std::unique_ptr<Impl> impl;
	};

	/**
	 * @brief The type of the result of the builtin @p id applied to @p args, by the same rules as infer(), but
	 * rank-polymorphic: `...` in the argument types stays a group of dimensions (the computegraph's node types).
	 * @details Where the result cannot be expressed exactly (e.g. the dimensions of a group are reduced, or a size
	 * depends on a group), its dimensions are `...` (unknown); a result whose shape cannot be inferred at all is
	 * `float[...]`. The masked flag is not set here (see Computegraph::addInvocationNode).
	 */
	DataType resultType(builtins::FunctionIDs id, const std::vector<std::reference_wrapper<const Node>>& args);

	/**
	 * @brief The rank-polymorphic shape of a type (`...` is a group, see Dim), e.g. to compare types; std::nullopt for
	 * types that are neither tensors nor scalars.
	 */
	std::optional<Shape> shapeOfType(const DataType& type);

	/**
	 * @brief Whether a computed type matches a declared one (e.g. a function's result against its declared return type).
	 * @details Only definite mismatches count: the shapes are compared from the end up to the first `...` of either,
	 * and a dimension that is not constant matches any. With @p elements, floating-point and integer/boolean element
	 * types differ as well. Float tensors of different float types never match (conversions are explicit). With
	 * @p strict, sizes that are different compile-time expressions (e.g. `indim` and `outdim`) do not match either:
	 * they may differ in an instance.
	 */
	bool matches(const DataType& declared, const DataType& computed, bool elements = true, bool strict = false);

	/**
	 * @brief Checks the arguments of the builtin @p id for definite mismatches (e.g. the inner dimensions of a matrix
	 * product, or shapes that do not broadcast). With @p strict, also for sizes that are different compile-time
	 * expressions (e.g. `indim @ outdim`), which may differ in an instance.
	 * @return a description of the mismatch, or std::nullopt.
	 */
	std::optional<std::string>
	check(builtins::FunctionIDs id, const std::vector<std::reference_wrapper<const Node>>& args, bool strict = false);

	/**
	 * @brief The float type of float results of non-float values (e.g. the mean of integers, or `x.mask` of an integer
	 * tensor): the type that `float` stands for (see parser::ParseOptions::defaultFloat).
	 */
	void setDefaultFloat(PrimitiveType type);

	/** @brief A type for messages, e.g. `float[..., 768]`. */
	std::string str(const DataType& type);

	std::expected<Shapes, std::string>
	infer(const FunctionDecl& func, unsigned ellipsisRank, const ConstantEvaluator& evaluate,
		  const std::unordered_map<const Node*, Shape>& known = {});

} // namespace mdl::shapes

#endif
