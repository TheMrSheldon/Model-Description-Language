#ifndef MDL_KERNELS_EXPR_HPP
#define MDL_KERNELS_EXPR_HPP

/**
 * @file
 * @brief The expressions of kernel packages: argument values and applicability conditions (see docs/dev/spec/kernel-packages.md).
 * @details One small typed language: integers, booleans, element types, shapes (lists of integers), and pointers,
 * over the captures of a match (`$x.shape[-1]`, `prod($x.shape[:-1])`, `$x.dtype in [float32, bfloat16]`). Parsing and
 * type checking are independent of the compiler backend; evaluation over a match is done by the backend.
 */

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace mdl::kernels::expr {

	enum class Type { integer, boolean, dtype, shape, pointer, real }; ///< real: an attribute (float arguments)
	std::string str(Type type);

	struct Node;
	using NodePtr = std::shared_ptr<const Node>;

	/** @brief A node of an expression; the type is known after parsing. */
	struct Node {
		enum class Kind {
			integer,   ///< a literal (value)
			dtypeName, ///< a literal element type (name), e.g. float32
			list,	   ///< `[a, b, ...]` of integers (a shape) or element types (for `in`)
			/**
			 * `$capture.field`: of a tensor shape, ndim, dtype, data (a pointer; NULL if the capture is not a tensor),
			 * is_tensor; of a scalar value (integer) or real; of a shape value (e.g. a permutation) values; of an
			 * operator `attr.NAME` (real). `$out` (or `$out0`, `$out1`, ... for several results): the result.
			 */
			access,
			unary,	   ///< op: `-`, `not`
			binary,	   ///< op: + - * // % == != < <= > >= and or
			in,		   ///< args[0] in args[1] (a list)
			index,	   ///< args[0][args[1]] of a shape (negative indices count from the end)
			slice,	   ///< args[0][from:to] of a shape (absent bounds: hasFrom/hasTo false)
			/**
			 * name(args...): prod(shape), len(shape), strides(shape) (the row-major strides of a contiguous tensor),
			 * permute(list, perm) (entry i is list[perm[i]]), unpermute(list, perm) (its inverse); an empty permutation is
			 * the identity
			 */
			call,
			conditional, ///< args[1] if args[0] else args[2] (only the chosen branch is evaluated if args[0] is constant)
			null,		 ///< the null pointer (e.g. `null if $x.is_tensor else $y.data`)
		};
		Kind kind;
		Type type;
		int64_t value = 0;
		std::string name, field, op;
		std::vector<NodePtr> args;
		int64_t from = 0, to = 0;
		bool hasFrom = false, hasTo = false;
	};

	/**
	 * @brief Parses and type-checks @p text; @p captures are the names that `$name` may refer to (plus `out`, the
	 * result). The error message gives the column (from 1) of the problem.
	 */
	std::expected<NodePtr, std::string> parse(const std::string& text, const std::vector<std::string>& captures);

	/** @brief The expression as text (normalized), e.g. for explanations. */
	std::string str(const Node& node);

} // namespace mdl::kernels::expr

#endif
