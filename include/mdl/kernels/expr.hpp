#ifndef MDL_KERNELS_EXPR_HPP
#define MDL_KERNELS_EXPR_HPP

/**
 * @file
 * @brief The expressions of kernel packages: argument values and applicability conditions (see docs/kernels.md).
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

	enum class Type { integer, boolean, dtype, shape, pointer };
	std::string str(Type type);

	struct Node;
	using NodePtr = std::shared_ptr<const Node>;

	/** @brief A node of an expression; the type is known after parsing. */
	struct Node {
		enum class Kind {
			integer,   ///< a literal (value)
			dtypeName, ///< a literal element type (name), e.g. float32
			list,	   ///< `[a, b, ...]` of integers (a shape) or element types (for `in`)
			access,	   ///< `$capture.field` (name = capture, field = shape, ndim, dtype, data, or attr.<name>)
			unary,	   ///< op: `-`, `not`
			binary,	   ///< op: + - * // % == != < <= > >= and or
			in,		   ///< args[0] in args[1] (a list)
			index,	   ///< args[0][args[1]] of a shape (negative indices count from the end)
			slice,	   ///< args[0][from:to] of a shape (absent bounds: hasFrom/hasTo false)
			call,	   ///< name(args...): prod(shape), len(shape)
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
