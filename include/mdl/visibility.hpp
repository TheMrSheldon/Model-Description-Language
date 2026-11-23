#ifndef MDL_VISIBILITY_HPP
#define MDL_VISIBILITY_HPP

/**
 * @file
 * @brief The visibility of a definition or member (see docs/language-spec.md, Visibility).
 */

namespace mdl {

	/** @brief Where a name is visible: the file (a member: its module or struct), the package, or everywhere. */
	enum class Visibility { Private, Package, Public };

} // namespace mdl

#endif
