#ifndef MDLS_PARTIAL_HPP
#define MDLS_PARTIAL_HPP

#include <string>

namespace mdls {
	/**
	 * @brief @p text with the definitions that do not parse replaced by spaces (newlines are kept, so all positions stay
	 * the same): top-level definitions, and inside a module that does not parse its members that do not. The rest of a
	 * document with syntax errors can then be analyzed as usual.
	 */
	std::string blankBrokenDefinitions(const std::string& text);
} // namespace mdls

#endif
