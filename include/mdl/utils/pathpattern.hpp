#ifndef MDL_UTILS_PATHPATTERN_HPP
#define MDL_UTILS_PATHPATTERN_HPP

#include <string>
#include <string_view>
#include <vector>

namespace mdl {

	/**
	 * @brief Selects dotted paths (e.g. weight paths like `layers.3.attn.W`) with gitignore-style patterns.
	 * @details
	 *  - Segments are separated by `.`; within a segment, `*` matches any characters and `?` a single character.
	 *  - A segment `**` matches any number (including zero) of segments.
	 *  - A pattern that matches a prefix of a path selects the whole subtree (`layers.0` selects `layers.0.W`).
	 *  - `!pattern` deselects; patterns are applied in order and the last matching one decides.
	 *  - If there are no patterns or the first one is a negation, all paths start out selected (so `!embed.**` alone
	 *    selects everything but the embedding); otherwise all paths start out deselected.
	 *  - Empty patterns and patterns starting with `#` are ignored.
	 */
	class PathPatterns {
	public:
		PathPatterns() = default;
		explicit PathPatterns(std::vector<std::string> patterns);

		/** @brief Parses newline-separated patterns (e.g. the contents of a file). */
		static PathPatterns parse(std::string_view text);

		/** @brief Whether @p path is selected. */
		bool selects(std::string_view path) const;

		bool empty() const { return patterns.empty(); }

	private:
		struct Pattern {
			std::vector<std::string> segments;
			bool negated;
		};
		std::vector<Pattern> patterns;
	};

} // namespace mdl

#endif
