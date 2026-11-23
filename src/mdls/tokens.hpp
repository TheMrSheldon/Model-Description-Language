#ifndef MDLS_TOKENS_HPP
#define MDLS_TOKENS_HPP

#include <mdl/diagnostics.hpp>

#include <string>
#include <string_view>
#include <vector>

namespace mdls {
	/** @brief A token of a document (lines from 1, columns in code points; @p end of the last line if multi-line). */
	struct Token {
		size_t type;
		std::string text;
		mdl::SourcePosition begin, end;

		bool is(std::string_view t) const { return text == t; }
		bool identifier() const;
		bool comment() const;
	};

	/** @brief The tokens of @p text (without whitespace; with comments). The text need not parse. */
	std::vector<Token> tokenize(const std::string& text);
} // namespace mdls

#endif
