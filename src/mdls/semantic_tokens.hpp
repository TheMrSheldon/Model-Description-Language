#ifndef MDLS_SEMANTIC_TOKENS_HPP
#define MDLS_SEMANTIC_TOKENS_HPP

#include "./protocol.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace mdls {
	/** @brief A classified token: line and column from 0 (code points), its length, type and modifiers (bit set). */
	struct SemanticToken {
		size_t line, column, length;
		unsigned type, modifiers;
	};

	/**
	 * @brief The token types (indices into this list; the legend of the protocol). Besides the standard types, `axis`
	 * marks the axis names of einops patterns (`x[... (h d) -> h ... d | h=heads]`).
	 */
	const std::vector<std::string>& semanticTokenTypes();
	const std::vector<std::string>& semanticTokenModifiers();

	/**
	 * @brief Classifies the identifiers of @p text from its parse tree (no type checking, so it works on code with
	 * errors): definitions and references of modules, structs, functions, hyperparameters, members, parameters,
	 * variables; annotations; einops axes. Keywords, literals, and operators are left to the editor's grammar (they
	 * need no context, and a grammar can distinguish them further, e.g. control-flow keywords).
	 */
	std::vector<SemanticToken> semanticTokens(std::string_view text);

	/** @brief The protocol's relative encoding of @p tokens (UTF-16 columns). */
	json encodeSemanticTokens(const std::vector<SemanticToken>& tokens, const LineIndex& lines);
} // namespace mdls

#endif
