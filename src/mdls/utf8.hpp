#ifndef MDLS_UTF8_HPP
#define MDLS_UTF8_HPP

// UTF-8 text positions, decoded like the parser's input (antlrcpp::Utf8, used by ANTLRInputStream): columns in code
// points match the frontend's also for invalid UTF-8 (each invalid byte is one replacement character).
#include <support/Utf8.h>

#include <cstddef>
#include <string_view>

namespace mdls::utf8 {
	/** @brief The byte length of the code point at the start of @p text (at least 1; 0 if @p text is empty). */
	inline size_t sequenceLength(std::string_view text) {
		return text.empty() ? 0 : antlrcpp::Utf8::decode(text).second;
	}

	/** @brief The number of UTF-16 code units of the code point at the start of @p text (1 or 2; 0 if empty). */
	inline size_t utf16Units(std::string_view text) {
		return text.empty() ? 0 : antlrcpp::Utf8::decode(text).first > 0xFFFF ? 2 : 1;
	}

	/** @brief The number of code points of @p text. */
	inline size_t codePoints(std::string_view text) {
		size_t count = 0;
		for (size_t i = 0; i < text.size(); i += sequenceLength(text.substr(i)))
			++count;
		return count;
	}

	/** @brief The byte offset @p count code points after byte offset @p offset of @p text (at most its size). */
	inline size_t advance(std::string_view text, size_t offset, size_t count) {
		for (size_t i = 0; i < count && offset < text.size(); ++i)
			offset += sequenceLength(text.substr(offset));
		return offset;
	}
} // namespace mdls::utf8

#endif
