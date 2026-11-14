#ifndef MDLS_PROTOCOL_HPP
#define MDLS_PROTOCOL_HPP

#include <nlohmann/json.hpp>

#include <filesystem>
#include <istream>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

/** @brief The transport and data conversions of the Language Server Protocol (JSON-RPC over stdin/stdout). */
namespace mdls {
	using json = nlohmann::json;

	/** @brief Reads one message (`Content-Length` header, then the JSON body); std::nullopt at the end of the input. */
	std::optional<json> readMessage(std::istream& in);

	/** @brief Writes one message. */
	void writeMessage(std::ostream& out, const json& message);

	/** @brief The path of a `file://` URI (percent-decoded; `file:///c%3A/x` is `c:/x` on Windows). */
	std::optional<std::filesystem::path> uriToPath(std::string_view uri);
	std::string pathToUri(const std::filesystem::path& path);

	/**
	 * @brief A document's text split into lines, to convert between the frontend's columns (Unicode code points) and
	 * the protocol's (UTF-16 code units).
	 */
	class LineIndex {
	public:
		explicit LineIndex(std::string_view text);

		size_t lineCount() const { return lines.size(); }
		/** @brief The UTF-16 column of code-point column @p column on @p line (both from 0). */
		size_t toUtf16(size_t line, size_t column) const;
		/** @brief The code-point column of UTF-16 column @p column on @p line. */
		size_t fromUtf16(size_t line, size_t column) const;
		/** @brief The UTF-16 length of @p count code points starting at code-point @p column on @p line. */
		size_t utf16Length(size_t line, size_t column, size_t count) const { return toUtf16(line, column + count) - toUtf16(line, column); }
		std::string_view line(size_t line) const;

	private:
		std::vector<std::string> lines;
	};
} // namespace mdls

#endif
