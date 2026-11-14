#include "./document.hpp"

namespace {
	/** @brief The numbers of equal lines at the start and at the end of @p a and @p b. */
	std::pair<size_t, size_t> commonLines(const mdls::LineIndex& a, const mdls::LineIndex& b) {
		const size_t n = a.lineCount(), m = b.lineCount();
		size_t prefix = 0;
		while (prefix < n && prefix < m && a.line(prefix) == b.line(prefix))
			++prefix;
		size_t suffix = 0;
		while (suffix < n - prefix && suffix < m - prefix && a.line(n - 1 - suffix) == b.line(m - 1 - suffix))
			++suffix;
		return {prefix, suffix};
	}
} // namespace

mdl::SourcePosition mdls::Document::toIndex(mdl::SourcePosition position) const {
	if (indexText == text)
		return position;
	const LineIndex now{text}, then{indexText};
	const auto [prefix, suffix] = commonLines(now, then);
	const size_t n = now.lineCount(), m = then.lineCount();
	const size_t line = position.line - 1; // from 0
	if (line < prefix)
		return position;
	if (line >= n - suffix)
		return {line + m - n + 1, position.column};
	return {prefix + 1, 0}; // in the changed lines
}

std::optional<mdl::SourcePosition> mdls::Document::fromIndex(mdl::SourcePosition position) const {
	if (indexText == text)
		return position;
	const LineIndex now{text}, then{indexText};
	const auto [prefix, suffix] = commonLines(now, then);
	const size_t n = now.lineCount(), m = then.lineCount();
	const size_t line = position.line - 1;
	if (line < prefix)
		return position;
	if (line >= m - suffix)
		return mdl::SourcePosition{line + n - m + 1, position.column};
	return std::nullopt;
}

mdl::SourcePosition mdls::toSource(const LineIndex& lines, const json& position) {
	const size_t line = position.at("line");
	return {line + 1, lines.fromUtf16(line, position.at("character"))};
}

mdls::json mdls::toProtocol(const LineIndex& lines, mdl::SourcePosition position) {
	const size_t line = position.line > 0 ? position.line - 1 : 0;
	return {{"line", line}, {"character", lines.toUtf16(line, position.column)}};
}

mdls::json mdls::toProtocol(const LineIndex& lines, const mdl::SourceRange& range) {
	return {{"start", toProtocol(lines, range.begin)}, {"end", toProtocol(lines, range.end)}};
}
