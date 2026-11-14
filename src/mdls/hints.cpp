// textDocument/inlayHint: inferred types after `auto` declarations, the sizes of einops axes, and the names of
// positional hyperparameters (`Linear[indim: 768, outdim: 3]`).
#include "./features.hpp"

#include <set>

namespace {
	std::string joined(const std::set<std::string>& values) {
		std::string s;
		for (const auto& v : values)
			s += (s.empty() ? "" : " | ") + v;
		return s;
	}
} // namespace

mdls::json mdls::inlayHints(const Document& doc, const json& params) {
	json hints = json::array();
	if (doc.index == nullptr || doc.indexText != doc.text)
		return hints; // (positions of an earlier text)
	const LineIndex lines{doc.text};
	const auto& requested = params.at("range");
	const mdl::SourceRange range{toSource(lines, requested.at("start")), toSource(lines, requested.at("end"))};
	auto inRange = [&](mdl::SourcePosition p) { return range.begin <= p && p <= range.end; };
	auto hint = [&](mdl::SourcePosition at, const std::string& label, int kind, bool padLeft, bool padRight) {
		if (inRange(at))
			hints.push_back(
					{{"position", toProtocol(lines, at)},
					 {"label", label},
					 {"kind", kind},
					 {"paddingLeft", padLeft},
					 {"paddingRight", padRight}}
			);
	};
	const std::string file = doc.sourceName();

	// `auto h = ...`: the inferred type after the name.
	for (const auto& s : doc.index->symbols())
		if (s.file == file && s.kind == mdl::analysis::SymbolKind::variable && s.detail.starts_with("auto ") &&
			!s.types.empty())
			hint(s.range.end, ": " + joined(s.types), 1, false, false);

	// Einops axes: the size after the first occurrence of each axis in a pattern (per line).
	std::set<std::pair<size_t, std::string>> shown;
	for (const auto& a : doc.index->axes())
		if (a.file == file && shown.emplace(a.range.begin.line, a.name).second)
			hint(a.range.end, "=" + joined(a.sizes), 1, false, false);

	// Positional hyperparameters of instantiations: their names.
	const auto tokens = tokenize(doc.text);
	for (size_t i = 1; i < tokens.size(); ++i) {
		if (!tokens[i].is("["))
			continue;
		const auto callee = calleeAt(doc, tokens, i);
		if (callee.symbol == nullptr || !callee.hyperparameters)
			continue;
		const auto spans = parameterSpans(*callee.symbol, '[');
		size_t argument = 0;
		bool atStart = true;
		int depth = 0;
		for (size_t k = i + 1; k < tokens.size(); ++k) {
			const Token& t = tokens[k];
			if (t.comment())
				continue;
			if (depth == 0 && atStart) {
				if (t.is("]"))
					break;
				if (!t.is(".") && argument < spans.size()) // a positional value (named ones start with `.`)
					hint(t.begin, spans[argument].name + ":", 2, false, true);
				atStart = false;
			}
			if (t.is("(") || t.is("["))
				++depth;
			else if ((t.is(")") || t.is("]")) && depth-- == 0)
				break;
			else if (t.is(",") && depth == 0) {
				++argument;
				atStart = true;
			}
		}
	}
	return hints;
}
