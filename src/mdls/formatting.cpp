// textDocument/formatting: mdl-format's style.
#include "./features.hpp"

mdls::json
mdls::formatting(const Document& doc, const json& params, const std::optional<mdl::format::FormatOptions>& style) {
	mdl::format::FormatOptions options;
	if (style)
		options = *style;
	else if (params.contains("options"))
		options.indentWidth = params["options"].value("tabSize", options.indentWidth);
	auto result = mdl::format::format(doc.text, options, doc.sourceName());
	if (!result.ok || result.text == doc.text)
		return json::array();
	const LineIndex lines{doc.text};
	const size_t last = lines.lineCount() - 1;
	json range{
			{"start", {{"line", 0}, {"character", 0}}},
			{"end", {{"line", last}, {"character", lines.toUtf16(last, lines.line(last).size() + 1)}}},
	};
	return json::array({{{"range", range}, {"newText", result.text}}});
}

mdls::json
mdls::rangeFormatting(const Document& doc, const json& params, const std::optional<mdl::format::FormatOptions>& style) {
	// The document is formatted as a whole; the changed lines are replaced if they overlap the requested range.
	const json edits = formatting(doc, params, style);
	if (edits.empty())
		return edits;
	const std::string formatted = edits[0]["newText"];
	const LineIndex before{doc.text}, after{formatted};
	const size_t n = before.lineCount(), m = after.lineCount();
	size_t prefix = 0;
	while (prefix < n && prefix < m && before.line(prefix) == after.line(prefix))
		++prefix;
	size_t suffix = 0;
	while (suffix < n - prefix && suffix < m - prefix && before.line(n - 1 - suffix) == after.line(m - 1 - suffix))
		++suffix;
	const size_t first = params.at("range").at("start").at("line"), last = params.at("range").at("end").at("line");
	if (last < prefix || first >= n - suffix)
		return json::array(); // the changes are outside of the range
	std::string text;
	for (size_t i = prefix; i < m - suffix; ++i)
		text += std::string{after.line(i)} + "\n";
	const json start{{"line", prefix}, {"character", 0}}, end{{"line", n - suffix}, {"character", 0}};
	return json::array({{{"range", {{"start", start}, {"end", end}}}, {"newText", text}}});
}
