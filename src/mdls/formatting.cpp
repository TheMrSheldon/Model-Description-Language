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
