// textDocument/codeAction: quick fixes for the diagnostics in the requested range (the frontend's fixes, renaming
// for naming rules, and imports for names of the standard library).
#include "./features.hpp"

#include <regex>

namespace {
	/** @brief Whether the ranges @p a and @p b overlap (or touch). */
	bool overlaps(const mdl::SourceRange& a, const mdl::SourceRange& b) {
		return !(a.end < b.begin || b.end < a.begin);
	}

	/** @brief The text in backticks of @p hint (`write `x``, `rename it to `x``), if any. */
	std::string quoted(const std::string& hint) {
		const auto open = hint.find('`'), close = hint.rfind('`');
		return open == std::string::npos || close == open ? std::string{} : hint.substr(open + 1, close - open - 1);
	}
} // namespace

mdls::json mdls::codeActions(
		const Document& doc, const json& params, const std::map<std::string, std::string>& openTexts,
		const std::map<std::string, std::string>& stdNames
) {
	json actions = json::array();
	const LineIndex lines{doc.text};
	const auto& requested = params.at("range");
	const mdl::SourceRange range{toSource(lines, requested.at("start")), toSource(lines, requested.at("end"))};
	auto action = [&](const std::string& title, json changes) {
		actions.push_back({{"title", title}, {"kind", "quickfix"}, {"edit", {{"changes", std::move(changes)}}}});
	};
	for (const auto& d : doc.diagnostics) {
		if (!overlaps(d.range, range))
			continue;
		// The frontend's fixes (e.g. `float32(x)`, `.bias=false`).
		for (const auto& fix : d.fixes) {
			json edits = json::array();
			for (const auto& e : fix.edits)
				edits.push_back({{"range", toProtocol(lines, e.range)}, {"newText", e.text}});
			action(fix.title, {{doc.uri, edits}});
		}
		// A name that breaks a naming rule: rename the definition and its uses.
		if (d.flag == "naming")
			if (const auto name = quoted(d.hint); !name.empty()) {
				const json at{
						{"textDocument", {{"uri", doc.uri}}},
						{"position", toProtocol(lines, d.range.begin)},
						{"newName", name}
				};
				try {
					action("Rename to `" + name + "`", rename(doc, at, openTexts)["changes"]);
				} catch (const RequestError&) {
				}
			}
		// A name (a value or a type) that the standard library defines: import it from its file.
		static const std::regex undefined{"Variable (\\w+) (is )?not defined|(\\w+) does not name a type"};
		std::smatch match;
		if (std::regex_match(d.message, match, undefined)) {
			const std::string name = match[1].matched ? match[1].str() : match[3].str();
			if (auto it = stdNames.find(name); it != stdNames.end()) {
				// After the last import (or at the top).
				size_t line = 0;
				for (size_t i = 0; i < lines.lineCount(); ++i)
					if (lines.line(i).starts_with("import ") || lines.line(i).starts_with("from "))
						line = i + 1;
				const json at{{"line", line}, {"character", 0}};
				action("Import " + name + " from " + it->second,
					   {{doc.uri, json::array(
										  {{{"range", {{"start", at}, {"end", at}}},
											{"newText", "from " + it->second + " import " + name + ";\n"}}}
								  )}});
			}
		}
	}
	return actions;
}
