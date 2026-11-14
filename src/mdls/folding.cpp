// textDocument/foldingRange: blocks, multi-line comments, and consecutive imports.
#include "./features.hpp"

mdls::json mdls::foldingRanges(const Document& doc) {
	json ranges = json::array();
	const auto tokens = tokenize(doc.text);
	auto add = [&](size_t first, size_t last, const char* kind) { // lines from 1
		if (last > first) {
			json r{{"startLine", first - 1}, {"endLine", last - 1}};
			if (kind != nullptr)
				r["kind"] = kind;
			ranges.push_back(std::move(r));
		}
	};
	std::vector<size_t> open; // lines of the open braces
	size_t importsFirst = 0, importsLast = 0;
	for (size_t i = 0; i < tokens.size(); ++i) {
		const Token& t = tokens[i];
		if (t.is("{")) {
			open.push_back(t.begin.line);
		} else if (t.is("}") && !open.empty()) {
			add(open.back(), t.begin.line - 1, nullptr); // (the closing brace stays visible)
			open.pop_back();
		} else if (t.comment() && t.end.line > t.begin.line) {
			add(t.begin.line, t.end.line, "comment");
		} else if (t.is("import")) {
			if (importsFirst == 0 || t.begin.line > importsLast + 1) {
				add(importsFirst, importsLast, "imports");
				importsFirst = t.begin.line;
			}
			importsLast = t.begin.line;
		}
	}
	add(importsFirst, importsLast, "imports");
	return ranges;
}
