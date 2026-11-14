// textDocument/definition and textDocument/documentSymbol.
#include "./features.hpp"

#include <fstream>
#include <iterator>
#include <map>

namespace {
	using mdl::analysis::SymbolKind;

	/** @brief The protocol's SymbolKind of a definition. */
	int symbolKind(SymbolKind kind) {
		switch (kind) {
		case SymbolKind::module:
			return 5; // Class
		case SymbolKind::structure:
			return 23; // Struct
		case SymbolKind::function:
			return 12; // Function
		case SymbolKind::method:
			return 6; // Method
		case SymbolKind::hyperparameter:
		case SymbolKind::templateParameter:
			return 26; // TypeParameter
		case SymbolKind::member:
			return 8; // Field
		default:
			return 13; // Variable
		}
	}
} // namespace

mdls::json mdls::definition(const Document& doc, const json& params) {
	if (doc.index == nullptr)
		return nullptr;
	const LineIndex lines{doc.text};
	const auto* ref = doc.index->referenceAt(doc.sourceName(), doc.toIndex(toSource(lines, params.at("position"))));
	if (ref == nullptr || !ref->symbol)
		return nullptr;
	const auto& s = doc.index->symbols()[*ref->symbol];
	if (s.file == doc.sourceName()) {
		// In this document: its position in the current text (the start of the change if it is in changed lines).
		auto begin = doc.fromIndex(s.range.begin);
		if (!begin)
			begin = doc.toIndex(s.range.begin);
		const mdl::SourceRange range{
				*begin, {begin->line, begin->column + (s.range.end.column - s.range.begin.column)}
		};
		return {{"uri", doc.uri}, {"range", toProtocol(lines, range)}};
	}
	// In another file: its columns are converted with that file's text.
	std::ifstream in{s.file, std::ios::binary};
	const std::string text{std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
	return {{"uri", pathToUri(s.file)}, {"range", toProtocol(LineIndex{text}, s.range)}};
}

mdls::json mdls::documentSymbols(const Document& doc) {
	json result = json::array();
	if (doc.index == nullptr || doc.indexText != doc.text)
		return result; // (positions of an earlier text)
	const LineIndex lines{doc.text};
	std::map<std::string, size_t> containers; // module/struct name -> index in result
	auto item = [&](const mdl::analysis::Symbol& s) {
		const auto range = toProtocol(lines, s.range);
		json symbol{{"name", s.name}, {"kind", symbolKind(s.kind)}, {"range", range}, {"selectionRange", range}};
		if (!s.types.empty())
			symbol["detail"] = *s.types.begin();
		return symbol;
	};
	for (const auto& s : doc.index->symbols())
		if (s.file == doc.sourceName() && s.container.empty() &&
			(s.kind == SymbolKind::module || s.kind == SymbolKind::structure || s.kind == SymbolKind::function)) {
			containers[s.name] = result.size();
			result.push_back(item(s));
			result.back()["children"] = json::array();
		}
	for (const auto& s : doc.index->symbols())
		if (s.file == doc.sourceName() && !s.container.empty())
			if (auto it = containers.find(s.container); it != containers.end())
				result[it->second]["children"].push_back(item(s));
	return result;
}
