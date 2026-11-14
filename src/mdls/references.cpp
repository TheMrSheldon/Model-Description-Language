// textDocument/references, documentHighlight, prepareRename, and rename.
#include "./features.hpp"

#include <fstream>
#include <iterator>
#include <regex>
#include <set>

namespace {
	/** @brief The definition at the cursor (a reference to it, or its declaration). */
	const mdl::analysis::Reference* referenceAtCursor(const mdls::Document& doc, const mdls::json& params) {
		if (doc.index == nullptr || doc.indexText != doc.text)
			return nullptr; // (positions of an earlier text)
		return doc.index->referenceAt(
				doc.sourceName(), mdls::toSource(mdls::LineIndex{doc.text}, params.at("position"))
		);
	}

	/** @brief The text of @p file: an open document's, else from disk. */
	std::string textOf(const std::string& file, const std::map<std::string, std::string>& openTexts) {
		if (auto it = openTexts.find(file); it != openTexts.end())
			return it->second;
		std::ifstream in{file, std::ios::binary};
		return std::string{std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
	}

	/** @brief The references to @p symbol by file. */
	std::map<std::string, std::vector<mdl::SourceRange>>
	referencesTo(const mdl::analysis::SemanticIndex& index, size_t symbol) {
		std::map<std::string, std::vector<mdl::SourceRange>> result;
		for (const auto& r : index.references())
			if (r.symbol == symbol)
				result[r.file].push_back(r.range);
		return result;
	}

	/** @brief Whether @p file is in the standard library (not renamed). */
	bool inStandardLibrary(const std::string& file) {
		std::error_code ec;
		for (const auto& dir : mdl::parser::defaultStdlibDirs()) {
			auto relative = std::filesystem::relative(file, dir, ec);
			if (!ec && !relative.empty() && *relative.begin() != "..")
				return true;
		}
		return false;
	}

	const std::set<std::string> keywords{"import", "module",   "struct", "masked", "for",  "in",
										 "array",  "template", "auto",	 "return", "true", "false"};
} // namespace

mdls::json
mdls::references(const Document& doc, const json& params, const std::map<std::string, std::string>& openTexts) {
	json locations = json::array();
	const auto* ref = referenceAtCursor(doc, params);
	if (ref == nullptr || !ref->symbol)
		return locations;
	const bool declaration = params.contains("context") && params["context"].value("includeDeclaration", true);
	const auto& definition = doc.index->symbols()[*ref->symbol];
	for (const auto& [file, ranges] : referencesTo(*doc.index, *ref->symbol)) {
		const LineIndex lines{file == doc.sourceName() ? doc.text : textOf(file, openTexts)};
		for (const auto& range : ranges)
			if (declaration || !(file == definition.file && range == definition.range))
				locations.push_back(
						{{"uri", file == doc.sourceName() ? doc.uri : pathToUri(file)},
						 {"range", toProtocol(lines, range)}}
				);
	}
	return locations;
}

mdls::json mdls::documentHighlight(const Document& doc, const json& params) {
	json highlights = json::array();
	if (doc.index == nullptr || doc.indexText != doc.text)
		return highlights;
	const LineIndex lines{doc.text};
	const auto cursor = toSource(lines, params.at("position"));
	const std::string file = doc.sourceName();
	// An einops axis: its occurrences in the same pattern (on the same line).
	for (const auto& a : doc.index->axes())
		if (a.file == file && a.range.begin.line == cursor.line && a.range.begin.column <= cursor.column &&
			cursor.column <= a.range.end.column) {
			for (const auto& b : doc.index->axes())
				if (b.file == file && b.name == a.name && b.range.begin.line == a.range.begin.line)
					highlights.push_back({{"range", toProtocol(lines, b.range)}, {"kind", 1}});
			return highlights;
		}
	const auto* ref = doc.index->referenceAt(file, cursor);
	if (ref == nullptr || !ref->symbol)
		return highlights;
	const auto& definition = doc.index->symbols()[*ref->symbol];
	for (const auto& range : referencesTo(*doc.index, *ref->symbol)[file])
		highlights.push_back(
				{{"range", toProtocol(lines, range)},
				 {"kind", definition.file == file && definition.range == range ? 3 : 2}}
		); // write : read
	return highlights;
}

mdls::json mdls::prepareRename(const Document& doc, const json& params) {
	const auto* ref = referenceAtCursor(doc, params);
	if (ref == nullptr || !ref->symbol)
		throw RequestError(RequestFailed, "nothing to rename here");
	const auto& definition = doc.index->symbols()[*ref->symbol];
	if (inStandardLibrary(definition.file))
		throw RequestError(RequestFailed, "definitions of the standard library cannot be renamed");
	return {{"range", toProtocol(LineIndex{doc.text}, ref->range)}, {"placeholder", definition.name}};
}

mdls::json mdls::rename(const Document& doc, const json& params, const std::map<std::string, std::string>& openTexts) {
	const std::string name = params.at("newName");
	static const std::regex identifier{"[_a-zA-Z][_a-zA-Z0-9]*"};
	if (!std::regex_match(name, identifier) || keywords.contains(name))
		throw RequestError(InvalidParams, "`" + name + "` is not a valid name");
	prepareRename(doc, params); // (checks)
	const auto* ref = referenceAtCursor(doc, params);
	json changes = json::object();
	for (const auto& [file, ranges] : referencesTo(*doc.index, *ref->symbol)) {
		const LineIndex lines{file == doc.sourceName() ? doc.text : textOf(file, openTexts)};
		json edits = json::array();
		for (const auto& range : ranges)
			edits.push_back({{"range", toProtocol(lines, range)}, {"newText", name}});
		changes[file == doc.sourceName() ? doc.uri : pathToUri(file)] = std::move(edits);
	}
	return {{"changes", changes}};
}
