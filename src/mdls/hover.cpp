// textDocument/hover: the declaration, kind, inferred types, and documentation of the name at the cursor.
#include "./features.hpp"

#include <filesystem>
#include <set>

namespace {
	using mdl::analysis::SymbolKind;

	std::string kindName(SymbolKind kind) {
		switch (kind) {
		case SymbolKind::module:
			return "module";
		case SymbolKind::structure:
			return "struct";
		case SymbolKind::function:
			return "function";
		case SymbolKind::method:
			return "member function";
		case SymbolKind::hyperparameter:
			return "hyperparameter";
		case SymbolKind::templateParameter:
			return "template parameter";
		case SymbolKind::member:
			return "member";
		case SymbolKind::parameter:
			return "parameter";
		case SymbolKind::variable:
			return "variable";
		case SymbolKind::loopVariable:
			return "loop variable";
		}
		return {};
	}

	std::string listTypes(const std::set<std::string>& types) {
		std::string s = types.size() > 1 ? "Types: " : "Type: ";
		bool first = true;
		for (const auto& t : types) {
			s += (first ? "`" : ", `") + t + "`";
			first = false;
		}
		return s;
	}
} // namespace

mdls::json mdls::hover(const Document& doc, const json& params) {
	if (doc.index == nullptr)
		return nullptr;
	const LineIndex lines{doc.text};
	// The index may be of an earlier text (while the current one does not parse): positions are mapped.
	const auto position = doc.toIndex(toSource(lines, params.at("position")));
	// A part of an import path: the documentation of the file (`@file`), or the files of the package.
	if (const auto* import = doc.index->importAt(doc.sourceName(), position)) {
		const bool package = import->targets.size() > 1 || std::filesystem::path{import->targets.front()}.stem().string(
														   ) != import->name.substr(import->name.rfind('.') + 1);
		std::string md = "```mdl\nimport " + import->name + "\n```\n";
		md += package ? "*package*"
					  : "*file* `" + std::filesystem::path{import->targets.front()}.filename().string() + "`";
		if (!import->documentation.empty())
			md += "\n\n---\n\n" + import->documentation;
		json result{{"contents", {{"kind", "markdown"}, {"value", md}}}};
		if (auto begin = doc.fromIndex(import->range.begin))
			result["range"] = toProtocol(
					lines,
					mdl::SourceRange{
							*begin,
							{begin->line, begin->column + (import->range.end.column - import->range.begin.column)}
					}
			);
		return result;
	}
	// `auto`: the inferred types of the declared names (also without inlay hints).
	const auto tokens = tokenize(doc.text);
	const auto cursor = toSource(lines, params.at("position"));
	for (size_t i = 0; i < tokens.size(); ++i)
		if (tokens[i].is("auto") && tokens[i].begin <= cursor && cursor <= tokens[i].end) {
			std::string md;
			for (size_t k = i + 1; k < tokens.size() && (tokens[k].identifier() || tokens[k].is(",")); ++k) {
				if (!tokens[k].identifier())
					continue;
				const auto* r = doc.index->referenceAt(doc.sourceName(), doc.toIndex(tokens[k].begin));
				if (r == nullptr)
					continue;
				for (const auto& type : r->types) // the declaration with its inferred type
					md += "auto " + tokens[k].text + ": " + type + "\n";
			}
			if (md.empty())
				return nullptr;
			md = "```mdl\n" + md + "```\n*inferred type*";
			return {{"contents", {{"kind", "markdown"}, {"value", md}}},
					{"range", toProtocol(lines, mdl::SourceRange{tokens[i].begin, tokens[i].end})}};
		}
	// An einops axis: its size.
	for (const auto& a : doc.index->axes())
		if (a.file == doc.sourceName() && a.range.begin.line == position.line &&
			a.range.begin.column <= position.column && position.column <= a.range.end.column) {
			std::string sizes;
			for (const auto& s : a.sizes)
				sizes += (sizes.empty() ? "`" : ", `") + s + "`";
			return {{"contents", {{"kind", "markdown"}, {"value", "*einops axis* `" + a.name + "`: size " + sizes}}}};
		}
	const auto* ref = doc.index->referenceAt(doc.sourceName(), position);
	if (ref == nullptr)
		return nullptr;
	const auto range = doc.fromIndex(ref->range.begin);
	std::string md;
	if (ref->symbol) {
		const auto& s = doc.index->symbols()[*ref->symbol];
		md = "```mdl\n" + (s.detail.empty() ? s.name : s.detail) + "\n```\n";
		md += "*" + kindName(s.kind) + "*" + (s.container.empty() ? "" : " of `" + s.container + "`");
		const auto& types = ref->types.empty() ? s.types : ref->types;
		if (!types.empty() && s.kind != SymbolKind::module && s.kind != SymbolKind::structure)
			md += "\n\n" + listTypes(types);
		for (const auto& weight : ref->weights) // the path of the weight (e.g. in a checkpoint)
			md += "\n\nWeight: `" + weight + "`";
		if (!s.documentation.empty())
			md += "\n\n---\n\n" + s.documentation;
	} else if (!ref->types.empty()) {
		md = listTypes(ref->types);
		for (const auto& weight : ref->weights)
			md += "\n\nWeight: `" + weight + "`";
	} else {
		return nullptr;
	}
	json result{{"contents", {{"kind", "markdown"}, {"value", md}}}};
	if (range) {
		const size_t length = ref->range.end.column - ref->range.begin.column;
		result["range"] = toProtocol(lines, mdl::SourceRange{*range, {range->line, range->column + length}});
	}
	return result;
}
