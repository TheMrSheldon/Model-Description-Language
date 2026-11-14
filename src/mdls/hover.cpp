// textDocument/hover: the declaration, kind, inferred types, and documentation of the name at the cursor.
#include "./features.hpp"

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
	const auto* ref = doc.index->referenceAt(doc.sourceName(), doc.toIndex(toSource(lines, params.at("position"))));
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
		if (!s.documentation.empty())
			md += "\n\n---\n\n" + s.documentation;
	} else if (!ref->types.empty()) {
		md = listTypes(ref->types);
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
