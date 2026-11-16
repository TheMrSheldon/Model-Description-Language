// Navigation by structure: document links, workspace symbols, selection ranges, call hierarchy, and linked editing of
// einops axes.
#include "./features.hpp"

#include <mdlLexer.h>
#include <mdlParser.h>

#include <fstream>
#include <iterator>
#include <regex>

using antlr4::mdlLexer;
using antlr4::mdlParser;

namespace {
	using mdl::analysis::SymbolKind;

	std::string readText(const std::string& file) {
		std::ifstream in{file, std::ios::binary};
		return std::string{std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
	}

	/** @brief A location in another file (its columns converted with that file's text). */
	mdls::json location(const mdls::Document& doc, const std::string& file, const mdl::SourceRange& range) {
		if (file == doc.sourceName())
			return {{"uri", doc.uri}, {"range", mdls::toProtocol(mdls::LineIndex{doc.text}, range)}};
		return {{"uri", mdls::pathToUri(file)}, {"range", mdls::toProtocol(mdls::LineIndex{readText(file)}, range)}};
	}

	/** @brief The einops axis at @p position of @p doc and its occurrences in the pattern (same line, same name). */
	std::vector<mdl::SourceRange> axisOccurrences(const mdls::Document& doc, mdl::SourcePosition position) {
		std::vector<mdl::SourceRange> ranges;
		if (doc.index == nullptr || doc.indexText != doc.text)
			return ranges;
		for (const auto& a : doc.index->axes())
			if (a.file == doc.sourceName() && a.range.begin.line == position.line &&
				a.range.begin.column <= position.column && position.column <= a.range.end.column) {
				for (const auto& b : doc.index->axes())
					if (b.file == a.file && b.name == a.name && b.range.begin.line == a.range.begin.line)
						ranges.push_back(b.range);
				break;
			}
		return ranges;
	}
} // namespace

std::vector<mdl::SourceRange> mdls::einopsAxisAt(const Document& doc, const json& params) {
	return axisOccurrences(doc, toSource(LineIndex{doc.text}, params.at("position")));
}

mdls::json mdls::linkedEditingRanges(const Document& doc, const json& params) {
	const auto ranges = einopsAxisAt(doc, params);
	if (ranges.empty())
		return nullptr;
	const LineIndex lines{doc.text};
	json result = json::array();
	for (const auto& r : ranges)
		result.push_back(toProtocol(lines, r));
	return {{"ranges", result}, {"wordPattern", "[_a-zA-Z][_a-zA-Z0-9]*"}};
}

mdls::json mdls::documentLinks(const Document& doc) {
	json links = json::array();
	if (doc.index == nullptr || doc.indexText != doc.text)
		return links;
	const LineIndex lines{doc.text};
	// Import paths: the imported file.
	for (const auto& i : doc.index->imports())
		if (i.file == doc.sourceName() && i.targets.size() == 1)
			links.push_back(
					{{"range", toProtocol(lines, i.range)},
					 {"target", pathToUri(i.targets.front())},
					 {"tooltip", "Open " + std::filesystem::path{i.targets.front()}.filename().string()}}
			);
	// `\ref name` in comments: the definition.
	static const std::regex ref{R"([@\\]ref\s+([_a-zA-Z][_a-zA-Z0-9]*))"};
	for (const auto& t : tokenize(doc.text)) {
		if (!t.comment())
			continue;
		for (std::sregex_iterator it{t.text.begin(), t.text.end(), ref}, end; it != end; ++it) {
			const std::string name = (*it)[1];
			const mdl::analysis::Symbol* target = nullptr;
			for (const auto& s : doc.index->symbols())
				if (s.name == name && s.container.empty() &&
					(s.kind == SymbolKind::module || s.kind == SymbolKind::structure || s.kind == SymbolKind::function))
					target = &s;
			if (target == nullptr)
				continue;
			// The position of the name in the comment.
			const size_t offset = static_cast<size_t>(it->position(1));
			const std::string_view before{t.text.data(), offset};
			const size_t newlines = std::count(before.begin(), before.end(), '\n');
			const size_t lineStart = before.rfind('\n');
			const size_t column = newlines == 0 ? t.begin.column + offset : offset - lineStart - 1; // (ASCII before)
			const mdl::SourceRange range{
					{t.begin.line + newlines, column}, {t.begin.line + newlines, column + name.size()}
			};
			const auto loc = location(doc, target->file, target->range);
			const size_t line = loc["range"]["start"]["line"].get<size_t>() + 1;
			links.push_back(
					{{"range", toProtocol(lines, range)},
					 {"target", loc["uri"].get<std::string>() + "#L" + std::to_string(line)},
					 {"tooltip", "Go to " + name}}
			);
		}
	}
	return links;
}

mdls::json mdls::workspaceSymbols(
		const std::vector<std::filesystem::path>& roots, const std::vector<const Document*>& open,
		const std::string& query
) {
	// The definitions of a source text (from its parse tree; cached per file and modification time).
	struct Entry {
		std::string name, container;
		int kind;
		mdl::SourceRange range;
	};
	static std::map<std::string, std::pair<std::filesystem::file_time_type, std::vector<Entry>>> cache;
	auto definitions = [](const std::string& text) {
		std::vector<Entry> entries;
		antlr4::ANTLRInputStream input;
		input.load(text, /*lenient=*/true); // (like the parser, see mdl::invalidUtf8)
		mdlLexer lexer{&input};
		lexer.removeErrorListeners();
		antlr4::CommonTokenStream stream{&lexer};
		mdlParser parser{&stream};
		parser.removeErrorListeners();
		auto* program = parser.program();
		auto range = [](antlr4::tree::TerminalNode* n) {
			auto* t = n->getSymbol();
			return mdl::SourceRange{
					{t->getLine(), t->getCharPositionInLine()},
					{t->getLine(), t->getCharPositionInLine() + t->getText().size()}
			};
		};
		for (auto* m : program->module_def()) {
			if (m->IDENT() == nullptr)
				continue;
			entries.push_back({m->IDENT()->getText(), "", 5, range(m->IDENT())});
			for (auto* f : m->funcdef())
				if (f->IDENT())
					entries.push_back({f->IDENT()->getText(), m->IDENT()->getText(), 6, range(f->IDENT())});
		}
		for (auto* s : program->struct_def())
			if (s->IDENT())
				entries.push_back({s->IDENT()->getText(), "", 23, range(s->IDENT())});
		for (auto* f : program->funcdef())
			if (f->IDENT())
				entries.push_back({f->IDENT()->getText(), "", 12, range(f->IDENT())});
		return entries;
	};
	// Matches: the query's characters in order (case-insensitive), like editors' fuzzy search.
	auto matches = [&](const std::string& name) {
		size_t k = 0;
		for (char c : name)
			if (k < query.size() &&
				std::tolower(static_cast<unsigned char>(c)) == std::tolower(static_cast<unsigned char>(query[k])))
				++k;
		return k == query.size();
	};
	json result = json::array();
	auto add = [&](const std::string& file, const std::string& uri, const std::string& text,
				   const std::vector<Entry>& entries) {
		const LineIndex lines{text};
		for (const auto& e : entries)
			if (matches(e.name) && result.size() < 500)
				result.push_back(
						{{"name", e.name},
						 {"kind", e.kind},
						 {"containerName", e.container},
						 {"location", {{"uri", uri}, {"range", toProtocol(lines, e.range)}}}}
				);
		(void)file;
	};
	std::set<std::string> done;
	for (const auto* doc : open) {
		done.insert(doc->sourceName());
		add(doc->sourceName(), doc->uri, doc->text, definitions(doc->text));
	}
	for (const auto& root : roots) {
		std::error_code ec;
		for (auto it = std::filesystem::recursive_directory_iterator(
					 root, std::filesystem::directory_options::skip_permission_denied, ec
			 );
			 it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
			if (ec)
				break;
			const auto name = it->path().filename().string();
			if (it->is_directory(ec) && (name.starts_with(".") || name == "out" || name == "build" || name == "_deps" ||
										 name == "node_modules")) {
				it.disable_recursion_pending();
				continue;
			}
			if (it->path().extension() != ".dlm" || done.contains(it->path().string()))
				continue;
			const auto modified = std::filesystem::last_write_time(it->path(), ec);
			auto& cached = cache[it->path().string()];
			const std::string text = readText(it->path().string());
			if (cached.first != modified || cached.second.empty())
				cached = {modified, definitions(text)};
			add(it->path().string(), pathToUri(it->path()), text, cached.second);
		}
	}
	return result;
}

mdls::json mdls::selectionRanges(const Document& doc, const json& params) {
	antlr4::ANTLRInputStream input;
	input.load(doc.text, /*lenient=*/true); // (like the parser, see mdl::invalidUtf8)
	mdlLexer lexer{&input};
	lexer.removeErrorListeners();
	antlr4::CommonTokenStream stream{&lexer};
	mdlParser parser{&stream};
	parser.removeErrorListeners();
	auto* program = parser.program();
	const LineIndex lines{doc.text};
	auto rangeOf = [](antlr4::ParserRuleContext* ctx) -> std::optional<mdl::SourceRange> {
		if (ctx->start == nullptr || ctx->stop == nullptr || ctx->stop->getTokenIndex() < ctx->start->getTokenIndex())
			return std::nullopt;
		auto* stop = ctx->stop;
		const std::string text = stop->getText();
		const size_t newlines = std::count(text.begin(), text.end(), '\n');
		const mdl::SourcePosition end =
				newlines == 0 ? mdl::SourcePosition{stop->getLine(), stop->getCharPositionInLine() + text.size()}
							  : mdl::SourcePosition{stop->getLine() + newlines, text.size() - text.rfind('\n') - 1};
		return mdl::SourceRange{{ctx->start->getLine(), ctx->start->getCharPositionInLine()}, end};
	};
	json result = json::array();
	for (const auto& position : params.at("positions")) {
		const auto p = toSource(lines, position);
		// The contexts that contain the position, outermost first.
		std::vector<mdl::SourceRange> chain;
		antlr4::tree::ParseTree* node = program;
		while (node != nullptr) {
			auto* ctx = dynamic_cast<antlr4::ParserRuleContext*>(node);
			if (ctx == nullptr)
				break;
			if (auto r = rangeOf(ctx); r && (chain.empty() || chain.back() != *r))
				chain.push_back(*r);
			antlr4::tree::ParseTree* next = nullptr;
			for (auto* child : node->children) {
				if (auto* c = dynamic_cast<antlr4::ParserRuleContext*>(child)) {
					auto r = rangeOf(c);
					if (r && r->begin <= p && p <= r->end) {
						next = child;
						break;
					}
				} else if (auto* t = dynamic_cast<antlr4::tree::TerminalNode*>(child)) {
					auto* s = t->getSymbol();
					const mdl::SourceRange r{
							{s->getLine(), s->getCharPositionInLine()},
							{s->getLine(), s->getCharPositionInLine() + s->getText().size()}
					};
					if (r.begin <= p && p <= r.end && s->getType() != antlr4::Token::EOF) {
						if (chain.empty() || chain.back() != r)
							chain.push_back(r);
						break;
					}
				}
			}
			node = next;
		}
		json selection = nullptr;
		for (const auto& r : chain) {
			json s{{"range", toProtocol(lines, r)}};
			if (!selection.is_null())
				s["parent"] = std::move(selection);
			selection = std::move(s);
		}
		if (selection.is_null())
			selection = {{"range", {{"start", position}, {"end", position}}}};
		result.push_back(std::move(selection));
	}
	return result;
}

namespace {
	/** @brief A call hierarchy item of a definition (the data identifies it in the index). */
	mdls::json callItem(const mdls::Document& doc, const mdl::analysis::Symbol& s) {
		const int kind = s.kind == SymbolKind::module ? 5 : s.kind == SymbolKind::method ? 6 : 12;
		const auto range = location(doc, s.file, s.range);
		const auto extent = s.extent == mdl::SourceRange{} ? range : location(doc, s.file, s.extent);
		return {{"name", s.name},
				{"kind", kind},
				{"detail", s.container},
				{"uri", range["uri"]},
				{"range", extent["range"]},
				{"selectionRange", range["range"]},
				{"data", {{"file", s.file}, {"line", s.range.begin.line}, {"column", s.range.begin.column}}}};
	}

	bool callable(SymbolKind kind) {
		return kind == SymbolKind::function || kind == SymbolKind::method || kind == SymbolKind::module;
	}

	std::optional<size_t> itemSymbol(const mdls::Document& doc, const mdls::json& item) {
		if (doc.index == nullptr || !item.contains("data"))
			return std::nullopt;
		const auto& d = item["data"];
		return doc.index->find(
				d.at("file").get<std::string>(), {d.at("line").get<size_t>(), d.at("column").get<size_t>()}
		);
	}

	/** @brief The callable that a reference calls: its definition, or the `forward` of an instance's module. */
	std::optional<size_t> calledBy(const mdl::analysis::SemanticIndex& index, const mdl::analysis::Reference& r) {
		if (!r.symbol)
			return std::nullopt;
		if (callable(index.symbols()[*r.symbol].kind))
			return r.symbol;
		for (size_t i = 0; i < index.symbols().size(); ++i) { // an instance: `lin(x)` calls Linear.forward
			const auto& s = index.symbols()[i];
			if (s.kind == SymbolKind::method && s.name == "forward" && r.types.contains(s.container))
				return i;
		}
		return std::nullopt;
	}

	/** @brief The innermost callable whose definition contains @p range in @p file. */
	std::optional<size_t>
	enclosing(const mdl::analysis::SemanticIndex& index, const std::string& file, const mdl::SourceRange& range) {
		std::optional<size_t> best;
		for (size_t i = 0; i < index.symbols().size(); ++i) {
			const auto& s = index.symbols()[i];
			if (s.file != file || !callable(s.kind) || !s.extent.contains(range.begin))
				continue;
			if (!best || index.symbols()[*best].extent.begin < s.extent.begin)
				best = i;
		}
		return best;
	}
} // namespace

mdls::json mdls::prepareCallHierarchy(const Document& doc, const json& params) {
	if (doc.index == nullptr || doc.indexText != doc.text)
		return nullptr;
	const auto* ref = doc.index->referenceAt(doc.sourceName(), toSource(LineIndex{doc.text}, params.at("position")));
	if (ref == nullptr)
		return nullptr;
	auto target = calledBy(*doc.index, *ref);
	if (!target)
		return nullptr;
	return json::array({callItem(doc, doc.index->symbols()[*target])});
}

mdls::json mdls::incomingCalls(const Document& doc, const json& item) {
	json calls = json::array();
	const auto target = itemSymbol(doc, item);
	if (!target)
		return calls;
	const auto& index = *doc.index;
	const auto& definition = index.symbols()[*target];
	std::map<size_t, std::vector<std::pair<std::string, mdl::SourceRange>>> callers;
	for (const auto& r : index.references()) {
		if (calledBy(index, r) != target || (r.file == definition.file && r.range == definition.range))
			continue;
		if (auto caller = enclosing(index, r.file, r.range))
			callers[*caller].emplace_back(r.file, r.range);
	}
	for (const auto& [caller, ranges] : callers) {
		json from = json::array();
		for (const auto& [file, range] : ranges)
			from.push_back(location(doc, file, range)["range"]);
		calls.push_back({{"from", callItem(doc, index.symbols()[caller])}, {"fromRanges", from}});
	}
	return calls;
}

mdls::json mdls::outgoingCalls(const Document& doc, const json& item) {
	json calls = json::array();
	const auto caller = itemSymbol(doc, item);
	if (!caller)
		return calls;
	const auto& index = *doc.index;
	const auto& definition = index.symbols()[*caller];
	std::map<size_t, std::vector<mdl::SourceRange>> callees;
	for (const auto& r : index.references()) {
		if (r.file != definition.file || !definition.extent.contains(r.range.begin) || r.range == definition.range)
			continue;
		if (auto callee = calledBy(index, r); callee && *callee != *caller)
			callees[*callee].push_back(r.range);
	}
	for (const auto& [callee, ranges] : callees) {
		json from = json::array();
		for (const auto& range : ranges)
			from.push_back(location(doc, definition.file, range)["range"]);
		calls.push_back({{"to", callItem(doc, index.symbols()[callee])}, {"fromRanges", from}});
	}
	return calls;
}
