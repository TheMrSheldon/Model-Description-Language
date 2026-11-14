#include "./server.hpp"

#include "./semantic_tokens.hpp"

#include <mdl/format.hpp>

#include <algorithm>
#include <set>

#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace {
	/** @brief JSON-RPC error codes. */
	enum ErrorCode : int {
		MethodNotFound = -32601,
		InvalidParams = -32602,
		InternalError = -32603,
		ServerNotInitialized = -32002,
		RequestFailed = -32803,
	};

	struct RpcError : std::runtime_error {
		int code;
		RpcError(int code, const std::string& message) : std::runtime_error(message), code(code) {}
	};

	const std::string version = "0.1.0";
} // namespace

int mdls::Server::run() {
	while (auto message = readMessage(in)) {
		if (message->is_discarded())
			continue;
		handle(*message);
		if (message->value("method", "") == "exit")
			return shutdownRequested ? 0 : 1;
	}
	return shutdownRequested ? 0 : 1;
}

void mdls::Server::handle(const json& message) {
	const std::string method = message.value("method", "");
	const json params = message.contains("params") ? message["params"] : json::object();
	if (!message.contains("id")) {
		try {
			handleNotification(method, params);
		} catch (const std::exception& e) {
			logLine("error in " + method + ": " + e.what());
		}
		return;
	}
	json response{{"jsonrpc", "2.0"}, {"id", message["id"]}};
	try {
		response["result"] = handleRequest(method, params);
	} catch (const RpcError& e) {
		response["error"] = {{"code", e.code}, {"message", e.what()}};
	} catch (const std::exception& e) {
		response["error"] = {{"code", InternalError}, {"message", e.what()}};
	}
	writeMessage(out, response);
}

mdls::json mdls::Server::handleRequest(const std::string& method, const json& params) {
	if (method == "initialize")
		return initialize(params);
	if (method == "shutdown") {
		shutdownRequested = true;
		return nullptr;
	}
	if (method == "textDocument/formatting")
		return formatting(params);
	if (method == "textDocument/semanticTokens/full")
		return semanticTokens(params);
	if (method == "textDocument/hover")
		return hover(params);
	if (method == "textDocument/definition")
		return definition(params);
	if (method == "textDocument/completion")
		return completion(params);
	if (method == "textDocument/documentSymbol")
		return documentSymbols(params);
	throw RpcError(MethodNotFound, "unsupported method " + method);
}

void mdls::Server::handleNotification(const std::string& method, const json& params) {
	if (method == "textDocument/didOpen") {
		const auto& item = params.at("textDocument");
		const std::string uri = item.at("uri");
		auto path = uriToPath(uri);
		auto& doc = documents[uri];
		doc = Document{path.value_or(uri), item.at("text"), item.value("version", 0)};
		analyze(doc);
	} else if (method == "textDocument/didChange") {
		const std::string uri = params.at("textDocument").at("uri");
		auto it = documents.find(uri);
		if (it == documents.end())
			return;
		for (const auto& change : params.at("contentChanges"))
			if (!change.contains("range"))
				it->second.text = change.at("text"); // full synchronization
		it->second.version = params.at("textDocument").value("version", it->second.version);
		analyze(it->second);
	} else if (method == "textDocument/didClose") {
		const std::string uri = params.at("textDocument").at("uri");
		documents.erase(uri);
		notify("textDocument/publishDiagnostics", {{"uri", uri}, {"diagnostics", json::array()}});
	}
	// initialized, didSave, $/cancelRequest, ...: nothing to do
}

mdls::json mdls::Server::initialize(const json& params) {
	// Initialization options mirror mdlc's: {"importDirs": [...], "float": "f32|bf16|f16"}.
	if (params.contains("initializationOptions") && params["initializationOptions"].is_object()) {
		const auto& options = params["initializationOptions"];
		if (options.contains("importDirs"))
			for (const auto& dir : options["importDirs"])
				parseOptions.importDirs.emplace_back(dir.get<std::string>());
		const std::string type = options.value("float", "f32");
		parseOptions.defaultFloat = type == "bf16"	 ? mdl::PrimitiveType::bfloat16
									: type == "f16" ? mdl::PrimitiveType::float16
													: mdl::PrimitiveType::float32;
	}
	json legend{{"tokenTypes", semanticTokenTypes()}, {"tokenModifiers", semanticTokenModifiers()}};
	return {
			{"capabilities",
			 {
					 {"textDocumentSync", {{"openClose", true}, {"change", 1}}},
					 {"documentFormattingProvider", true},
					 {"hoverProvider", true},
					 {"definitionProvider", true},
					 {"documentSymbolProvider", true},
					 {"completionProvider", {{"triggerCharacters", json::array({"."})}}},
					 {"semanticTokensProvider", {{"legend", legend}, {"full", true}}},
			 }},
			{"serverInfo", {{"name", "mdls"}, {"version", version}}},
	};
}

mdls::Server::Document& mdls::Server::document(const json& params) {
	const std::string uri = params.at("textDocument").at("uri");
	auto it = documents.find(uri);
	if (it == documents.end())
		throw RpcError(InvalidParams, "unknown document " + uri);
	return it->second;
}

void mdls::Server::analyze(Document& document) {
	auto options = parseOptions;
	options.sourceDir = document.path.parent_path();
	// Open documents override the files on disk (also for imports).
	options.readFile = [this](const std::filesystem::path& file) -> std::optional<std::string> {
		for (const auto& [_, doc] : documents) {
			std::error_code ec;
			if (doc.path == file || std::filesystem::equivalent(doc.path, file, ec))
				return doc.text;
		}
		return std::nullopt;
	};
	std::istringstream stream{document.text};
	auto result = mdl::parser::analyze(stream, document.path.string(), options);
	// Keep the last index that has the document's definitions (the text may not parse while it is being typed).
	const bool translated = std::any_of(result.index->references().begin(), result.index->references().end(), [&](const auto& r) {
		return r.file == document.path.string();
	});
	if (translated || document.index == nullptr) {
		document.index = result.index;
		document.indexText = document.text;
	}

	// The naming rules of the style file (see mdl-format).
	auto all = result.diagnostics;
	if (auto file = mdl::format::findStyleFile(document.path))
		if (auto style = mdl::format::loadStyle(*file)) {
			auto naming = mdl::format::checkNaming(document.text, style->naming, document.path.string());
			all.insert(all.end(), naming.begin(), naming.end());
		}
	const LineIndex lines{document.text};
	json diagnostics = json::array();
	for (const auto& d : all) {
		if (d.file != document.path.string())
			continue; // in an imported file
		auto position = [&](mdl::SourcePosition p) {
			const size_t line = p.line > 0 ? p.line - 1 : 0;
			return json{{"line", line}, {"character", lines.toUtf16(line, p.column)}};
		};
		json diagnostic{
				{"range", {{"start", position(d.range.begin)}, {"end", position(d.range.end)}}},
				{"severity", d.severity == mdl::Severity::error ? 1 : d.severity == mdl::Severity::warning ? 2 : 3},
				{"source", "mdls"},
				{"message", d.hint.empty() ? d.message : d.message + "\nhint: " + d.hint},
		};
		if (!d.flag.empty())
			diagnostic["code"] = d.flag;
		diagnostics.push_back(std::move(diagnostic));
	}
	notify("textDocument/publishDiagnostics",
		   {{"uri", pathToUri(document.path)}, {"version", document.version}, {"diagnostics", diagnostics}});
}

mdls::json mdls::Server::formatting(const json& params) {
	const auto& doc = document(params);
	mdl::format::FormatOptions style;
	if (auto file = mdl::format::findStyleFile(doc.path)) {
		auto loaded = mdl::format::loadStyle(*file);
		if (!loaded)
			throw RpcError(RequestFailed, loaded.error());
		style = *loaded;
	} else if (params.contains("options")) {
		style.indentWidth = params["options"].value("tabSize", style.indentWidth);
	}
	auto result = mdl::format::format(doc.text, style, doc.path.string());
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

mdls::json mdls::Server::semanticTokens(const json& params) {
	const auto& doc = document(params);
	return encodeSemanticTokens(mdls::semanticTokens(doc.text), LineIndex{doc.text});
}

void mdls::Server::notify(const std::string& method, json params) {
	writeMessage(out, {{"jsonrpc", "2.0"}, {"method", method}, {"params", std::move(params)}});
}

void mdls::Server::logLine(const std::string& text) {
	if (log != nullptr)
		*log << text << std::endl;
}

mdl::SourcePosition mdls::Server::Document::toIndex(mdl::SourcePosition position) const {
	if (indexText == text)
		return position;
	const LineIndex now{text}, then{indexText};
	const size_t n = now.lineCount(), m = then.lineCount();
	size_t prefix = 0;
	while (prefix < n && prefix < m && now.line(prefix) == then.line(prefix))
		++prefix;
	size_t suffix = 0;
	while (suffix < n - prefix && suffix < m - prefix && now.line(n - 1 - suffix) == then.line(m - 1 - suffix))
		++suffix;
	const size_t line = position.line - 1; // from 0
	if (line < prefix)
		return position;
	if (line >= n - suffix)
		return {line + m - n + 1, position.column};
	return {prefix + 1, 0}; // in the changed lines
}

namespace {
	using mdl::analysis::SymbolKind;

	/** @brief The frontend's position (line from 1, code points) of the protocol's position in @p doc. */
	mdl::SourcePosition toSource(const mdls::LineIndex& lines, const mdls::json& position) {
		const size_t line = position.at("line");
		return {line + 1, lines.fromUtf16(line, position.at("character"))};
	}

	mdls::json toRange(const mdls::LineIndex& lines, const mdl::SourceRange& range) {
		auto position = [&](mdl::SourcePosition p) {
			const size_t line = p.line > 0 ? p.line - 1 : 0;
			return mdls::json{{"line", line}, {"character", lines.toUtf16(line, p.column)}};
		};
		return {{"start", position(range.begin)}, {"end", position(range.end)}};
	}

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

	/** @brief The protocol's CompletionItemKind of a symbol. */
	int completionKind(SymbolKind kind) {
		switch (kind) {
		case SymbolKind::module:
			return 7; // Class
		case SymbolKind::structure:
			return 22; // Struct
		case SymbolKind::function:
			return 3; // Function
		case SymbolKind::method:
			return 2; // Method
		case SymbolKind::hyperparameter:
		case SymbolKind::templateParameter:
			return 25; // TypeParameter
		case SymbolKind::member:
			return 5; // Field
		default:
			return 6; // Variable
		}
	}

	/** @brief The protocol's SymbolKind of a symbol (document symbols). */
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

	std::string joinTypes(const std::set<std::string>& types) {
		std::string s;
		for (const auto& t : types)
			s += (s.empty() ? "" : ", ") + ("`" + t + "`");
		return s;
	}
} // namespace

mdls::json mdls::Server::hover(const json& params) {
	const auto& doc = document(params);
	if (doc.index == nullptr)
		return nullptr;
	const LineIndex lines{doc.text};
	const auto position = toSource(lines, params.at("position"));
	if (doc.indexText != doc.text)
		return nullptr; // the positions of the index are of an earlier text
	const auto* ref = doc.index->referenceAt(doc.path.string(), position);
	if (ref == nullptr)
		return nullptr;
	std::string md;
	if (ref->symbol) {
		const auto& s = doc.index->symbols()[*ref->symbol];
		const std::string declaration = s.detail.empty() ? s.name : s.detail;
		md = "```mdl\n" + declaration + "\n```\n";
		md += "*" + kindName(s.kind) + "*" + (s.container.empty() ? "" : " of `" + s.container + "`");
		const auto& types = ref->types.empty() ? s.types : ref->types;
		if (!types.empty() && s.kind != SymbolKind::module && s.kind != SymbolKind::structure)
			md += "\n\n" + std::string{types.size() > 1 ? "Types: " : "Type: "} + joinTypes(types);
		if (!s.documentation.empty())
			md += "\n\n---\n\n" + s.documentation;
	} else if (!ref->types.empty()) {
		md = (ref->types.size() > 1 ? "Types: " : "Type: ") + joinTypes(ref->types);
	} else {
		return nullptr;
	}
	return {{"contents", {{"kind", "markdown"}, {"value", md}}}, {"range", toRange(lines, ref->range)}};
}

mdls::json mdls::Server::definition(const json& params) {
	const auto& doc = document(params);
	if (doc.index == nullptr)
		return nullptr;
	const LineIndex lines{doc.text};
	if (doc.indexText != doc.text)
		return nullptr; // the positions of the index are of an earlier text
	const auto* ref = doc.index->referenceAt(doc.path.string(), toSource(lines, params.at("position")));
	if (ref == nullptr || !ref->symbol)
		return nullptr;
	const auto& s = doc.index->symbols()[*ref->symbol];
	// Ranges in other files are converted with their own text (the positions of their lines).
	std::string text = s.file == doc.path.string() ? doc.text : std::string{};
	if (text.empty()) {
		std::ifstream in{s.file, std::ios::binary};
		text.assign(std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{});
	}
	return {{"uri", pathToUri(s.file)}, {"range", toRange(LineIndex{text}, s.range)}};
}

mdls::json mdls::Server::completion(const json& params) {
	const auto& doc = document(params);
	const LineIndex lines{doc.text};
	const auto position = toSource(lines, params.at("position"));
	const std::string_view line = lines.line(position.line - 1);
	// The text of the line before the cursor (code points = bytes for the ASCII identifiers that matter here).
	const std::string before{line.substr(0, std::min(line.size(), position.column))};
	size_t start = before.size();
	while (start > 0 && (std::isalnum(static_cast<unsigned char>(before[start - 1])) || before[start - 1] == '_'))
		--start;
	json items = json::array();
	std::set<std::string> seen;
	auto add = [&](const std::string& label, int kind, const std::string& detail, const std::string& documentation) {
		if (!seen.insert(label).second)
			return;
		json item{{"label", label}, {"kind", kind}};
		if (!detail.empty())
			item["detail"] = detail;
		if (!documentation.empty())
			item["documentation"] = {{"kind", "markdown"}, {"value", documentation}};
		items.push_back(std::move(item));
	};

	// `import std.` and `import `: the files of the standard library.
	if (before.starts_with("import ")) {
		const std::string path = before.substr(7, start - 7);
		if (path.empty())
			add("std", 9, "the standard library", "");
		if (path == "std.") {
			auto dirs = parseOptions.stdlibDirs.empty() ? mdl::parser::defaultStdlibDirs() : parseOptions.stdlibDirs;
			for (const auto& dir : dirs) {
				std::error_code ec;
				for (const auto& entry : std::filesystem::directory_iterator(dir, ec))
					if (entry.path().extension() == ".dlm")
						add(entry.path().stem().string(), 17, "std." + entry.path().stem().string(), "");
			}
		}
		return items;
	}
	if (doc.index == nullptr)
		return items;
	const auto& index = *doc.index;
	const std::string file = doc.path.string();

	if (start > 0 && before[start - 1] == '.') {
		// `.name=` in a module instantiation: the hyperparameters of the module.
		size_t k = start - 1;
		while (k > 0 && before[k - 1] == ' ')
			--k;
		if (k > 0 && (before[k - 1] == ',' || before[k - 1] == '[')) {
			int depth = 0;
			for (size_t i = start - 1; i-- > 0;) {
				if (before[i] == ']')
					++depth;
				else if (before[i] == '[' && depth-- == 0) {
					size_t end = i, begin = i;
					while (begin > 0 && (std::isalnum(static_cast<unsigned char>(before[begin - 1])) || before[begin - 1] == '_'))
						--begin;
					const std::string module = before.substr(begin, end - begin);
					for (const auto& s : index.symbols())
						if (s.kind == SymbolKind::hyperparameter && s.container == module)
							add(s.name, 25, s.detail, "");
					break;
				}
			}
			return items;
		}
		// `x.`: the members of the value before the dot: the reference there (if the index is current), else the
		// references of the visible definition of that name (the index may be from an earlier text).
		const size_t dot = start - 1;
		size_t nameStart = dot;
		while (nameStart > 0 && (std::isalnum(static_cast<unsigned char>(before[nameStart - 1])) || before[nameStart - 1] == '_'))
			--nameStart;
		const std::string name = before.substr(nameStart, dot - nameStart);
		const mdl::analysis::Reference* target = nullptr;
		const auto mapped = doc.toIndex({position.line, dot});
		for (const auto& r : index.references())
			if (r.file == file && r.range.end.line == mapped.line && r.range.end.column == mapped.column &&
				r.range.end.column - r.range.begin.column == name.size() &&
				(!r.symbol || index.symbols()[*r.symbol].name == name))
				target = &r;
		if (target == nullptr)
			for (const auto* s : index.visibleAt(file, doc.toIndex(position)))
				if (s->name == name) {
					const size_t id = static_cast<size_t>(s - index.symbols().data());
					for (const auto& r : index.references())
						if (r.symbol == id && !r.members.empty())
							target = &r;
					break;
				}
		for (const auto* r = target; r != nullptr; r = nullptr) {
			{
				for (const auto& m : r->members) {
					// The member's definition, if the module is known: its kind and documentation.
					const mdl::analysis::Symbol* def = nullptr;
					for (const auto& s : index.symbols())
						if (s.name == m && (s.kind == SymbolKind::member || s.kind == SymbolKind::method) &&
							r->types.contains(s.container))
							def = &s;
					if (def != nullptr)
						add(m, completionKind(def->kind), def->detail, def->documentation);
					else
						add(m, m == "T" ? 2 : 5, "", "");
				}
			}
		}
		return items;
	}

	for (const auto* s : index.visibleAt(file, doc.toIndex(position)))
		add(s->name, completionKind(s->kind), s->detail, s->documentation);
	for (const char* keyword : {"auto", "return", "for", "in", "masked", "module", "struct", "import", "template", "array",
								"true", "false"})
		add(keyword, 14, "", "");
	for (const char* type : {"float", "float32", "bfloat16", "float16", "int", "uint", "bool"})
		add(type, 25, "", "");
	for (const char* annotation : {"conditional"})
		if (start > 0 && before[start - 1] == '@')
			add(annotation, 14, "@conditional(c): the member exists only if c holds", "");
	return items;
}

mdls::json mdls::Server::documentSymbols(const json& params) {
	const auto& doc = document(params);
	json result = json::array();
	if (doc.index == nullptr)
		return result;
	const LineIndex lines{doc.text};
	const std::string file = doc.path.string();
	std::map<std::string, size_t> containers; // module/struct name -> index in result
	auto symbol = [&](const mdl::analysis::Symbol& s) {
		const auto range = toRange(lines, s.range);
		json item{{"name", s.name}, {"kind", symbolKind(s.kind)}, {"range", range}, {"selectionRange", range}};
		if (!s.types.empty())
			item["detail"] = *s.types.begin();
		return item;
	};
	for (const auto& s : doc.index->symbols())
		if (s.file == file && s.container.empty() &&
			(s.kind == SymbolKind::module || s.kind == SymbolKind::structure || s.kind == SymbolKind::function)) {
			containers[s.name] = result.size();
			result.push_back(symbol(s));
			result.back()["children"] = json::array();
		}
	for (const auto& s : doc.index->symbols())
		if (s.file == file && !s.container.empty())
			if (auto it = containers.find(s.container); it != containers.end())
				result[it->second]["children"].push_back(symbol(s));
	return result;
}
