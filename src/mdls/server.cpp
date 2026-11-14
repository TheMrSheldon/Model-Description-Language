#include "./server.hpp"

#include "./features.hpp"
#include "./partial.hpp"
#include "./semantic_tokens.hpp"

#include <mdl/version.hpp>

#include <algorithm>
#include <format>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>

int mdls::Server::run() {
	while (true) {
		std::optional<json> message;
		try {
			message = readMessage(in);
		} catch (const ProtocolError& e) {
			logLine(std::string{"protocol error: "} + e.what());
			continue; // the header is skipped; the next message starts with a header again
		}
		if (!message)
			break;
		if (message->is_discarded()) {
			writeMessage(
					out,
					{{"jsonrpc", "2.0"}, {"id", nullptr}, {"error", {{"code", ParseError}, {"message", "invalid JSON"}}}
					}
			);
			continue;
		}
		handle(*message);
		if (message->value("method", "") == "exit")
			break;
	}
	return shutdownRequested ? 0 : 1;
}

void mdls::Server::handle(const json& message) {
	if (!message.contains("method"))
		return; // a response to a request of the server (e.g. client/registerCapability)
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
	} catch (const RequestError& e) {
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
	if (method == "textDocument/formatting") {
		const auto& doc = document(params);
		return formatting(doc, params, styleOf(doc));
	}
	if (method == "textDocument/semanticTokens/full") {
		const auto& doc = document(params);
		return encodeSemanticTokens(semanticTokens(doc.text), LineIndex{doc.text});
	}
	if (method == "textDocument/hover")
		return hover(document(params), params);
	if (method == "textDocument/definition")
		return definition(document(params), params);
	if (method == "textDocument/completion")
		return completion(document(params), params, parseOptions);
	if (method == "textDocument/documentSymbol")
		return documentSymbols(document(params));
	if (method == "textDocument/signatureHelp")
		return signatureHelp(document(params), params);
	if (method == "textDocument/inlayHint")
		return inlayHints(document(params), params);
	if (method == "textDocument/references")
		return references(document(params), params, openTexts());
	if (method == "textDocument/documentHighlight")
		return documentHighlight(document(params), params);
	if (method == "textDocument/prepareRename")
		return prepareRename(document(params), params);
	if (method == "textDocument/rename")
		return rename(document(params), params, openTexts());
	if (method == "textDocument/foldingRange")
		return foldingRanges(document(params));
	if (method == "textDocument/codeAction")
		return codeActions(document(params), params, openTexts(), standardLibraryNames());
	if (method == "textDocument/codeLens")
		return codeLenses(document(params));
	if (method == "mdl/moduleTree")
		return moduleTree(document(params), params);
	if (method == "textDocument/documentLink")
		return documentLinks(document(params));
	if (method == "textDocument/selectionRange")
		return selectionRanges(document(params), params);
	if (method == "textDocument/linkedEditingRange")
		return linkedEditingRanges(document(params), params);
	if (method == "textDocument/prepareCallHierarchy")
		return prepareCallHierarchy(document(params), params);
	if (method == "callHierarchy/incomingCalls" || method == "callHierarchy/outgoingCalls") {
		const auto& item = params.at("item");
		// The document the item belongs to (its index), else any open document whose index has it.
		const Document* doc = nullptr;
		for (const auto& [uri, d] : documents)
			if (uri == item.value("uri", "") || doc == nullptr)
				doc = &d;
		if (doc == nullptr)
			return json::array();
		return method == "callHierarchy/incomingCalls" ? incomingCalls(*doc, item) : outgoingCalls(*doc, item);
	}
	if (method == "workspace/symbol") {
		std::vector<const Document*> open;
		for (const auto& [_, d] : documents)
			open.push_back(&d);
		return workspaceSymbols(workspaceRoots, open, params.value("query", ""));
	}
	if (method == "textDocument/rangeFormatting") {
		const auto& doc = document(params);
		return rangeFormatting(doc, params, styleOf(doc));
	}
	throw RequestError(MethodNotFound, "unsupported method " + method);
}

void mdls::Server::applySettings(const json& settings) {
	// Settings mirror mdlc's options: {"importDirs": [...], "float": "f32|bf16|f16", "strict": false}, also nested in
	// {"mdl": {...}} (as VS Code sends a section of the configuration).
	if (!settings.is_object())
		return;
	const json& options = settings.contains("mdl") && settings["mdl"].is_object() ? settings["mdl"] : settings;
	if (options.contains("importDirs") && options["importDirs"].is_array()) {
		parseOptions.importDirs.clear();
		for (const auto& dir : options["importDirs"])
			parseOptions.importDirs.emplace_back(dir.get<std::string>());
	}
	if (options.contains("strict")) // what may be wrong (e.g. sizes that may differ) is an error
		parseOptions.possibleMismatch = options.value("strict", false) ? mdl::parser::DiagnosticLevel::error
																	   : mdl::parser::DiagnosticLevel::warning;
	if (options.contains("float")) {
		const std::string type = options.value("float", "f32");
		parseOptions.defaultFloat = type == "bf16"	? mdl::PrimitiveType::bfloat16
									: type == "f16" ? mdl::PrimitiveType::float16
													: mdl::PrimitiveType::float32;
	}
	stdNames.reset();
}

void mdls::Server::analyzeAll() {
	for (auto& [_, doc] : documents)
		analyze(doc);
}

void mdls::Server::handleNotification(const std::string& method, const json& params) {
	if (method == "initialized" && watchFiles) {
		// Imported files and style files that change outside of the editor.
		json watchers = json::array({{{"globPattern", "**/*.dlm"}}, {{"globPattern", "**/.mdl-format"}}});
		writeMessage(
				out, {{"jsonrpc", "2.0"},
					  {"id", "mdls-watch"},
					  {"method", "client/registerCapability"},
					  {"params",
					   {{"registrations", json::array(
												  {{{"id", "mdls-watch"},
													{"method", "workspace/didChangeWatchedFiles"},
													{"registerOptions", {{"watchers", watchers}}}}}
										  )}}}}
		);
		return;
	}
	if (method == "workspace/didChangeWatchedFiles") {
		analyzeAll();
		return;
	}
	if (method == "workspace/didChangeConfiguration") {
		applySettings(params.value("settings", json::object()));
		analyzeAll();
		return;
	}
	if (method == "textDocument/didOpen") {
		const auto& item = params.at("textDocument");
		const std::string uri = item.at("uri");
		auto path = uriToPath(uri);
		auto& doc = documents[uri];
		doc = Document{
				.uri = uri,
				.path = path.value_or(std::filesystem::path{uri}),
				.onDisk = path.has_value(),
				.text = item.at("text"),
				.version = item.value("version", 0),
		};
		analyzeWithImporters(doc);
	} else if (method == "textDocument/didChange") {
		const std::string uri = params.at("textDocument").at("uri");
		auto it = documents.find(uri);
		if (it == documents.end())
			return;
		for (const auto& change : params.at("contentChanges"))
			if (!change.contains("range"))
				it->second.text = change.at("text"); // full synchronization
		it->second.version = params.at("textDocument").value("version", it->second.version);
		analyzeWithImporters(it->second);
	} else if (method == "textDocument/didClose") {
		const std::string uri = params.at("textDocument").at("uri");
		auto it = documents.find(uri);
		if (it == documents.end())
			return;
		Document closed = std::move(it->second);
		documents.erase(it);
		notify("textDocument/publishDiagnostics", {{"uri", uri}, {"diagnostics", json::array()}});
		// Importers now read the file from disk.
		for (auto& [_, doc] : documents)
			if (std::find(doc.sources.begin(), doc.sources.end(), closed.path) != doc.sources.end())
				analyze(doc);
	}
	// initialized, didSave, $/cancelRequest, ...: nothing to do
}

mdls::json mdls::Server::initialize(const json& params) {
	if (params.contains("initializationOptions"))
		applySettings(params["initializationOptions"]);
	// The workspace (for the symbol search).
	if (params.contains("workspaceFolders") && params["workspaceFolders"].is_array())
		for (const auto& folder : params["workspaceFolders"])
			if (auto path = uriToPath(folder.value("uri", "")))
				workspaceRoots.push_back(*path);
	if (workspaceRoots.empty() && params.contains("rootUri") && params["rootUri"].is_string())
		if (auto path = uriToPath(params["rootUri"].get<std::string>()))
			workspaceRoots.push_back(*path);
	watchFiles =
			params.contains("capabilities") &&
			params["capabilities"].value("/workspace/didChangeWatchedFiles/dynamicRegistration"_json_pointer, false);
	json legend{{"tokenTypes", semanticTokenTypes()}, {"tokenModifiers", semanticTokenModifiers()}};
	return {
			{"capabilities",
			 {
					 {"textDocumentSync", {{"openClose", true}, {"change", 1}}},
					 {"documentFormattingProvider", true},
					 {"hoverProvider", true},
					 {"definitionProvider", true},
					 {"documentSymbolProvider", true},
					 {"completionProvider", {{"triggerCharacters", json::array({".", "@"})}}},
					 {"signatureHelpProvider", {{"triggerCharacters", json::array({"(", "[", ","})}}},
					 {"inlayHintProvider", true},
					 {"referencesProvider", true},
					 {"documentHighlightProvider", true},
					 {"renameProvider", {{"prepareProvider", true}}},
					 {"foldingRangeProvider", true},
					 {"codeActionProvider", {{"codeActionKinds", json::array({"quickfix"})}}},
					 {"documentRangeFormattingProvider", true},
					 {"codeLensProvider", {{"resolveProvider", false}}},
					 {"documentLinkProvider", {{"resolveProvider", false}}},
					 {"selectionRangeProvider", true},
					 {"linkedEditingRangeProvider", true},
					 {"callHierarchyProvider", true},
					 {"workspaceSymbolProvider", true},
					 {"semanticTokensProvider", {{"legend", legend}, {"full", true}}},
			 }},
			{"serverInfo", {{"name", "mdls"}, {"version", mdl::version}}},
	};
}

mdls::Document& mdls::Server::document(const json& params) {
	const std::string uri = params.at("textDocument").at("uri");
	auto it = documents.find(uri);
	if (it == documents.end())
		throw RequestError(InvalidParams, "unknown document " + uri);
	return it->second;
}

void mdls::Server::analyzeWithImporters(Document& document) {
	analyze(document);
	// The open documents that import it (also indirectly), each once.
	std::set<std::string> done{document.uri};
	std::vector<const Document*> changed{&document};
	while (!changed.empty()) {
		const Document* source = changed.back();
		changed.pop_back();
		for (auto& [uri, doc] : documents) {
			if (done.contains(uri) || !source->onDisk)
				continue;
			const bool imports = std::any_of(doc.sources.begin(), doc.sources.end(), [&](const auto& p) {
				std::error_code ec;
				return p == source->path || std::filesystem::equivalent(p, source->path, ec);
			});
			if (!imports)
				continue;
			done.insert(uri);
			analyze(doc);
			changed.push_back(&doc);
		}
	}
}

void mdls::Server::analyze(Document& document) {
	auto options = parseOptions;
	options.sourceDir = document.onDisk ? document.path.parent_path() : std::filesystem::current_path();
	// Open documents override the files on disk (also for imports).
	options.readFile = [this](const std::filesystem::path& file) -> std::optional<std::string> {
		for (const auto& [_, doc] : documents) {
			std::error_code ec;
			if (doc.onDisk && (doc.path == file || std::filesystem::equivalent(doc.path, file, ec)))
				return doc.text;
		}
		return std::nullopt;
	};
	std::istringstream stream{document.text};
	auto result = mdl::parser::analyze(stream, document.sourceName(), options);
	document.sources = result.unit.sources;
	// With syntax errors, the definitions that parse are analyzed anyway (the others are blanked out).
	const bool syntaxErrors = std::any_of(result.diagnostics.begin(), result.diagnostics.end(), [&](const auto& d) {
		return d.syntax && d.file == document.sourceName();
	});
	const bool hadIndex = document.index != nullptr;
	if (syntaxErrors) {
		const std::string partialText = blankBrokenDefinitions(document.text);
		std::istringstream partialStream{partialText};
		auto partial = mdl::parser::analyze(partialStream, document.sourceName(), options);
		// The syntax errors of the whole text, everything else from the analysis of the rest.
		auto diagnostics = std::move(result.diagnostics);
		std::erase_if(diagnostics, [](const auto& d) { return !d.syntax; });
		for (auto& d : partial.diagnostics)
			if (!d.syntax)
				diagnostics.push_back(std::move(d));
		result = std::move(partial);
		result.diagnostics = std::move(diagnostics);
	}
	// Keep the last index that has the document's definitions (the text may not parse while it is being typed).
	const bool translated =
			std::any_of(result.index->references().begin(), result.index->references().end(), [&](const auto& r) {
				return r.file == document.sourceName();
			});
	// (With syntax errors, the analysis of the rest gives the diagnostics; completion and hovers keep the last full
	// analysis, which has the definition being typed.)
	if ((translated && !syntaxErrors) || !hadIndex) {
		document.index = result.index;
		document.indexText = document.text;
	}

	auto all = result.diagnostics;
	if (auto style = styleOf(document)) { // the naming rules of the style (see mdl-format)
		auto naming = mdl::format::checkNaming(document.text, style->naming, document.sourceName());
		all.insert(all.end(), naming.begin(), naming.end());
	}
	const LineIndex lines{document.text};
	json diagnostics = json::array();
	document.diagnostics.clear();
	// Errors of imported files: at the import (with the location in the other file).
	std::map<std::string, std::string> importedTexts;
	for (const auto& d : all) {
		if (d.file == document.sourceName() || d.severity != mdl::Severity::error)
			continue;
		std::error_code ec;
		for (const auto& import : result.index->imports())
			if (import.file == document.sourceName() &&
				std::any_of(import.targets.begin(), import.targets.end(), [&](const auto& t) {
					return t == d.file || std::filesystem::equivalent(t, d.file, ec);
				})) {
				auto& text = importedTexts[d.file];
				if (text.empty()) {
					std::ifstream in{d.file, std::ios::binary};
					text.assign(std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{});
				}
				diagnostics.push_back({
						{"range", toProtocol(lines, import.range)},
						{"severity", 1},
						{"source", "mdls"},
						{"message", std::format("{} has errors: {}", import.name, d.message)},
						{"relatedInformation",
						 json::array(
								 {{{"location",
									{{"uri", pathToUri(d.file)}, {"range", toProtocol(LineIndex{text}, d.range)}}},
								   {"message", d.message}}}
						 )},
				});
				break;
			}
	}
	for (const auto& d : all) {
		if (d.file != document.sourceName())
			continue; // in an imported file (above)
		document.diagnostics.push_back(d);
		json diagnostic{
				{"range", toProtocol(lines, d.range)},
				{"severity", d.severity == mdl::Severity::error		? 1
							 : d.severity == mdl::Severity::warning ? 2
																	: 3},
				{"source", "mdls"},
				{"message", d.hint.empty() ? d.message : d.message + "\nhint: " + d.hint},
		};
		if (!d.flag.empty())
			diagnostic["code"] = d.flag;
		diagnostics.push_back(std::move(diagnostic));
	}
	for (auto& unused : unusedDiagnostics(document, *result.index))
		diagnostics.push_back(std::move(unused));
	notify("textDocument/publishDiagnostics",
		   {{"uri", document.uri}, {"version", document.version}, {"diagnostics", diagnostics}});
}

std::optional<mdl::format::FormatOptions> mdls::Server::styleOf(const Document& document) {
	if (!document.onDisk)
		return std::nullopt;
	auto file = mdl::format::findStyleFile(document.path);
	if (!file)
		return std::nullopt;
	std::error_code ec;
	const auto modified = std::filesystem::last_write_time(*file, ec);
	auto it = styles.find(*file);
	if (it == styles.end() || it->second.modified != modified) {
		auto loaded = mdl::format::loadStyle(*file);
		if (!loaded)
			logLine(loaded.error());
		it = styles.insert_or_assign(*file, CachedStyle{modified, loaded ? std::optional{*loaded} : std::nullopt})
					 .first;
	}
	return it->second.style;
}

void mdls::Server::notify(const std::string& method, json params) {
	writeMessage(out, {{"jsonrpc", "2.0"}, {"method", method}, {"params", std::move(params)}});
}

void mdls::Server::logLine(const std::string& text) {
	if (log != nullptr)
		*log << text << std::endl;
}

const std::map<std::string, std::string>& mdls::Server::standardLibraryNames() {
	if (!stdNames) {
		stdNames.emplace();
		std::istringstream source{"import std;"};
		auto result = mdl::parser::analyze(source, "<std>", parseOptions);
		for (const auto& s : result.index->symbols())
			if (s.container.empty() &&
				(s.kind == mdl::analysis::SymbolKind::function || s.kind == mdl::analysis::SymbolKind::module ||
				 s.kind == mdl::analysis::SymbolKind::structure))
				stdNames->try_emplace(s.name, "std." + std::filesystem::path{s.file}.stem().string());
	}
	return *stdNames;
}

std::map<std::string, std::string> mdls::Server::openTexts() const {
	std::map<std::string, std::string> texts;
	for (const auto& [_, doc] : documents)
		texts[doc.sourceName()] = doc.text;
	return texts;
}
