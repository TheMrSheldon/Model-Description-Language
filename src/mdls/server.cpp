#include "./server.hpp"

#include "./features.hpp"
#include "./semantic_tokens.hpp"

#include <mdl/version.hpp>

#include <algorithm>
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
	throw RequestError(MethodNotFound, "unsupported method " + method);
}

void mdls::Server::handleNotification(const std::string& method, const json& params) {
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
	// Initialization options mirror mdlc's: {"importDirs": [...], "float": "f32|bf16|f16"}.
	if (params.contains("initializationOptions") && params["initializationOptions"].is_object()) {
		const auto& options = params["initializationOptions"];
		if (options.contains("importDirs"))
			for (const auto& dir : options["importDirs"])
				parseOptions.importDirs.emplace_back(dir.get<std::string>());
		const std::string type = options.value("float", "f32");
		parseOptions.defaultFloat = type == "bf16"	? mdl::PrimitiveType::bfloat16
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
					 {"completionProvider", {{"triggerCharacters", json::array({".", "@"})}}},
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
	// Keep the last index that has the document's definitions (the text may not parse while it is being typed).
	const bool translated =
			std::any_of(result.index->references().begin(), result.index->references().end(), [&](const auto& r) {
				return r.file == document.sourceName();
			});
	if (translated || document.index == nullptr) {
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
	for (const auto& d : all) {
		if (d.file != document.sourceName())
			continue; // in an imported file
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
