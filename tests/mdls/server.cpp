// The language server (mdls) driven through its protocol: diagnostics, hover, definition, completion, formatting,
// semantic tokens, document symbols.
#include "mdls/server.hpp"
#include "mdls/protocol.hpp"
#include "mdls/semantic_tokens.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

using mdls::json;

namespace {
	/** @brief Runs a session of @p messages (initialize, the messages, shutdown, exit); returns the server's messages. */
	std::vector<json> session(const std::vector<json>& messages) {
		std::stringstream in, out;
		auto send = [&](json m) {
			m["jsonrpc"] = "2.0";
			mdls::writeMessage(in, m);
		};
		send({{"id", 0}, {"method", "initialize"}, {"params", {{"capabilities", json::object()}}}});
		for (const auto& m : messages)
			send(m);
		send({{"id", 999}, {"method", "shutdown"}});
		send({{"method", "exit"}});
		mdls::Server server{in, out};
		CHECK(server.run() == 0);
		std::vector<json> result;
		while (auto m = mdls::readMessage(out))
			result.push_back(*m);
		return result;
	}

	const json* response(const std::vector<json>& messages, int id) {
		for (const auto& m : messages)
			if (m.contains("id") && m["id"] == id)
				return &m;
		return nullptr;
	}

	const std::string uri = mdls::pathToUri(std::filesystem::path{MDL_SOURCE_DIR} / "tests" / "model.dlm");

	json open(const std::string& text) {
		return {{"method", "textDocument/didOpen"},
				{"params", {{"textDocument", {{"uri", uri}, {"languageId", "mdl"}, {"version", 1}, {"text", text}}}}}};
	}

	json request(int id, const std::string& method, int line = -1, int character = -1) {
		json params{{"textDocument", {{"uri", uri}}}};
		if (line >= 0)
			params["position"] = {{"line", line}, {"character", character}};
		return {{"id", id}, {"method", method}, {"params", params}};
	}

	const std::string model = R"(import std;
/** @brief A residual block. */
module Block[d: uint] {
    lin: Linear[d, d]
    float[..., d] forward(x: float[..., d]) {
        auto h = lin(x);
        return h + x[... (a b) -> ... (a b)];
    }
}
)";
} // namespace

TEST_CASE("mdls publishes diagnostics", "[mdls]") {
	auto messages = session({open("float[...] f(x: float[...]) => y;")});
	const json* published = nullptr;
	for (const auto& m : messages)
		if (m.value("method", "") == "textDocument/publishDiagnostics")
			published = &m;
	REQUIRE(published != nullptr);
	const auto& diagnostics = (*published)["params"]["diagnostics"];
	REQUIRE(diagnostics.size() == 1);
	CHECK(diagnostics[0]["message"] == "Variable y not defined");
	CHECK(diagnostics[0]["range"]["start"] == json{{"line", 0}, {"character", 31}});
}

TEST_CASE("mdls shows types and documentation on hover", "[mdls]") {
	auto messages =
			session({open(model), request(1, "textDocument/hover", 5, 13), request(2, "textDocument/hover", 3, 10)});
	const auto* variable = response(messages, 1);
	REQUIRE(variable != nullptr);
	const std::string text = (*variable)["result"]["contents"]["value"];
	CHECK(text.find("auto h = lin(x);") != std::string::npos);
	CHECK(text.find("`float32[..., d]`") != std::string::npos);
	const std::string linear = (*response(messages, 2))["result"]["contents"]["value"];
	CHECK(linear.find("module Linear[indim: uint, outdim: uint, bias: bool = true]") != std::string::npos);
	CHECK(linear.find("linear transformation") != std::string::npos);
}

TEST_CASE("mdls goes to definitions, also into the standard library", "[mdls]") {
	auto messages = session(
			{open(model), request(1, "textDocument/definition", 3, 10), request(2, "textDocument/definition", 6, 19)}
	);
	const auto& linear = (*response(messages, 1))["result"];
	CHECK(linear["uri"].get<std::string>().ends_with("/std/nn.dlm"));
	const auto& x = (*response(messages, 2))["result"];
	CHECK(x["uri"] == uri);
	CHECK(x["range"]["start"] == json{{"line", 4}, {"character", 26}});
}

TEST_CASE("mdls completes members and names in scope", "[mdls]") {
	std::string typing = model;
	typing.replace(typing.find("        return"), 0, "        auto y = lin.\n");
	auto change = json{
			{"method", "textDocument/didChange"},
			{"params",
			 {{"textDocument", {{"uri", uri}, {"version", 2}}}, {"contentChanges", json::array({{{"text", typing}}})}}},
	};
	auto messages = session(
			{open(model), change, request(1, "textDocument/completion", 6, 21),
			 request(2, "textDocument/completion", 7, 15)}
	);
	auto labels = [&](int id) {
		std::set<std::string> set;
		for (const auto& item : (*response(messages, id))["result"])
			set.insert(item["label"].get<std::string>());
		return set;
	};
	CHECK(labels(1) == std::set<std::string>{"W", "b", "forward"});
	const auto names = labels(2);
	CHECK(names.contains("h"));
	CHECK(names.contains("lin"));
	CHECK(names.contains("softmax"));
	CHECK(names.contains("return"));
}

TEST_CASE("mdls formats documents", "[mdls]") {
	auto messages = session({open("module M[]{W:float[3]}"), request(1, "textDocument/formatting")});
	const auto& edits = (*response(messages, 1))["result"];
	REQUIRE(edits.size() == 1);
	CHECK(edits[0]["newText"] == "module M[] {\n    W: float[3]\n}\n");
}

TEST_CASE("mdls classifies tokens, including einops axes", "[mdls]") {
	const auto tokens = mdls::semanticTokens(model);
	const auto& types = mdls::semanticTokenTypes();
	auto typeAt = [&](size_t line, size_t column) -> std::string {
		for (const auto& t : tokens)
			if (t.line == line && t.column == column)
				return types[t.type];
		return "";
	};
	CHECK(typeAt(2, 0).empty());			 // module (a keyword: left to the editor's grammar)
	CHECK(typeAt(2, 7) == "class");			 // Block
	CHECK(typeAt(2, 13) == "typeParameter"); // d
	CHECK(typeAt(3, 4) == "property");		 // lin
	CHECK(typeAt(5, 13) == "variable");		 // h
	CHECK(typeAt(6, 26) == "axis");			 // a
}

TEST_CASE("mdls lists the definitions of a document", "[mdls]") {
	auto messages = session({open(model), request(1, "textDocument/documentSymbol")});
	const auto& symbols = (*response(messages, 1))["result"];
	REQUIRE(symbols.size() == 1);
	CHECK(symbols[0]["name"] == "Block");
	std::set<std::string> children;
	for (const auto& c : symbols[0]["children"])
		children.insert(c["name"].get<std::string>());
	CHECK(children == std::set<std::string>{"d", "lin", "forward"});
}

TEST_CASE("mdls publishes diagnostics under the URI the client sent", "[mdls]") {
	// VS Code encodes the drive letter's colon on Windows; an untitled document has no file.
	for (const std::string clientUri : {"file:///c%3A/models/model.dlm", "untitled:Untitled-1"}) {
		std::stringstream in, out;
		auto send = [&](mdls::json m) {
			m["jsonrpc"] = "2.0";
			mdls::writeMessage(in, m);
		};
		send({{"id", 0}, {"method", "initialize"}, {"params", json::object()}});
		send({{"method", "textDocument/didOpen"},
			  {"params",
			   {{"textDocument", {{"uri", clientUri}, {"version", 1}, {"text", "float[...] f(x: float[...]) => y;"}}}}}}
		);
		send({{"id", 1}, {"method", "shutdown"}});
		send({{"method", "exit"}});
		mdls::Server{in, out}.run();
		bool found = false;
		while (auto m = mdls::readMessage(out))
			if (m->value("method", "") == "textDocument/publishDiagnostics") {
				CHECK((*m)["params"]["uri"] == clientUri);
				CHECK((*m)["params"]["diagnostics"].size() == 1);
				found = true;
			}
		CHECK(found);
	}
}

TEST_CASE("mdls survives invalid headers and bodies", "[mdls]") {
	std::stringstream in, out;
	in << "Content-Length: abc\r\n\r\n";
	in << "Content-Length: 99999999999999999999\r\n\r\n";
	in << "Content-Length: 5\r\n\r\n{oops";
	mdls::writeMessage(in, {{"jsonrpc", "2.0"}, {"id", 1}, {"method", "shutdown"}});
	mdls::writeMessage(in, {{"jsonrpc", "2.0"}, {"method", "exit"}});
	CHECK(mdls::Server{in, out}.run() == 0);
	std::vector<json> messages;
	while (auto m = mdls::readMessage(out))
		messages.push_back(*m);
	REQUIRE(messages.size() == 2);
	CHECK(messages[0]["error"]["code"] == -32700); // the invalid JSON
	CHECK(messages[1]["id"] == 1);
}

TEST_CASE("mdls hovers and goes to definitions while the text does not parse", "[mdls]") {
	std::string typing = model;
	typing.replace(typing.find("        return"), 0, "        auto y = lin.\n");
	auto change = json{
			{"method", "textDocument/didChange"},
			{"params",
			 {{"textDocument", {{"uri", uri}, {"version", 2}}}, {"contentChanges", json::array({{{"text", typing}}})}}},
	};
	// `return h + x` moved one line down; its x is defined in the unchanged line 4.
	auto messages = session(
			{open(model), change, request(1, "textDocument/hover", 7, 19), request(2, "textDocument/definition", 7, 19)}
	);
	const std::string text = (*response(messages, 1))["result"]["contents"]["value"];
	CHECK(text.find("x: float[..., d]") != std::string::npos);
	CHECK((*response(messages, 1))["result"]["range"]["start"] == json{{"line", 7}, {"character", 19}});
	CHECK((*response(messages, 2))["result"]["range"]["start"] == json{{"line", 4}, {"character", 26}});
}

TEST_CASE("mdls completes named hyperparameters and nothing in comments", "[mdls]") {
	const std::string text = "import std;\nmodule M[] {\n    l: Linear[4, 3, .\n    // l.\n}\n";
	auto messages = session(
			{open("import std;\nmodule M[] {\n    l: Linear[4, 3]\n    // l\n}\n"),
			 {{"method", "textDocument/didChange"},
			  {"params",
			   {{"textDocument", {{"uri", uri}, {"version", 2}}}, {"contentChanges", json::array({{{"text", text}}})}}}
			 },
			 request(1, "textDocument/completion", 2, 21),
			 request(2, "textDocument/completion", 3, 9)}
	);
	std::set<std::string> labels;
	for (const auto& item : (*response(messages, 1))["result"])
		labels.insert(item["label"].get<std::string>());
	CHECK(labels == std::set<std::string>{"indim", "outdim", "bias"});
	CHECK((*response(messages, 2))["result"].empty());
}

TEST_CASE("mdls re-analyzes the open documents that import a changed document", "[mdls]") {
	const auto dir = std::filesystem::temp_directory_path() / "mdls-import-test";
	std::filesystem::create_directories(dir);
	{
		std::ofstream{dir / "lib.dlm"} << "float[...] twice(x: float[...]) => x + x;\n";
	}
	const std::string libUri = mdls::pathToUri(dir / "lib.dlm"), mainUri = mdls::pathToUri(dir / "main.dlm");
	auto openDoc = [](const std::string& u, const std::string& text) {
		return json{
				{"method", "textDocument/didOpen"},
				{"params", {{"textDocument", {{"uri", u}, {"version", 1}, {"text", text}}}}}
		};
	};
	// Renaming the function in the open library makes the importer's call an error.
	auto messages = session(
			{openDoc(libUri, "float[...] twice(x: float[...]) => x + x;\n"),
			 openDoc(mainUri, "import lib;\nfloat[...] f(x: float[...]) => twice(x);\n"),
			 {{"method", "textDocument/didChange"},
			  {"params",
			   {{"textDocument", {{"uri", libUri}, {"version", 2}}},
				{"contentChanges", json::array({{{"text", "float[...] double(x: float[...]) => x + x;\n"}}})}}}}}
	);
	std::filesystem::remove_all(dir);
	std::vector<size_t> mainDiagnostics;
	for (const auto& m : messages)
		if (m.value("method", "") == "textDocument/publishDiagnostics" && m["params"]["uri"] == mainUri)
			mainDiagnostics.push_back(m["params"]["diagnostics"].size());
	CHECK(mainDiagnostics == std::vector<size_t>{0, 1});
}

TEST_CASE("mdls shows the documentation of imported files and goes to them", "[mdls]") {
	auto messages = session({open("import std.nn;\n"), request(1, "textDocument/hover", 0, 12),
							 request(2, "textDocument/hover", 0, 8), request(3, "textDocument/definition", 0, 12),
							 request(4, "textDocument/definition", 0, 8)});
	const std::string file = (*response(messages, 1))["result"]["contents"]["value"];
	CHECK(file.find("import std.nn") != std::string::npos);
	CHECK(file.find("Basic neural network layers.") != std::string::npos); // the @file comment of std/nn.dlm
	const std::string package = (*response(messages, 2))["result"]["contents"]["value"];
	CHECK(package.find("*package*") != std::string::npos);
	CHECK(package.find("`std.nn`: Basic neural network layers.") != std::string::npos);
	CHECK((*response(messages, 3))["result"]["uri"].get<std::string>().ends_with("/std/nn.dlm"));
	CHECK((*response(messages, 4))["result"].size() > 3); // all files of std
}
