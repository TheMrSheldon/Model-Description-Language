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
	json diagnostics = json::array();
	for (const auto& d : (*published)["params"]["diagnostics"])
		if (d.value("code", "") != "unused") // (hints for the unused parameter)
			diagnostics.push_back(d);
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
				CHECK(std::count_if(
							  (*m)["params"]["diagnostics"].begin(), (*m)["params"]["diagnostics"].end(),
							  [](const auto& d) { return d.value("code", "") != "unused"; }
					  ) == 1);
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
			mainDiagnostics.push_back(std::count_if(
					m["params"]["diagnostics"].begin(), m["params"]["diagnostics"].end(),
					[](const auto& d) { return d["severity"] == 1; } // errors (not the hints for unused definitions)
			));
	CHECK(mainDiagnostics == std::vector<size_t>{0, 1});
}

TEST_CASE("mdls shows the documentation of imported files and goes to them", "[mdls]") {
	auto messages = session(
			{open("import std.nn;\n"), request(1, "textDocument/hover", 0, 12), request(2, "textDocument/hover", 0, 8),
			 request(3, "textDocument/definition", 0, 12), request(4, "textDocument/definition", 0, 8)}
	);
	const std::string file = (*response(messages, 1))["result"]["contents"]["value"];
	CHECK(file.find("import std.nn") != std::string::npos);
	CHECK(file.find("Basic neural network layers.") != std::string::npos); // the @file comment of std/nn.dlm
	const std::string package = (*response(messages, 2))["result"]["contents"]["value"];
	CHECK(package.find("*package*") != std::string::npos);
	CHECK(package.find("`std.nn`: Basic neural network layers.") != std::string::npos);
	CHECK((*response(messages, 3))["result"]["uri"].get<std::string>().ends_with("/std/nn.dlm"));
	CHECK((*response(messages, 4))["result"].size() > 3); // all files of std
}

namespace {
	const std::string features = R"(import std;
struct Conf { dims: struct { model: uint heads: uint } }
/** @brief Attention. @param conf the sizes */
module Att[conf: Conf] {
    proj: Linear[conf.dims.model, conf.dims.model]
    W: float[conf.dims.model, conf.dims.model]
    float[..., conf.dims.model] forward(x: float[..., conf.dims.model]) {
        auto q = proj(x) @ W;
        auto split = q[... (h d) -> h ... d | h=conf.dims.heads];
        return split[h ... d -> ... (h d)];
    }
}
module Net[] {
    l: Linear[8, 3, true]
    w: bfloat16[8]
    float[..., 3] forward(x: float[..., 8]) => l(x * w);
}
)";

	json range(int fromLine, int toLine) {
		return {{"start", {{"line", fromLine}, {"character", 0}}}, {"end", {{"line", toLine}, {"character", 0}}}};
	}
} // namespace

TEST_CASE("mdls shows signatures of calls and instantiations", "[mdls]") {
	auto messages = session({open(features), request(1, "textDocument/signatureHelp", 4, 33)});
	const auto& help = (*response(messages, 1))["result"];
	REQUIRE(help["signatures"].size() == 1);
	CHECK(help["signatures"][0]["label"] == "module Linear[indim: uint, outdim: uint, bias: bool = true]");
	CHECK(help["activeParameter"] == 1); // after the first comma
	CHECK(help["signatures"][0]["parameters"].size() == 3);
}

TEST_CASE("mdls shows inferred types, axis sizes, and hyperparameter names inline", "[mdls]") {
	auto request_ = request(1, "textDocument/inlayHint");
	request_["params"]["range"] = range(0, 30);
	auto messages = session({open(features), request_});
	std::set<std::string> labels;
	for (const auto& hint : (*response(messages, 1))["result"])
		labels.insert(hint["label"].get<std::string>());
	CHECK(labels.contains(": float32[..., conf.dims.model]")); // auto q
	CHECK(labels.contains("=conf.dims.heads"));				   // the axis h
	CHECK(labels.contains("indim:"));						   // Linear[8, ...]
	CHECK(labels.contains("outdim:"));
}

TEST_CASE("mdls hovers auto, einops axes, and weights", "[mdls]") {
	auto messages = session(
			{open(features), request(1, "textDocument/hover", 7, 9), request(2, "textDocument/hover", 8, 28),
			 request(3, "textDocument/hover", 7, 28)}
	);
	const std::string autoHover = (*response(messages, 1))["result"]["contents"]["value"];
	CHECK(autoHover.find("auto q: float32[..., conf.dims.model]") != std::string::npos);
	const std::string axis = (*response(messages, 2))["result"]["contents"]["value"];
	CHECK(axis == "*einops axis* `h`: size `conf.dims.heads`");
	const std::string weight = (*response(messages, 3))["result"]["contents"]["value"];
	CHECK(weight.find("Weight: `W`") != std::string::npos);
}

TEST_CASE("mdls finds references, highlights, and renames", "[mdls]") {
	auto references = request(1, "textDocument/references", 7, 13);
	references["params"]["context"] = {{"includeDeclaration", true}};
	auto renaming = request(2, "textDocument/rename", 7, 13);
	renaming["params"]["newName"] = "query";
	auto library = request(3, "textDocument/prepareRename", 4, 12); // Linear (std)
	auto messages =
			session({open(features), references, renaming, library, request(4, "textDocument/documentHighlight", 8, 28)}
			);
	CHECK((*response(messages, 1))["result"].size() == 2); // q's declaration and use
	const auto& edits = (*response(messages, 2))["result"]["changes"][uri];
	REQUIRE(edits.size() == 2);
	CHECK(edits[0]["newText"] == "query");
	CHECK((*response(messages, 3)).contains("error"));	   // definitions of std are not renamed
	CHECK((*response(messages, 4))["result"].size() == 3); // h on both sides of `->` and in the binding
}

TEST_CASE("mdls offers quick fixes", "[mdls]") {
	auto actions = request(1, "textDocument/codeAction");
	actions["params"]["range"] = range(13, 16);
	actions["params"]["context"] = {{"diagnostics", json::array()}};
	auto imports = request(2, "textDocument/codeAction");
	imports["params"]["range"] = range(0, 2);
	imports["params"]["context"] = {{"diagnostics", json::array()}};
	auto messages = session({open(features), actions});
	std::set<std::string> titles;
	for (const auto& a : (*response(messages, 1))["result"])
		titles.insert(a["title"].get<std::string>());
	CHECK(titles == std::set<std::string>{"Write `.bias=true`", "Convert to float32"});

	auto missing = session({open("float[...] f(x: float[...]) => softmax(x);\n"), imports});
	const auto& add = (*response(missing, 2))["result"];
	REQUIRE(add.size() == 1);
	CHECK(add[0]["title"] == "Import std.activations");
}

TEST_CASE("mdls goes to struct members, folds, and formats ranges", "[mdls]") {
	auto formatRange = request(3, "textDocument/rangeFormatting");
	formatRange["params"]["range"] = range(1, 2);
	auto messages = session(
			{open(features), request(1, "textDocument/definition", 4, 28), request(2, "textDocument/foldingRange")}
	);
	CHECK((*response(messages, 1))["result"]["range"]["start"] == json{{"line", 1}, {"character", 29}}); // dims
	CHECK((*response(messages, 2))["result"].size() >= 3);

	auto formatted = session({open("module A[]{W:float[3]}\nmodule B[]{W:float[3]}\n"), formatRange});
	const auto& edits = (*response(formatted, 3))["result"];
	REQUIRE(edits.size() == 1);
	CHECK(edits[0]["range"]["start"]["line"] == 0); // the changed lines (both modules change)
}

TEST_CASE("mdls highlights Doxygen commands and their arguments in doc comments", "[mdls]") {
	const std::string text =
			"/**\n * @brief Uses \\p x and \\f$x^2\\f$ (see \\cite layernorm).\n * @param[in] x the input"
			"\n * \\ref Linear, mail@example.com\n */\nfloat[...] f(x: float[...]) => x;\n";
	const auto tokens = mdls::semanticTokens(text);
	const auto& types = mdls::semanticTokenTypes();
	auto at = [&](size_t line, size_t column) -> std::pair<std::string, size_t> {
		for (const auto& t : tokens)
			if (t.line == line && t.column == column)
				return {types[t.type], t.length};
		return {"", 0};
	};
	CHECK(at(1, 3) == std::pair<std::string, size_t>{"keyword", 6});	// @brief
	CHECK(at(1, 15) == std::pair<std::string, size_t>{"keyword", 2});	// \p
	CHECK(at(1, 18) == std::pair<std::string, size_t>{"parameter", 1}); // x
	CHECK(at(1, 24) == std::pair<std::string, size_t>{"string", 9});	// \f$x^2\f$
	CHECK(at(1, 45) == std::pair<std::string, size_t>{"label", 9});		// layernorm
	CHECK(at(2, 3) == std::pair<std::string, size_t>{"keyword", 6});	// @param
	CHECK(at(2, 14) == std::pair<std::string, size_t>{"parameter", 1}); // x (after [in])
	CHECK(at(3, 8) == std::pair<std::string, size_t>{"label", 6});		// Linear
	CHECK(at(3, 19).first.empty());										// mail@example.com is no command
	const auto& modifiers = mdls::semanticTokenModifiers();
	CHECK(std::find(modifiers.begin(), modifiers.end(), "documentation") != modifiers.end());
}

namespace {
	const std::string models = R"(import std;
import std.losses;
/** @brief Uses \ref Block. */
module Block[d: uint] {
    lin: Linear[d, d]
    norm: LayerNorm[d]
    spare: float[d]
    float[..., d] forward(x: float[..., d]) {
        auto h = lin(x) + x;
        auto unused = h;
        return norm(h);
    }
    float[..., d] split(x: float[..., d], extra: float[...]) => x[... (a b) -> ... (a b) | a=2];
}
module Net[] {
    b: Block[8]
    float[..., 8] forward(x: float[..., 8]) => b(x);
}
)";
} // namespace

TEST_CASE("mdls shows parameter counts and memory, and the module tree", "[mdls]") {
	auto tree = request(2, "textDocument/codeLens");
	auto messages = session({open(models), request(1, "textDocument/codeLens"), request(2, "mdl/moduleTree")});
	std::set<std::string> titles;
	for (const auto& lens : (*response(messages, 1))["result"])
		titles.insert(lens["command"]["title"].get<std::string>());
	// Block[8]: Linear[8, 8] (72) + LayerNorm[8] (16) + spare (8) = 96 values.
	CHECK(titles.contains(
			"Block[8]: 96 parameters · 384 B (float32) · 192 B (bfloat16/float16) · 96 B (int8) · 48 B (int4)"
	));
	CHECK(titles.contains("Block[d]: d * d + 4 * d parameters")); // the generic module (checked symbolically)
	CHECK(titles.contains("Net: 96 parameters · 384 B (float32) · 192 B (bfloat16/float16) · 96 B (int8) · 48 B (int4)")
	);
	const auto& roots = (*response(messages, 2))["result"];
	const auto net = std::find_if(roots.begin(), roots.end(), [](const auto& r) { return r["name"] == "Net"; });
	REQUIRE(net != roots.end());
	CHECK((*net)["parameters"] == 96);
	REQUIRE((*net)["children"].size() == 1);
	CHECK((*net)["children"][0]["name"] == "b");
	CHECK((*net)["children"][0]["children"].size() == 3); // lin, norm, spare
}

TEST_CASE("mdls shows the result types of operators and calls", "[mdls]") {
	auto messages =
			session({open(models), request(1, "textDocument/hover", 8, 24), request(2, "textDocument/hover", 8, 18)});
	// +, once symbolic (Block[d]) and once concrete (Block[8] in Net)
	CHECK((*response(messages, 1))["result"]["contents"]["value"] == "Results: `float32[..., 8]`, `float32[..., d]`");
	const std::string call = (*response(messages, 2))["result"]["contents"]["value"]; // lin(x)
	CHECK(call.find("`float32[..., d]`") != std::string::npos);
}

TEST_CASE("mdls marks unused variables, parameters, members, and imports", "[mdls]") {
	auto messages = session({open(models)});
	std::set<std::string> unused;
	for (const auto& m : messages)
		if (m.value("method", "") == "textDocument/publishDiagnostics")
			for (const auto& d : m["params"]["diagnostics"])
				if (d.value("code", "") == "unused") {
					CHECK(d["tags"] == json::array({1}));
					unused.insert(d["message"].get<std::string>());
				}
	CHECK(unused.contains("unused is never used"));
	CHECK(unused.contains("The parameter extra is never used"));
	CHECK(unused.contains("The member spare of Block is never used"));
	CHECK_FALSE(unused.contains("The import std is never used"));
}

TEST_CASE("mdls completes einops axes and renames them within the pattern", "[mdls]") {
	std::string typing = models;
	typing.replace(typing.find("| a=2]"), 6, "| ]");
	auto change = json{
			{"method", "textDocument/didChange"},
			{"params",
			 {{"textDocument", {{"uri", uri}, {"version", 2}}}, {"contentChanges", json::array({{{"text", typing}}})}}},
	};
	auto renaming = request(2, "textDocument/rename", 12, 71);
	renaming["params"]["newName"] = "heads";
	auto messages = session(
			{open(models), renaming, request(3, "textDocument/linkedEditingRange", 12, 71), change,
			 request(4, "textDocument/completion", 12, 91)}
	);
	const auto& edits = (*response(messages, 2))["result"]["changes"][uri];
	CHECK(edits.size() == 3); // a in both sides and in the binding
	CHECK((*response(messages, 3))["result"]["ranges"].size() == 3);
	std::set<std::string> labels;
	for (const auto& item : (*response(messages, 4))["result"])
		labels.insert(item["label"].get<std::string>());
	CHECK(labels == std::set<std::string>{"a", "b"});
}

TEST_CASE("mdls links imports and references in comments", "[mdls]") {
	auto messages = session({open(models), request(1, "textDocument/documentLink")});
	std::set<std::string> targets;
	for (const auto& link : (*response(messages, 1))["result"])
		targets.insert(link["target"].get<std::string>());
	CHECK(std::any_of(targets.begin(), targets.end(), [](const auto& t) { return t.ends_with("/std/losses.dlm"); }));
	CHECK(targets.contains(uri + "#L4")); // \ref Block
}

TEST_CASE("mdls shows the call hierarchy", "[mdls]") {
	auto prepare = session({open(models), request(1, "textDocument/prepareCallHierarchy", 3, 8)});
	const auto items = (*response(prepare, 1))["result"];
	REQUIRE(items.size() == 1);
	CHECK(items[0]["name"] == "Block");
	auto forward = session({open(models), request(1, "textDocument/prepareCallHierarchy", 7, 20)});
	const auto item = (*response(forward, 1))["result"][0];
	auto incoming = json{{"id", 2}, {"method", "callHierarchy/incomingCalls"}, {"params", {{"item", item}}}};
	auto outgoing = json{{"id", 3}, {"method", "callHierarchy/outgoingCalls"}, {"params", {{"item", item}}}};
	auto calls = session({open(models), incoming, outgoing});
	std::set<std::string> callers, callees;
	for (const auto& c : (*response(calls, 2))["result"])
		callers.insert(c["from"]["name"].get<std::string>());
	for (const auto& c : (*response(calls, 3))["result"])
		callees.insert(c["to"]["name"].get<std::string>());
	CHECK(callers.contains("forward")); // Net.forward calls b(x)
	CHECK(callees.contains("forward")); // lin(x), norm(h): the forwards of Linear and LayerNorm
}

TEST_CASE("mdls expands selections by syntax and finds symbols in the workspace", "[mdls]") {
	auto select = request(1, "textDocument/selectionRange");
	select["params"]["positions"] = json::array({{{"line", 8}, {"character", 19}}});
	auto messages = session({open(models), select});
	const auto& s = (*response(messages, 1))["result"][0];
	CHECK(s["range"]["start"] == json{{"line", 8}, {"character", 17}}); // `lin`
	size_t depth = 0;
	for (const json* p = &s; p->contains("parent"); p = &(*p)["parent"])
		++depth;
	CHECK(depth >= 4); // lin, lin(x), lin(x) + x, the statement, ...

	const auto dir = std::filesystem::temp_directory_path() / "mdls-workspace-test";
	std::filesystem::create_directories(dir / "sub");
	{
		std::ofstream{dir / "sub" / "layers.dlm"} << "module FancyLayer[] { float f(x: float) => x; }\n";
	}
	std::stringstream in, out;
	mdls::writeMessage(
			in,
			{{"jsonrpc", "2.0"}, {"id", 0}, {"method", "initialize"}, {"params", {{"rootUri", mdls::pathToUri(dir)}}}}
	);
	mdls::writeMessage(
			in, {{"jsonrpc", "2.0"}, {"id", 1}, {"method", "workspace/symbol"}, {"params", {{"query", "fncy"}}}}
	);
	mdls::writeMessage(in, {{"jsonrpc", "2.0"}, {"id", 2}, {"method", "shutdown"}});
	mdls::writeMessage(in, {{"jsonrpc", "2.0"}, {"method", "exit"}});
	mdls::Server{in, out}.run();
	std::filesystem::remove_all(dir);
	std::vector<json> replies;
	while (auto m = mdls::readMessage(out))
		replies.push_back(*m);
	const auto& symbols = (*response(replies, 1))["result"];
	REQUIRE(symbols.size() == 1);
	CHECK(symbols[0]["name"] == "FancyLayer");
}

TEST_CASE("mdls analyzes what parses and reports errors of imported files and new settings", "[mdls]") {
	const std::string broken = "module M[n: uint] {\n    float[..., n] good(x: float[..., n]) => x + z;\n"
							   "    float[..., n] bad(x: float[..., n]) => x +;\n}\n";
	auto messages = session({open(broken)});
	std::set<std::string> found;
	for (const auto& m : messages)
		if (m.value("method", "") == "textDocument/publishDiagnostics")
			for (const auto& d : m["params"]["diagnostics"])
				found.insert(d["message"].get<std::string>());
	CHECK(found.contains("Variable z not defined")); // in the function that parses, despite the syntax error

	const std::string mismatch = "module T[a: uint, b: uint] { W: float[a, b] float[b] f(x: float[a]) => W @ x; }\n";
	auto strict =
			json{{"method", "workspace/didChangeConfiguration"},
				 {"params", {{"settings", {{"mdl", {{"strict", true}}}}}}}};
	auto configured = session({open(mismatch), strict});
	std::vector<int> severities;
	for (const auto& m : configured)
		if (m.value("method", "") == "textDocument/publishDiagnostics")
			for (const auto& d : m["params"]["diagnostics"])
				if (d.value("code", "") == "permits-illegal-instantiation")
					severities.push_back(d["severity"]);
	REQUIRE_FALSE(severities.empty());
	CHECK(severities.front() == 2); // a warning, then (strict) an error
	CHECK(severities.back() == 1);
}

TEST_CASE("mdls reports uses of conditional members where they may not exist", "[mdls]") {
	const std::string source = R"(module Test[input: uint, output: uint] {
    W: float[input, output]
    float[output] forward(x: float[input]) => x @ W;
}
module Test2[bias: bool = true] {
    @conditional(bias)
    test: Test[4, 5]
    other: Test[4, 5]
    float[5] forward(x: float[4]) => test.forward(x);
    float[5] guarded(x: float[4]) => bias ? test(x) : other(x);
    float[4] shadowed(test: float[4]) => test;
}
module Test3[] {
    test: Test2[.bias = false]
    float[5] forward(x: float[4]) => test(x);
}
)";
	auto messages = session({open(source)});
	std::vector<std::tuple<int, int, std::string>> found; // line, severity, code
	for (const auto& m : messages)
		if (m.value("method", "") == "textDocument/publishDiagnostics")
			for (const auto& d : m["params"]["diagnostics"])
				if (d.value("code", "") != "unused" && d.value("code", "") != "shadow")
					found.emplace_back(d["range"]["start"]["line"], d["severity"], d.value("code", ""));
	// Test2: a warning where `test` is used unguarded (not in `bias ? ... : ...`, not the parameter `test`);
	// Test3: the error of the instance Test2[false] at its instantiation (and inside Test2.forward).
	CHECK(std::ranges::contains(found, std::tuple{8, 2, std::string{"permits-illegal-instantiation"}}));
	CHECK(std::ranges::count_if(found, [](const auto& f) { return std::get<0>(f) == 9 || std::get<0>(f) == 10; }) == 0);
	CHECK(std::ranges::count_if(found, [](const auto& f) { return std::get<0>(f) == 13 && std::get<1>(f) == 1; }) == 1);
}

TEST_CASE("mdls highlights `_` as a discarded value", "[mdls]") {
	const std::string text = "float[4] f(x: float[4]) {\n    for (_ in 0..2) { x = x + 1.0; }\n    auto a, _ = g(x);\n"
							 "    return a[_ d -> d];\n}\n";
	const auto tokens = mdls::semanticTokens(text);
	const auto& types = mdls::semanticTokenTypes();
	std::vector<std::pair<size_t, size_t>> discarded;
	for (const auto& t : tokens)
		if (types[t.type] == "discarded")
			discarded.emplace_back(t.line, t.column);
	CHECK(discarded == std::vector<std::pair<size_t, size_t>>{{1, 9}, {2, 12}, {3, 13}});
	auto messages = session({open(text), request(1, "textDocument/hover", 2, 12)});
	CHECK((*response(messages, 1))["result"]["contents"]["value"] ==
		  "`_`: a discarded value (no value, type, or members)");
}
