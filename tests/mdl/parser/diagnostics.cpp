// The diagnostics of the frontend as data (parser::analyze): messages, ranges, severities, hints, and flags.
#include <mdl/parser.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace mdl;

namespace {
	parser::AnalysisResult analyzeSource(const std::string& code, parser::ParseOptions options = {}) {
		std::istringstream source{code};
		return parser::analyze(source, "<test>", options);
	}

	/** @brief The range as `line:begin-end` (columns from 1, end inclusive) for compact checks. */
	std::string where(const Diagnostic& d) {
		return std::to_string(d.range.begin.line) + ":" + std::to_string(d.range.begin.column + 1) + "-" +
			   std::to_string(d.range.end.column);
	}
} // namespace

TEST_CASE("diagnostics are returned as data with their ranges", "[parser][diagnostics]") {
	auto result = analyzeSource("module M[] { W: float[4, 3]\n    float[..., 4] f(x: float[..., 4]) => x @ W; }");
	REQUIRE_FALSE(result.ok);
	REQUIRE(result.diagnostics.size() == 1);
	const auto& d = result.diagnostics[0];
	CHECK(d.severity == Severity::error);
	CHECK(d.file == "<test>");
	CHECK(d.message == "f returns float32[..., 3], but its declared return type is float32[..., 4]");
	CHECK(where(d) == "2:19-19"); // the function name
}

TEST_CASE("a range spans the marked expression", "[parser][diagnostics]") {
	auto result = analyzeSource(R"(
		module First[] { W: float[3, 2] float[..., 3] forward(x: float[..., 2]) => x @ W.T(); }
		module Net[] {
			first: First[]
			float[..., 3] forward(x: float[..., 4]) => first(x);
		})");
	REQUIRE(result.diagnostics.size() == 1);
	CHECK(result.diagnostics[0].message == "Argument 1 of forward is float32[..., 4], but the parameter x is float32[..., 2]");
	CHECK(where(result.diagnostics[0]) == "5:53-53");
}

TEST_CASE("syntax and lexer errors are diagnostics", "[parser][diagnostics]") {
	auto syntax = analyzeSource("float[...] f(x: float[...]) => x +;");
	REQUIRE_FALSE(syntax.ok);
	REQUIRE(syntax.diagnostics.size() == 1);
	CHECK(where(syntax.diagnostics[0]) == "1:35-35"); // the `;`

	// An unknown character: an error (it was printed but not counted before).
	auto lexer = analyzeSource("float[...] f(x: float[...]) => x $ x;");
	REQUIRE_FALSE(lexer.ok);
	REQUIRE_FALSE(lexer.diagnostics.empty());
	CHECK(lexer.diagnostics[0].message.find("token recognition error") != std::string::npos);
	CHECK(where(lexer.diagnostics[0]) == "1:34-34");
}

TEST_CASE("a hint and a flag are separate from the message", "[parser][diagnostics]") {
	auto hint = analyzeSource(R"(
		module Lin[indim: uint, outdim: uint, bias: bool = true] {
			W: float[outdim, indim]
			float[..., outdim] forward(x: float[..., indim]) => x @ W.T();
		}
		module Net[] { l: Lin[4, 3, false] float[..., 3] f(x: float[..., 4]) => l(x); })");
	REQUIRE(hint.diagnostics.size() == 1);
	CHECK(hint.diagnostics[0].hint == "write `.bias=false`");

	auto flag = analyzeSource("module M[] { W: bfloat16[4] float32[..., 4] f(x: float32[..., 4]) => x * W; }");
	REQUIRE(flag.diagnostics.size() == 1);
	CHECK(flag.diagnostics[0].flag == "float-widening");
	CHECK(flag.diagnostics[0].message == "Operator *: implicit widening from bfloat16 to float32");
	CHECK(flag.diagnostics[0].hint == "convert explicitly, e.g. `float32(x)`");

	parser::ParseOptions warn;
	warn.floatWidening = parser::DiagnosticLevel::warning;
	auto warning = analyzeSource("module M[] { W: bfloat16[4] float32[..., 4] f(x: float32[..., 4]) => x * W; }", warn);
	CHECK(warning.ok);
	REQUIRE(warning.diagnostics.size() == 1);
	CHECK(warning.diagnostics[0].severity == Severity::warning);
}

TEST_CASE("code elaborated several times reports an error once", "[parser][diagnostics]") {
	// The error is in the module body, which is expanded for both instances.
	auto result = analyzeSource(R"(
		module Bad[n: uint] { float[..., n] forward(x: float[..., n]) => y; }
		module Net[] {
			a: Bad[2]
			b: Bad[3]
			float[..., 2] forward(x: float[..., 2]) => a(x);
		})");
	REQUIRE_FALSE(result.ok);
	CHECK(result.diagnostics.size() == 1);
}

TEST_CASE("analyze keeps the compilation unit when there are errors", "[parser][diagnostics]") {
	auto result = analyzeSource(R"(
		float[...] good(x: float[...]) => x + 1;
		float[...] bad(x: float[...]) => y;)");
	REQUIRE_FALSE(result.ok);
	CHECK(result.unit.definitions.contains("good"));
	// parse() fails on the same code (and prints the diagnostic to a sink if one is given).
	size_t reported = 0;
	parser::ParseOptions options;
	options.diagnostics = [&](const Diagnostic&) { ++reported; };
	std::istringstream source{"float[...] bad(x: float[...]) => y;"};
	CHECK_FALSE(parser::parse(source, "<test>", options).has_value());
	CHECK(reported == 1);
}

TEST_CASE("sources can come from a reader (unsaved editor buffers)", "[parser][diagnostics]") {
	const auto dir = std::filesystem::temp_directory_path() / "mdl-overlay-test"; // never written: only read via readFile
	parser::ParseOptions options;
	options.sourceDir = dir;
	options.readFile = [&](const std::filesystem::path& file) -> std::optional<std::string> {
		if (file.filename() == "helpers.dlm")
			return "float[...] twice(x: float[...]) => x + x;";
		return std::nullopt;
	};
	// Imports must exist on disk to be found; the reader provides their contents.
	std::filesystem::create_directories(dir);
	{ std::ofstream{dir / "helpers.dlm"} << "this is not valid"; }
	auto result = analyzeSource("import helpers;\nfloat[...] f(x: float[...]) => twice(x);", options);
	std::filesystem::remove_all(dir);
	CHECK(result.ok);
}

TEST_CASE("modules with hyperparameters are checked symbolically", "[parser][diagnostics]") {
	// Not instantiated anywhere: parse() does not check the body, analyze() does (with symbolic hyperparameters).
	const std::string code = R"(
		module Lin[indim: uint, outdim: uint, bias: bool = true] {
			W: float[outdim, indim]
			@conditional(bias)
			b: float[outdim]
			float[..., outdim] forward(x: float[..., indim]) => bias ? x @ W.T() + b : x @ W.T();
		}
		module Stack[n: uint, d: uint] {
			layers: [Lin[d, d] for i in 0..n]
			float[n, ..., d] hidden(x: float[..., d]) {
				float[n, ..., d] hs;
				auto h = x;
				for (i in 0..n) { h = layers[i](h); hs[i] = h; }
				return hs;
			}
			float[..., d] last(x: float[..., d]) => layers[n - 1](x) + layers[0](x);
		})";
	CHECK(analyzeSource(code).ok);
	// Definite mismatches are found also with symbolic sizes; sizes that are different hyperparameters (e.g. indim
	// and outdim) may be equal in an instance, so they are not reported.
	auto wrong = analyzeSource(R"(
		module M[outdim: uint] {
			W: float[outdim, 3]
			float[..., outdim] forward(x: float[..., 4]) => x @ W.T();
		})");
	REQUIRE(wrong.diagnostics.size() == 1);
	CHECK(wrong.diagnostics[0].message ==
		  "Operator @: the inner dimensions of the matrix product differ ([..., 4] @ [3, outdim])");
}

TEST_CASE("the symbolic analysis reports no errors in the repository's models", "[parser][diagnostics]") {
	size_t count = 0;
	for (const char* dir : {"std", "examples"})
		for (const auto& entry : std::filesystem::recursive_directory_iterator(std::filesystem::path{MDL_SOURCE_DIR} / dir)) {
			if (entry.path().extension() != ".dlm")
				continue;
			auto result = parser::analyze(entry.path());
			INFO(entry.path().string());
			for (const auto& d : result.diagnostics)
				INFO(d.file + ":" + std::to_string(d.range.begin.line) + ": " + d.message);
			CHECK(result.diagnostics.empty());
			++count;
		}
	CHECK(count > 5);
}

TEST_CASE("sizes that may differ are warnings, and errors in strict mode", "[parser][diagnostics]") {
	const std::string code = R"(
		module Test[input: uint, output: uint] {
			W: float[input, output]
			b: float[output]
			float[output] forward(x: float[input]) => W @ x + b;
		})";
	auto lenient = analyzeSource(code);
	CHECK(lenient.ok);
	REQUIRE(lenient.diagnostics.size() == 2);
	CHECK(lenient.diagnostics[0].severity == Severity::warning);
	CHECK(lenient.diagnostics[0].flag == "possible-mismatch");
	CHECK(lenient.diagnostics[0].message ==
		  "Operator @: the inner dimensions of the matrix product may differ ([input, output] @ [input])");
	CHECK(lenient.diagnostics[1].message == "Operator +: cannot broadcast [input] and [output] (the sizes may differ)");

	parser::ParseOptions strict;
	strict.possibleMismatch = parser::DiagnosticLevel::error;
	auto strictResult = analyzeSource(code, strict);
	CHECK_FALSE(strictResult.ok);
	REQUIRE(strictResult.diagnostics.size() == 2);
	CHECK(strictResult.diagnostics[0].severity == Severity::error);

	// Correct generic code has no warnings (x @ W.T() is [..., output]).
	auto correct = analyzeSource(R"(
		module Test[input: uint, output: uint] {
			W: float[output, input]
			b: float[output]
			float[..., output] forward(x: float[..., input]) => x @ W.T() + b;
		})",
								 strict);
	CHECK(correct.ok);
	CHECK(correct.diagnostics.empty());
}
