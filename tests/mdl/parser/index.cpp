// The semantic index of parser::analyze (for tools): definitions, references, types, documentation, visible names.
#include <mdl/parser.hpp>

#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>

using namespace mdl;
using analysis::SymbolKind;

namespace {
	parser::AnalysisResult analyzeSource(const std::string& code) {
		std::istringstream source{code};
		return parser::analyze(source, "<test>");
	}

	/** @brief The reference at line @p line (from 1) on the first occurrence of @p name on that line. */
	const analysis::Reference* at(const parser::AnalysisResult& r, const std::string& code, size_t line, const std::string& name) {
		std::istringstream in{code};
		std::string text;
		for (size_t i = 0; i < line && std::getline(in, text); ++i) {
		}
		const auto column = text.find(name);
		REQUIRE(column != std::string::npos);
		return r.index->referenceAt("<test>", {line, column});
	}
} // namespace

const std::string code = R"(import std;
/** @brief A residual block. @param d the width */
module Block[d: uint] {
    lin: Linear[d, d]
    float[..., d] forward(x: float[..., d]) {
        auto h = lin(x);
        return h + x;
    }
}
module Net[] {
    block: Block[768]
    float[..., 768] forward(x: float[..., 768]) => block(x);
})";

TEST_CASE("the index has the types of variables, symbolic and concrete", "[parser][index]") {
	auto result = analyzeSource(code);
	REQUIRE(result.ok);
	const auto* h = at(result, code, 6, "h");
	REQUIRE(h != nullptr);
	CHECK(h->types.contains("float32[..., d]"));   // the module checked with a symbolic width
	CHECK(h->types.contains("float32[..., 768]")); // Block[768] in Net
	REQUIRE(h->symbol);
	CHECK(result.index->symbols()[*h->symbol].kind == SymbolKind::variable);
}

TEST_CASE("references resolve to their definitions", "[parser][index]") {
	auto result = analyzeSource(code);
	const auto* x = at(result, code, 7, "x");
	REQUIRE(x != nullptr);
	REQUIRE(x->symbol);
	const auto& param = result.index->symbols()[*x->symbol];
	CHECK(param.kind == SymbolKind::parameter);
	CHECK(param.range.begin.line == 5);
	CHECK(param.detail == "x: float[..., d]");

	const auto* block = at(result, code, 11, "Block");
	REQUIRE(block != nullptr);
	REQUIRE(block->symbol);
	const auto& module = result.index->symbols()[*block->symbol];
	CHECK(module.kind == SymbolKind::module);
	CHECK(module.detail == "module Block[d: uint]");
	CHECK(module.documentation == "A residual block.\n\n**Parameters**\n- `d`: the width");
}

TEST_CASE("the documentation of the standard library is in the index", "[parser][index]") {
	auto result = analyzeSource(code);
	const auto* lin = at(result, code, 4, "Linear");
	REQUIRE(lin != nullptr);
	REQUIRE(lin->symbol);
	const auto& linear = result.index->symbols()[*lin->symbol];
	CHECK(linear.file.ends_with("nn.dlm"));
	CHECK_FALSE(linear.documentation.empty());
}

TEST_CASE("module instances know their members", "[parser][index]") {
	auto result = analyzeSource(code);
	const auto* lin = at(result, code, 6, "lin");
	REQUIRE(lin != nullptr);
	CHECK(lin->members.contains("W"));
	CHECK(lin->members.contains("forward"));
	CHECK(lin->types.contains("Linear"));
}

TEST_CASE("the visible names at a position", "[parser][index]") {
	auto result = analyzeSource(code);
	auto names = [&](SourcePosition p) {
		std::set<std::string> set;
		for (const auto* s : result.index->visibleAt("<test>", p))
			set.insert(s->name);
		return set;
	};
	const auto inForward = names({7, 8}); // in `return h + x;`
	CHECK(inForward.contains("h"));
	CHECK(inForward.contains("x"));
	CHECK(inForward.contains("lin"));
	CHECK(inForward.contains("d"));
	CHECK(inForward.contains("Net"));
	CHECK(inForward.contains("softmax")); // imported
	CHECK_FALSE(names({6, 8}).contains("h")); // before its declaration
	CHECK_FALSE(names({12, 8}).contains("h")); // in another function
}
