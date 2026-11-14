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

TEST_CASE("doc comments become Markdown, including math in all spellings", "[parser][index]") {
	using analysis::docCommentToMarkdown;
	CHECK(docCommentToMarkdown("/** @brief Computes \\f$x^2\\f$ and @f$y@f$. */") == "Computes $x^2$ and $y$.");
	CHECK(docCommentToMarkdown("/** @brief Inline \\f(a\\f) and @f(b@f). */") == "Inline $a$ and $b$.");
	CHECK(docCommentToMarkdown("/** @brief Display @f[ x = y @f] */") == "Display $$ x = y $$");
	CHECK(docCommentToMarkdown("/** @brief Env \\f{align}{a &= b\\f} */") == "Env $$\\begin{align}a &= b\\end{align}$$");
	CHECK(analysis::docParameter("/** @param[in] x the input @tparam N the size */", "x") == "the input");
	CHECK(analysis::docParameter("/** @param[in] x the input @tparam N the size */", "N") == "the size");
}

TEST_CASE("parameters get their @param documentation", "[parser][index]") {
	const std::string source = R"(/**
 * @brief A layer.
 * @param width the number of features
 */
module Layer[width: uint] {
    W: float[width, width]
    /**
     * @brief Applies it.
     * @param x the input
     */
    float[..., width] forward(x: float[..., width]) => x @ W;
}
/**
 * @tparam N the size
 * @param y the values
 */
template<N: uint>
float[...] total(y: float[..., N]) => __builtin_sum(y);
)";
	auto result = analyzeSource(source);
	auto doc = [&](size_t line, const std::string& name) {
		const auto* ref = at(result, source, line, name);
		REQUIRE(ref != nullptr);
		REQUIRE(ref->symbol);
		return result.index->symbols()[*ref->symbol].documentation;
	};
	CHECK(doc(5, "width") == "the number of features"); // hyperparameter
	CHECK(doc(11, "x") == "the input");				   // function parameter
	CHECK(doc(17, "N") == "the size");				   // template parameter
	CHECK(doc(18, "y") == "the values");
}

TEST_CASE("doc comments accept both command spellings, sections, code blocks, and word styles", "[parser][index]") {
	using analysis::docCommentToMarkdown;
	// `\cmd` and `@cmd` are the same command (as in Doxygen).
	CHECK(docCommentToMarkdown("/** \\brief Short. \\param x the input \\return y \\todo more */") ==
		  docCommentToMarkdown("/** @brief Short. @param x the input @return y @todo more */"));
	CHECK(docCommentToMarkdown("/** \\brief Short. \\todo more */") == "Short.\n\n**Todo** more");
	CHECK(analysis::docParameter("/** \\param x the input */", "x") == "the input");
	// Sections in order, after the parameters and the result.
	CHECK(docCommentToMarkdown("/** @brief B. @note N. @warning W. @see S. \\deprecated D. */") ==
		  "B.\n\n**Note** N.\n\n**Warning** W.\n\n**See also** S.\n\n**Deprecated** D.");
	// Word styles.
	CHECK(docCommentToMarkdown("/** @brief An \\a italic, \\b bold, and \\c code word. */") ==
		  "An *italic*, **bold**, and `code` word.");
	// Code blocks keep their lines.
	CHECK(docCommentToMarkdown("/**\n * @brief Usage:\n * @code\n * auto y = f(x);\n *   z\n * @endcode\n */") ==
		  "Usage:\n\n```\nauto y = f(x);\n  z\n```");
}
