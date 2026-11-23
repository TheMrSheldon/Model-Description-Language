// The formatter (mdl::format): style rules, options, idempotence, and that only whitespace changes.
#include <mdl/format.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>

using namespace mdl;
namespace fs = std::filesystem;

namespace {
	std::string formatted(const std::string& source, const format::FormatOptions& options = {}) {
		auto result = format::format(source, options);
		REQUIRE(result.ok);
		return result.text;
	}
} // namespace

TEST_CASE("the formatter normalizes indentation, spaces, and line breaks", "[format]") {
	const std::string source = R"(import std;
module   M[n:uint,   bias:bool=true]{ W:float[n,n] @conditional(bias) b:float[n]   // bias
  layers:[Linear[n,n,.bias=false] for i in 0..3]
float[...,n] forward(x:float[...,n]){auto h=x;for(i in 0..3){h=layers[i](h);}


  auto y=-h[...,0,:];   return bias?h@W.T()+b:h@W.T();}
  template<N:uint> float[...] f(x:float[...,N])=>x[... (a b) -> ... a b|a=2,b=N/2];
}
struct S { dims: struct { a: uint b: uint } k: array[uint;  4] }
)";
	CHECK(formatted(source) == R"(import std;
module M[n: uint, bias: bool = true] {
    W: float[n, n]
    @conditional(bias) b: float[n]  // bias
    layers: [Linear[n, n, .bias=false] for i in 0..3]
    float[..., n] forward(x: float[..., n]) {
        auto h = x;
        for (i in 0..3) {
            h = layers[i](h);
        }

        auto y = -h[..., 0, :];
        return bias ? h @ W.T() + b : h @ W.T();
    }
    template<N: uint> float[...] f(x: float[..., N]) => x[... (a b) -> ... a b | a=2, b=N / 2];
}
struct S {
    dims: struct { a: uint b: uint }
    k: array[uint; 4]
}
)");
}

TEST_CASE("the formatter keeps comments and chosen line breaks", "[format]") {
	// A block comment moves as a whole (its lines keep their indentation relative to `/**`).
	const std::string source = "  /**\n   * @brief Doc.\n   */\n  template<N: uint>\nfloat[...] sum(x: float[..., N]) "
							   "=> __builtin_sum(x);\nmodule M[] {\n@conditional(true)\nW: float[3]\n}\n";
	CHECK(formatted(source) == "/**\n * @brief Doc.\n */\ntemplate<N: uint>\nfloat[...] sum(x: float[..., N]) => "
							   "__builtin_sum(x);\nmodule M[] {\n    @conditional(true)\n    W: float[3]\n}\n");
	// Consecutive trailing comments are aligned.
	CHECK(formatted("module M[] {\n W: float[3] // a\n bb: float[3]   // b\n}\n") ==
		  "module M[] {\n    W: float[3]   // a\n    bb: float[3]  // b\n}\n");
}

TEST_CASE("long lines are wrapped", "[format]") {
	format::FormatOptions narrow;
	narrow.columnLimit = 50;
	CHECK(formatted("float[...] f(x: float[...], y: float[...]) => x + y + x * y;", narrow) ==
		  "float[...] f(x: float[...], y: float[...]) =>\n    x + y + x * y;\n");
	CHECK(formatted(
				  "float[...] f(first: float[...], second: float[...], third: float[...]) { return first; }", narrow
		  ) ==
		  "float[...] f(first: float[...],\n        second: float[...], third: float[...]) {\n    return first;\n}\n");
}

TEST_CASE("the indentation width and the blank lines are options", "[format]") {
	format::FormatOptions options;
	options.indentWidth = 2;
	options.maxEmptyLines = 0;
	CHECK(formatted("module M[] {\n W: float[3]\n\n\n b: float[3]\n}\n", options) ==
		  "module M[] {\n  W: float[3]\n  b: float[3]\n}\n");
}

TEST_CASE("a file with syntax errors is not formatted", "[format]") {
	auto result = format::format("float[...] f(x: float[...]) => x +;");
	CHECK_FALSE(result.ok);
	CHECK(result.text == "float[...] f(x: float[...]) => x +;");
	CHECK(result.diagnostics.size() == 1);
}

TEST_CASE("every source file of the repository formats idempotently with the same tokens", "[format]") {
	size_t count = 0;
	for (const char* dir : {"std", "examples", "tests"})
		for (const auto& entry : fs::recursive_directory_iterator(fs::path{MDL_SOURCE_DIR} / dir)) {
			if (entry.path().extension() != ".dlm")
				continue;
			std::ifstream in{entry.path()};
			const std::string source{std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
			INFO(entry.path().string());
			auto once = format::format(source); // checks internally that only whitespace changes
			REQUIRE(once.ok);
			auto twice = format::format(once.text);
			REQUIRE(twice.ok);
			CHECK(twice.text == once.text);
			++count;
		}
	CHECK(count > 10);
}

TEST_CASE("style files are read and found", "[format]") {
	const auto dir = fs::temp_directory_path() / ("mdl-format-test-" + std::to_string(std::random_device{}()));
	fs::create_directories(dir / "sub");
	{
		std::ofstream{dir / ".mdl-format"} << "# style\nIndentWidth: 2\nColumnLimit: 80  # narrow\n";
	}
	auto found = format::findStyleFile(dir / "sub" / "model.dlm");
	REQUIRE(found);
	CHECK(fs::equivalent(*found, dir / ".mdl-format"));
	auto style = format::loadStyle(*found);
	REQUIRE(style);
	CHECK(style->indentWidth == 2);
	CHECK(style->columnLimit == 80);
	CHECK(style->maxEmptyLines == 1);
	{
		std::ofstream{dir / "bad"} << "IndentWidht: 2\n";
	}
	auto bad = format::loadStyle(dir / "bad");
	REQUIRE_FALSE(bad);
	CHECK(bad.error().find("unknown key IndentWidht") != std::string::npos);
	fs::remove_all(dir);
}

TEST_CASE("naming conventions are recognized and converted", "[format][naming]") {
	using format::Naming;
	CHECK(format::isSnakeCase("hidden_size"));
	CHECK(format::isSnakeCase("d"));
	CHECK_FALSE(format::isSnakeCase("hiddenSize"));
	CHECK_FALSE(format::isSnakeCase("hidden__size"));
	CHECK(format::isCamelCase("hiddenSize"));
	CHECK_FALSE(format::isCamelCase("HiddenSize"));
	CHECK(format::isPascalCase("TransformerEncoder"));
	CHECK_FALSE(format::isPascalCase("transformer_encoder"));
	CHECK(format::isAllCaps("MAX_LEN"));
	CHECK(format::isAllCaps("N"));
	CHECK_FALSE(format::isAllCaps("Max_Len"));
	CHECK(format::isLowercase("forward"));
	CHECK_FALSE(format::isLowercase("for_ward"));
	CHECK(format::convert("hiddenSize", Naming::snakeCase) == "hidden_size");
	CHECK(format::convert("hidden_size", Naming::camelCase) == "hiddenSize");
	CHECK(format::convert("HTTPServer", Naming::snakeCase) == "http_server");
	CHECK(format::convert("num_attn_heads", Naming::pascalCase) == "NumAttnHeads");
	CHECK(format::convert("maxLen", Naming::allCaps) == "MAX_LEN");
}

TEST_CASE("the style's naming rules are checked for each kind of definition", "[format][naming]") {
	format::FormatOptions options;
	options.naming.module = format::Naming::pascalCase;
	options.naming.hyperparameter = format::Naming::snakeCase;
	options.naming.variable = format::Naming::camelCase;
	options.naming.function = format::Naming::camelCase;
	auto result = format::format(
			R"(module my_block[hiddenSize: uint] {
    float[..., hiddenSize] forward(x: float[..., hiddenSize]) {
        auto out_value = x;
        return out_value;
    }
}
)",
			options
	);
	REQUIRE(result.ok);
	std::vector<std::string> messages, hints;
	for (const auto& d : result.diagnostics) {
		CHECK(d.flag == "naming");
		CHECK(d.severity == Severity::warning);
		messages.push_back(d.message);
		hints.push_back(d.hint);
	}
	CHECK(messages == std::vector<std::string>{
							  "The module name my_block is not PascalCase",
							  "The hyperparameter name hiddenSize is not snake_case",
							  "The variable name out_value is not camelCase",
					  });
	CHECK(hints[0] == "rename it to `MyBlock`");
}

TEST_CASE("comments are reflowed at the column limit if enabled", "[format]") {
	format::FormatOptions options;
	options.columnLimit = 34;
	const std::string source =
			"// one two three four five six seven eight\n/**\n * alpha beta gamma delta epsilon zeta\n */\n"
			"float[...] f(x: float[...]) => x;\n";
	CHECK(formatted(source, options) == source); // off by default
	options.reflowComments = true;
	const auto text = formatted(source, options);
	CHECK(text ==
		  "// one two three four five six\n// seven eight\n/**\n * alpha beta gamma delta epsilon\n * zeta\n */\n"
		  "float[...] f(x: float[...]) => x;\n");
	CHECK(formatted(text, options) == text);
}

TEST_CASE("style files set the reflow and naming options", "[format][naming]") {
	const auto dir = fs::temp_directory_path() / ("mdl-format-naming-" + std::to_string(std::random_device{}()));
	fs::create_directories(dir);
	{
		std::ofstream{dir / ".mdl-format"}
				<< "ReflowComments: true\nModuleNaming: PascalCase\nVariableNaming: snake_case\n";
	}
	auto style = format::loadStyle(dir / ".mdl-format");
	REQUIRE(style);
	CHECK(style->reflowComments);
	CHECK(style->naming.module == format::Naming::pascalCase);
	CHECK(style->naming.variable == format::Naming::snakeCase);
	CHECK(style->naming.function == format::Naming::any);
	{
		std::ofstream{dir / "bad"} << "ModuleNaming: kebab-case\n";
	}
	CHECK_FALSE(format::loadStyle(dir / "bad"));
	fs::remove_all(dir);
}

TEST_CASE("invalid UTF-8 is reported and left unchanged", "[format]") {
	const std::string source = "// caf\xE9\nfloat[...] f(x: float[...])=>x;\n"; // (Latin-1 é)
	auto result = format::format(source);
	CHECK(result.text == source);
	REQUIRE(result.diagnostics.size() == 1);
	CHECK(result.diagnostics[0].message == "Invalid UTF-8 (byte 0xE9): source files must be UTF-8");
	CHECK(result.diagnostics[0].range.begin == SourcePosition{1, 6});
}

TEST_CASE("the formatter spaces imports and visibility", "[format]") {
	CHECK(formatted("from  .lib import a ,b as  c;\nfrom ..  import  d;\npublic  from ...x.y import *;\nimport  .w  as "
					"v;\n") ==
		  "from .lib import a, b as c;\nfrom .. import d;\npublic from ...x.y import *;\nimport .w as v;\n");
	CHECK(formatted("private float[...] f(x: float[...]) => x;\nmodule M[] {\n  public W: float[4]\n}\n") ==
		  "private float[...] f(x: float[...]) => x;\nmodule M[] {\n    public W: float[4]\n}\n");
}
