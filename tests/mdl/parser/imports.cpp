#include <mdl/parser.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

using namespace mdl;
namespace fs = std::filesystem;

namespace {
	/** A temporary directory that is removed when the object goes out of scope. */
	class TempDir {
	public:
		TempDir() {
			static std::atomic<int> counter{0};
			path = fs::temp_directory_path() /
				   ("mdl-test-" + std::to_string(std::random_device{}()) + "-" + std::to_string(counter++));
			fs::create_directories(path);
		}
		~TempDir() { fs::remove_all(path); }

		fs::path write(const std::string& relative, const std::string& content) const {
			auto file = path / relative;
			fs::create_directories(file.parent_path());
			std::ofstream{file} << content;
			return file;
		}

		fs::path path;
	};
} // namespace

TEST_CASE("the standard library can be imported", "[parser][imports]") {
	std::istringstream source{R"(
		import std;
		module Block[dim: uint] {
			lin: Linear[dim, dim]
			norm: LayerNorm[dim]
			float[..., dim] forward(x: float[..., dim]) => norm(gelu(lin(x)) + softmax(x));
		}
	)"};
	auto unit = parser::parse(source, "<test>");
	REQUIRE(unit.has_value());
	// Only the definitions of the parsed file are part of the compilation unit.
	CHECK(unit->definitions.size() == 1);
	CHECK(unit->definitions.contains("Block"));
	// Block has a hyperparameter: its members are expanded when it is instantiated.
	auto block = unit->instantiate("Block", {{"dim", Object{&unit->graph().addConstNode(Atomic{8})}}});
	REQUIRE(block.has_value());
	CHECK(weightPaths(block->get()) == std::vector<std::string>{"lin.W", "lin.b", "norm.beta", "norm.gamma"});
}

TEST_CASE("single files of the standard library can be imported", "[parser][imports]") {
	std::istringstream source{"import std.arithmetic;\nfloat[...] f(x: float[...]) => exp(x) + tanh(x);"};
	CHECK(parser::parse(source, "<test>").has_value());
}

TEST_CASE("imports are resolved relative to the importing file and deduplicated", "[parser][imports]") {
	TempDir dir;
	dir.write("lib/base.dlm", "float[...] twice(x: float[...]) => x * 2;");
	dir.write("lib/left.dlm", "import base;\nfloat[...] left(x: float[...]) => twice(x) + 1;");
	dir.write("lib/right.dlm", "import base;\nfloat[...] right(x: float[...]) => twice(x) - 1;");
	auto main = dir.write("main.dlm", "import lib.left;\nimport lib.right;\nfloat[...] f(x: float[...]) => left(x) * right(x);");
	auto unit = parser::parse(main);
	REQUIRE(unit.has_value()); // `base` is imported twice but translated once (no redefinition errors)
	CHECK(unit->definitions.size() == 1);
}

TEST_CASE("import directories are searched", "[parser][imports]") {
	TempDir libs, src;
	libs.write("mylib.dlm", "float[...] helper(x: float[...]) => x + 1;");
	auto main = src.write("main.dlm", "import mylib;\nfloat[...] f(x: float[...]) => helper(x);");
	CHECK_FALSE(parser::parse(main).has_value());
	CHECK(parser::parse(main, {.importDirs = {libs.path}}).has_value());
}

TEST_CASE("importing a missing file is an error", "[parser][imports]") {
	std::istringstream source{"import does.not.exist;"};
	CHECK_FALSE(parser::parse(source, "<test>").has_value());
}

TEST_CASE("redefining an imported function is an error", "[parser][imports]") {
	std::istringstream source{"import std.arithmetic;\nfloat[...] exp(x: float[...]) => x;"};
	CHECK_FALSE(parser::parse(source, "<test>").has_value());
}
