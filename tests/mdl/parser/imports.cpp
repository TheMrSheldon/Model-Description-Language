#include <mdl/parser.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

	/** The messages of the diagnostics of parsing @p file (errors and warnings). */
	std::vector<std::string> messages(const fs::path& file, parser::ParseOptions options = {}) {
		std::vector<std::string> result;
		options.diagnostics = [&](const Diagnostic& d) { result.push_back(d.message); };
		parser::parse(file, options);
		return result;
	}

	bool contains(const std::vector<std::string>& messages, const std::string& part) {
		for (const auto& m : messages)
			if (m.find(part) != std::string::npos)
				return true;
		return false;
	}

	std::optional<CompilationUnit> parseText(const std::string& text) {
		std::istringstream source{text};
		return parser::parse(source, "<test>");
	}
} // namespace

TEST_CASE("names of the standard library are imported explicitly; the prelude implicitly", "[parser][imports]") {
	auto unit = parseText(R"(
		from std.nn import Linear, LayerNorm;
		module Block[dim: uint] {
			lin: Linear[dim, dim]
			norm: LayerNorm[dim]
			public float[..., dim] forward(x: float[..., dim]) => norm(gelu(lin(x)) + softmax(x));
		}
	)");
	REQUIRE(unit.has_value());
	// Only the definitions of the parsed file are part of the compilation unit.
	CHECK(unit->definitions.size() == 1);
	CHECK(unit->definitions.contains("Block"));
	auto block = unit->instantiate("Block", {{"dim", Object{&unit->graph().addConstNode(Atomic{8})}}});
	REQUIRE(block.has_value());
	CHECK(weightPaths(block->get()) == std::vector<std::string>{"lin.W", "lin.b", "norm.beta", "norm.gamma"});
}

TEST_CASE("an import binds a namespace (its last component, or an alias)", "[parser][imports]") {
	CHECK(parseText(R"(
		import std.nn;
		import std.losses as l;
		import std;
		module M[] {
			a: nn.Linear[4, 4]
			b: std.nn.Linear[4, 4]
			float f(x: float[..., 4], y: float[..., 4]) => l.mse(a(x) + b(x), std.activations.softmax(y));
		}
	)")
				  .has_value());
	// Nothing else of the file becomes visible.
	CHECK_FALSE(parseText("import std.nn;\nmodule M[] {\n a: Linear[4, 4]\n}").has_value());
	// A namespace is not a value.
	CHECK_FALSE(parseText("import std.arithmetic;\nfloat[...] f(x: float[...]) => arithmetic + x;").has_value());
}

TEST_CASE("local definitions and explicit imports take precedence over the prelude", "[parser][imports]") {
	TempDir dir;
	// A local `exp` hides the prelude's silently.
	auto local = dir.write("local.dlm", "float[...] exp(x: float[...]) => x;\nfloat[...] f(x: float[...]) => exp(x);");
	CHECK(messages(local).empty());
	// A local definition that hides an explicit import is reported (-Wshadow).
	auto hidden = dir.write(
			"hidden.dlm", "from std.losses import mse;\nfloat mse(a: float[...], b: float[...]) => mean_flat(a);"
	);
	CHECK(contains(messages(hidden), "mse shadows the imported mse"));
}

TEST_CASE("relative imports, translated once", "[parser][imports]") {
	TempDir dir;
	dir.write("lib/base.dlm", "float[...] twice(x: float[...]) => x * 2;");
	dir.write("lib/left.dlm", "from .base import twice;\nfloat[...] left(x: float[...]) => twice(x) + 1;");
	dir.write("lib/right.dlm", "from .base import twice;\nfloat[...] right(x: float[...]) => twice(x) - 1;");
	dir.write("lib/sub/deep.dlm", "from ..base import twice;\nfloat[...] deep(x: float[...]) => twice(x);");
	auto main = dir.write(
			"main.dlm", "from lib.left import left;\nfrom lib.right import right;\nfrom lib.sub.deep import deep;\n"
						"float[...] f(x: float[...]) => left(x) * right(x) + deep(x);"
	);
	auto unit = parser::parse(main);
	REQUIRE(unit.has_value()); // `base` is imported three times but translated once (no redefinition errors)
	CHECK(unit->definitions.size() == 1);
}

TEST_CASE("import directories are searched; another package sees public names only", "[parser][imports]") {
	TempDir libs, src;
	libs.write(
			"mylib.dlm", "public float[...] helper(x: float[...]) => x + 1;\nfloat[...] internal(x: float[...]) => x;"
	);
	auto main = src.write("main.dlm", "from mylib import helper;\nfloat[...] f(x: float[...]) => helper(x);");
	CHECK_FALSE(parser::parse(main).has_value());
	CHECK(parser::parse(main, {.importDirs = {libs.path}}).has_value());
	auto other = src.write("other.dlm", "from mylib import internal;\nfloat[...] f(x: float[...]) => internal(x);");
	CHECK(contains(messages(other, {.importDirs = {libs.path}}), "internal is package-visible in mylib.dlm"));
}

TEST_CASE("package visibility with re-exports: the implementation can move between files", "[parser][imports]") {
	TempDir libs, src;
	libs.write("mylib/_details.dlm", "float[...] scale(x: float[...]) => x * 2;");
	libs.write(
			"mylib/_fileutils.dlm",
			"from ._details import scale;\nfloat[...] normalize(x: float[...]) => scale(x) / 3;\n"
			"private float[...] secret(x: float[...]) => x;"
	);
	libs.write("mylib/utils.dlm", "public from ._fileutils import normalize;");
	const parser::ParseOptions options{.importDirs = {libs.path}};
	auto good =
			src.write("good.dlm", "from mylib.utils import normalize;\nfloat[...] f(x: float[...]) => normalize(x);");
	CHECK(parser::parse(good, options).has_value());
	auto direct = src.write(
			"direct.dlm", "from mylib.utils import normalize as n;\nfrom mylib._fileutils import normalize;\n"
						  "float[...] f(x: float[...]) => normalize(x) + n(x);"
	);
	auto m = messages(direct, options);
	CHECK(contains(m, "normalize is package-visible in _fileutils.dlm (it is re-exported by utils.dlm)"));
	auto secret = src.write("secret.dlm", "from mylib._fileutils import secret;");
	CHECK(contains(messages(secret, options), "secret is private in _fileutils.dlm"));
	// Imports are private: the re-exporting file's own import of `scale` is not visible to importers.
	auto transitive = src.write("transitive.dlm", "from mylib.utils import scale;");
	CHECK(contains(messages(transitive, options), "Cannot import scale"));
}

TEST_CASE("directory imports: the index file, or the files as namespaces", "[parser][imports]") {
	TempDir libs, src;
	libs.write("indexed/indexed.dlm", "public from .impl import run;");
	libs.write("indexed/impl.dlm", "public float[...] run(x: float[...]) => x;");
	libs.write("plain/tools.dlm", "public float[...] tool(x: float[...]) => x;");
	const parser::ParseOptions options{.importDirs = {libs.path}};
	auto main = src.write(
			"main.dlm", "import indexed;\nimport plain;\nfrom plain.tools import *;\n"
						"float[...] f(x: float[...]) => indexed.run(x) + plain.tools.tool(x) + tool(x);"
	);
	CHECK(parser::parse(main, options).has_value());
}

TEST_CASE("importing a missing file or name is an error", "[parser][imports]") {
	CHECK_FALSE(parseText("import does.not.exist;").has_value());
	CHECK_FALSE(parseText("from std.nn import Nope;").has_value());
}

TEST_CASE("members: private only inside the module, package-visible only in its package", "[parser][imports]") {
	TempDir libs, src;
	libs.write("mylib.dlm", R"(
		public module Inner[] {
			W: float[4, 4]
			public V: float[4, 4]
			public float[..., 4] forward(x: float[..., 4]) => x @ W;
		}
	)");
	auto main = src.write("main.dlm", R"(
		from mylib import Inner;
		module Local[] {
			private s: float[4]
			public float[..., 4] forward(x: float[..., 4]) => x * s;
		}
		module M[] {
			inner: Inner[]
			local: Local[]
			public float[..., 4] f(x: float[..., 4]) => inner(x) @ inner.V + x @ inner.W + local(x) * local.s;
		}
	)");
	auto m = messages(main, {.importDirs = {libs.path}});
	CHECK(contains(m, "W is a package-visible member of Inner (in another package: mylib.dlm)"));
	CHECK(contains(m, "s is a private member of Local"));
	CHECK_FALSE(contains(m, "V is"));
}
