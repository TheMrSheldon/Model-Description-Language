// Kernel packages: manifests (package.yml), kernel directories, and platform compatibility (see docs/kernels.md).
#include <mdl/kernels.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <fstream>
#include <string>

using namespace mdl::kernels;

namespace {
	const std::filesystem::path kTestPackages = std::filesystem::path{MDL_SOURCE_DIR} / "tests" / "kernels";

	/** @brief A temporary directory, removed at the end of the scope. */
	struct TempDir {
		std::filesystem::path path;
		explicit TempDir(const std::string& name) : path(std::filesystem::temp_directory_path() / name) {
			std::filesystem::remove_all(path);
			std::filesystem::create_directories(path);
		}
		~TempDir() { std::filesystem::remove_all(path); }
	};

	void write(const std::filesystem::path& file, const std::string& text) {
		std::filesystem::create_directories(file.parent_path());
		std::ofstream{file} << text;
	}

	std::string manifest(const std::string& name, const std::string& version) {
		return "schema: 1\npackage: " + name + "\nversion: \"" + version + "\"\n";
	}
} // namespace

TEST_CASE("a package manifest is read with its sections", "[kernels]") {
	std::vector<Problem> problems;
	auto p = loadPackage(kTestPackages / "valid" / "scaled_add" / "package.yml", problems);
	REQUIRE(problems.empty());
	REQUIRE(p.has_value());
	CHECK(p->name == "scaled_add");
	CHECK(p->version == "1.0");

	REQUIRE(p->artifacts.size() == 1);
	const auto& a = p->artifacts[0];
	CHECK(a.kind == Artifact::Kind::staticLibrary);
	CHECK(a.path == kTestPackages / "valid" / "scaled_add" / "lib" / "libscaled_add.a"); // relative to the package
	CHECK(a.platform.os == "linux");
	CHECK(a.platform.arch == "x86_64");
	REQUIRE(a.dependencies.size() == 1);
	CHECK(a.dependencies[0].kind == Dependency::Kind::systemLibrary);
	CHECK(a.dependencies[0].name == "m");

	const auto* pattern = p->pattern("scaled-add");
	REQUIRE(pattern != nullptr);
	CHECK(pattern->root.op == "relax.add");
	REQUIRE(pattern->root.args.size() == 2);
	const auto& mul = pattern->root.args[0];
	CHECK(mul.op == "relax.multiply");
	CHECK(mul.args[0].kind == PatternNode::Kind::capture);
	CHECK(mul.args[0].capture == "x");
	CHECK(mul.args[1].kind == PatternNode::Kind::constant);
	CHECK(mul.args[1].value == 2.0);
	CHECK_FALSE(mul.args[1].tolerance.has_value()); // exact by default

	REQUIRE(p->implementations.size() == 1);
	const auto& impl = p->implementations[0];
	CHECK(impl.symbol == "test_scaled_add_f32");
	CHECK(impl.priority == 10);
	REQUIRE(impl.arguments.size() == 4);
	CHECK(impl.arguments[0].type == "pointer");
	CHECK(impl.arguments[0].access == "write");
	CHECK(impl.arguments[0].align == 64);
	CHECK(impl.arguments[3].value == "prod($x.shape)");
	CHECK(impl.dtypes.at("x") == std::vector<std::string>{"float32"});
	CHECK(impl.guarantees.at("numerics.exact") == "true");
}

TEST_CASE("problems of a manifest are reported with their positions", "[kernels]") {
	std::vector<Problem> problems;
	CHECK_FALSE(loadPackage(kTestPackages / "broken" / "bad" / "package.yml", problems).has_value());
	std::vector<std::pair<size_t, std::string>> found;
	for (const auto& p : problems)
		found.emplace_back(p.line, p.message);
	CHECK(found == std::vector<std::pair<size_t, std::string>>{
						   {6, "kind 'static-libary' is not one of: static-library, shared-library, cubin, ptx"},
						   {12, "output 'z' of pattern 'p' is not a capture"},
						   {15, "implementation 'i' refers to an unknown pattern 'missing'"},
						   {16, "implementation 'i' refers to an unknown artifact 'nope'"},
						   {20, "a pointer argument needs 'access' (read, write, readwrite)"},
						   {21, "unknown key 'colour' in an implementation (expected: pattern, artifacts, priority, "
								"symbol, abi, adapter, signature, applicability, guarantees)"},
				   });

	TempDir dir{"mdl-kernels-schema"};
	write(dir.path / "p" / "package.yml", "schema: 2\npackage: p\nversion: \"1\"\n");
	problems.clear();
	CHECK_FALSE(loadPackage(dir.path / "p" / "package.yml", problems).has_value());
	REQUIRE(problems.size() == 1);
	CHECK(problems[0].message == "unsupported schema version 2 (this compiler reads schema 1)");
}

TEST_CASE("more specific kernel directories hide packages of the same name", "[kernels]") {
	TempDir project{"mdl-kernels-project"}, global{"mdl-kernels-global"};
	write(global.path / "fast" / "package.yml", manifest("fast", "1.0"));
	write(global.path / "other" / "package.yml", manifest("other", "2.0"));
	write(global.path / "wrong" / "package.yml", manifest("notwrong", "1.0"));
	// The project's .mdl/kernels is found upward from the source file.
	write(project.path / ".mdl" / "kernels" / "fast" / "package.yml", manifest("fast", "1.1"));
	write(project.path / "models" / "deep" / "model.dlm", "");

	auto dirs = kernelDirs({global.path}, project.path / "models" / "deep" / "model.dlm");
	REQUIRE(dirs.size() >= 2);
	CHECK(dirs[0].origin == DirOrigin::explicitDir);
	CHECK(dirs[1].origin == DirOrigin::project);
	CHECK(dirs[1].path == std::filesystem::weakly_canonical(project.path / ".mdl" / "kernels"));

	// Precedence: the project's package, then the explicit directory's.
	auto found = discover({dirs[1], dirs[0]});
	REQUIRE(found.packages.size() == 2);
	CHECK(found.packages[0].name == "fast");
	CHECK(found.packages[0].version == "1.1");
	CHECK(found.packages[1].name == "other");
	REQUIRE(found.hidden.size() == 1);
	CHECK(found.hidden[0].first.version == "1.0");
	// A package must be in a directory of its name.
	REQUIRE(found.problems.size() == 1);
	CHECK(found.problems[0].message == "the package 'notwrong' must be in a directory of the same name (not 'wrong')");
}

TEST_CASE("artifacts run on compatible platforms", "[kernels]") {
	const Platform host{.os = "linux", .arch = "x86_64", .abi = "gnu", .libc = "2.39"};
	CHECK(compatible({.os = "linux", .arch = "x86_64", .abi = "gnu", .libc = ">=2.28"}, host));
	CHECK(compatible({.os = "linux", .arch = "x86_64", .abi = "", .libc = ""}, host));
	CHECK_FALSE(compatible({.os = "linux", .arch = "x86_64", .abi = "gnu", .libc = ">=2.40"}, host));
	CHECK_FALSE(compatible({.os = "linux", .arch = "aarch64", .abi = "gnu", .libc = ""}, host));
	CHECK_FALSE(compatible(
			{.os = "linux", .arch = "x86_64", .abi = "gnu", .libc = ">=2.28"},
			{.os = "linux", .arch = "x86_64", .abi = "gnu", .libc = ""}
	));																						   // no glibc (e.g. musl)
	CHECK(compatible({.os = "linux", .arch = "x86_64", .abi = "gnu", .libc = ">=2.9"}, host)); // numerically
}
