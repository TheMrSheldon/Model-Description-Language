#include <mdl/computegraphv2/function.hpp>
#include <mdl/parser.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

using namespace mdl;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::optional<CompilationUnit> parseSource(const std::string& source) {
	std::istringstream stream{source};
	return parser::parse(stream, "<test>");
}

static const FunctionDecl& moduleFunction(const CompilationUnit& unit, const std::string& module, const std::string& fn) {
	return std::get<ModuleType>(unit.definitions.at(module)).functions.at(fn);
}

/** Visits every node reachable from the outputs of @p func exactly once. */
static void forEachNode(const FunctionDecl& func, const std::function<void(const Node&)>& visit) {
	std::unordered_set<const Node*> seen;
	std::function<void(const Node&)> walk = [&](const Node& node) {
		if (!seen.insert(&node).second)
			return;
		visit(node);
		if (const auto* inv = std::get_if<InvocationNode>(&node))
			for (const auto& arg : inv->arguments)
				walk(arg.get());
	};
	for (const auto& output : func.outputs)
		walk(output.get());
}

static std::vector<std::string> weightNames(const FunctionDecl& func) {
	std::vector<std::string> names;
	forEachNode(func, [&](const Node& node) {
		if (const auto* pn = std::get_if<ParameterNode>(&node))
			names.push_back(pn->name);
	});
	std::sort(names.begin(), names.end());
	return names;
}

static size_t countOps(const FunctionDecl& func, builtins::FunctionIDs id) {
	size_t count = 0;
	forEachNode(func, [&](const Node& node) {
		if (const auto* inv = std::get_if<InvocationNode>(&node); inv && inv->id == id)
			++count;
	});
	return count;
}

// ---------------------------------------------------------------------------
// Grammar
// ---------------------------------------------------------------------------

TEST_CASE("subtraction of a literal without spaces parses", "[parser][grammar]") {
	auto unit = parseSource("float[...] f(x: float[...]) => x-1;");
	REQUIRE(unit.has_value());
	CHECK(countOps(std::get<FunctionDecl>(unit->definitions.at("f")), builtins::FunctionIDs::TENSOR_SUB) == 1);
}

TEST_CASE("identifiers may contain digits", "[parser][grammar]") {
	auto unit = parseSource("float[...] layer1(x2: float[...]) => x2 * 2;");
	REQUIRE(unit.has_value());
	CHECK(unit->definitions.contains("layer1"));
}

TEST_CASE("float literals with signed exponents parse", "[parser][grammar]") {
	auto unit = parseSource("float[...] f(x: float[...]) => x + 1e-12 + 2.5e3;");
	REQUIRE(unit.has_value());
}

TEST_CASE("negated literals are folded into constants", "[parser][grammar]") {
	auto unit = parseSource("float[...] f(x: float[...]) => x * -2;");
	REQUIRE(unit.has_value());
	const auto& f = std::get<FunctionDecl>(unit->definitions.at("f"));
	CHECK(countOps(f, builtins::FunctionIDs::TENSOR_SUB) == 0);
	bool foundMinusTwo = false;
	forEachNode(f, [&](const Node& node) {
		if (const auto* cn = std::get_if<ConstNode>(&node))
			foundMinusTwo |= std::holds_alternative<int>(cn->value) && std::get<int>(cn->value) == -2;
	});
	CHECK(foundMinusTwo);
}

TEST_CASE("unary minus on a tensor emits a subtraction", "[parser][grammar]") {
	auto unit = parseSource("float[...] f(x: float[...]) => -x;");
	REQUIRE(unit.has_value());
	CHECK(countOps(std::get<FunctionDecl>(unit->definitions.at("f")), builtins::FunctionIDs::TENSOR_SUB) == 1);
}

TEST_CASE("a function without return statement is an error", "[parser][errors]") {
	CHECK_FALSE(parseSource("float[...] f(x: float[...]) { auto y = x; }").has_value());
}

TEST_CASE("a function with two return statements is an error", "[parser][errors]") {
	CHECK_FALSE(parseSource("float[...] f(x: float[...]) { return x; return x; }").has_value());
}

// ---------------------------------------------------------------------------
// Inlining
// ---------------------------------------------------------------------------

TEST_CASE("inlining keeps shared subexpressions shared", "[parser][inline]") {
	// `e` is used twice in g; inlining g into f must not duplicate the exponential.
	auto unit = parseSource(R"(
		float[...] g(x: float[...]) {
			auto e = __builtin_exp(x);
			return e * e + e;
		}
		float[...] f(x: float[...]) => g(g(x));
	)");
	REQUIRE(unit.has_value());
	const auto& f = std::get<FunctionDecl>(unit->definitions.at("f"));
	CHECK(countOps(f, builtins::FunctionIDs::EXP) == 2);
	CHECK(countOps(f, builtins::FunctionIDs::TENSOR_PROD) == 2);
}

TEST_CASE("function arguments are input nodes", "[parser][inline]") {
	auto unit = parseSource("float[...] f(x: float[...], y: float[...]) => x + y;");
	REQUIRE(unit.has_value());
	const auto& f = std::get<FunctionDecl>(unit->definitions.at("f"));
	REQUIRE(f.parameters.size() == 2);
	CHECK(f.parameters[0].second.get().name == "x");
	size_t inputs = 0;
	forEachNode(f, [&](const Node& node) { inputs += std::holds_alternative<InputNode>(node); });
	CHECK(inputs == 2);
}

// ---------------------------------------------------------------------------
// Module instancing and weights
// ---------------------------------------------------------------------------

static const char* nestedModules = R"(
	module Inner[] {
		W: float[2, 2]
		float[..., 2] forward(x: float[..., 2]) => x @ W;
	}
	module Mid[] {
		a: Inner[]
		b: Inner[]
		float[..., 2] forward(x: float[..., 2]) => b(a(x));
	}
	module Outer[] {
		m1: Mid[]
		m2: Mid[]
		float[..., 2] forward(x: float[..., 2]) => m2(m1(x));
	}
)";

TEST_CASE("nested module instances get distinct weights with full paths", "[parser][weights]") {
	auto unit = parseSource(nestedModules);
	REQUIRE(unit.has_value());
	CHECK(weightNames(moduleFunction(*unit, "Outer", "forward")) ==
		  std::vector<std::string>{"m1.a.W", "m1.b.W", "m2.a.W", "m2.b.W"});
	CHECK(weightNames(moduleFunction(*unit, "Mid", "forward")) == std::vector<std::string>{"a.W", "b.W"});
	CHECK(countOps(moduleFunction(*unit, "Outer", "forward"), builtins::FunctionIDs::TENSOR_DOT) == 4);
}

TEST_CASE("weightPaths lists weights in declaration order", "[parser][weights]") {
	auto unit = parseSource(nestedModules);
	REQUIRE(unit.has_value());
	CHECK(weightPaths(std::get<ModuleType>(unit->definitions.at("Outer"))) ==
		  std::vector<std::string>{"m1.a.W", "m1.b.W", "m2.a.W", "m2.b.W"});
}

TEST_CASE("repeated calls of the same instance share its weights", "[parser][weights]") {
	auto unit = parseSource(R"(
		module Inner[] {
			W: float[2, 2]
			float[..., 2] forward(x: float[..., 2]) => x @ W;
		}
		module Shared[] {
			lin: Inner[]
			float[..., 2] forward(x: float[..., 2]) {
				auto h = lin(x);
				return lin(h);
			}
		}
	)");
	REQUIRE(unit.has_value());
	const auto& f = moduleFunction(*unit, "Shared", "forward");
	CHECK(weightNames(f) == std::vector<std::string>{"lin.W"});
	CHECK(countOps(f, builtins::FunctionIDs::TENSOR_DOT) == 2);
}

TEST_CASE("module hyperparameters are substituted in weight types", "[parser][weights]") {
	auto unit = parseSource(R"(
		module Lin[n: uint] {
			W: float[n, n]
			float[..., n] forward(x: float[..., n]) => x @ W;
		}
		module Two[] {
			small: Lin[2]
			large: Lin[3]
			float[..., 3] forward(x: float[..., 2]) => large(small(x));
		}
	)");
	REQUIRE(unit.has_value());
	std::map<std::string, std::vector<int>> dims;
	forEachNode(moduleFunction(*unit, "Two", "forward"), [&](const Node& node) {
		const auto* pn = std::get_if<ParameterNode>(&node);
		if (pn == nullptr)
			return;
		for (const auto& dim : std::get<TensorType>(pn->type).shape) {
			const Node* dimNode = std::get<const Node*>(dim);
			const auto* cn = std::get_if<ConstNode>(dimNode);
			REQUIRE(cn != nullptr);
			dims[pn->name].push_back(std::get<int>(cn->value));
		}
	});
	CHECK(dims["small.W"] == std::vector<int>{2, 2});
	CHECK(dims["large.W"] == std::vector<int>{3, 3});
}

// ---------------------------------------------------------------------------
// Multiple outputs
// ---------------------------------------------------------------------------

TEST_CASE("functions can return multiple values", "[parser][outputs]") {
	auto unit = parseSource(R"(
		(float[...], float[...]) split(x: float[...]) {
			return x + 1.0, x * 2.0;
		}
		float[...] f(x: float[...]) {
			auto a, b = split(x);
			return a - b;
		}
	)");
	REQUIRE(unit.has_value());
	CHECK(std::get<FunctionDecl>(unit->definitions.at("split")).outputs.size() == 2);
	const auto& f = std::get<FunctionDecl>(unit->definitions.at("f"));
	REQUIRE(f.outputs.size() == 1);
	CHECK(countOps(f, builtins::FunctionIDs::TENSOR_ADD) == 1);
	CHECK(countOps(f, builtins::FunctionIDs::TENSOR_PROD) == 1);
	CHECK(countOps(f, builtins::FunctionIDs::TENSOR_SUB) == 1);
}

TEST_CASE("using a multi-value call as a single value is an error", "[parser][outputs]") {
	CHECK_FALSE(parseSource(R"(
		(float[...], float[...]) split(x: float[...]) { return x, x; }
		float[...] f(x: float[...]) => split(x) + 1;
	)")
						.has_value());
}

TEST_CASE("destructuring with the wrong number of names is an error", "[parser][outputs]") {
	CHECK_FALSE(parseSource(R"(
		(float[...], float[...]) split(x: float[...]) { return x, x; }
		float[...] f(x: float[...]) {
			auto a, b, c = split(x);
			return a;
		}
	)")
						.has_value());
}
