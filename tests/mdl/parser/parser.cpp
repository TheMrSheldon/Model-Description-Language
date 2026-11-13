#include <mdl/computegraphv2/function.hpp>
#include <mdl/parser.hpp>
#include <mdl/passes/shapes.hpp>

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
		module Lin[m: uint, n: uint] {
			W: float[m, n]
			float[..., n] forward(x: float[..., m]) => x @ W;
		}
		module Two[] {
			small: Lin[2, 3]
			large: Lin[3, 3]
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
	CHECK(dims["small.W"] == std::vector<int>{2, 3});
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

// ---------------------------------------------------------------------------
// Module templates, loops, and module arrays
// ---------------------------------------------------------------------------

static const char* layerStack = R"(
	module Layer[dim: uint] {
		W: float[dim, dim]
		float[..., dim] forward(x: float[..., dim]) => x @ W;
	}
	module Stack[depth: uint, dim: uint] {
		layers: [Layer[dim] for i in 0..depth]
		float[..., dim] forward(x: float[..., dim]) {
			auto h = x;
			for (i in 0..depth) {
				h = layers[i](h);
			}
			return h;
		}
	}
	module Model[] {
		stack: Stack[3, 4]
		float[..., 4] forward(x: float[..., 4]) => stack(x);
	}
)";

TEST_CASE("loops over module arrays are unrolled with per-layer weights", "[parser][loops]") {
	auto unit = parseSource(layerStack);
	REQUIRE(unit.has_value());
	const auto& f = moduleFunction(*unit, "Model", "forward");
	CHECK(countOps(f, builtins::FunctionIDs::TENSOR_DOT) == 3);
	CHECK(weightNames(f) == std::vector<std::string>{"stack.layers.0.W", "stack.layers.1.W", "stack.layers.2.W"});
	CHECK(weightPaths(std::get<ModuleType>(unit->definitions.at("Model"))) ==
		  std::vector<std::string>{"stack.layers.0.W", "stack.layers.1.W", "stack.layers.2.W"});
}

TEST_CASE("modules are expanded per instantiation (hyperparameter loop bounds)", "[parser][loops]") {
	auto unit = parseSource(layerStack);
	REQUIRE(unit.has_value());
	auto value = [&](int v) { return Object{&unit->graph().addConstNode(Atomic{v})}; };
	auto deep = unit->instantiate("Stack", {{"depth", value(5)}, {"dim", value(8)}});
	REQUIRE(deep.has_value());
	CHECK(countOps(deep->get().functions.at("forward"), builtins::FunctionIDs::TENSOR_DOT) == 5);
	auto shallow = unit->instantiate("Stack", {{"depth", value(2)}, {"dim", value(8)}});
	REQUIRE(shallow.has_value());
	CHECK(countOps(shallow->get().functions.at("forward"), builtins::FunctionIDs::TENSOR_DOT) == 2);
}

TEST_CASE("structs can be passed as module hyperparameters", "[parser][templates]") {
	auto unit = parseSource(R"(
		struct Conf {
			dim: uint
			depth: uint
		}
		module Layer[dim: uint] {
			W: float[dim, dim]
			float[..., dim] forward(x: float[..., dim]) => x @ W;
		}
		module Net[conf: Conf] {
			layers: [Layer[conf.dim] for i in 0..conf.depth]
			float[..., conf.dim] forward(x: float[..., conf.dim]) {
				auto h = x;
				for (i in 0..conf.depth) { h = layers[i](h); }
				return h;
			}
		}
		module Outer[conf: Conf] {
			net: Net[conf]
			float[..., conf.dim] forward(x: float[..., conf.dim]) => net(x);
		}
	)");
	REQUIRE(unit.has_value());
	auto conf = AggregateInstance::create(
			std::get<AggregateType>(unit->definitions.at("Conf")),
			{{"dim", Object{&unit->graph().addConstNode(Atomic{4})}},
			 {"depth", Object{&unit->graph().addConstNode(Atomic{2})}}}
	);
	REQUIRE(conf.has_value());
	auto outer = unit->instantiate("Outer", {{"conf", Object{*conf}}});
	REQUIRE(outer.has_value());
	CHECK(weightNames(outer->get().functions.at("forward")) == std::vector<std::string>{"net.layers.0.W", "net.layers.1.W"});
}

TEST_CASE("assigning an undeclared variable is an error", "[parser][loops]") {
	CHECK_FALSE(parseSource("float[...] f(x: float[...]) { y = x; return x; }").has_value());
}

TEST_CASE("indexing a module array out of bounds is an error", "[parser][loops]") {
	CHECK_FALSE(parseSource(R"(
		module Layer[] {
			W: float[2, 2]
			float[..., 2] forward(x: float[..., 2]) => x @ W;
		}
		module Stack[] {
			layers: [Layer[] for i in 0..2]
			float[..., 2] forward(x: float[..., 2]) => layers[2](x);
		}
	)")
						.has_value());
}

TEST_CASE("loop bounds must be compile-time constants", "[parser][loops]") {
	CHECK_FALSE(parseSource(R"(
		float[...] f(x: float[...]) {
			auto h = x;
			for (i in 0..x) { h = h + x; }
			return h;
		}
	)")
						.has_value());
}

TEST_CASE("syntax errors are reported without translating the broken tree", "[parser]") {
	std::istringstream stream{"module M[] { (float[4], float[4]) f(x: float[4]) => x, x; }"};
	auto unit = mdl::parser::parse(stream, "<test>");
	CHECK_FALSE(unit.has_value());
}

// ---------------------------------------------------------------------------
// Types (shape rules shared with mdl::shapes)
// ---------------------------------------------------------------------------

TEST_CASE("node types follow the shape rules: broadcasting and matrix products", "[parser][types]") {
	auto unit = parseSource(R"(
		module M[] {
			b: float[4]
			W: float[4, 3]
			float[..., 3] f(x: float[..., 4]) => (b + x) @ W;
		}
	)");
	REQUIRE(unit.has_value());
	const auto& f = moduleFunction(*unit, "M", "f");
	const auto& out = std::get<InvocationNode>(f.outputs[0].get());
	// b + x is [..., 4] (not b's type [4]); the product is [..., 3].
	CHECK(shapes::str(getDataType(out.arguments[0].get())) == "float32[..., 4]");
	CHECK(shapes::str(getDataType(f.outputs[0].get())) == "float32[..., 3]");
}

TEST_CASE("results must have the declared return types", "[parser][types]") {
	CHECK(parseSource("module M[] { W: float[4, 3] float[..., 3] f(x: float[..., 4]) => x @ W; }").has_value());
	CHECK_FALSE(parseSource("module M[] { W: float[4, 3] float[..., 4] f(x: float[..., 4]) => x @ W; }").has_value());
	CHECK_FALSE(parseSource("module M[] { W: float[4, 3] float[4] f(x: float[4]) => x @ W; }").has_value());
}

TEST_CASE("arguments must match the parameters, operands the operators", "[parser][types]") {
	// A layer from 2 to 3 dimensions followed by one that expects 4.
	CHECK_FALSE(parseSource(R"(
		module Lin[m: uint, n: uint] {
			W: float[m, n]
			float[..., n] forward(x: float[..., m]) => x @ W;
		}
		module Net[] {
			first: Lin[2, 3]
			second: Lin[4, 4]
			float[..., 4] forward(x: float[..., 2]) => second(first(x));
		}
	)").has_value());
	// Inner dimensions of a matrix product, and shapes that do not broadcast.
	CHECK_FALSE(parseSource("module M[] { W: float[3, 3] float[..., 3] f(x: float[..., 4]) => x @ W; }").has_value());
	CHECK_FALSE(parseSource("module M[] { b: float[3] float[..., 4] f(x: float[..., 4]) => x + b; }").has_value());
}

TEST_CASE("float types do not mix implicitly; conversions are explicit", "[parser][types]") {
	// Mixed operands, an argument of another float type, a result of another float type.
	CHECK_FALSE(parseSource("module M[] { W: bfloat16[4] float32[..., 4] f(x: float32[..., 4]) => x * W; }").has_value());
	CHECK_FALSE(parseSource(R"(
		float32[...] g(x: float32[...]) => x;
		module M[] { bfloat16[..., 4] f(x: bfloat16[..., 4]) => g(x); }
	)").has_value());
	CHECK_FALSE(parseSource("module M[] { float32[..., 4] f(x: bfloat16[..., 4]) => x; }").has_value());
	// Explicit conversions; literals take the tensor's type.
	auto unit = parseSource(
			"module M[] { W: bfloat16[4] float32[..., 4] f(x: float32[..., 4]) => x * float32(W) + 0.5; }"
	);
	REQUIRE(unit.has_value());
	CHECK(shapes::str(getDataType(moduleFunction(*unit, "M", "f").outputs[0].get())) == "float32[..., 4]");
}

TEST_CASE("`float` stands for the configured default float type", "[parser][types]") {
	std::istringstream stream{"module M[] { W: float[4] float[..., 4] f(x: float[..., 4]) => x * bfloat16(W); }"};
	auto unit = parser::parse(stream, "<test>", {.defaultFloat = PrimitiveType::bfloat16});
	REQUIRE(unit.has_value());
	const auto& f = moduleFunction(*unit, "M", "f");
	CHECK(shapes::str(getDataType(f.outputs[0].get())) == "bfloat16[..., 4]");
	// bfloat16(W) of a bfloat16 W is no conversion.
	CHECK(std::get<InvocationNode>(f.outputs[0].get()).id == builtins::FunctionIDs::TENSOR_PROD);
	CHECK(std::holds_alternative<ParameterNode>(std::get<InvocationNode>(f.outputs[0].get()).arguments[1].get()));
}
