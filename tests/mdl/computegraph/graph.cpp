#include <mdl/computegraphv2/graph.hpp>
#include <mdl/parser.hpp>

#include <catch2/catch_test_macros.hpp>

#include <sstream>

using namespace mdl;
using builtins::FunctionIDs;

TEST_CASE("constants are interned", "[graph][cse]") {
	Computegraph graph;
	CHECK(&graph.addConstNode(Atomic{1.0}) == &graph.addConstNode(Atomic{1.0}));
	CHECK(&graph.addConstNode(Atomic{1.0}) != &graph.addConstNode(Atomic{1}));	   // double vs. int
	CHECK(&graph.addConstNode(Atomic{0.0}) != &graph.addConstNode(Atomic{-0.0})); // bitwise comparison
	CHECK(&graph.addConstNode(Atomic{std::vector<int>{1, 0}}) == &graph.addConstNode(Atomic{std::vector<int>{1, 0}}));
	CHECK(&graph.addConstNode(Atomic{std::vector<int>{1, 0}}) != &graph.addConstNode(Atomic{std::vector<int>{0, 1}}));
}

TEST_CASE("builtin invocations are interned", "[graph][cse]") {
	Computegraph graph;
	DataType type = TensorType{PrimitiveType::float32, {4u}};
	const Node& x = graph.addInputNode(type, "x");
	const Node& y = graph.addInputNode(type, "y");
	const Node& a = graph.addInvocationNode(FunctionIDs::TENSOR_PROD, {x, y});
	const size_t nodes = graph.getNumNodes();
	CHECK(&graph.addInvocationNode(FunctionIDs::TENSOR_PROD, {x, y}) == &a);
	CHECK(graph.getNumNodes() == nodes);
	// Different argument order or builtin: a different node (the type follows from the arguments).
	CHECK(&graph.addInvocationNode(FunctionIDs::TENSOR_PROD, {y, x}) != &a);
	CHECK(&graph.addInvocationNode(FunctionIDs::TENSOR_ADD, {x, y}) != &a);
	// Inputs are never merged.
	CHECK(&graph.addInputNode(type, "x") != &x);
}

TEST_CASE("repeated calls of a function with the same arguments share their nodes", "[graph][cse]") {
	std::istringstream source{R"(
		module M[] {
			float[..., 4] f(x: float[..., 4]) => __builtin_exp(x) * x;
			float[..., 4] forward(x: float[..., 4]) => f(x) + f(x);
		}
	)"};
	auto unit = parser::parse(source, "<test>");
	REQUIRE(unit.has_value());
	const auto& m = std::get<ModuleType>(unit->definitions.at("M"));
	const Node& out = m.functions.at("forward").outputs.at(0).get();
	const auto& add = std::get<InvocationNode>(out);
	CHECK(&add.arguments[0].get() == &add.arguments[1].get());
}
