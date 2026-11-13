#include <mdl/computegraphv2/builtins.hpp>
#include <mdl/computegraphv2/function.hpp>

#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <variant>

using namespace mdl;
namespace ph = std::placeholders;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Builds a three-parameter FunctionDecl (x:int, y:float, z:int) whose output
 * is TENSOR_ADD(x, y) — a builtin InvocationNode that uses paramX and paramY.
 * paramZ is a "dead" parameter kept to test binding scenarios that involve
 * parameters unused by the output.
 * All nodes live in @p graph.
 *
 * addInvocationNode(FunctionDecl, args) inlines by substituting ParameterNodes
 * in the output tree.  A builtin InvocationNode output therefore produces a
 * new InvocationNode after inlining, while a ConstNode would return itself
 * unchanged (no new node created).
 */
static FunctionDecl makeDecl(Computegraph& graph) {
	const Node& paramX = graph.addInputNode(PrimitiveType::integer);
	const Node& paramY = graph.addInputNode(PrimitiveType::float32);
	const Node& paramZ = graph.addInputNode(PrimitiveType::integer);
	// output: TENSOR_ADD(x, y)  [z is intentionally unused in the output]
	const Node& output = graph.addInvocationNode(
			builtins::FunctionIDs::TENSOR_ADD, {std::cref(paramX), std::cref(paramY)}
	);

	return FunctionDecl{
			.name = "f",
			.templateParameters = {},
			.parameters =
					{
							{"x", std::cref(std::get<InputNode>(paramX))},
							{"y", std::cref(std::get<InputNode>(paramY))},
							{"z", std::cref(std::get<InputNode>(paramZ))},
					},
			.outputs = {output},
	};
}

// ---------------------------------------------------------------------------
// Structural properties (name, templateParameters, output kind)
// ---------------------------------------------------------------------------

TEST_CASE("bind preserves function name", "[bind]") {
	Computegraph graph;
	FunctionDecl decl = makeDecl(graph);

	const Node& bound = graph.addConstNode(Atomic{1});
	FunctionDecl result = mdl::bind(graph, decl, bound, std::placeholders::_1, std::placeholders::_2);

	REQUIRE(result.name == decl.name);
}

TEST_CASE("bind preserves template parameters", "[bind]") {
	Computegraph graph;
	FunctionDecl decl = makeDecl(graph);
	const Node& hp = graph.addHyperparameterNode(PrimitiveType::integer);
	decl.templateParameters.push_back({"N", std::cref(std::get<HyperparameterNode>(hp))});

	const Node& bound = graph.addConstNode(Atomic{0});
	FunctionDecl result = mdl::bind(graph, decl, bound, std::placeholders::_1, std::placeholders::_2);

	REQUIRE(result.templateParameters.size() == decl.templateParameters.size());
	REQUIRE(&result.templateParameters[0].second.get() == &decl.templateParameters[0].second.get());
}

TEST_CASE("bind output is an InvocationNode", "[bind]") {
	Computegraph graph;
	FunctionDecl decl = makeDecl(graph); // output = TENSOR_ADD(x, y)

	const Node& bound = graph.addConstNode(Atomic{7});
	FunctionDecl result = mdl::bind(graph, decl, bound, std::placeholders::_1, std::placeholders::_2);

	// Inlining TENSOR_ADD(x, y) with x=bound, y=newParam produces a new InvocationNode.
	REQUIRE(std::holds_alternative<InvocationNode>(result.outputs[0].get()));
}

TEST_CASE("bind invocation reflects the arity of the output op (TENSOR_ADD = 2 args)", "[bind]") {
	Computegraph graph;
	FunctionDecl decl = makeDecl(graph); // output = TENSOR_ADD(x, y)

	const Node& bound = graph.addConstNode(Atomic{0});
	FunctionDecl result = mdl::bind(graph, decl, std::placeholders::_1, bound, std::placeholders::_2);

	const auto& inv = std::get<InvocationNode>(result.outputs[0].get());
	// The inlined InvocationNode mirrors the original builtin call (2 operands), not the
	// number of parameters of the FunctionDecl being bound (3).
	REQUIRE(inv.arguments.size() == 2);
}

// ---------------------------------------------------------------------------
// Parameter count in the resulting FunctionDecl
// ---------------------------------------------------------------------------

TEST_CASE("bind with all args bound produces zero parameters", "[bind]") {
	Computegraph graph;
	FunctionDecl decl = makeDecl(graph);

	const Node& a = graph.addConstNode(Atomic{1});
	const Node& b = graph.addConstNode(Atomic{2.0});
	const Node& c = graph.addConstNode(Atomic{3});

	FunctionDecl result = mdl::bind(graph, decl, a, b, c);

	REQUIRE(result.parameters.empty());
}

TEST_CASE("bind with all placeholders produces same parameter count", "[bind]") {
	Computegraph graph;
	FunctionDecl decl = makeDecl(graph); // 3 params

	FunctionDecl result = mdl::bind(graph, decl, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

	REQUIRE(result.parameters.size() == 3);
}

TEST_CASE("bind partial: one bound, two placeholders", "[bind]") {
	Computegraph graph;
	FunctionDecl decl = makeDecl(graph); // x:int, y:float, z:int

	const Node& bound = graph.addConstNode(Atomic{99});
	// Bind x, keep y as _1, keep z as _2
	FunctionDecl result = mdl::bind(graph, decl, bound, std::placeholders::_1, std::placeholders::_2);

	REQUIRE(result.parameters.size() == 2);
}

// ---------------------------------------------------------------------------
// DataType propagation into new ParameterNodes
// ---------------------------------------------------------------------------

TEST_CASE("bind assigns correct DataType to new parameters from placeholder positions", "[bind]") {
	Computegraph graph;
	FunctionDecl decl = makeDecl(graph); // x:int, y:float, z:int

	const Node& bound = graph.addConstNode(Atomic{0});
	// Keep y (_1:float) and z (_2:int); bind x
	FunctionDecl result = mdl::bind(graph, decl, bound, std::placeholders::_1, std::placeholders::_2);

	REQUIRE(result.parameters.size() == 2);
	// DataType's variant members lack operator==; extract PrimitiveType directly.
	CHECK(std::get<PrimitiveType>(result.parameters[0].second.get().datatype()) == PrimitiveType::float32); // _1 ← y
	CHECK(std::get<PrimitiveType>(result.parameters[1].second.get().datatype()) == PrimitiveType::integer);	 // _2 ← z
}

TEST_CASE("bind preserves parameter names from placeholder positions", "[bind]") {
	Computegraph graph;
	FunctionDecl decl = makeDecl(graph); // x:int, y:float, z:int

	const Node& bound = graph.addConstNode(Atomic{0});
	FunctionDecl result = mdl::bind(graph, decl, bound, std::placeholders::_1, std::placeholders::_2);

	REQUIRE(result.parameters.size() == 2);
	CHECK(result.parameters[0].first == "y"); // _1 ← y
	CHECK(result.parameters[1].first == "z"); // _2 ← z
}

// ---------------------------------------------------------------------------
// Placeholder reordering
// ---------------------------------------------------------------------------

TEST_CASE("bind with swapped placeholders reorders parameter types", "[bind]") {
	Computegraph graph;
	FunctionDecl decl = makeDecl(graph); // x:int, y:float, z:int

	const Node& bound = graph.addConstNode(Atomic{5});
	// Bind x; put z first (_1) and y second (_2) — types swap
	FunctionDecl result = mdl::bind(graph, decl, bound, std::placeholders::_2, std::placeholders::_1);

	REQUIRE(result.parameters.size() == 2);
	CHECK(std::get<PrimitiveType>(result.parameters[0].second.get().datatype()) ==
		  PrimitiveType::integer); // _1 ← z:int
	CHECK(std::get<PrimitiveType>(result.parameters[1].second.get().datatype()) ==
		  PrimitiveType::float32); // _2 ← y:float
}

TEST_CASE("bind with swapped placeholders reorders parameter names", "[bind]") {
	Computegraph graph;
	FunctionDecl decl = makeDecl(graph); // x:int, y:float, z:int

	const Node& bound = graph.addConstNode(Atomic{5});
	FunctionDecl result = mdl::bind(graph, decl, bound, std::placeholders::_2, std::placeholders::_1);

	REQUIRE(result.parameters.size() == 2);
	CHECK(result.parameters[0].first == "z"); // _1 ← z
	CHECK(result.parameters[1].first == "y"); // _2 ← y
}

// ---------------------------------------------------------------------------
// Invocation argument wiring
// ---------------------------------------------------------------------------

TEST_CASE("bind wires bound node into invocation at correct position", "[bind]") {
	Computegraph graph;
	FunctionDecl decl = makeDecl(graph); // output = TENSOR_ADD(x, y)

	const Node& boundA = graph.addConstNode(Atomic{10}); // binds x (operand 0 of TENSOR_ADD)
	const Node& boundC = graph.addConstNode(Atomic{30}); // binds z (not in output — no effect on inv)

	FunctionDecl result = mdl::bind(graph, decl, boundA, std::placeholders::_1, boundC);

	// Inlining TENSOR_ADD(x, y) with x=boundA, y=newParamY, z=boundC:
	// → new InvocationNode(TENSOR_ADD, [boundA, newParamY])
	const auto& inv = std::get<InvocationNode>(result.outputs[0].get());
	REQUIRE(inv.arguments.size() == 2);		   // TENSOR_ADD arity
	CHECK(&inv.arguments[0].get() == &boundA); // x operand ← boundA
}

TEST_CASE("bind wires new ParameterNode into invocation at placeholder position", "[bind]") {
	Computegraph graph;
	FunctionDecl decl = makeDecl(graph); // output = TENSOR_ADD(x, y)

	const Node& bound = graph.addConstNode(Atomic{0}); // binds x (operand 0)
	FunctionDecl result = mdl::bind(graph, decl, bound, std::placeholders::_1, std::placeholders::_2);

	// Inlining TENSOR_ADD(x, y) with x=bound, y=newParamY, z=newParamZ:
	// → new InvocationNode(TENSOR_ADD, [bound, newParamY])
	const auto& inv = std::get<InvocationNode>(result.outputs[0].get());
	REQUIRE(result.parameters.size() == 2);

	// result.parameters[0] = ("y", newParamY)  [_1 → y, index 1 in original params]
	// The y operand (inv.arguments[1]) must point to the same ParameterNode.
	const InputNode& newParamY = result.parameters[0].second.get();
	CHECK(&std::get<InputNode>(inv.arguments[1].get()) == &newParamY);
}

// ---------------------------------------------------------------------------
// Graph growth
// ---------------------------------------------------------------------------

TEST_CASE("bind adds exactly (num_placeholders + 1) nodes to the graph", "[bind]") {
	Computegraph graph;
	FunctionDecl decl = makeDecl(graph); // adds 4 nodes: 3 ParameterNodes + 1 InvocationNode

	const Node& bound = graph.addConstNode(Atomic{0}); // +1 before bind
	size_t nodesBeforeBind = graph.getNumNodes();

	FunctionDecl result = mdl::bind(graph, decl, bound, std::placeholders::_1, std::placeholders::_2);
	// bind adds: 2 new ParameterNodes (for _1 and _2) + 1 new InvocationNode (inlined output)
	CHECK(graph.getNumNodes() == nodesBeforeBind + 3);
	(void)result;
}

TEST_CASE("bind all-bound adds exactly 1 node (the inlined InvocationNode)", "[bind]") {
	Computegraph graph;
	FunctionDecl decl = makeDecl(graph);

	const Node& a = graph.addConstNode(Atomic{1});
	const Node& b = graph.addConstNode(Atomic{2.0});
	const Node& c = graph.addConstNode(Atomic{3});
	size_t nodesBeforeBind = graph.getNumNodes();

	FunctionDecl result = mdl::bind(graph, decl, a, b, c);
	// bind adds: 0 new ParameterNodes + 1 new InvocationNode (inlined TENSOR_ADD(a, b))
	CHECK(graph.getNumNodes() == nodesBeforeBind + 1);
	(void)result;
}

// ---------------------------------------------------------------------------
// Edge case: zero-parameter function
// ---------------------------------------------------------------------------

TEST_CASE("bind on a zero-parameter function with no args returns a ConstNode output", "[bind]") {
	Computegraph graph;
	const Node& output = graph.addConstNode(Atomic{0});

	FunctionDecl decl{.name = "constant", .outputs = {output}};

	FunctionDecl result = mdl::bind(graph, decl);

	// Inlining a ConstNode output with an empty substitution map returns the same
	// ConstNode directly — no new node is added and the output is not an InvocationNode.
	CHECK(result.parameters.empty());
	CHECK(std::holds_alternative<ConstNode>(result.outputs[0].get()));
	CHECK(&result.outputs[0].get() == &output);
}
