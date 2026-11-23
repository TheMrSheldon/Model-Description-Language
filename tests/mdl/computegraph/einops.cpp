#include <mdl/utils/einops.hpp>
#include <mdl/exports.hpp>
#include "mdl/argvals.hpp"
#include <mdl/computegraphv2/builtins.hpp>
#include <mdl/computegraphv2/datatype.hpp>
#include <mdl/computegraphv2/graph.hpp>
#include <mdl/computegraphv2/node.hpp>

#include <catch2/catch_test_macros.hpp>

#include <variant>

using namespace mdl;
using namespace mdl::einops;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a SpecItem with a single Named atom.
static SpecItem named(std::string n) {
	return SpecItem{.is_group = false, .atoms = {{AtomKind::Named, std::move(n), 0}}};
}

/// Build a SpecItem with a single Literal atom.
static SpecItem literal(int64_t v) {
	return SpecItem{.is_group = false, .atoms = {{AtomKind::Literal, {}, v}}};
}

/// Build a SpecItem that is the ellipsis token.
static SpecItem ellipsis() {
	return SpecItem{.is_group = false, .atoms = {{AtomKind::Ellipsis, {}, 0}}};
}

/// Build a grouped SpecItem from several named atoms.
static SpecItem group(std::initializer_list<std::string> names) {
	SpecItem item{.is_group = true, .atoms = {}};
	for (auto& n : names)
		item.atoms.push_back({AtomKind::Named, n, 0});
	return item;
}

/// The value of a statically known dimension (a literal or a constant expression in the graph).
static std::optional<int64_t> staticDim(const TensorType::Shape::value_type& dim) {
	if (const auto* u = std::get_if<unsigned>(&dim))
		return *u;
	if (const auto* n = std::get_if<const Node*>(&dim))
		return evalConstant(**n, ArgVals{});
	return std::nullopt;
}

/// Make a static TensorType with known integer dims.
static TensorType tensorType(PrimitiveType elem, std::initializer_list<unsigned> dims) {
	TensorType::Shape shape;
	for (unsigned d : dims)
		shape.push_back(d);
	return TensorType{.type = elem, .shape = std::move(shape)};
}

// ---------------------------------------------------------------------------
// Pure permute: a b c -> b a c
// ---------------------------------------------------------------------------

TEST_CASE("einops pure permute emits PERMUTE node", "[einops][permute]") {
	Computegraph graph;
	const Node& input = graph.addParameterNode(
			tensorType(PrimitiveType::float32, {4, 8, 16}));

	Spec lhs = {named("a"), named("b"), named("c")};
	Spec rhs = {named("b"), named("a"), named("c")};

	auto result = apply(graph, input, lhs, rhs, {});
	REQUIRE(result.has_value());

	const Node& out = result->get();
	REQUIRE(std::holds_alternative<InvocationNode>(out));
	const auto& inv = std::get<InvocationNode>(out);
	REQUIRE(inv.id == builtins::FunctionIDs::PERMUTE);
	// Arguments: the input and the permutation [1, 0, 2] as one constant vector.
	REQUIRE(inv.arguments.size() == 2);
	REQUIRE(&inv.arguments[0].get() == &input);
	const auto* perm = std::get_if<ConstNode>(&inv.arguments[1].get());
	REQUIRE(perm != nullptr);
	REQUIRE(std::get<std::vector<int>>(perm->value) == std::vector<int>{1, 0, 2});
}

TEST_CASE("einops identity permute emits no PERMUTE node", "[einops][permute]") {
	Computegraph graph;
	const Node& input = graph.addParameterNode(
			tensorType(PrimitiveType::float32, {4, 8}));

	Spec lhs = {named("a"), named("b")};
	Spec rhs = {named("a"), named("b")};

	auto result = apply(graph, input, lhs, rhs, {});
	REQUIRE(result.has_value());
	// No permute needed — result should be the original input node
	REQUIRE(&result->get() == &input);
}

// ---------------------------------------------------------------------------
// Merge (group in RHS): a b -> (a b)
// ---------------------------------------------------------------------------

TEST_CASE("einops merge emits RESHAPE node", "[einops][reshape]") {
	Computegraph graph;
	const Node& input = graph.addParameterNode(
			tensorType(PrimitiveType::float32, {4, 8}));

	Spec lhs = {named("a"), named("b")};
	Spec rhs = {group({"a", "b"})};

	auto result = apply(graph, input, lhs, rhs, {});
	REQUIRE(result.has_value());

	const Node& out = result->get();
	REQUIRE(std::holds_alternative<InvocationNode>(out));
	const auto& inv = std::get<InvocationNode>(out);
	REQUIRE(inv.id == builtins::FunctionIDs::RESHAPE);
	// Output shape should be [32]
	const auto& outType = std::get<TensorType>(inv.type);
	REQUIRE(outType.shape.size() == 1);
	REQUIRE(std::get<unsigned>(outType.shape[0]) == 32u);
}

// ---------------------------------------------------------------------------
// Split (group in LHS): (a b) -> a b
// ---------------------------------------------------------------------------

TEST_CASE("einops split emits RESHAPE node", "[einops][reshape]") {
	Computegraph graph;
	const Node& input = graph.addParameterNode(
			tensorType(PrimitiveType::float32, {32}));

	Spec lhs = {group({"a", "b"})};
	Spec rhs = {named("a"), named("b")};

	// Must provide sizes for grouped dims
	const Node& sizeA = graph.addConstNode(Atomic{4});
	const Node& sizeB = graph.addConstNode(Atomic{8});

	auto result = apply(graph, input, lhs, rhs, {{"a", &sizeA}, {"b", &sizeB}});
	REQUIRE(result.has_value());

	const Node& out = result->get();
	REQUIRE(std::holds_alternative<InvocationNode>(out));
	const auto& inv = std::get<InvocationNode>(out);
	REQUIRE(inv.id == builtins::FunctionIDs::RESHAPE);
}

// ---------------------------------------------------------------------------
// Unsqueeze: add literal dim 1 in RHS  (a b -> a 1 b)
// ---------------------------------------------------------------------------

TEST_CASE("einops unsqueeze adds literal dim", "[einops][reshape]") {
	Computegraph graph;
	const Node& input = graph.addParameterNode(
			tensorType(PrimitiveType::float32, {4, 8}));

	Spec lhs = {named("a"), named("b")};
	Spec rhs = {named("a"), literal(1), named("b")};

	auto result = apply(graph, input, lhs, rhs, {});
	REQUIRE(result.has_value());

	const Node& out = result->get();
	REQUIRE(std::holds_alternative<InvocationNode>(out));
	const auto& inv = std::get<InvocationNode>(out);
	REQUIRE(inv.id == builtins::FunctionIDs::RESHAPE);
	const auto& outType = std::get<TensorType>(inv.type);
	REQUIRE(outType.shape.size() == 3);
	REQUIRE(std::get<unsigned>(outType.shape[1]) == 1u);
}

// ---------------------------------------------------------------------------
// Ellipsis passthrough: ... a b -> ... b a
// ---------------------------------------------------------------------------

TEST_CASE("einops ellipsis permute emits PERMUTE with negative indices", "[einops][ellipsis]") {
	Computegraph graph;
	// Shape has Elipsis for the batch dims
	TensorType::Shape shape{Elipsis{}, 4u, 8u};
	const Node& input = graph.addParameterNode(
			TensorType{.type = PrimitiveType::float32, .shape = shape});

	Spec lhs = {ellipsis(), named("a"), named("b")};
	Spec rhs = {ellipsis(), named("b"), named("a")};

	auto result = apply(graph, input, lhs, rhs, {});
	REQUIRE(result.has_value());

	const Node& out = result->get();
	REQUIRE(std::holds_alternative<InvocationNode>(out));
	const auto& inv = std::get<InvocationNode>(out);
	REQUIRE(inv.id == builtins::FunctionIDs::PERMUTE);
	// The (possibly multi-dimensional) ellipsis is marked; post-ellipsis dims use negative indices.
	const auto& perm = std::get<std::vector<int>>(std::get<ConstNode>(inv.arguments[1].get()).value);
	REQUIRE(perm == std::vector<int>{builtins::kEllipsisPermMarker, -1, -2});
}

TEST_CASE("einops splits after an ellipsis mark the ellipsis group in the reshape", "[einops][ellipsis]") {
	Computegraph graph;
	const Node& input = graph.addParameterNode(TensorType{.type = PrimitiveType::float32, .shape = {Elipsis{}, 768u}});
	Spec lhs = {ellipsis(), group({"h", "d"})};
	Spec rhs = {named("h"), ellipsis(), named("d")};
	const Node& heads = graph.addConstNode(Atomic{12});
	auto result = apply(graph, input, lhs, rhs, {{"h", &heads}});
	REQUIRE(result.has_value());

	// PERMUTE(RESHAPE(x, {0, 1}, 12, 64), [h, ..., d])
	const auto& permute = std::get<InvocationNode>(result->get());
	REQUIRE(permute.id == builtins::FunctionIDs::PERMUTE);
	const auto& perm = std::get<std::vector<int>>(std::get<ConstNode>(permute.arguments[1].get()).value);
	REQUIRE(perm == std::vector<int>{-2, builtins::kEllipsisPermMarker, -1});
	const auto& reshape = std::get<InvocationNode>(permute.arguments[0].get());
	REQUIRE(reshape.id == builtins::FunctionIDs::RESHAPE);
	// The ellipsis group: no input dims before it, one (the merged `(h d)`) after it.
	REQUIRE(std::get<std::vector<int>>(std::get<ConstNode>(reshape.arguments[1].get()).value) == std::vector<int>{0, 1});
}

// ---------------------------------------------------------------------------
// Inference: one unknown in a group is inferred from the merged size
// ---------------------------------------------------------------------------

TEST_CASE("einops infers b from merged size and a=2 binding", "[einops][inference]") {
	Computegraph graph;
	// Input dim 0 = 24, so b = 24 / 2 = 12
	const Node& input = graph.addParameterNode(
			tensorType(PrimitiveType::float32, {24}));

	Spec lhs = {group({"a", "b"})};
	Spec rhs = {named("a"), named("b")};

	const Node& sizeA = graph.addConstNode(Atomic{2});
	auto result = apply(graph, input, lhs, rhs, {{"a", &sizeA}});
	REQUIRE(result.has_value());

	const auto& inv = std::get<InvocationNode>(result->get());
	REQUIRE(inv.id == builtins::FunctionIDs::RESHAPE);
	const auto& outType = std::get<TensorType>(inv.type);
	REQUIRE(outType.shape.size() == 2);
	REQUIRE(staticDim(outType.shape[1]) == 12); // b inferred as 24/2
}

TEST_CASE("einops infers a from merged size and b=3 binding", "[einops][inference]") {
	Computegraph graph;
	// Input dim 0 = 12, so a = 12 / 3 = 4
	const Node& input = graph.addParameterNode(
			tensorType(PrimitiveType::float32, {12}));

	Spec lhs = {group({"a", "b"})};
	Spec rhs = {named("a"), named("b")};

	const Node& sizeB = graph.addConstNode(Atomic{3});
	auto result = apply(graph, input, lhs, rhs, {{"b", &sizeB}});
	REQUIRE(result.has_value());

	const auto& inv = std::get<InvocationNode>(result->get());
	REQUIRE(inv.id == builtins::FunctionIDs::RESHAPE);
	const auto& outType = std::get<TensorType>(inv.type);
	REQUIRE(outType.shape.size() == 2);
	REQUIRE(staticDim(outType.shape[0]) == 4); // a inferred as 12/3
}

TEST_CASE("einops infers b from RHS literal: (a b) -> 2 b", "[einops][inference]") {
	Computegraph graph;
	// Input dim 0 = 16, RHS has literal 2 → a=2, b = 16/2 = 8
	const Node& input = graph.addParameterNode(
			tensorType(PrimitiveType::float32, {16}));

	// (a b) -> 2 b  means: a is absent from RHS by name, replaced by literal 2
	SpecItem lit2{.is_group = false, .atoms = {{AtomKind::Literal, {}, 2}}};
	Spec lhs = {group({"a", "b"})};
	Spec rhs = {lit2, named("b")};

	auto result = apply(graph, input, lhs, rhs, {});
	REQUIRE(result.has_value());

	const auto& inv = std::get<InvocationNode>(result->get());
	REQUIRE(inv.id == builtins::FunctionIDs::RESHAPE);
	const auto& outType = std::get<TensorType>(inv.type);
	REQUIRE(outType.shape.size() == 2);
	REQUIRE(std::get<unsigned>(outType.shape[0]) == 2u);
	REQUIRE(std::get<unsigned>(outType.shape[1]) == 8u);
}

TEST_CASE("einops infers a from RHS literal: (a b) -> a 3", "[einops][inference]") {
	Computegraph graph;
	// Input dim 0 = 12, RHS literal 3 → b=3, a = 12/3 = 4
	const Node& input = graph.addParameterNode(
			tensorType(PrimitiveType::float32, {12}));

	SpecItem lit3{.is_group = false, .atoms = {{AtomKind::Literal, {}, 3}}};
	Spec lhs = {group({"a", "b"})};
	Spec rhs = {named("a"), lit3};

	auto result = apply(graph, input, lhs, rhs, {});
	REQUIRE(result.has_value());

	const auto& inv = std::get<InvocationNode>(result->get());
	REQUIRE(inv.id == builtins::FunctionIDs::RESHAPE);
	const auto& outType = std::get<TensorType>(inv.type);
	REQUIRE(outType.shape.size() == 2);
	REQUIRE(std::get<unsigned>(outType.shape[0]) == 4u);
	REQUIRE(std::get<unsigned>(outType.shape[1]) == 3u);
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

TEST_CASE("einops error on non-tensor input", "[einops][error]") {
	Computegraph graph;
	const Node& input = graph.addParameterNode(PrimitiveType::integer);

	Spec lhs = {named("a")};
	Spec rhs = {named("a")};

	auto result = apply(graph, input, lhs, rhs, {});
	REQUIRE(!result.has_value());
	REQUIRE(!result.error().empty());
}

TEST_CASE("einops error on LHS rank mismatch", "[einops][error]") {
	Computegraph graph;
	const Node& input = graph.addParameterNode(
			tensorType(PrimitiveType::float32, {4, 8}));

	// LHS has 3 items but tensor has 2 dims
	Spec lhs = {named("a"), named("b"), named("c")};
	Spec rhs = {named("a"), named("b"), named("c")};

	auto result = apply(graph, input, lhs, rhs, {});
	REQUIRE(!result.has_value());
}

TEST_CASE("einops error when both grouped dim sizes are unknown (ambiguous)", "[einops][error]") {
	Computegraph graph;
	const Node& input = graph.addParameterNode(
			tensorType(PrimitiveType::float32, {32}));

	Spec lhs = {group({"a", "b"})};
	Spec rhs = {named("a"), named("b")};

	// Neither a nor b provided — cannot infer both
	auto result = apply(graph, input, lhs, rhs, {});
	REQUIRE(!result.has_value());
}
