#include <mdl/computegraphv2/graph.hpp>
#include <mdl/computegraphv2/object.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace mdl;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static AggregateType makeType(std::initializer_list<std::string> names) {
	AggregateType type;
	for (const auto& name : names)
		type.members[name] = DataType{PrimitiveType::integer};
	return type;
}

// ---------------------------------------------------------------------------
// AggregateInstance::create — error cases
// ---------------------------------------------------------------------------

TEST_CASE("create fails when a member is missing from values", "[AggregateInstance]") {
	AggregateType type = makeType({"x", "y"});
	auto result = AggregateInstance::create(type, {{"x", static_cast<const Node*>(nullptr)}});

	REQUIRE_FALSE(result.has_value());
	CHECK(result.error().find("y") != std::string::npos);
}

TEST_CASE("create fails when values contains a superfluous key", "[AggregateInstance]") {
	AggregateType type = makeType({"x"});
	auto result = AggregateInstance::create(
			type, {{"x", static_cast<const Node*>(nullptr)}, {"z", static_cast<const Node*>(nullptr)}}
	);

	REQUIRE_FALSE(result.has_value());
	CHECK(result.error().find("z") != std::string::npos);
}

TEST_CASE("create succeeds with exactly the right keys", "[AggregateInstance]") {
	AggregateType type = makeType({"x"});
	auto result = AggregateInstance::create(type, {{"x", static_cast<const Node*>(nullptr)}});

	CHECK(result.has_value());
}

TEST_CASE("create succeeds for an empty type with empty values", "[AggregateInstance]") {
	auto result = AggregateInstance::create(AggregateType{}, {});

	CHECK(result.has_value());
}

// ---------------------------------------------------------------------------
// getMember — nonexistent name
// ---------------------------------------------------------------------------

TEST_CASE("getMember returns nullopt for nonexistent member name", "[AggregateInstance]") {
	AggregateType type = makeType({"x"});
	auto inst = AggregateInstance::create(type, {{"x", static_cast<const Node*>(nullptr)}}).value();

	CHECK_FALSE(inst.getMember("y").has_value());
}

// ---------------------------------------------------------------------------
// getMember — member present in type always has a value
// ---------------------------------------------------------------------------

TEST_CASE("getMember returns a value for every member in the type", "[AggregateInstance]") {
	AggregateType type = makeType({"a", "b"});
	auto inst = AggregateInstance::create(
						  type, {{"a", static_cast<const Node*>(nullptr)}, {"b", static_cast<const Node*>(nullptr)}}
	)
						  .value();

	CHECK(inst.getMember("a").has_value());
	CHECK(inst.getMember("b").has_value());
	CHECK_FALSE(inst.getMember("c").has_value());
}

// ---------------------------------------------------------------------------
// getMember — returned reference points to the stored object
// ---------------------------------------------------------------------------

TEST_CASE("getMember returns reference to the correct object among multiple members", "[AggregateInstance]") {
	Computegraph graph;
	const Node& nodeA = graph.addConstNode(Atomic{1});
	const Node& nodeB = graph.addConstNode(Atomic{2});

	// Map order: "a" < "b" → index 0 = "a", index 1 = "b"
	AggregateType type = makeType({"a", "b"});
	auto inst = AggregateInstance::create(type, {{"a", &nodeA}, {"b", &nodeB}}).value();

	REQUIRE(inst.getMember("a").has_value());
	CHECK(std::get<const Node*>(inst.getMember("a")->get()) == &nodeA);

	REQUIRE(inst.getMember("b").has_value());
	CHECK(std::get<const Node*>(inst.getMember("b")->get()) == &nodeB);
}
