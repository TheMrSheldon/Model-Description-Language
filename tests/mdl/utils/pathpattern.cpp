#include <mdl/utils/pathpattern.hpp>

#include <catch2/catch_test_macros.hpp>

using mdl::PathPatterns;

TEST_CASE("no patterns select everything", "[pathpattern]") {
	PathPatterns patterns;
	CHECK(patterns.selects("layers.0.W"));
}

TEST_CASE("a plain path selects its subtree", "[pathpattern]") {
	auto patterns = PathPatterns::parse("layer1.sublayer");
	CHECK(patterns.selects("layer1.sublayer"));
	CHECK(patterns.selects("layer1.sublayer.W"));
	CHECK_FALSE(patterns.selects("layer1.other.W"));
	CHECK_FALSE(patterns.selects("layer1"));
}

TEST_CASE("* matches one segment, ** any number", "[pathpattern]") {
	auto one = PathPatterns::parse("layer1.*");
	CHECK(one.selects("layer1.W"));
	CHECK(one.selects("layer1.sub.W")); // subtree of layer1.sub
	CHECK_FALSE(one.selects("layer2.W"));

	auto weights = PathPatterns::parse("layers.*.W");
	CHECK(weights.selects("layers.3.W"));
	CHECK_FALSE(weights.selects("layers.3.b"));

	auto deep = PathPatterns::parse("**.gamma");
	CHECK(deep.selects("gamma"));
	CHECK(deep.selects("layers.0.norm.gamma"));
	CHECK_FALSE(deep.selects("layers.0.norm.beta"));
}

TEST_CASE("globs within a segment", "[pathpattern]") {
	auto patterns = PathPatterns::parse("weight*.?");
	CHECK(patterns.selects("weightQuery.W"));
	CHECK(patterns.selects("weightKey.b"));
	CHECK_FALSE(patterns.selects("mhaNorm.gamma"));
	CHECK_FALSE(patterns.selects("weightQuery.Wb"));
}

TEST_CASE("negations deselect and the last matching pattern wins", "[pathpattern]") {
	auto patterns = PathPatterns::parse("layers.**\n!layers.0.**\n# comment\n\nlayers.0.norm");
	CHECK(patterns.selects("layers.1.W"));
	CHECK_FALSE(patterns.selects("layers.0.W"));
	CHECK(patterns.selects("layers.0.norm.gamma"));
	CHECK_FALSE(patterns.selects("embed.W"));
}

TEST_CASE("a leading negation starts from everything selected", "[pathpattern]") {
	auto patterns = PathPatterns::parse("!layer2.sublayer");
	CHECK(patterns.selects("layer1.W"));
	CHECK(patterns.selects("layer2.other.W"));
	CHECK_FALSE(patterns.selects("layer2.sublayer.W"));
}
