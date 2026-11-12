#include <mdl/parser.hpp>
#include <mdl/passes/shapes.hpp>

#include <catch2/catch_test_macros.hpp>

#include <sstream>

using namespace mdl;

namespace {
	CompilationUnit parseOrFail(const std::string& source) {
		std::istringstream stream{source};
		auto unit = parser::parse(stream, "<test>");
		REQUIRE(unit.has_value());
		return std::move(*unit);
	}

	std::optional<int64_t> evaluate(const Node& node) {
		if (const auto* c = std::get_if<ConstNode>(&node))
			if (const auto* i = std::get_if<int>(&c->value))
				return *i;
		return std::nullopt;
	}

	/** @brief The inferred shape of @p node (which must have one). */
	const shapes::Shape& shapeAt(const shapes::Shapes& s, const Node& node) {
		const auto* shape = s.find(node);
		REQUIRE(shape != nullptr);
		return *shape;
	}
} // namespace

TEST_CASE("shapes are inferred exactly where `...` stands for several dimensions", "[shapes]") {
	// The attention scores q @ k^T: the transposition moves the sequence dimension out of `...`.
	auto unit = parseOrFail(R"(
		module Attention[] {
			float[..., 8] scores(x: float[..., 8]) => x[... (h d) -> h ... d | h=2] @ x[... (h d) -> h ... d | h=2].T();
		}
	)");
	const auto& m = std::get<ModuleType>(unit.definitions.at("Attention"));
	const auto& scores = m.functions.at("scores");
	auto result = shapes::infer(scores, 2, evaluate);
	REQUIRE(result.has_value());
	const auto& out = shapeAt(*result, scores.outputs[0].get());
	CHECK(shapes::str(out) == "[2, B0, B1, B1]");

	auto one = shapes::infer(scores, 1, evaluate);
	REQUIRE(one.has_value());
	CHECK(shapes::str(shapeAt(*one, scores.outputs[0].get())) == "[2, B, B]");
}

TEST_CASE("reductions, reshapes, and embeddings", "[shapes]") {
	auto unit = parseOrFail(R"(
		module M[] {
			table: float[10, 4]
			float[...] means(x: float[..., 4]) => __builtin_mean(x);
			float[..., 4] sums(x: float[..., 4]) => x - __builtin_sum(x);
			float[..., 4] embed(ids: int[...]) => __builtin_take(table, ids);
			float[...] flat(x: float[..., 4]) => x[... d -> (...) d];
			float total(x: float[..., 4]) => __builtin_mean_all(x);
		}
	)");
	const auto& m = std::get<ModuleType>(unit.definitions.at("M"));
	auto check = [&](const char* fn, const char* expected) {
		const auto& f = m.functions.at(fn);
		auto r = shapes::infer(f, 2, evaluate);
		REQUIRE(r.has_value());
		CHECK(shapes::str(shapeAt(*r, f.outputs[0].get())) == expected);
	};
	check("means", "[B0, B1]");
	check("sums", "[B0, B1, 4]");
	check("embed", "[B0, B1, 4]");
	check("flat", "[B0*B1, 4]");
	check("total", "[]");
}
