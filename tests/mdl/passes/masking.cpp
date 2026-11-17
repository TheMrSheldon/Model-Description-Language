#include <mdl/computegraphv2/builtins.hpp>
#include <mdl/parser.hpp>
#include <mdl/passes/masking.hpp>
#include <mdl/passes/shapes.hpp>

#include <catch2/catch_test_macros.hpp>

#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace mdl;
using FIDs = builtins::FunctionIDs;

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

	/** @brief The invocations of @p id reachable from the outputs of @p f. */
	std::vector<const InvocationNode*> reachable(const FunctionDecl& f, FIDs id) {
		std::vector<const Node*> stack;
		std::set<const Node*> seen;
		for (const auto& out : f.outputs)
			stack.push_back(&out.get());
		std::vector<const InvocationNode*> found;
		while (!stack.empty()) {
			const Node* node = stack.back();
			stack.pop_back();
			if (!seen.insert(node).second)
				continue;
			if (const auto* inv = std::get_if<InvocationNode>(node)) {
				if (inv->id == id)
					found.push_back(inv);
				for (const auto& arg : inv->arguments)
					stack.push_back(&arg.get());
			}
		}
		return found;
	}

	FunctionDecl masked(CompilationUnit& unit, const FunctionDecl& f) {
		auto shapes = shapes::infer(f, 2, evaluate);
		REQUIRE(shapes.has_value());
		auto result = masking::apply(unit.graph(), f, *shapes);
		REQUIRE(result.has_value());
		return result->function;
	}

	bool argumentIs(const InvocationNode& inv, FIDs id) {
		const auto* arg = std::get_if<InvocationNode>(&inv.arguments[0].get());
		return arg != nullptr && arg->id == id;
	}

	// The attention scores [heads, batch, seq, seq] of x [batch, seq, 8]; max reduces over the (masked) key dimension.
	const char* attention = R"(
		module M[] {
			float[...] generic(x: masked float[..., 8]) =>
				__builtin_max(x[... (h d) -> h ... d | h=2] @ x[... (h d) -> h ... d | h=2].T());
			float[...] explicit(x: masked float[..., 8]) =>
				__builtin_max(x.data[... (h d) -> h ... d | h=2] @ x.data[... (h d) -> h ... d | h=2].T());
		}
	)";
} // namespace

TEST_CASE("a reduction over a masked dimension ignores the masked elements", "[masking]") {
	auto unit = parseOrFail(attention);
	const auto& m = std::get<ModuleType>(unit.definitions.at("M"));
	auto f = masked(unit, m.functions.at("generic"));
	CHECK(f.parameters.back().first == "mask");
	auto maxima = reachable(f, FIDs::MAX);
	REQUIRE(maxima.size() == 1);
	CHECK(argumentIs(*maxima.front(), FIDs::WHERE));
}

TEST_CASE("a reduction over a flattening fills the masked elements before it (in any order)", "[masking]") {
	auto unit = parseOrFail(R"(
		module M[] {
			float all(x: masked float[..., 4]) => __builtin_mean(x[... -> (...)]);
			float reordered(x: masked float[..., 4]) => __builtin_mean(x[... d -> (d ...)]);
		}
	)");
	const auto& m = std::get<ModuleType>(unit.definitions.at("M"));
	for (const char* name : {"all", "reordered"}) {
		auto f = masked(unit, m.functions.at(name));
		// The mean of the filled values, divided by the fraction of valid elements (both flattened).
		auto divisions = reachable(f, FIDs::TENSOR_DIV);
		REQUIRE(divisions.size() == 1);
		auto means = reachable(f, FIDs::MEAN);
		REQUIRE(means.size() == 2);
		for (const auto* mean : means) {
			REQUIRE(argumentIs(*mean, FIDs::RESHAPE));
			const auto& reshape = std::get<InvocationNode>(mean->arguments[0].get());
			const auto* filled = std::get_if<InvocationNode>(&reshape.arguments[0].get());
			REQUIRE(filled != nullptr);
			CHECK((filled->id == FIDs::WHERE || filled->id == FIDs::BROADCAST_LIKE)); // values or the valid indicator
		}
	}
}

TEST_CASE("a reduction over a flattened masked dimension after other operations is an error", "[masking]") {
	auto unit = parseOrFail(R"(
		module M[] {
			float scaled(x: masked float[..., 4]) => __builtin_mean(x[... -> (...)] * 2.0);
		}
	)");
	const auto& f = std::get<ModuleType>(unit.definitions.at("M")).functions.at("scaled");
	auto shapes = shapes::infer(f, 2, evaluate);
	REQUIRE(shapes.has_value());
	auto result = masking::apply(unit.graph(), f, *shapes);
	REQUIRE_FALSE(result.has_value());
	CHECK(result.error().find("merges the masked dimension") != std::string::npos);
}

TEST_CASE("values computed from x.data and x.mask are plain: the code took care of the mask", "[masking]") {
	auto unit = parseOrFail(attention);
	const auto& m = std::get<ModuleType>(unit.definitions.at("M"));
	auto f = masked(unit, m.functions.at("explicit"));
	auto maxima = reachable(f, FIDs::MAX);
	REQUIRE(maxima.size() == 1);
	CHECK_FALSE(argumentIs(*maxima.front(), FIDs::WHERE));
}

TEST_CASE("floating-point inputs are zeroed at masked positions", "[masking]") {
	auto unit = parseOrFail(R"(
		module M[] {
			float[..., 8] f(x: masked float[..., 8]) => x * 2;
		}
	)");
	const auto& m = std::get<ModuleType>(unit.definitions.at("M"));
	auto f = masked(unit, m.functions.at("f"));
	// x * 2 reads the filled input (and the output is filled as well).
	const auto* out = std::get_if<InvocationNode>(&f.outputs[0].get());
	REQUIRE(out != nullptr);
	CHECK(out->id == FIDs::WHERE);
	const auto* product = std::get_if<InvocationNode>(&out->arguments[1].get());
	REQUIRE(product != nullptr);
	CHECK(argumentIs(*product, FIDs::WHERE));
}
