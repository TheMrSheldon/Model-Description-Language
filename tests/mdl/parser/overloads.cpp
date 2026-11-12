#include <mdl/parser.hpp>

#include <catch2/catch_test_macros.hpp>

#include <set>
#include <sstream>

using namespace mdl;
using builtins::FunctionIDs;

namespace {
	std::expected<CompilationUnit, std::string> parse(const std::string& source) {
		std::istringstream stream{source};
		auto unit = parser::parse(stream, "<test>");
		if (!unit)
			return std::unexpected(std::string{"parse error"});
		return std::move(*unit);
	}

	/** @brief The builtins reachable from the outputs of @p f. */
	std::multiset<FunctionIDs> builtinsOf(const FunctionDecl& f) {
		std::multiset<FunctionIDs> ids;
		std::set<const Node*> seen;
		std::vector<const Node*> stack;
		for (const auto& out : f.outputs)
			stack.push_back(&out.get());
		while (!stack.empty()) {
			const Node* n = stack.back();
			stack.pop_back();
			if (!seen.insert(n).second)
				continue;
			if (const auto* inv = std::get_if<InvocationNode>(n)) {
				ids.insert(inv->id);
				for (const auto& a : inv->arguments)
					stack.push_back(&a.get());
			}
		}
		return ids;
	}

	// `reduce` has a masked overload; `outer` does not know about masks but calls `reduce`.
	const char* library = R"(
		float[...] reduce(x: float[...]) => __builtin_sum(x);
		float[...] reduce(x: masked float[...]) => __builtin_mean(x.data) * 2.0;
		float[...] outer(x: float[...]) => reduce(x) + 1.0;
	)";
} // namespace

TEST_CASE("a masked argument selects the masked overload, also in nested calls", "[parser][overloads]") {
	auto unit = parse(std::string(library) + R"(
		module M[] {
			float[...] plain(x: float[..., 4]) => outer(x);
			float[...] direct(x: masked float[..., 4]) => reduce(x);
			float[...] nested(x: masked float[..., 4]) => outer(x);
			float[..., 4] scaled(x: masked float[..., 4]) => __builtin_exp(x) * 2.0;
		}
	)");
	REQUIRE(unit.has_value());
	const auto& m = std::get<ModuleType>(unit->definitions.at("M"));

	auto plain = builtinsOf(m.functions.at("plain"));
	CHECK(plain.contains(FunctionIDs::SUM));
	CHECK_FALSE(plain.contains(FunctionIDs::MEAN));

	for (const char* fn : {"direct", "nested"}) {
		auto ids = builtinsOf(m.functions.at(fn));
		CHECK(ids.contains(FunctionIDs::MEAN));
		CHECK(ids.contains(FunctionIDs::UNMASK));
		CHECK_FALSE(ids.contains(FunctionIDs::SUM));
	}
	// Computations on a masked tensor yield masked tensors (`x.data`, as in the masked `reduce`, is unmasked).
	const auto& out = m.functions.at("scaled").outputs.at(0).get();
	const auto* type = std::get_if<TensorType>(&getDataType(out));
	REQUIRE(type != nullptr);
	CHECK(type->masked);
}

TEST_CASE("masked parameters need masked tensors, and `.mask` needs a masked tensor", "[parser][overloads]") {
	CHECK_FALSE(parse(R"(
		float[...] onlyMasked(x: masked float[...]) => x.data;
		module M[] { float[...] forward(x: float[..., 4]) => onlyMasked(x); }
	)").has_value());
	CHECK_FALSE(parse(R"(
		module M[] { float[...] forward(x: float[..., 4]) => x.mask; }
	)").has_value());
	// The same signature twice is a redefinition.
	CHECK_FALSE(parse(R"(
		float[...] f(x: float[...]) => x;
		float[...] f(y: float[...]) => y;
	)").has_value());
}
