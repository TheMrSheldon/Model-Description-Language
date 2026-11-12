#include <mdl/parser.hpp>
#include <mdl/passes/autodiff.hpp>

#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>
#include <vector>

using namespace mdl;

namespace {
	CompilationUnit parseOrFail(const std::string& source) {
		std::istringstream stream{source};
		auto unit = parser::parse(stream, "<test>");
		REQUIRE(unit.has_value());
		return std::move(*unit);
	}

	const char* twoLayers = R"(
		module Layer[] {
			W: float[4, 4]
			b: float[4]
			float[..., 4] forward(x: float[..., 4]) => x @ W.T() + b;
		}
		module Net[] {
			first: Layer[]
			second: Layer[]
			unused: float[4]
			float[..., 4] forward(x: float[..., 4]) => second(__builtin_tanh(first(x)));
		}
	)";

	std::vector<std::string> paramNames(const FunctionDecl& f) {
		std::vector<std::string> names;
		for (const auto& [name, _] : f.parameters)
			names.push_back(name);
		return names;
	}
} // namespace

TEST_CASE("the gradient function has cotangent inputs and gradient outputs", "[autodiff]") {
	auto unit = parseOrFail(twoLayers);
	const auto& net = std::get<ModuleType>(unit.definitions.at("Net"));
	auto grad = autodiff::vjp(unit.graph(), net.functions.at("forward"), weightPaths(net));
	REQUIRE(grad.has_value());
	CHECK(grad->name == "forward_vjp");
	CHECK(paramNames(*grad) == std::vector<std::string>{"x", "d_output"});
	// Unused weights are not part of the function and get no gradient output.
	CHECK(grad->outputNames ==
		  std::vector<std::string>{"output", "d_first_W", "d_first_b", "d_second_W", "d_second_b"});
	CHECK(grad->outputs.size() == grad->returnTypes.size());
}

TEST_CASE("trainable patterns select the differentiated weights", "[autodiff]") {
	auto unit = parseOrFail(twoLayers);
	const auto& net = std::get<ModuleType>(unit.definitions.at("Net"));
	autodiff::Options options;
	options.trainable = PathPatterns::parse("!first.**");
	options.inputs = true;
	auto grad = autodiff::vjp(unit.graph(), net.functions.at("forward"), weightPaths(net), options);
	REQUIRE(grad.has_value());
	CHECK(grad->outputNames == std::vector<std::string>{"output", "d_second_W", "d_second_b", "d_x"});
}

TEST_CASE("gradients can be differentiated again (higher order)", "[autodiff]") {
	auto unit = parseOrFail(twoLayers);
	const auto& net = std::get<ModuleType>(unit.definitions.at("Net"));
	auto grad = autodiff::vjp(unit.graph(), net.functions.at("forward"), weightPaths(net));
	REQUIRE(grad.has_value());
	auto second = autodiff::vjp(unit.graph(), *grad, weightPaths(net));
	REQUIRE(second.has_value());
	CHECK(second->name == "forward_vjp_vjp");
	// One cotangent per output of the first gradient function.
	CHECK(second->parameters.size() == grad->parameters.size() + grad->outputs.size());
}

TEST_CASE("builtins without gradient rule are reported", "[autodiff]") {
	auto unit = parseOrFail(R"(
		module M[] {
			W: float[4]
			float[...] forward(x: float[..., 4]) => __builtin_var(x * W, 1);
		}
	)");
	const auto& m = std::get<ModuleType>(unit.definitions.at("M"));
	auto grad = autodiff::vjp(unit.graph(), m.functions.at("forward"), weightPaths(m));
	CHECK_FALSE(grad.has_value());
}
