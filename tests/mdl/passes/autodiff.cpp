#include <mdl/parser.hpp>
#include <mdl/passes/autodiff.hpp>

#include <catch2/catch_test_macros.hpp>

#include <set>
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

namespace {
	/** @brief Counts the invocations of @p id reachable from the outputs of @p f. */
	size_t countReachable(const FunctionDecl& f, builtins::FunctionIDs id) {
		std::vector<const Node*> stack;
		std::set<const Node*> seen;
		for (const auto& out : f.outputs)
			stack.push_back(&out.get());
		size_t count = 0;
		while (!stack.empty()) {
			const Node* node = stack.back();
			stack.pop_back();
			if (!seen.insert(node).second)
				continue;
			if (const auto* inv = std::get_if<InvocationNode>(node)) {
				count += inv->id == id;
				for (const auto& arg : inv->arguments)
					stack.push_back(&arg.get());
			}
		}
		return count;
	}

	// `Scaled.gradient` is deliberately not the derivative of forward (it uses cos), so its use is observable.
	const char* customGradient = R"(
		module Scaled[] {
			s: float[4]
			float[..., 4] forward(x: float[..., 4]) => __builtin_exp(x) * s;
			(float[..., 4], float[4]) gradient(x: float[..., 4], dy: float[..., 4]) {
				return dy * __builtin_cos(x), __builtin_sum_like(dy * __builtin_exp(x), s);
			}
		}
		module Inner[] {
			a: Scaled[]
			float[..., 4] forward(x: float[..., 4]) => __builtin_tanh(a(x));
		}
		module Outer[] {
			first: Inner[]
			second: Inner[]
			float[..., 4] forward(x: float[..., 4]) => second(first(x));
		}
	)";
} // namespace

TEST_CASE("a custom gradient function of a module replaces differentiation", "[autodiff][custom]") {
	auto unit = parseOrFail(customGradient);
	const auto& outer = std::get<ModuleType>(unit.definitions.at("Outer"));
	autodiff::Options options;
	options.inputs = true;
	auto grad = autodiff::vjp(unit.graph(), outer.functions.at("forward"), weightPaths(outer), options);
	REQUIRE(grad.has_value());
	CHECK(grad->outputNames == std::vector<std::string>{"output", "d_first_a_s", "d_second_a_s", "d_x"});
	// Both nested instances (two levels deep) use the custom gradient: one cos per instance.
	CHECK(countReachable(*grad, builtins::FunctionIDs::COS) == 2);
	// The exp recomputed in the custom gradient is the forward's node (one exp per instance).
	CHECK(countReachable(*grad, builtins::FunctionIDs::EXP) == 2);
}

TEST_CASE("a custom gradient with a wrong signature is reported", "[autodiff][custom]") {
	auto unit = parseOrFail(R"(
		module Bad[] {
			s: float[4]
			float[..., 4] forward(x: float[..., 4]) => x * s;
			float[..., 4] gradient(x: float[..., 4], dy: float[..., 4]) => dy * s;
		}
		module Net[] {
			bad: Bad[]
			float[..., 4] forward(x: float[..., 4]) => bad(x);
		}
	)");
	const auto& net = std::get<ModuleType>(unit.definitions.at("Net"));
	auto grad = autodiff::vjp(unit.graph(), net.functions.at("forward"), weightPaths(net));
	REQUIRE_FALSE(grad.has_value());
	CHECK(grad.error().find("must return 2 value(s)") != std::string::npos);
}

TEST_CASE("no gradient flows through stop_gradient", "[autodiff]") {
	auto unit = parseOrFail(R"(
		module M[] {
			W: float[4]
			V: float[4]
			float[..., 4] forward(x: float[..., 4]) => x * __builtin_stop_gradient(W) + x * V;
		}
	)");
	const auto& m = std::get<ModuleType>(unit.definitions.at("M"));
	auto grad = autodiff::vjp(unit.graph(), m.functions.at("forward"), weightPaths(m));
	REQUIRE(grad.has_value());
	REQUIRE(grad->outputNames == std::vector<std::string>{"output", "d_W", "d_V"});
	// d_W is a broadcast zero constant.
	const auto& dW = std::get<InvocationNode>(grad->outputs[1].get());
	CHECK(dW.id == builtins::FunctionIDs::BROADCAST_LIKE);
	CHECK(std::holds_alternative<ConstNode>(dW.arguments[0].get()));
}

TEST_CASE("a custom gradient can be called directly with the instance's weights", "[autodiff][custom]") {
	auto unit = parseOrFail(R"(
		module Scaled[] {
			s: float[4]
			float[..., 4] forward(x: float[..., 4]) => x * s;
			(float[..., 4], float[4]) gradient(x: float[..., 4], dy: float[..., 4]) {
				return dy * s, __builtin_sum_like(dy * x, s);
			}
		}
		module Net[] {
			inst: Scaled[]
			float[..., 4] forward(x: float[..., 4]) {
				auto dx, ds = inst.gradient(x, x);
				return dx;
			}
		}
	)");
	const auto& net = std::get<ModuleType>(unit.definitions.at("Net"));
	const auto& out = std::get<InvocationNode>(net.functions.at("forward").outputs.at(0).get());
	const auto* weight = std::get_if<ParameterNode>(&out.arguments[1].get());
	REQUIRE(weight != nullptr);
	CHECK(weight->name == "inst.s");
}
