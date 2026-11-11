// Soundness of the TVM backend's pattern rewrites: a pattern must only fire if the matched subgraph computes exactly
// the replacement (constants, axes, and shapes are checked).
#include "mdl/backends/tvm/details/pattern_registry.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/tensor.h>

#include <functional>
#include <string>
#include <vector>

using namespace tvm::relax;
using tvm::ffi::Array;

namespace {
	Expr constant(float value) {
		auto data = tvm::runtime::Tensor::Empty(tvm::ffi::Shape{}, tvm::DataType::Float(32), DLDevice{kDLCPU, 0});
		data.CopyFromBytes(&value, sizeof(float));
		return Constant(data);
	}

	Expr op(const char* name, Array<Expr> args) { return Call(tvm::Op::Get(name), args); }

	Expr reduce(const char* name, Expr x, bool keepdims) {
		auto fn = tvm::ffi::Function::GetGlobalRequired(std::string("relax.op.") + name);
		return fn(x, Array<tvm::Integer>{-1}, keepdims).cast<Expr>();
	}

	/** Builds `f(x[, gamma, beta]) = body(...)` with x of shape [rows, 8] and applies the backend patterns. */
	Function rewrite(
			const std::function<Expr(BlockBuilder&, const std::vector<Var>&)>& body, size_t numParams = 1,
			int64_t rows = 4
	) {
		auto bb = BlockBuilder::Create(std::nullopt);
		std::vector<Var> params{Var("x", TensorStructInfo(ShapeExpr(Array<tvm::PrimExpr>{tvm::IntImm(tvm::DataType::Int(64), rows), 8}), tvm::DataType::Float(32)))};
		for (size_t i = 1; i < numParams; ++i)
			params.push_back(Var("p" + std::to_string(i), TensorStructInfo(ShapeExpr({8}), tvm::DataType::Float(32))));
		bb->BeginScope(Array<Var>(params.begin(), params.end()));
		bb->BeginDataflowBlock();
		auto out = bb->EmitOutput(body(bb, params));
		auto block = bb->EndBlock();
		bb->EndScope();
		Function func(Array<Var>(params.begin(), params.end()), SeqExpr({block}, out), GetStructInfo(out));
		auto mod = tvm::IRModule({{tvm::GlobalVar("main"), func}});
		mod = mdl::backend::tvm::applyPatterns(mod, mdl::backend::tvm::defaultPatterns());
		return tvm::Downcast<Function>(mod->Lookup("main"));
	}

	/** All calls of operator @p name in @p func. */
	std::vector<Call> callsOf(const Function& func, const std::string& name) {
		std::vector<Call> calls;
		PostOrderVisit(func, [&](const Expr& e) {
			if (const auto* call = e.as<CallNode>())
				if (const auto* o = call->op.as<tvm::OpNode>(); o && o->name == name)
					calls.push_back(tvm::ffi::GetRef<Call>(call));
		});
		return calls;
	}

	Expr gelu(BlockBuilder& bb, const Expr& x, float half, float one, float sqrt2) {
		auto erf = bb->Emit(op("relax.erf", {bb->Emit(op("relax.divide", {x, constant(sqrt2)}))}));
		auto lhs = bb->Emit(op("relax.multiply", {x, constant(half)}));
		return op("relax.multiply", {lhs, bb->Emit(op("relax.add", {constant(one), erf}))});
	}
} // namespace

TEST_CASE("the GELU decomposition is recognized", "[tvm][patterns]") {
	auto f = rewrite([](auto& bb, auto& p) { return gelu(bb, p[0], 0.5f, 1.0f, 1.41421356f); });
	CHECK(callsOf(f, "relax.nn.gelu").size() == 1);
}

TEST_CASE("a GELU-shaped expression with other constants is not rewritten", "[tvm][patterns]") {
	auto f = rewrite([](auto& bb, auto& p) { return gelu(bb, p[0], 3.0f, 5.0f, 7.0f); });
	CHECK(callsOf(f, "relax.nn.gelu").empty());
}

TEST_CASE("softmax is only recognized for reductions with keepdims", "[tvm][patterns]") {
	auto softmax = [](bool keepdims) {
		return [keepdims](BlockBuilder& bb, const std::vector<Var>& p) -> Expr {
			auto max = bb->Emit(reduce("max", p[0], keepdims));
			auto e = bb->Emit(op("relax.exp", {bb->Emit(op("relax.subtract", {p[0], max}))}));
			return op("relax.divide", {e, bb->Emit(reduce("sum", e, keepdims))});
		};
	};
	CHECK(callsOf(rewrite(softmax(true)), "relax.nn.softmax").size() == 1);
	// Without keepdims, x - max(x) broadcasts the row maxima along the wrong axis (valid for square inputs only).
	CHECK(callsOf(rewrite(softmax(false), 1, 8), "relax.nn.softmax").empty());
}

TEST_CASE("layer norm keeps the model's epsilon and checks the divisor", "[tvm][patterns]") {
	auto layerNorm = [](float n, float eps) {
		return [n, eps](BlockBuilder& bb, const std::vector<Var>& p) -> Expr {
			static const auto reshape = tvm::ffi::Function::GetGlobalRequired("relax.op.reshape");
			auto keep = [&](Expr e) { return bb->Emit(reshape(e, Array<tvm::PrimExpr>{4, 1}).template cast<Expr>()); };
			auto mu = keep(bb->Emit(op("relax.divide", {bb->Emit(reduce("sum", p[0], false)), constant(n)})));
			auto xc = bb->Emit(op("relax.subtract", {p[0], mu}));
			auto sq = bb->Emit(op("relax.multiply", {xc, xc}));
			auto var = keep(bb->Emit(op("relax.divide", {bb->Emit(reduce("sum", sq, false)), constant(n)})));
			auto s = bb->Emit(op("relax.rsqrt", {bb->Emit(op("relax.add", {var, constant(eps)}))}));
			auto scaled = bb->Emit(op("relax.multiply", {bb->Emit(op("relax.multiply", {xc, s})), p[1]}));
			return op("relax.add", {scaled, p[2]});
		};
	};
	auto calls = callsOf(rewrite(layerNorm(8.0f, 1e-5f), 3), "relax.nn.layer_norm");
	REQUIRE(calls.size() == 1);
	CHECK_THAT(calls[0]->attrs.as<LayerNormAttrs>()->epsilon, Catch::Matchers::WithinRel(1e-5, 1e-3));
	// Dividing by something other than the normalized dimension is not a layer norm.
	CHECK(callsOf(rewrite(layerNorm(5.0f, 1e-5f), 3), "relax.nn.layer_norm").empty());
}

TEST_CASE("x @ W^T + b becomes mdl.matmul with transposed weight and bias", "[tvm][patterns]") {
	auto bb = BlockBuilder::Create(std::nullopt);
	auto f = [](BlockBuilder& bb, const std::vector<Var>& p) -> Expr {
		static const auto permute = tvm::ffi::Function::GetGlobalRequired("relax.op.permute_dims");
		auto wt = bb->Emit(permute(p[1], Array<tvm::Integer>{1, 0}).template cast<Expr>());
		static const auto matmul = tvm::ffi::Function::GetGlobalRequired("relax.op.matmul");
		auto prod = bb->Emit(matmul(p[0], wt, tvm::ffi::Optional<tvm::DataType>()).template cast<Expr>());
		return op("relax.add", {prod, p[2]});
	};
	// Build with a [8, 8] weight (the generic helper only creates 1-D extra parameters, so do it manually).
	std::vector<Var> params{
			Var("x", TensorStructInfo(ShapeExpr({4, 8}), tvm::DataType::Float(32))),
			Var("W", TensorStructInfo(ShapeExpr({8, 8}), tvm::DataType::Float(32))),
			Var("b", TensorStructInfo(ShapeExpr({8}), tvm::DataType::Float(32))),
	};
	bb->BeginScope(Array<Var>(params.begin(), params.end()));
	bb->BeginDataflowBlock();
	auto out = bb->EmitOutput(f(bb, params));
	auto block = bb->EndBlock();
	bb->EndScope();
	Function func(Array<Var>(params.begin(), params.end()), SeqExpr({block}, out), GetStructInfo(out));
	auto mod = mdl::backend::tvm::applyPatterns(
			tvm::IRModule({{tvm::GlobalVar("main"), func}}), mdl::backend::tvm::defaultPatterns()
	);
	auto rewritten = tvm::Downcast<Function>(mod->Lookup("main"));
	auto calls = callsOf(rewritten, "mdl.matmul");
	REQUIRE(calls.size() == 1);
	CHECK(GetStructInfoAs<TensorStructInfoNode>(calls[0]->args[2]) != nullptr); // bias fused
}
