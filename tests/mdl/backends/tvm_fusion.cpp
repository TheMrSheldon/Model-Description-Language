// The fusion planner (src/mdl/backends/tvm/details/fusion): which legalized kernel calls become one kernel.
#include "mdl/backends/tvm/details/fusion/fusion.hpp"
#include "mdl/backends/tvm/details/legalize.hpp"

#include <mdl/compiler.hpp>

#include <catch2/catch_test_macros.hpp>

#include <tvm/ir/function.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>

#include <functional>
#include <vector>

using namespace tvm::relax;
using tvm::Call;
using tvm::Expr;
using tvm::Var;
using tvm::ffi::Array;

namespace {
	TensorType tensor(int64_t rows, int64_t columns) {
		return TensorType(
				ShapeExpr(Array<tvm::PrimExpr>{tvm::IntImm(tvm::PrimType::Int(64), rows), tvm::IntImm(tvm::PrimType::Int(64), columns)}),
				tvm::PrimType::Float(32)
		);
	}

	Expr op(const char* name, Array<Expr> args) { return Call(tvm::Type::Missing(), tvm::Op::Get(name), args); }

	Expr sum(Expr x, int64_t axis) {
		static const auto fn = tvm::ffi::Function::GetGlobalRequired("relax.op.sum");
		return fn(x, Array<int64_t>{axis}, true).cast<Expr>();
	}

	Expr scalar(double value) { return tvm::FloatImm(tvm::PrimType::Float(32), value); }

	/** @brief The numbers of kernel calls of the groups (primitive functions) after fusing `f(x, y) = body(x, y)`. */
	std::vector<size_t> groups(const std::function<Expr(BlockBuilder&, const Var&, const Var&)>& body) {
		auto bb = BlockBuilder::Create(std::nullopt);
		const Var x("x", tensor(256, 64)), y("y", tensor(256, 64));
		bb->BeginScope(Array<Var>{x, y});
		bb->BeginDataflowBlock();
		auto out = bb->EmitOutput(body(bb, x, y));
		auto block = bb->EndBlock();
		bb->EndScope();
		const Function f({x, y}, SeqExpr({block}, out), GetType(out));
		// (exported, as the compiler's functions: dead code elimination keeps it)
		auto mod = tvm::IRModule({{tvm::GlobalVar("main"), tvm::WithAttr(f, tvm::attr::kGlobalSymbol, tvm::ffi::String("main"))}});
		mod = transform::Normalize()(mod);
		mod = transform::LegalizeOps(mdl::backend::tvm::legalizeMap(mdl::CompileOptions{}), std::nullopt, false)(mod);
		mod = mdl::backend::tvm::fusion::fuseKernels()(mod);
		std::vector<size_t> sizes;
		for (const auto& [gv, f] : mod->functions)
			if (const auto* fn = f.as<FunctionNode>(); fn && fn->HasNonzeroAttr(attr::kPrimitive)) {
				size_t calls = 0;
				PostOrderVisit(fn->body, [&](const Expr& e) {
					const auto* c = e.as<tvm::CallNode>();
					calls += c != nullptr && c->op.same_as(tvm::Op::Get("relax.call_tir"));
				});
				sizes.push_back(calls);
			}
		return sizes;
	}
} // namespace

TEST_CASE("a normalization (two reductions over the same rows and its epilogue) is one kernel", "[tvm][fusion]") {
	auto sizes = groups([](BlockBuilder& bb, const Var& x, const Var&) -> Expr {
		auto mu = bb->Emit(op("relax.divide", {bb->Emit(sum(x, -1)), scalar(64)}));
		auto xc = bb->Emit(op("relax.subtract", {x, mu}));
		auto var = bb->Emit(op("relax.divide", {bb->Emit(sum(bb->Emit(op("relax.multiply", {xc, xc})), -1)), scalar(64)}));
		return op("relax.multiply", {xc, bb->Emit(op("relax.rsqrt", {bb->Emit(op("relax.add", {var, scalar(1e-5)}))}))});
	});
	REQUIRE(sizes.size() == 1);
	CHECK(sizes[0] >= 7);
}

TEST_CASE("kernels that read the same tensor become one kernel with several results", "[tvm][fusion]") {
	auto sizes = groups([](BlockBuilder& bb, const Var& x, const Var&) -> Expr {
		return Tuple({bb->Emit(op("relax.exp", {x})), bb->Emit(op("relax.sin", {x}))});
	});
	CHECK(sizes == std::vector<size_t>{2});
}

TEST_CASE("a shared elementwise value is computed in each of its consumers' kernel", "[tvm][fusion]") {
	// (GELU and its gradient share 1 + erf(x / √2))
	auto sizes = groups([](BlockBuilder& bb, const Var& x, const Var& y) -> Expr {
		auto phi = bb->Emit(op("relax.add", {scalar(1), bb->Emit(op("relax.erf", {x}))}));
		return Tuple({bb->Emit(op("relax.multiply", {x, phi})), bb->Emit(op("relax.multiply", {y, phi}))});
	});
	REQUIRE(sizes.size() == 1);
}

TEST_CASE("sums over the same columns are one kernel", "[tvm][fusion]") {
	auto sizes = groups([](BlockBuilder& bb, const Var& x, const Var& y) -> Expr {
		return Tuple({bb->Emit(sum(x, 0)), bb->Emit(sum(bb->Emit(op("relax.multiply", {x, y})), 0))});
	});
	REQUIRE(sizes.size() == 1);
	CHECK(sizes[0] == 3);
}

TEST_CASE("kernels that a call outside the group connects are not fused (no cycle)", "[tvm][fusion]") {
	auto sizes = groups([](BlockBuilder& bb, const Var& x, const Var&) -> Expr {
		auto a = bb->Emit(op("relax.exp", {x}));
		auto external = bb->Emit(Call(
				tvm::Type::Missing(), tvm::Op::Get("relax.call_dps_packed"), {ExternFunc("external"), Tuple({a})},
				tvm::Attrs(), {tensor(256, 64)}
		));
		return op("relax.add", {a, external});
	});
	CHECK(sizes.empty());
}

TEST_CASE("a normalization that also returns its per-row statistics is one kernel", "[tvm][fusion]") {
	// (the statistics saved for the backward pass: the mean without the reduced axis, reshaped for the normalization)
	auto sizes = groups([](BlockBuilder& bb, const Var& x, const Var&) -> Expr {
		static const auto sumOp = tvm::ffi::Function::GetGlobalRequired("relax.op.sum");
		static const auto reshape = tvm::ffi::Function::GetGlobalRequired("relax.op.reshape");
		static const auto astype = tvm::ffi::Function::GetGlobalRequired("relax.op.astype");
		// (as in bfloat16 builds: sums in float32)
		auto wide = bb->Emit(astype(x, tvm::PrimType::Float(64)->dtype).cast<Expr>());
		auto total = bb->Emit(astype(bb->Emit(sumOp(wide, Array<int64_t>{-1}, false).cast<Expr>()),
									 tvm::PrimType::Float(32)->dtype)
									  .cast<Expr>());
		auto mean = bb->Emit(op("relax.divide", {total, scalar(64)}));
		auto mu = bb->Emit(reshape(mean, Array<tvm::PrimExpr>{tvm::IntImm(tvm::PrimType::Int(64), 256),
														  tvm::IntImm(tvm::PrimType::Int(64), 1)})
								   .cast<Expr>());
		auto xc = bb->Emit(op("relax.subtract", {x, mu}));
		auto var = bb->Emit(op("relax.divide", {bb->Emit(sum(bb->Emit(op("relax.multiply", {xc, xc})), -1)), scalar(64)}));
		auto rstd = bb->Emit(op("relax.rsqrt", {bb->Emit(op("relax.add", {var, scalar(1e-5)}))}));
		return Tuple({bb->Emit(op("relax.multiply", {xc, rstd})), mean, rstd});
	});
	CHECK(sizes.size() == 1);
}
