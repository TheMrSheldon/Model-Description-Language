#include "./scatter_add_rows.hpp"

#include "../../details/legalize.hpp"
#include "../../details/tvm_compat.hpp"

#include <tvm/ir/expr.h>
#include <tvm/relax/type.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include <string>

namespace mdl::backend::tvm::gpu {

	namespace {
		namespace tirx = ::tvm::tirx;
		using ::tvm::IntImm;
		using ::tvm::PrimExpr;

		constexpr int64_t kThreads = 256;

		/** @brief `for i in [0, extent)` over blockIdx.x * kThreads + threadIdx.x (guarded). */
		tirx::Stmt grid(const PrimExpr& extent, const std::string& name, const std::function<tirx::Stmt(PrimExpr)>& body) {
			const auto i64 = ::tvm::PrimType::Int(64);
			::tvm::PrimVar bx(name + "_bx", i64), tx(name + "_tx", i64);
			const PrimExpr threads = IntImm(i64, kThreads);
			const PrimExpr blocks = ::tvm::floordiv(extent + IntImm(i64, kThreads - 1), threads);
			const PrimExpr i = bx * threads + tx;
			auto bind = [](const ::tvm::PrimVar& v, const PrimExpr& n, const char* tag, tirx::Stmt s) {
				return tirx::For(
						v, IntImm(v.ty(), 0), n, tirx::ForKind::kThreadBinding, std::move(s),
						tirx::IterVar(::tvm::Range(IntImm(v.ty(), 0), n), v, tirx::IterVarType::kThreadIndex, tag)
				);
			};
			return bind(bx, blocks, "blockIdx.x", bind(tx, threads, "threadIdx.x", tirx::IfThenElse(i < extent, body(i))));
		}

		tirx::PrimFunc kernel(::tvm::PrimType idsType) {
			const auto i64 = ::tvm::PrimType::Int(64), f32 = ::tvm::PrimType::Float(32);
			::tvm::PrimVar n("n", i64), r("r", i64), v("v", i64);
			auto G = tirx::decl_buffer({n, r}, f32, "g");
			auto I = tirx::decl_buffer({n}, idsType, "ids");
			auto O = tirx::decl_buffer({v, r}, f32, "out");
			auto zero = grid(v * r, "zero", [&](PrimExpr e) -> tirx::Stmt {
				return tirx::BufferStore(O, ::tvm::FloatImm(f32, 0.0), {::tvm::floordiv(e, r), ::tvm::floormod(e, r)});
			});
			auto add = grid(n * r, "add", [&](PrimExpr e) -> tirx::Stmt {
				const PrimExpr k = ::tvm::floordiv(e, r), c = ::tvm::floormod(e, r);
				const PrimExpr row =
						::tvm::max(::tvm::min(::tvm::prim::cast(i64, tirx::BufferLoad(I, {k})), v - 1), IntImm(i64, 0));
				// (a pointer: an Expr, not a PrimExpr)
				const ::tvm::Expr address =
						::tvm::Call(O.DataPointerType(), tirx::builtin::address_of(), {tirx::BufferLoad(O, {row, c})});
				const PrimExpr atomic =
						::tvm::Call(f32, tirx::builtin::call_extern(),
									{::tvm::StringImm("atomicAdd"), address, tirx::BufferLoad(G, {k, c})})
								.as_or_throw<PrimExpr>();
				return tirx::Evaluate(atomic);
			});
			auto body = ::tvm::s_tir::SBlockRealize(
					{}, IntImm::Bool(true),
					::tvm::s_tir::SBlock("root", tirx::SeqStmt(::tvm::ffi::Array<tirx::Stmt>{zero, add}))
			);
			auto func = tirx::PrimFunc({G, I, O}, body, ::tvm::VoidType());
			func = ::tvm::WithAttr(std::move(func), "op_pattern", 8); // opaque: not fused
			return ::tvm::WithAttr(std::move(func), "tirx.is_scheduled", IntImm::Bool(true));
		}
	} // namespace

	::tvm::ffi::Function scatterAddRows() {
		return legalizeHandler([](const ::tvm::relax::BlockBuilder& bb, const ::tvm::Call& call) -> ::tvm::relax::Expr {
			const auto* out = ::tvm::relax::GetTypeAs<::tvm::relax::TensorTypeNode>(call);
			return callPrimFunc(
					bb, kernel(tensorInfo(call->args[1])->dtype.value()), "scatter_add_rows",
					{call->args[0], call->args[1]}, ::tvm::ffi::GetRef<::tvm::relax::TensorType>(out)
			);
		});
	}

} // namespace mdl::backend::tvm::gpu
