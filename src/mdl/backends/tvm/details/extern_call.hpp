#pragma once

/**
 * @file extern_call.hpp
 * @brief Calls of extern (C) kernels from Relax: a PrimFunc that `call_extern`s the kernel with buffer pointers and
 * scalars, called through `call_tir` (used by the legalizer's vendor kernels and by kernel packages).
 */

#include "./tvm_compat.hpp"

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mdl::backend::tvm {

	using ::tvm::Call;
	using ::tvm::Expr;
	using ::tvm::PrimExpr;
	using ::tvm::ffi::Array;
	using ::tvm::relax::BlockBuilder;
	using ::tvm::relax::ShapeExpr;
	using ::tvm::relax::TensorType;
	namespace tirx = ::tvm::tirx;

	/**
	 * @brief The symbolic variables in @p shapes that no shape determines directly (as a whole dimension), e.g. `B0`
	 * and `B1` of a flattened dimension `B0 * B1`. A kernel cannot recover them from its buffers, so they are passed
	 * as extra (scalar) arguments.
	 */
	inline Array<::tvm::Var>
	unboundVars(const std::vector<Array<PrimExpr>>& shapes, const std::vector<PrimExpr>& scalars = {}) {
		std::vector<const tirx::VarNode*> direct;
		for (const auto& shape : shapes)
			for (const auto& d : shape)
				if (const auto* v = d.as<tirx::VarNode>())
					direct.push_back(v);
		std::vector<Array<PrimExpr>> all = shapes;
		all.emplace_back(scalars.begin(), scalars.end()); // e.g. a runtime dimension as a divisor
		Array<::tvm::Var> unbound;
		for (const auto& shape : all)
			for (const auto& d : shape)
				for (const auto& v : tirx::UndefinedVars(d))
					if (std::find(direct.begin(), direct.end(), v.get()) == direct.end() &&
						std::find_if(unbound.begin(), unbound.end(), [&](const ::tvm::Var& u) {
							return u.same_as(v);
						}) == unbound.end())
						unbound.push_back(v);
		return unbound;
	}

	/**
	 * @brief `call_tir(gv, (inputs..., vars...), out)`: the unbound symbolic variables are scalar arguments after
	 * the inputs (the PrimFunc's parameters: inputs, variables, output).
	 */
	inline Expr
	callTir(const ::tvm::GlobalVar& gv, const Array<Expr>& inputs, const ::tvm::Type& out,
			const Array<::tvm::Var>& vars) {
		static const auto& op = ::tvm::Op::Get("relax.call_tir");
		Array<Expr> args = inputs;
		for (const auto& v : vars)
			args.push_back(v);
		return Call(::tvm::Type::Missing(), op, {gv, ::tvm::relax::Tuple(args)}, ::tvm::Attrs(), {out});
	}

	/** @brief Builds a PrimFunc that calls an extern kernel and emits a `call_tir` to it. */
	class ExternCall {
	public:
		ExternCall(BlockBuilder bb, std::string kernel) : bb(std::move(bb)), kernel(std::move(kernel)) {}

		/** @brief Adds a tensor argument; returns the pointer expression to pass to the kernel. */
		Expr input(const Expr& arg, const std::string& argName) {
			inputs.push_back(arg);
			auto shape = knownShape(arg);
			if (!shape)
				throw std::runtime_error("mdl backend: expected a tensor with known shape");
			return buffer(*shape, tensorInfo(arg)->dtype.value(), argName, /*write=*/false);
		}

		/** @brief Adds an output (several: the call's result is a tuple); returns its pointer expression. */
		Expr output(const Array<PrimExpr>& shape, ::tvm::PrimType dtype) {
			outInfos.push_back(TensorType(ShapeExpr(shape), dtype));
			auto ptr =
					buffer(shape, dtype, outInfos.size() == 1 ? "out" : "out" + std::to_string(outInfos.size() - 1),
						   true);
			outputBuffers.push_back(params.back());
			return ptr;
		}

		/**
		 * @brief An int64 array with @p values on the caller's stack (an `int64[]` argument, e.g. the shape of a strided
		 * view); returns its pointer expression.
		 */
		Expr array(const Array<PrimExpr>& values) {
			const auto i64 = ::tvm::PrimType::Int(64);
			const auto n = ::tvm::IntImm(i64, std::max<int64_t>(static_cast<int64_t>(values.size()), 1));
			auto buffer = tirx::BufferVar(
					"array" + std::to_string(prologue.size()),
					tirx::BufferType("", i64, {n}, {}, ::tvm::IntImm(i64, 0), 0, 0)
			);
			prologue.push_back(tirx::DeclBuffer(
					buffer,
					Call(::tvm::PointerType(i64), tirx::builtin::tvm_stack_alloca(), {::tvm::StringImm("shape"), n})
			));
			for (size_t i = 0; i < values.size(); ++i)
				prologue.push_back(tirx::BufferStore(
						buffer, ::tvm::prim::cast(i64, values[i]), {::tvm::IntImm(i64, static_cast<int64_t>(i))}
				));
			return buffer.data();
		}

		/** @param scalars the integer arguments (their symbolic variables become parameters if no buffer binds them) */
		Expr finish(std::vector<Expr> kernelArgs, const std::vector<PrimExpr>& scalars = {}) {
			// The parameters: the inputs, the unbound symbolic variables, then the output (destination passing style).
			std::vector<::tvm::Var> ordered;
			for (const auto& p : params)
				if (std::none_of(outputBuffers.begin(), outputBuffers.end(), [&](const auto& o) {
						return o.same_as(p);
					}))
					ordered.push_back(p);
			auto vars = unboundVars(shapes, scalars);
			ordered.insert(ordered.end(), vars.begin(), vars.end());
			ordered.insert(ordered.end(), outputBuffers.begin(), outputBuffers.end());
			Array<Expr> callArgs{::tvm::StringImm(kernel)};
			for (auto& arg : kernelArgs)
				callArgs.push_back(std::move(arg));
			auto call = Call(::tvm::PrimType::Int(32), tirx::builtin::call_extern(), callArgs).as_or_throw<PrimExpr>();
			auto check = tirx::AssertStmt(
					call == 0, ::tvm::StringImm("RuntimeError"), {::tvm::StringImm("mdl kernel " + kernel + " failed")}
			);
			std::vector<tirx::Stmt> stmts = prologue;
			stmts.push_back(check);
			// Wrap the call in a root block like any other (TE-generated) PrimFunc; Relax analyses expect it.
			auto body = ::tvm::s_tir::SBlockRealize(
					{}, ::tvm::IntImm::Bool(true),
					::tvm::s_tir::SBlock("root", tirx::SeqStmt::Flatten(Array<tirx::Stmt>(stmts.begin(), stmts.end())))
			);
			auto func = tirx::PrimFunc(Array<::tvm::Var>(ordered.begin(), ordered.end()), body, ::tvm::VoidType());
			// Opaque (kOpaque = 8): extern calls must not be fused with other kernels.
			func = ::tvm::WithAttr(std::move(func), "op_pattern", 8);
			auto gv = bb->AddFunction(func, kernel);
			if (outInfos.size() == 1)
				return callTir(gv, inputs, outInfos.front(), vars);
			return callTir(gv, inputs, ::tvm::TupleType(Array<::tvm::Type>(outInfos.begin(), outInfos.end())), vars);
		}

	private:
		BlockBuilder bb;
		std::string kernel;
		Array<Expr> inputs;
		std::vector<::tvm::Var> params; ///< the buffers
		std::vector<Array<PrimExpr>> shapes;
		std::vector<tirx::Stmt> prologue; ///< the arrays on the stack
		std::vector<TensorType> outInfos;
		std::vector<::tvm::Var> outputBuffers;

		Expr buffer(const Array<PrimExpr>& shape, ::tvm::PrimType dtype, const std::string& n, bool write) {
			auto buf = tirx::decl_buffer(shape, dtype, n);
			shapes.push_back(shape);
			params.push_back(buf);
			(void
			)write; // (the kernels get the plain data pointer: tvm_access_ptr carries a dtype marker that the bfloat16
			// legalization would rewrite)
			return buf.data();
		}
	};

} // namespace mdl::backend::tvm
