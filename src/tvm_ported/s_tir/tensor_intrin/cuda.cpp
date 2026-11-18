// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/tensor_intrin/cuda.py
// (conventions: src/tvm_ported/README.md)
// The TVMScript functions (`@T.prim_func`) are built with the C++ IR builder that TVMScript itself drives
// (tvm/tirx/script/builder/ir.h): one builder call per TVMScript statement, in the same order.
#include "./cuda.hpp"

#include <tvm/ffi/dtype.h>
#include <tvm/ir/op.h>
#include <tvm/s_tir/tensor_intrin.h>
#include <tvm/script/ir_builder/base.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/script/builder/ir.h>

#include <mutex>
#include <stdexcept>
#include <vector>

namespace mdl::tvm_ported::tensor_intrin::cuda {

	namespace {
		namespace T = ::tvm::script::ir_builder::tirx;
		using ::tvm::script::ir_builder::IRBuilder;
		using ::tvm::With;
		using ::tvm::PrimType;

		PrimType dtype_of(const std::string& name) { return PrimType(ffi::StringToDLDataType(name)); }

		/** @brief A Python int in TVMScript: an int32 constant. */
		PrimExpr I(int64_t v) { return ::tvm::IntImm::Int32(v); }

		/** @brief `A[0:m, 0:n]` */
		TensorRegion slice(const tirx::BufferVar& buffer, std::vector<PrimExpr> extents) {
			Array<Range> region;
			for (const auto& e : extents)
				region.push_back(Range::FromMinExtent(I(0), e));
			return tirx::BufferRegion(buffer, region);
		}

		PrimExpr load(const tirx::BufferVar& buffer, Array<PrimExpr> indices) {
			return tirx::BufferLoad(buffer, std::move(indices)).as_or_throw<PrimExpr>();
		}

		/** @brief `@T.prim_func(s_tir=True) def <name>(...)`: @p body issues the statements of the function. */
		template <typename F>
		PrimFunc prim_func(const char* name, F&& body) {
			IRBuilder builder;
			{
				With<IRBuilder> _(builder);
				{
					With<T::PrimFuncFrame> frame(T::PrimFunc(/*is_private=*/false, /*s_tir=*/true));
					T::FuncName(name);
					body();
				}
			}
			return builder->Get<PrimFunc>();
		}

		/** @brief `T.match_buffer(param, shape, dtype, align=64, offset_factor=..., scope=..., strides=...)` */
		tirx::BufferVar match_buffer(
				const ::tvm::Var& param, std::vector<PrimExpr> shape, const std::string& dtype, int offset_factor,
				const std::string& scope, Array<PrimExpr> strides = {}
		) {
			return T::MatchBuffer(
					param, Array<PrimExpr>(shape.begin(), shape.end()), dtype_of(dtype), std::nullopt, strides,
					PrimExpr(), scope, /*align=*/64, offset_factor
			);
		}

		/** @brief The @p i-th loop variable of `T.grid(...)`. */
		PrimExpr v_(const With<T::ForFrame>& grid, size_t i) { return (*grid.get())->vars[i].as_or_throw<PrimVar>(); }

		/** @brief `T.axis.remap(kinds, bindings)` */
		std::vector<PrimExpr> remap(const char* kinds, Array<PrimExpr> bindings) {
			std::vector<PrimExpr> result;
			for (const auto& v : T::axis::Remap(kinds, bindings))
				result.push_back(v.as_or_throw<PrimVar>());
			return result;
		}

		PrimVar int32(const char* name) { return PrimVar(name, PrimType::Int(32)); }

		::tvm::Expr call(const char* op, Array<::tvm::Expr> args) {
			return ::tvm::Call(::tvm::VoidType(), ::tvm::Op::Get(op), std::move(args));
		}

		/** @brief `buffer.access_ptr("r" | "w", ptr_type=dtype)` */
		::tvm::Expr access_ptr(const tirx::BufferVar& buffer, int access_mask, const std::string& dtype) {
			return buffer.access_ptr(access_mask, ::tvm::PointerType(dtype_of(dtype)));
		}
	} // namespace

	// ######## WMMA intrinsics ########

	/** @brief Compute wmma fragment index using elem_offset of the buffer */
	PrimExpr get_wmma_fragment_index(const tirx::BufferVar& buffer, const PrimExpr& stride, int m_dim, int n_dim) {
		const PrimExpr frag_index_m = ::tvm::floordiv(::tvm::floordiv(buffer->elem_offset, stride), m_dim);
		const PrimExpr frag_index_n = ::tvm::floordiv(::tvm::floormod(buffer->elem_offset, stride), n_dim);

		const PrimExpr num_fragments_per_row = ::tvm::floordiv(stride, n_dim);
		return frag_index_m * num_fragments_per_row + frag_index_n;
	}

	/** @brief Generator of wmma_load intrins */
	Intrin get_wmma_load_intrin(
			int m_dim, int n_dim, int k_dim, const std::string& dtype, const std::string& shared_scope, bool is_b,
			bool is_col_major
	) {
		const std::string wmma_fragment_scope = std::string("wmma.matrix_") + (is_b ? "b" : "a");
		const std::string layout = is_col_major ? "col_major" : "row_major";

		auto [frag_m, frag_n] = is_b ? std::pair{k_dim, n_dim} : std::pair{m_dim, k_dim};
		if (is_col_major)
			std::swap(frag_m, frag_n);
		const int offset_factor = frag_n;

		const PrimFunc wmma_load_desc = prim_func("wmma_load_desc", [&] {
			const auto a = T::Arg("a", T::Handle()), c = T::Arg("c", T::Handle());
			const auto A = match_buffer(a, {I(frag_m), I(frag_n)}, dtype, offset_factor, shared_scope);
			const auto C = match_buffer(c, {I(frag_m), I(frag_n)}, dtype, offset_factor, wmma_fragment_scope);
			With<T::SBlockFrame> root(T::Block("root"));
			T::Reads({slice(A, {I(frag_m), I(frag_n)})});
			T::Writes({slice(C, {I(frag_m), I(frag_n)})});
			With<T::ForFrame> grid(T::Grid({I(frag_m), I(frag_n)}));
			const auto i = v_(grid, 0), j = v_(grid, 1);
			With<T::SBlockFrame> block(T::Block("load"));
			const auto v = remap("SS", {i, j});
			const PrimExpr vii = v[0], vjj = v[1];
			T::BufferStore(C, load(A, {vii, vjj}), {vii, vjj});
		});

		const PrimFunc wmma_load_impl = prim_func("wmma_load_impl", [&] {
			const auto a = T::Arg("a", T::Handle()), c = T::Arg("c", T::Handle());
			const PrimVar s1 = int32("s1"), s0 = int32("s0"), d1 = int32("d1"), d0 = int32("d0");
			const auto A = match_buffer(a, {I(frag_m), I(frag_n)}, dtype, offset_factor, shared_scope, {s1, s0});
			const auto C =
					match_buffer(c, {I(frag_m), I(frag_n)}, dtype, offset_factor, wmma_fragment_scope, {d1, d0});
			With<T::SBlockFrame> root(T::Block("root"));
			T::Reads({slice(A, {I(frag_m), I(frag_n)})});
			T::Writes({slice(C, {I(frag_m), I(frag_n)})});
			T::Evaluate(call(
					"tirx.tvm_load_matrix_sync",
					{C.data(), I(m_dim), I(n_dim), I(k_dim), get_wmma_fragment_index(C, d1, frag_m, frag_n),
					 access_ptr(A, 1, dtype), s1, ::tvm::StringImm(layout)}
			));
		});

		return {wmma_load_desc, wmma_load_impl};
	}

	/** @brief Generator of wmma_fill intrins */
	Intrin get_wmma_fill_intrin(int m_dim, int n_dim, int k_dim, const std::string& dtype) {
		const PrimExpr zero = ::tvm::prim::cast(dtype_of(dtype), ::tvm::IntImm(PrimType::Int(32), 0));
		const int offset_factor = n_dim;

		const PrimFunc wmma_fill_desc = prim_func("wmma_fill_desc", [&] {
			const auto c = T::Arg("c", T::Handle());
			const auto C = match_buffer(c, {I(m_dim), I(n_dim)}, dtype, offset_factor, "wmma.accumulator");
			With<T::SBlockFrame> root(T::Block("root"));
			T::Reads({});
			T::Writes({slice(C, {I(m_dim), I(n_dim)})});
			With<T::ForFrame> grid(T::Grid({I(m_dim), I(n_dim)}));
			const auto i = v_(grid, 0), j = v_(grid, 1);
			With<T::SBlockFrame> block(T::Block("init"));
			const auto v = remap("SS", {i, j});
			const PrimExpr vii = v[0], vjj = v[1];
			T::BufferStore(C, zero, {vii, vjj});
		});

		const PrimFunc wmma_fill_impl = prim_func("wmma_fill_impl", [&] {
			const auto c = T::Arg("c", T::Handle());
			const PrimVar d1 = int32("d1"), d0 = int32("d0");
			const auto C =
					match_buffer(c, {I(m_dim), I(n_dim)}, dtype, offset_factor, "wmma.accumulator", {d1, d0});
			With<T::SBlockFrame> root(T::Block("root"));
			T::Reads({});
			T::Writes({slice(C, {I(m_dim), I(n_dim)})});
			T::Evaluate(call(
					"tirx.tvm_fill_fragment",
					{C.data(), I(m_dim), I(n_dim), I(k_dim), get_wmma_fragment_index(C, d1, m_dim, n_dim),
					 ::tvm::FloatImm(PrimType::Float(32), 0)}
			));
		});

		return {wmma_fill_desc, wmma_fill_impl};
	}

	/** @brief Generator of wmma_store intrins */
	Intrin get_wmma_store_intrin(int m_dim, int n_dim, int k_dim, const std::string& dtype, const std::string& scope) {
		const int offset_factor = n_dim;

		const PrimFunc wmma_store_desc = prim_func("wmma_store_desc", [&] {
			const auto a = T::Arg("a", T::Handle()), c = T::Arg("c", T::Handle());
			const auto A = match_buffer(a, {I(m_dim), I(n_dim)}, dtype, offset_factor, "wmma.accumulator");
			const auto C = match_buffer(c, {I(m_dim), I(n_dim)}, dtype, offset_factor, scope);
			With<T::SBlockFrame> root(T::Block("root"));
			T::Reads({slice(A, {I(m_dim), I(n_dim)})});
			T::Writes({slice(C, {I(m_dim), I(n_dim)})});
			With<T::ForFrame> grid(T::Grid({I(m_dim), I(n_dim)}));
			const auto i = v_(grid, 0), j = v_(grid, 1);
			With<T::SBlockFrame> block(T::Block("store"));
			const auto v = remap("SS", {i, j});
			const PrimExpr vii = v[0], vjj = v[1];
			T::BufferStore(C, load(A, {vii, vjj}), {vii, vjj});
		});

		const PrimFunc wmma_store_impl = prim_func("wmma_store_impl", [&] {
			const auto a = T::Arg("a", T::Handle()), c = T::Arg("c", T::Handle());
			const PrimVar s1 = int32("s1"), s0 = int32("s0"), d1 = int32("d1"), d0 = int32("d0");
			const auto A =
					match_buffer(a, {I(m_dim), I(n_dim)}, dtype, offset_factor, "wmma.accumulator", {d1, d0});
			const auto C = match_buffer(c, {I(m_dim), I(n_dim)}, dtype, offset_factor, scope, {s1, s0});
			With<T::SBlockFrame> root(T::Block("root"));
			T::Reads({slice(A, {I(m_dim), I(n_dim)})});
			T::Writes({slice(C, {I(m_dim), I(n_dim)})});
			T::Evaluate(call(
					"tirx.tvm_store_matrix_sync",
					{A.data(), I(m_dim), I(n_dim), I(k_dim), get_wmma_fragment_index(A, d1, m_dim, n_dim),
					 access_ptr(C, 2, dtype), s1, ::tvm::StringImm("row_major")}
			));
		});

		return {wmma_store_desc, wmma_store_impl};
	}

	/** @brief Generator of wmma_sync intrins */
	Intrin get_wmma_sync_intrin(
			int m_dim, int n_dim, int k_dim, const std::string& in_dtype, const std::string& out_dtype,
			bool b_transposed
	) {
		auto maybe_cast = [&](const PrimExpr& v) -> PrimExpr {
			if (in_dtype != out_dtype)
				return ::tvm::prim::cast(dtype_of(out_dtype), v);
			return v;
		};

		auto maybe_swap = [&]<typename X>(X i, X j) -> std::pair<X, X> {
			if (b_transposed)
				return {j, i};
			return {i, j};
		};

		const auto [b_shape_0, b_shape_1] = maybe_swap(k_dim, n_dim);

		const int A_offset_factor = k_dim;
		const int B_offset_factor = b_shape_1;
		const int out_offset_factor = n_dim;

		const PrimFunc wmma_sync_desc = prim_func("wmma_sync_desc", [&] {
			const auto a = T::Arg("a", T::Handle()), b = T::Arg("b", T::Handle()), c = T::Arg("c", T::Handle());
			const auto A = match_buffer(a, {I(m_dim), I(k_dim)}, in_dtype, A_offset_factor, "wmma.matrix_a");
			const auto B = match_buffer(b, {I(b_shape_0), I(b_shape_1)}, in_dtype, B_offset_factor, "wmma.matrix_b");
			const auto C = match_buffer(c, {I(m_dim), I(n_dim)}, out_dtype, out_offset_factor, "wmma.accumulator");

			With<T::SBlockFrame> root(T::Block("root"));
			T::Reads(
					{slice(C, {I(m_dim), I(n_dim)}), slice(A, {I(m_dim), I(k_dim)}),
					 slice(B, {I(b_shape_0), I(b_shape_1)})}
			);
			T::Writes({slice(C, {I(m_dim), I(n_dim)})});
			With<T::ForFrame> grid(T::Grid({I(m_dim), I(n_dim), I(k_dim)}));
			const auto i = v_(grid, 0), j = v_(grid, 1), k = v_(grid, 2);
			With<T::SBlockFrame> block(T::Block(""));
			const auto v = remap("SSR", {i, j, k});
			const PrimExpr vii = v[0], vjj = v[1], vkk = v[2];
			const auto [B_index_0, B_index_1] = maybe_swap(vkk, vjj);
			T::BufferStore(
					C,
					load(C, {vii, vjj}) + maybe_cast(load(A, {vii, vkk})) * maybe_cast(load(B, {B_index_0, B_index_1})),
					{vii, vjj}
			);
		});

		const PrimFunc wmma_sync_impl = prim_func("wmma_sync_impl", [&] {
			const auto a = T::Arg("a", T::Handle()), b = T::Arg("b", T::Handle()), c = T::Arg("c", T::Handle());
			const PrimVar a1 = int32("a1"), a0 = int32("a0"), b1 = int32("b1"), b0 = int32("b0"), c1 = int32("c1"),
						  c0 = int32("c0");

			const auto A =
					match_buffer(a, {I(m_dim), I(k_dim)}, in_dtype, A_offset_factor, "wmma.matrix_a", {a1, a0});
			const auto B = match_buffer(
					b, {I(b_shape_0), I(b_shape_1)}, in_dtype, B_offset_factor, "wmma.matrix_b", {b1, b0}
			);
			const auto C = match_buffer(
					c, {I(m_dim), I(n_dim)}, out_dtype, out_offset_factor, "wmma.accumulator", {c1, c0}
			);

			With<T::SBlockFrame> root(T::Block("root"));
			T::Reads(
					{slice(C, {I(m_dim), I(n_dim)}), slice(A, {I(m_dim), I(k_dim)}),
					 slice(B, {I(b_shape_0), I(b_shape_1)})}
			);
			T::Writes({slice(C, {I(m_dim), I(n_dim)})});
			T::Evaluate(call(
					"tirx.tvm_mma_sync",
					{C.data(), get_wmma_fragment_index(C, c1, m_dim, n_dim), A.data(),
					 get_wmma_fragment_index(A, a1, m_dim, k_dim), B.data(),
					 get_wmma_fragment_index(B, b1, b_shape_0, b_shape_1), C.data(),
					 get_wmma_fragment_index(C, c1, m_dim, n_dim)}
			));
		});

		return {wmma_sync_desc, wmma_sync_impl};
	}

	void register_wmma_intrinsics() {
		static std::once_flag once;
		std::call_once(once, [] {
			auto reg = [](const std::string& name, const Intrin& intrin) {
				s_tir::TensorIntrin::Register(name, s_tir::TensorIntrin(intrin.desc, intrin.impl), /*override=*/true);
			};
			reg("wmma_sync_16x16x16_f16f16f32", get_wmma_sync_intrin(16, 16, 16, "float16", "float32", false));
			reg("wmma_sync_16x16x16_f16f16f32_trans", get_wmma_sync_intrin(16, 16, 16, "float16", "float32", true));
			reg("wmma_sync_16x16x16_f16f16f16", get_wmma_sync_intrin(16, 16, 16, "float16", "float16", false));
			reg("wmma_sync_16x16x16_f16f16f16_trans", get_wmma_sync_intrin(16, 16, 16, "float16", "float16", true));
			reg("wmma_sync_16x16x16_s8s8s32", get_wmma_sync_intrin(16, 16, 16, "int8", "int32", false));
			reg("wmma_sync_16x16x16_s8s8s32_trans", get_wmma_sync_intrin(16, 16, 16, "int8", "int32", true));
			reg("wmma_sync_8x8x32_s4s4s32_trans", get_wmma_sync_intrin(8, 8, 32, "int4", "int32", true));
			for (const char* scope : {"shared", "shared.dyn"}) {
				const std::string s = std::string(scope) == "shared" ? "shared" : "shared_dyn";
				reg("wmma_load_16x16x16_f16_a_" + s, get_wmma_load_intrin(16, 16, 16, "float16", scope, false, false));
				reg("wmma_load_16x16x16_f16_b_" + s, get_wmma_load_intrin(16, 16, 16, "float16", scope, true, false));
				reg("wmma_load_16x16x16_f16_a_trans_" + s,
					get_wmma_load_intrin(16, 16, 16, "float16", scope, false, true));
				reg("wmma_load_16x16x16_f16_b_trans_" + s,
					get_wmma_load_intrin(16, 16, 16, "float16", scope, true, true));
				reg("wmma_load_16x16x16_s8_a_" + s, get_wmma_load_intrin(16, 16, 16, "int8", scope, false, false));
				reg("wmma_load_16x16x16_s8_b_" + s, get_wmma_load_intrin(16, 16, 16, "int8", scope, true, false));
				reg("wmma_load_16x16x16_s8_a_trans_" + s, get_wmma_load_intrin(16, 16, 16, "int8", scope, false, true));
				reg("wmma_load_16x16x16_s8_b_trans_" + s, get_wmma_load_intrin(16, 16, 16, "int8", scope, true, true));
				reg("wmma_load_8x8x32_s4_a_" + s, get_wmma_load_intrin(8, 8, 32, "int4", scope, false, false));
				reg("wmma_load_8x8x32_s4_b_trans_" + s, get_wmma_load_intrin(8, 8, 32, "int4", scope, true, true));
			}
			reg("wmma_fill_16x16x16_f32", get_wmma_fill_intrin(16, 16, 16, "float32"));
			reg("wmma_fill_16x16x16_f16", get_wmma_fill_intrin(16, 16, 16, "float16"));
			reg("wmma_fill_16x16x16_s32", get_wmma_fill_intrin(16, 16, 16, "int32"));
			reg("wmma_fill_8x8x32_s32", get_wmma_fill_intrin(8, 8, 32, "int32"));
			for (const char* scope : {"shared", "shared.dyn", "global"}) {
				const std::string s = std::string(scope) == "shared.dyn" ? "shared_dyn" : scope;
				reg("wmma_store_16x16x16_f32_" + s, get_wmma_store_intrin(16, 16, 16, "float32", scope));
				reg("wmma_store_16x16x16_f16_" + s, get_wmma_store_intrin(16, 16, 16, "float16", scope));
				reg("wmma_store_16x16x16_s32_" + s, get_wmma_store_intrin(16, 16, 16, "int32", scope));
				reg("wmma_store_8x8x32_s32_" + s, get_wmma_store_intrin(8, 8, 32, "int32", scope));
			}
		});
	}

	std::map<std::string, std::string> get_wmma_intrin_group(
			const std::string& load_scope_, const std::string& store_scope_, const std::string& in_dtype_,
			const std::string& out_dtype_, bool trans_b_
	) {
		if (load_scope_ != "shared" && load_scope_ != "shared.dyn")
			throw std::invalid_argument("get_wmma_intrin_group: load_scope");
		if (store_scope_ != "global" && store_scope_ != "shared" && store_scope_ != "shared.dyn")
			throw std::invalid_argument("get_wmma_intrin_group: store_scope");
		if (in_dtype_ != "float16" && in_dtype_ != "int8")
			throw std::invalid_argument("get_wmma_intrin_group: in_dtype");
		if (out_dtype_ != "float16" && out_dtype_ != "float32" && out_dtype_ != "int32")
			throw std::invalid_argument("get_wmma_intrin_group: out_dtype");

		const std::string shape = "16x16x16";
		const std::string in_dtype = in_dtype_ == "float16" ? "f16" : "s8";
		const std::string out_dtype = out_dtype_ == "float16" ? "f16" : out_dtype_ == "float32" ? "f32" : "s32";
		// convert "shared.dyn" to "shared_dyn"
		auto underscored = [](std::string s) {
			std::replace(s.begin(), s.end(), '.', '_');
			return s;
		};
		const std::string load_scope = underscored(load_scope_), store_scope = underscored(store_scope_);
		const std::string trans_a;
		const std::string trans_b = trans_b_ ? "_trans" : "";

		// e.g. wmma_load_16x16x16_f16_a_shared
		const std::string load_a_intrin = "wmma_load_" + shape + "_" + in_dtype + "_a" + trans_a + "_" + load_scope;
		// e.g. wmma_load_16x16x16_f16_b_trans_shared_dyn
		const std::string load_b_intrin = "wmma_load_" + shape + "_" + in_dtype + "_b" + trans_b + "_" + load_scope;
		// e.g. wmma_sync_16x16x16_f16f16f32_trans
		const std::string compute_intrin = "wmma_sync_" + shape + "_" + in_dtype + in_dtype + out_dtype + trans_b;
		// e.g. wmma_fill_16x16x16_f16
		const std::string init_intrin = "wmma_fill_" + shape + "_" + out_dtype;
		// e.g. wmma_store_16x16x16_f16_shared_dyn
		const std::string store_intrin = "wmma_store_" + shape + "_" + out_dtype + "_" + store_scope;

		return {
				{"init", init_intrin},
				{"load_a", load_a_intrin},
				{"load_b", load_b_intrin},
				{"compute", compute_intrin},
				{"store", store_intrin},
		};
	}

} // namespace mdl::tvm_ported::tensor_intrin::cuda
