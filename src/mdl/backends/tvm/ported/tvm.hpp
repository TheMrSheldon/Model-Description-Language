#pragma once

/**
 * @file tvm.hpp
 * @brief The TVM names the port uses, under the names of TVM's Python API (part of the port, not of TVM's DLight).
 */

#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/sym/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/index_map.h>
#include <tvm/tirx/stmt.h>

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

namespace mdl::tvm_ported {

	namespace ffi = ::tvm::ffi;
	namespace tirx = ::tvm::tirx;
	namespace s_tir = ::tvm::s_tir;

	using ::tvm::IRModule;
	using ::tvm::PrimExpr;
	using ::tvm::PrimVar;
	using ::tvm::Range;
	using ::tvm::Target;
	using ::tvm::TensorRegion;
	using ffi::Array;
	using s_tir::LoopRV;
	using s_tir::SBlockRV;
	using s_tir::Schedule;
	using tirx::PrimFunc;

	/** @brief The value of a constant integer (Python's `isinstance(e, tirx.IntImm)` and `e.value`). */
	inline std::optional<int64_t> as_int(const PrimExpr& e) {
		if (const auto* imm = e.as<::tvm::IntImmNode>())
			return static_cast<int64_t>(imm->value);
		return std::nullopt;
	}

	/**
	 * @brief Split factors as Python writes them (`[None, 32]`): std::nullopt for None, integers as int32 constants
	 * (as Python's conversion of ints).
	 */
	inline Array<ffi::Optional<PrimExpr>> factors(std::initializer_list<std::optional<int64_t>> values) {
		Array<ffi::Optional<PrimExpr>> result;
		for (const auto& v : values)
			result.push_back(v ? ffi::Optional<PrimExpr>(::tvm::IntImm::Int32(*v)) : ffi::Optional<PrimExpr>());
		return result;
	}

	/** @brief The storage scope of a buffer (Python's `buffer.scope()`: "global" if unset). */
	inline std::string scope_of(const tirx::BufferVar& buffer) {
		const std::string scope = buffer->storage_scope;
		return scope.empty() ? "global" : scope;
	}

	/** @brief The buffer of a buffer region (Python's `region.source`). */
	inline tirx::BufferVar source_of(const TensorRegion& region) {
		return region->source.as_or_throw<tirx::BufferVar>();
	}

} // namespace mdl::tvm_ported
