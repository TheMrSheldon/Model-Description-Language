#pragma once

/**
 * @file aot.hpp
 * @brief Ahead-of-time lowering of Relax functions into TIR entry functions (no Relax VM).
 */

#include <tvm/ir/module.h>

#include <string>

namespace mdl::backend::tvm {

	/** @brief Attribute that marks the entry functions created by lowerToAOT (already flat, block-free TIR). */
	inline constexpr const char* kAOTEntryAttr = "mdl.aot_entry";

	/**
	 * @brief Replaces every exported Relax function by a TIR entry function that calls the kernels directly.
	 * @details Expects the Relax functions in the form produced by CallTIRRewrite and StaticPlanBlockMemory: a sequence
	 * of allocations (`relax.builtin.alloc_tensor`, `relax.memory.alloc_storage/alloc_tensor`), reshapes, and calls of
	 * PrimFuncs. The entry function with global symbol `S` (exported as `__tvm_ffi_S` by MakePackedAPI) takes the
	 * Relax parameters followed by one preallocated buffer per output (destination passing style). Intermediate tensors
	 * live in workspace allocations, and kernels are called through their packed C symbols.
	 *
	 * @param kernelPrefix prefix for the global symbols of the kernels (to avoid clashes between compiled models).
	 * @return A module that only contains PrimFuncs.
	 */
	::tvm::IRModule lowerToAOT(::tvm::IRModule mod, const std::string& kernelPrefix);

} // namespace mdl::backend::tvm
