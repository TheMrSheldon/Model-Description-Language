#pragma once

/**
 * @file aot.hpp
 * @brief Ahead-of-time lowering of Relax functions into TIR entry functions (no Relax VM).
 */

#include <tvm/ir/module.h>
#include <tvm/tirx/var.h>

#include <string>
#include <unordered_map>
#include <vector>

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
	/**
	 * @brief How the arguments of an exported function are passed (see mdl::ExportedFunction).
	 * @details Packed tensors (see mdl::ContextBuffer) are not arguments of the entry function; instead, it takes one
	 * more argument, a 1-D uint8 buffer (the context), in which the packed tensors are laid out in order at 64-byte
	 * aligned offsets that are computed from their (symbolic) shapes. Producer and consumer of a context list the same
	 * tensors in the same order, so both compute the same layout. Accumulated outputs are added to the caller's buffer.
	 */
	struct EntrySignature {
		std::vector<bool> params;	  ///< per Relax parameter: packed
		std::vector<bool> outputs;	  ///< per returned tensor: packed
		std::vector<bool> accumulate; ///< per returned tensor: added to the output buffer
		/**
		 * @brief Per Relax parameter: the index of the returned tensor whose output buffer it is (the previous value of
		 * an accumulated output, updated in place), or -1 for an argument of the entry.
		 */
		std::vector<int> aliases;
		bool context = false;		  ///< whether the function takes a context buffer (as its last argument)
		std::string contextName;	  ///< the context buffer's name (shared by its producer and consumers)
		/** @brief For the producer: global symbol of the size function to emit (`dims... -> bytes`); else empty. */
		std::string sizeSymbol;
		/**
		 * @brief The symbolic dimensions of `...` (the arguments of the size function). The producer writes them into
		 * the header of the context buffer; the consumer reads those that its other arguments do not determine.
		 */
		::tvm::ffi::Array<::tvm::PrimVar> dims;
	};

	/** @param packing the packed tensors of the exported functions by global symbol (see EntrySignature). */
	/** @param profile whether to call mdl_profile_begin/end around every kernel call. */
	::tvm::IRModule lowerToAOT(
			::tvm::IRModule mod, const std::string& kernelPrefix,
			const std::unordered_map<std::string, EntrySignature>& packing = {}, bool profile = false
	);

} // namespace mdl::backend::tvm
