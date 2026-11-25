// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/backend/cuda/codegen/registry.py
// (conventions: src/tvm_ported/README.md)
#pragma once

/**
 * @file registry.hpp
 * @brief The registry of codegen functions for CUDA device intrinsics (TVM's global `tirx.intrinsics.cuda.get_codegen`).
 * @details In TVM, the modules python/tvm/backend/cuda/cpp/*.py and ptx/*.py register codegens for the TIRX device
 * intrinsics (TMA, async copies, descriptors, ...). They are not ported: the registry is empty, and CodeGenCUDA
 * generates all other operations (including WMMA tensor cores) itself.
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <map>
#include <string>

namespace mdl::tvm_ported::backend::cuda::codegen {

	/** @brief CODEGEN_REGISTRY: codegen functions by op name. */
	std::map<std::string, ::tvm::ffi::Function>& CODEGEN_REGISTRY();

	/** @brief Get the codegen function for a given op (empty if none). */
	::tvm::ffi::Optional<::tvm::ffi::Function> get_codegen(const std::string& op);

} // namespace mdl::tvm_ported::backend::cuda::codegen
