#pragma once

/**
 * @file globals.hpp
 * @brief Registers the ported functions under the names that TVM's C++ looks up (in Python, the decorators
 * `register_global_func`/`TensorIntrin.register` do this when the modules are imported). Not a translation.
 */

namespace mdl::tvm_ported {

	/**
	 * @brief The callbacks of TVM's CUDA C code generation and the CUDA tensor intrinsics:
	 * - `tirx.intrinsics.cuda.header_generator` (backend/cuda/codegen/header.cpp)
	 * - `tirx.intrinsics.cuda.get_codegen` (backend/cuda/codegen/registry.cpp)
	 * - `tvm_callback_cuda_compile` (support/nvcc.cpp)
	 * - the WMMA tensor intrinsics (s_tir/tensor_intrin/cuda.cpp)
	 *
	 * Idempotent; functions registered by someone else (e.g. TVM's Python package) are kept.
	 */
	void register_cuda_globals();

} // namespace mdl::tvm_ported
