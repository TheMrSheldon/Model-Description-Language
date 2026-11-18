// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/support/nvcc.py
// (conventions: src/mdl/backends/tvm/ported/README.md)
#pragma once

/**
 * @file nvcc.hpp
 * @brief Utility to invoke nvcc compiler in the system.
 * @details Ported: the NVRTC path of compile_cuda (cubin and ptx) and tvm_callback_cuda_compile. Not ported: the nvcc
 * subprocess path (TVM_CUDA_COMPILE_MODE=nvcc, fatbin), NVSHMEM linking, and the version/feature queries.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mdl::tvm_ported::support::nvcc {

	/**
	 * @brief Compile CUDA code with NVRTC.
	 * @param target_format "cubin" (default) or "ptx".
	 * @param arch the CUDA architecture (sm_80 or compute_80); default: the "arch" of the current TVM target.
	 * @return the compiled binary.
	 */
	std::string compile_cuda(
			const std::string& code, std::optional<std::string> target_format = std::nullopt,
			std::optional<std::string> arch = std::nullopt, const std::vector<std::string>& options = {}
	);

	/** @brief Find the CUDA root (CUDA_PATH, the location of nvcc, /usr/local/cuda). */
	std::string find_cuda_path();

	/** @brief Compile CUDA code (TVM's global `tvm_callback_cuda_compile`; NVRTC, cubin). */
	std::string tvm_callback_cuda_compile(const std::string& code);

} // namespace mdl::tvm_ported::support::nvcc
