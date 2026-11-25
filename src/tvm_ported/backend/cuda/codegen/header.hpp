// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/backend/cuda/codegen/header.py
// (conventions: src/tvm_ported/README.md)
#pragma once

/**
 * @file header.hpp
 * @brief The header generator is used to generate the header for the CUDA code. It's controlled by the predefined
 * tags. The tags are used to identify the utility functions/classes necessary for the codegen.
 */

#include <string>
#include <vector>

namespace mdl::tvm_ported::backend::cuda::codegen {

	/** @brief The predefined tags. */
	const std::vector<std::string>& TAGS();

	/**
	 * @brief Generate the header for the CUDA code (TVM's global `tirx.intrinsics.cuda.header_generator`).
	 * @throws std::invalid_argument for an unknown tag.
	 */
	std::string header_generator(const std::vector<std::string>& tags);

} // namespace mdl::tvm_ported::backend::cuda::codegen
