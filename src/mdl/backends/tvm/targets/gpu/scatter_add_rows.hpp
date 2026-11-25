#pragma once

#include <tvm/ffi/function.h>

namespace mdl::backend::tvm::gpu {

	/**
	 * @brief The legalization of mdl.scatter_add_rows for GPUs: a zero fill, then one thread per element of the added
	 * rows, which adds it with atomicAdd (repeated indices accumulate in any order). Two thread-bound loop nests in
	 * one PrimFunc (marked as scheduled), which become two kernels launched in order.
	 */
	::tvm::ffi::Function scatterAddRows();

} // namespace mdl::backend::tvm::gpu
