// Stand-in for PyTorch's header: FlashAttention's launch code checks CUDA errors with these macros. They throw
// mdl_flash::CudaError, which the C ABI (src/flash.cu) turns into a non-zero return value.
#pragma once
#include <cuda_runtime.h>

#include <cstdio> // (PyTorch's header provides these; FlashAttention relies on them, e.g. fprintf)
#include <cstdlib>
#include <stdexcept>

namespace mdl_flash {
	struct CudaError {
		cudaError_t status;
	};
} // namespace mdl_flash

#define C10_CUDA_CHECK(expr)                                                                                          \
	do {                                                                                                              \
		const cudaError_t mdl_status_ = (expr);                                                                       \
		if (mdl_status_ != cudaSuccess)                                                                               \
			throw ::mdl_flash::CudaError{mdl_status_};                                                                \
	} while (false)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
