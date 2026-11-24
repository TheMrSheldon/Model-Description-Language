// Stand-in for PyTorch's header (see ../CUDAGeneratorImpl.h).
#pragma once
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cstdint>
#include <tuple>

namespace at::cuda::philox {
	__host__ __device__ __forceinline__ std::tuple<uint64_t, uint64_t> unpack(at::PhiloxCudaState state) {
		return {state.seed_, state.offset_};
	}
} // namespace at::cuda::philox
