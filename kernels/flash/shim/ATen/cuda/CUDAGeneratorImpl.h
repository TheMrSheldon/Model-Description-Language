// Stand-in for PyTorch's header (FlashAttention's sources include it for the dropout generator state). Our build has no
// dropout (FLASHATTENTION_DISABLE_DROPOUT): the state is never read.
#pragma once
#include <cstdint>

namespace at {
	struct PhiloxCudaState {
		uint64_t seed_ = 0;
		uint64_t offset_ = 0;
	};
} // namespace at
