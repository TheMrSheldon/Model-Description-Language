#pragma once

/**
 * @file cuda.hpp
 * @brief NVIDIA GPUs: device code through TVM's CUDA C code generation (compiled with NVRTC), with the host CPU as the host. Without MDL_ENABLE_CUDA, the target is
 * registered but unsupported.
 */

#include "./target.hpp"
#include "./unsupported.hpp"

namespace mdl::backend::tvm {

#if MDL_WITH_CUDA
	class CUDATarget final : public Target {
	public:
		/** @param opts CompileOptions::target: `cuda -arch=sm_XY` (the architecture is required), a tag, or JSON. */
		explicit CUDATarget(const mdl::CompileOptions& opts);

		const ::tvm::Target& tvm() const override { return target; }
		kernels::Platform kernelPlatform() const override;
		::tvm::IRModule schedule(::tvm::IRModule kernels) const override;
		std::vector<::tvm::transform::Pass> blockLowering() const override;
		std::vector<::tvm::transform::Pass> afterVectorize() const override;
		std::vector<::tvm::transform::Pass> beforeSplit() const override;
		void emit(::tvm::IRModule mod, const std::filesystem::path& output) const override;

	private:
		::tvm::Target target;
		std::string arch;					///< sm_XY
		std::filesystem::path dumpDir; ///< CompileOptions::dumpDir: the generated CUDA source is written there
	};
#else
	using CUDATarget = UnsupportedTarget<"cuda", "MDL_ENABLE_CUDA">;
#endif

} // namespace mdl::backend::tvm
