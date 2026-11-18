#pragma once

/** @file llvm.hpp @brief The host CPU (TVM's llvm target): the default target. */

#include "./target.hpp"

namespace mdl::backend::tvm {

	class LLVMTarget final : public Target {
	public:
		explicit LLVMTarget(const mdl::CompileOptions& opts);

		/** @brief The LLVM target of the host CPU (also the host of device targets). */
		static ::tvm::Target host(const mdl::CompileOptions& opts);

		/** @brief Finalizes host functions (lowers TVM builtins and intrinsics). */
		static std::vector<::tvm::transform::Pass> finalizeHost();

		const ::tvm::Target& tvm() const override { return target; }
		kernels::Platform kernelPlatform() const override { return kernels::hostPlatform(); }
		::tvm::IRModule schedule(::tvm::IRModule kernels) const override;
		std::vector<::tvm::transform::Pass> blockLowering() const override;
		void emit(::tvm::IRModule mod, const std::filesystem::path& output) const override;

	private:
		::tvm::Target target;
	};

} // namespace mdl::backend::tvm
