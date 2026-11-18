#pragma once

/**
 * @file target.hpp
 * @brief The devices the TVM backend compiles for. Everything that depends on the device (the TVM target, the schedule
 * of the TVM-generated kernels, the device-specific TIR passes, code generation, and which artifacts of kernel packages
 * can be used) is behind this interface; the rest of the compiler is device-independent.
 *
 * Targets are created by name through the registry (see registry.hpp): `--target llvm`, `--target cuda -arch=sm_80`.
 */

#include <mdl/compiler.hpp>
#include <mdl/kernels.hpp>

#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mdl::backend::tvm {

	class Target {
	public:
		virtual ~Target() = default;

		/** @brief The TVM target of the kernels, with the host CPU as its host. */
		virtual const ::tvm::Target& tvm() const = 0;

		/** @brief What an artifact of a kernel package must run on (see kernels::runsOn). */
		virtual kernels::Platform kernelPlatform() const = 0;

		/** @brief Schedules the TVM-generated kernels (PrimFuncs with blocks) of @p kernels for this device. */
		virtual ::tvm::IRModule schedule(::tvm::IRModule kernels) const = 0;

		/** @brief The passes that lower the scheduled kernels' blocks into loops (before buffers are flattened). */
		virtual std::vector<::tvm::transform::Pass> blockLowering() const = 0;

		/** @brief Device-specific passes after vectorization, before storage rewriting (virtual threads, ...). */
		virtual std::vector<::tvm::transform::Pass> afterVectorize() const { return {}; }

		/** @brief Device-specific passes before host and device code are split (thread synchronization, ...). */
		virtual std::vector<::tvm::transform::Pass> beforeSplit() const { return {}; }

		/**
		 * @brief Finalizes the lowered module (after MakePackedAPI) and writes it as one object file.
		 * @param mod host functions (the AOT entries, the kernels' launchers) and, for devices, the device kernels.
		 */
		virtual void emit(::tvm::IRModule mod, const std::filesystem::path& output) const = 0;
	};

	/**
	 * @brief The TVM target of a target string: a JSON config, a tag (`nvidia/nvidia-a100`), or a kind with options
	 * (`cuda -arch=sm_80`, which TVM itself no longer parses).
	 */
	::tvm::Target parseTarget(const std::string& spec);

	/** @brief A pass registered with TVM by name (passes without a C++ header). */
	template <typename... Args>
	::tvm::transform::Pass registeredPass(const std::string& name, Args&&... args) {
		auto factory = ::tvm::ffi::Function::GetGlobal(name);
		if (!factory)
			throw std::runtime_error("TVM pass not found: " + name);
		return (*factory)(std::forward<Args>(args)...).template cast<::tvm::transform::Pass>();
	}

	/** @brief Runs @p passes on @p mod with @p target as the current target. */
	::tvm::IRModule
	runPasses(::tvm::IRModule mod, const std::vector<::tvm::transform::Pass>& passes, const ::tvm::Target& target);

} // namespace mdl::backend::tvm
