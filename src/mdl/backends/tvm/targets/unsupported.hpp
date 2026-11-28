#pragma once

/**
 * @file unsupported.hpp
 * @brief The stand-in for a target that this build does not include (e.g. `using CUDATarget = UnsupportedTarget<...>`
 * without MDL_ENABLE_CUDA): it stays registered, so that the error says how to get it instead of "not found".
 */

#include "./target.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace mdl::backend::tvm {

	/** @brief A string literal as a template argument. */
	template <std::size_t N>
	struct Literal {
		char value[N]{};
		constexpr Literal(const char (&s)[N]) { std::copy_n(s, N, value); } // NOLINT(google-explicit-constructor)
	};

	/**
	 * @brief A target that is not part of this build: creating it fails.
	 * @tparam Kind the target's kind (`cuda`).
	 * @tparam Feature the CMake option that includes it (`MDL_ENABLE_CUDA`).
	 */
	template <Literal Kind, Literal Feature>
	class UnsupportedTarget final : public Target {
	public:
		explicit UnsupportedTarget(const mdl::CompileOptions& /*opts*/) {
			throw std::runtime_error(
					std::string("target '") + Kind.value +
					"' is not supported by this build of mdlc (build it with -D" + Feature.value + "=ON)"
			);
		}

		const ::tvm::Target& tvm() const override { unreachable(); }
		kernels::Platform kernelPlatform() const override { unreachable(); }
		::tvm::IRModule schedule(::tvm::IRModule) const override { unreachable(); }
		std::vector<::tvm::transform::Pass> blockLowering() const override { unreachable(); }
		void emit(::tvm::IRModule, const std::filesystem::path&) const override { unreachable(); }

	private:
		[[noreturn]] static void unreachable() { throw std::logic_error("UnsupportedTarget used"); }
	};

} // namespace mdl::backend::tvm
