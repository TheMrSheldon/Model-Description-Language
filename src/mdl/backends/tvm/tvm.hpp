#ifndef MVL_BACKENDS_TVM_TVM_HPP
#define MVL_BACKENDS_TVM_TVM_HPP

#include "../../argvals.hpp"
#include <mdl/compiler.hpp>

#include <filesystem>
#include <vector>

namespace mdl {
	struct ExportedFunction;
} // namespace mdl

namespace mdl::backends::tvm {
	/**
	 * @brief Compiles the exported functions into a relocatable object file.
	 * @details Each function is exported with the TVM FFI packed calling convention as `__tvm_ffi_<symbol>`, taking
	 * the inputs, the weights, and the preallocated outputs (see mdl::ExportedFunction).
	 */
	void compileToFile(
			const std::vector<mdl::ExportedFunction>& exports,
			const std::filesystem::path& output,
			const mdl::ArgVals& argvals,
			const mdl::CompileOptions& opts = {}
	);
} // namespace mdl::backends::tvm

#endif