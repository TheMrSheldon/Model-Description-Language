#ifndef MVL_BACKENDS_TVM_TVM_HPP
#define MVL_BACKENDS_TVM_TVM_HPP

#include "../../argvals.hpp"
#include <mdl/compiler.hpp>

#include <filesystem>

namespace mdl {
	struct CompilationUnit;
} // namespace mdl

namespace mdl::backends::tvm {
	void compileToFile(
			const mdl::CompilationUnit& unit,
			const std::filesystem::path& output,
			const mdl::ArgVals& argvals,
			const mdl::CompileOptions& opts = {}
	);
} // namespace mdl::backends::tvm

#endif