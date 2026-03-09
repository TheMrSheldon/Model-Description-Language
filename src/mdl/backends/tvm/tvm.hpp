#ifndef MVL_BACKENDS_TVM_TVM_HPP
#define MVL_BACKENDS_TVM_TVM_HPP

#include <filesystem>

namespace mdl {
	struct CompilationUnit;
	struct Namespace;
} // namespace mdl

namespace mdl::backends::tvm {
	void compileToFile(const mdl::CompilationUnit& unit, const std::filesystem::path& output);
} // namespace mdl::backends::tvm

#endif