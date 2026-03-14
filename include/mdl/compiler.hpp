#ifndef MDL_COMPILER_HPP
#define MDL_COMPILER_HPP

#include <filesystem>

namespace mdl {
	struct CompilationUnit;

	void compileToFile(const CompilationUnit& unit, const std::filesystem::path& output);
} // namespace mdl

#endif