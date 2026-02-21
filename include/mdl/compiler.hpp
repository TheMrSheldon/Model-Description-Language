#ifndef MDL_COMPILER_HPP
#define MDL_COMPILER_HPP

#include <filesystem>

namespace mdl {
	struct Module;

	void compileToFile(const Module& module, const std::filesystem::path& output);
} // namespace mdl

#endif