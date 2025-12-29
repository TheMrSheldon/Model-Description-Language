#ifndef MDL_COMPILER_HPP
#define MDL_COMPILER_HPP

#include <filesystem>

class Module;

namespace mdl {
    void compileToFile(const Module& module, const std::filesystem::path& output);
}

#endif