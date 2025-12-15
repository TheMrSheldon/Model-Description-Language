#ifndef MDL_COMPILER_HPP
#define MDL_COMPILER_HPP

#include <filesystem>

namespace mdl {
    class Computegraph;

    void compileToFile(const Computegraph& graph, const std::filesystem::path& output);
}

#endif