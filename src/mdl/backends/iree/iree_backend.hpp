#pragma once

#include <filesystem>

namespace mdl {
struct CompilationUnit;
class ArgVals;
struct CompileOptions;
} // namespace mdl

namespace mdl::backends::iree {

/**
 * Compile an MDL CompilationUnit to an IREE VM FlatBuffer (.vmfb).
 *
 * The compilation pipeline is:
 *   MDL IR  →  StableHLO MLIR (text, internal)  →  .vmfb  (via IREE compiler C API)
 *
 * The output path has its extension replaced with ".vmfb".
 * The IREE runtime is required to load and execute the resulting file.
 */
void compileToFile(
    const mdl::CompilationUnit& unit,
    const std::filesystem::path& output,
    const mdl::ArgVals& argvals,
    const mdl::CompileOptions& opts);

} // namespace mdl::backends::iree
