#pragma once

/**
 * @file pipeline.hpp
 * @brief C++ Relax compilation pipeline: all passes + VMCodeGen + export.
 *
 * Replaces the Python subprocess that previously handled LegalizeOps, FuseOps,
 * VMCodeGen, TIR build, VMLink, and export_library.
 *
 * Entry point: compilePipeline(irMod, targetStr, dnnlOpsSrc, output, opts)
 *
 * The function is responsible for everything after the IRModule is built and
 * C++ pattern rewrites are applied.  The caller retains the IRModule object.
 */

#include <tvm/ir/module.h>
#include <tvm/target/target.h>

#include <sstream>
#include <string>

namespace mdl::backend::tvm {

/**
 * @brief Run the full Relax→native compilation pipeline.
 *
 * @param irMod          The IRModule produced by the C++ IR builder (with C++
 *                       pattern rewrites already applied).
 * @param targetStr      CPU target string, e.g. "llvm -mcpu=skylake-avx512".
 * @param dnnlOpsSrc     Content of the oneDNN ops C++ source file.
 * @param output         Path to the model .so to produce.
 * @param optimization   Optimization level string ("default", "metaschedule", …).
 */
/**
 * @brief Parse "llvm -mcpu=xxx" → TVM Target with fast-math enabled.
 */
::tvm::Target parseTarget(const std::string& targetStr);

/**
 * @brief Run all Relax passes (LegalizeOps through AttachGlobalSymbol).
 *
 * The caller should have already applied C++ pattern rewrites.
 * Returns the fully-lowered IRModule ready for VMCodeGen.
 */
::tvm::IRModule runRelaxPipeline(::tvm::IRModule irMod, const ::tvm::Target& target);

} // namespace mdl::backend::tvm
