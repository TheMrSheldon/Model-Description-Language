#pragma once

/**
 * @file pipeline.hpp
 * @brief The C++ compilation pipeline of the TVM backend: Relax passes, AOT lowering, TIR passes, and LLVM codegen.
 */

#include <mdl/compiler.hpp>

#include <tvm/ir/module.h>
#include <tvm/target/target.h>

#include <filesystem>
#include <string>

namespace mdl::backend::tvm {

	/** @brief The LLVM target for the host CPU. */
	::tvm::Target makeTarget(const mdl::CompileOptions& opts);

	/**
	 * @brief The Relax module at the matching boundary of kernel packages (see docs/kernels.md): constants folded and the
	 * backend's canonical forms recognized (relax.nn.* operators, mdl.matmul with folded permutations), before
	 * legalization.
	 */
	::tvm::IRModule canonicalize(::tvm::IRModule mod, const ::tvm::Target& target);

	/** @brief The Relax passes after the matching boundary: legalization, fusion, and memory planning. */
	::tvm::IRModule lowerRelax(::tvm::IRModule mod, const ::tvm::Target& target, const mdl::CompileOptions& opts);

	/**
	 * @brief Runs the Relax passes: canonicalize, then lowerRelax.
	 * @return the module in the form expected by lowerToAOT.
	 */
	::tvm::IRModule runRelaxPipeline(::tvm::IRModule mod, const ::tvm::Target& target, const mdl::CompileOptions& opts);

	/**
	 * @brief Runs the TIR passes on a PrimFunc-only module and writes the generated code into an object file.
	 * @param systemLibPrefix symbol prefix of the "system library": the generated code calls the TVM runtime directly
	 * (instead of through function pointers that are only set when TVM loads a shared library), so the object file can
	 * be linked statically.
	 */
	void buildObject(
			::tvm::IRModule mod, const ::tvm::Target& target, const std::string& systemLibPrefix,
			const std::filesystem::path& output
	);

} // namespace mdl::backend::tvm
