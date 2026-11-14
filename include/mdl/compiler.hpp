#ifndef MDL_COMPILER_HPP
#define MDL_COMPILER_HPP

#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace mdl {
	struct ExportedFunction;
	class ArgVals;

	enum class Backend {
		Tvm,  ///< TVM/Relax backend (default) — produces a .so via TVM
		Iree, ///< IREE backend — produces a .vmfb via the IREE compiler API
	};

	/** Which kernels the backend may call for recognized subgraphs (matmul, softmax, layer norm, ...). */
	enum class Kernels {
		OneDnn, ///< Vendor kernels of the mdl_kernels library (oneDNN); the model must be linked against mdl_kernels
		Tvm,	///< Only TVM-generated kernels; the model has no dependency on mdl_kernels
	};

	struct CompileOptions {
		/**
		 * Optimization level string.
		 *
		 * ""             — default pipeline (tile+vectorize+FuseOps, fast compile)
		 * "metaschedule" — MetaSchedule auto-tuning over all operators (slow compile,
		 *                  near-optimal runtime performance)
		 */
		std::string optimization;

		/** Target backend. Defaults to TVM. */
		Backend backend = Backend::Tvm;

		/** Kernels that the backend may call. */
		Kernels kernels = Kernels::OneDnn;

		/**
		 * Number of dimensions that `...` stands for in the signatures of compiled functions (e.g. 2 for `[B, S, D]`
		 * inputs of a function declared with `float[..., D]`). All `...` of a function denote the same dimensions.
		 */
		unsigned ellipsisRank = 1;

		/**
		 * Inputs (by parameter name) whose `...` stands for fewer dimensions: the leading ones (e.g. labels: int[...] as
		 * [batch] while the token ids are [batch, seq]).
		 */
		std::map<std::string, unsigned> inputEllipsisRanks;

		/** Enable fast-math (approximate exp/erf/...); trades accuracy for speed. */
		bool fastMath = false;

		/** Instrument every kernel call (see mdl_profile_begin in mdl/runtime/kernels.h). */
		bool profile = false;

		/** Directory for intermediate debug artifacts (IR dumps); empty to disable. */
		std::filesystem::path dumpDir;
	};

	/** @brief Compiles the exported functions (see mdl::exportedFunctions) into @p output. */
	void compileToFile(
			const std::vector<ExportedFunction>& exports, const std::filesystem::path& output, const ArgVals& argvals,
			const CompileOptions& opts = {}
	);

	/**
	 * @brief Returns the mangled symbol name for a module member function.
	 * @details The scheme is `<module>__<function>`. Both the compiler backend and the binding
	 * generators must use this function so that generated bindings can resolve the correct symbol
	 * at runtime.
	 */
	inline std::string mangledName(std::string_view moduleName, std::string_view funcName) {
		return std::string{moduleName} + "__" + std::string{funcName};
	}
} // namespace mdl

#endif