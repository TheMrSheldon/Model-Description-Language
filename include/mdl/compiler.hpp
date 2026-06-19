#ifndef MDL_COMPILER_HPP
#define MDL_COMPILER_HPP

#include <filesystem>
#include <string>
#include <string_view>

namespace mdl {
	struct CompilationUnit;
	class ArgVals;

	enum class Backend {
		Tvm,  ///< TVM/Relax backend (default) — produces a .so via TVM
		Iree, ///< IREE backend — produces a .vmfb via the IREE compiler API
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
	};

	void compileToFile(
			const CompilationUnit& unit,
			const std::filesystem::path& output,
			const ArgVals& argvals,
			const CompileOptions& opts = {});

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