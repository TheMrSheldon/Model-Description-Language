#ifndef MDL_COMPILER_HPP
#define MDL_COMPILER_HPP

#include <filesystem>
#include <map>
#include <optional>
#include <ostream>
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

	/** Whether the backend calls optimized kernels of kernel packages (e.g. oneDNN, see docs/dev/spec/kernel-packages.md). */
	enum class Kernels {
		Packages, ///< The selected implementations of the kernel packages (the rest is TVM-generated)
		None,	  ///< Only TVM-generated kernels; the model depends on no kernel package
	};

	/** @brief A library of a kernel package that must be linked (see docs/dev/spec/kernel-packages.md). */
	struct LinkedArtifact {
		std::string package, id;
		std::filesystem::path path;
		std::string kind; ///< static-library, shared-library
		std::vector<std::string>
				dependencies; ///< `kind:name`, e.g. `system-library:m`, `runtime:openmp`, `artifact:dnnl`
	};

	/** @brief An implementation of a kernel package that the compiler used (see docs/dev/spec/kernel-packages.md). */
	struct SelectedKernel {
		std::string function; ///< the compiled function's symbol
		std::string package, version, implementation, symbol;
		std::filesystem::path manifest;		   ///< the package's package.yml
		std::vector<LinkedArtifact> artifacts; ///< the implementation's artifact, then those it depends on
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
		Kernels kernels = Kernels::Packages;

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

		/**
		 * The device to compile for, as a TVM target string (`llvm`, `cuda -arch=sm_80`). Only the backend interprets
		 * it (see backends/tvm/targets/); the frontend and the model are independent of it.
		 */
		std::string target = "llvm";

		/** Enable fast-math (approximate exp/erf/...); trades accuracy for speed. */
		bool fastMath = false;

		/** Instrument every kernel call (see mdl_profile_begin in mdl/runtime/profile.h). */
		bool profile = false;

		/** @brief Kernel directories searched first (`--kernels-dir`; see docs/dev/spec/kernel-packages.md and kernels::kernelDirs). */
		std::vector<std::filesystem::path> kernelDirs;

		/** @brief Kernel packages that may be used (`--kernel-package`; all if empty) and those that may not. */
		std::vector<std::string> kernelPackages, noKernelPackages;

		/** @brief Whether kernel packages are used at all (`--no-kernel-packages`; e.g. when the build links none). */
		bool useKernelPackages = true;

		/** @brief If set, receives the implementations of kernel packages that the compiled code calls. */
		std::vector<SelectedKernel>* selectedKernels = nullptr;

		/** @brief The source file (its project's `.mdl/kernels` is a kernel directory), if known. */
		std::optional<std::filesystem::path> source;

		/**
		 * @brief Inspection of kernel packages (`mdl-kernels`): the compiler stops at the matching boundary and writes to
		 * inspectOut instead of an output file: the selected implementations (brief), also every candidate with the
		 * verdicts of its constraints (explain), or the Relax module at the boundary (dump).
		 */
		enum class KernelInspection { none, brief, explain, dump } inspectKernels = KernelInspection::none;
		std::ostream* inspectOut = nullptr;

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
	/**
	 * @brief The symbol prefix of the object file @p output (`mdl_<stem>_`): the prefix of its kernels and of its TVM
	 * system library, which the C++ binding loads to attach device code (see mdl_detail::loadSystemLib).
	 */
	inline std::string systemLibPrefix(const std::filesystem::path& output) {
		std::string prefix = "mdl_";
		for (char c : output.stem().string())
			prefix += (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ? c : '_';
		return prefix + "_";
	}

	inline std::string mangledName(std::string_view moduleName, std::string_view funcName) {
		return std::string{moduleName} + "__" + std::string{funcName};
	}
} // namespace mdl

#endif