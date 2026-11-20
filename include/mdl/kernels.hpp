#ifndef MDL_KERNELS_HPP
#define MDL_KERNELS_HPP

/**
 * @file
 * @brief Kernel packages: optimized kernels from kernel directories (see docs/dev/spec/kernel-packages.md).
 * @details A kernel directory contains one package per subdirectory with a `package.yml`. A package describes what it
 * matches (patterns over the Relax graph at the matching boundary), how to call its prebuilt implementations (typed C
 * signatures), and which binaries (artifacts) provide them for which platforms. Loading a package only reads its
 * manifest: artifacts are never opened.
 */

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mdl::kernels {

	/** @brief The version of the `package.yml` format this compiler reads (`schema: 1`). */
	inline constexpr int kSchemaVersion = 1;

	/** @brief A problem in a package manifest (or a kernel directory), with its position if known. */
	struct Problem {
		std::filesystem::path file;
		size_t line = 0, column = 0; ///< from 1; 0 if unknown
		std::string message;

		std::string str() const; ///< `file:line:column: message`
	};

	/** @brief The binary compatibility of an artifact (or of the host, see hostPlatform). */
	struct Platform {
		std::string os;	  ///< linux, windows, darwin
		std::string arch; ///< x86_64, aarch64
		std::string abi;  ///< gnu, msvc, darwin
		std::string
				libc; ///< a requirement like `>=2.28` (artifacts), or a version like `2.39` (the host); may be empty
		std::string deviceArch; ///< the device of a compilation target (e.g. sm_80); empty: the CPU

		std::string str() const; ///< e.g. `linux x86_64 gnu libc>=2.28`
	};

	/** @brief A link dependency of an artifact (structured; integrations translate it, e.g. into CMake targets). */
	struct Dependency {
		enum class Kind {
			systemLibrary, ///< a library of the system, by name (`m`, `dl`)
			runtime,	   ///< a runtime the integration knows how to provide (`openmp`)
			artifact,	   ///< another artifact of the same package, by id
		};
		Kind kind;
		std::string name;
	};

	/** @brief A binary that provides implementations, for one platform. */
	struct Artifact {
		enum class Kind {
			staticLibrary,
			sharedLibrary,
			cubin, ///< reserved (CUDA): parsed, but not supported yet
			ptx,   ///< reserved (CUDA)
		};
		std::string id;
		Kind kind;
		std::filesystem::path path; ///< absolute (relative paths in the manifest are relative to the package)
		Platform platform;
		std::vector<std::string> cpuFeatures; ///< required instruction set extensions (e.g. avx2)
		std::string deviceArch;				  ///< reserved (CUDA), e.g. sm_80
		std::vector<Dependency> dependencies;
		size_t line = 0; ///< in the manifest

		bool supported() const { return kind == Kind::staticLibrary || kind == Kind::sharedLibrary; }
	};

	/**
	 * @brief A node of a pattern: an operator call over arguments, a captured value, or a constant.
	 * @details Captures with the same name must be the same value. Constants match exactly unless a tolerance is given.
	 */
	struct PatternNode {
		enum class Kind { op, capture, constant };
		Kind kind = Kind::op;
		std::string op;							  ///< e.g. relax.nn.softmax
		std::vector<PatternNode> args;			  ///< of an op
		std::map<std::string, std::string> attrs; ///< of an op, as written in the manifest (e.g. `axis: -1`)
		std::string capture;					  ///< the name of a capture
		std::vector<std::string> notOps;		  ///< of a capture: operators that must not produce the value (`not:`)
		double value = 0;						  ///< of a constant
		std::string valueExpr; ///< of a constant given as an integer expression over the captures (e.g. `$x.shape[-1]`)
		std::optional<double> tolerance; ///< of a constant (exact if absent)
		size_t line = 0;
	};

	struct Pattern {
		std::string id;
		PatternNode root;
		std::vector<std::string> outputs; ///< the region's results by capture name (empty: the root)
		size_t line = 0;
	};

	/** @brief An argument of the C signature of an implementation. */
	struct Argument {
		std::string type;	///< pointer, int64, int32, float32, float64, int64[] (a pointer to the values of a shape)
		std::string access; ///< pointers: read, write, readwrite
		std::string value;	///< an expression over the match, e.g. `$x.data`, `prod($x.shape[:-1])`
		int64_t align = 0;	///< pointers: the required alignment in bytes (0: none)
	};

	struct Implementation {
		std::string id;
		std::string pattern;				///< the id of its pattern
		std::vector<std::string> artifacts; ///< ids; the first compatible one is used
		int64_t priority = 0;
		std::string symbol;
		std::string abi = "c";
		std::vector<Argument> arguments;
		std::string returns = "int32";							  ///< the return type; nonzero is an error
		std::string aliasing = "none";							  ///< or `out=<capture>` (in place)
		std::vector<std::pair<std::string, std::string>> outputs; ///< (shape, dtype) expressions per result
		std::map<std::string, std::vector<std::string>> dtypes;	  ///< applicability: capture -> allowed dtypes
		std::map<std::string, std::string> layouts;				  ///< applicability: capture -> layout (row-major)
		std::vector<std::string> where;							  ///< applicability: boolean expressions
		std::map<std::string, std::string> guarantees;			  ///< declared by the author, e.g. numerics.max_ulp
		size_t line = 0;
	};

	struct Package {
		std::string name;
		std::string version;
		std::string description;
		std::filesystem::path directory; ///< the package's directory
		std::filesystem::path manifest;	 ///< its package.yml
		std::vector<Artifact> artifacts;
		std::vector<Pattern> patterns;
		std::vector<Implementation> implementations;

		const Artifact* artifact(const std::string& id) const;
		const Pattern* pattern(const std::string& id) const;
	};

	/**
	 * @brief Reads and validates the manifest @p file (unknown keys and schema versions, references between sections,
	 * value types). Returns the package, or std::nullopt with the problems.
	 */
	std::optional<Package> loadPackage(const std::filesystem::path& file, std::vector<Problem>& problems);

	/** @brief Where a kernel directory comes from (most specific first). */
	enum class DirOrigin { explicitDir, project, user, global };
	std::string str(DirOrigin origin);

	struct KernelDir {
		std::filesystem::path path;
		DirOrigin origin;
	};

	/**
	 * @brief The kernel directories, most specific first: @p explicitDirs (`--kernels-dir`), the project's `.mdl/kernels`
	 * found upward from @p source (if given), the user's directory (see userKernelDir), and the global ones
	 * (`MDL_KERNELS_PATH`, `<prefix>/share/mdl/kernels`, the source tree's `kernels/`). Only existing directories.
	 */
	std::vector<KernelDir> kernelDirs(
			const std::vector<std::filesystem::path>& explicitDirs, const std::optional<std::filesystem::path>& source
	);

	/**
	 * @brief The user's kernel directory: `$XDG_DATA_HOME/mdl/kernels` (default `~/.local/share/mdl/kernels`),
	 * `%LOCALAPPDATA%\mdl\kernels` on Windows, `~/Library/Application Support/mdl/kernels` on macOS.
	 */
	std::optional<std::filesystem::path> userKernelDir();

	/** @brief The packages of the kernel directories; a package hides one of the same name in a later directory. */
	struct Discovery {
		std::vector<KernelDir> dirs;
		std::vector<Package> packages; ///< in the order of the directories
		/** @brief Packages hidden by a package of the same name in a more specific directory: (hidden, by). */
		std::vector<std::pair<Package, std::filesystem::path>> hidden;
		std::vector<Problem> problems; ///< of packages that could not be loaded (they are skipped)
	};

	Discovery discover(const std::vector<KernelDir>& dirs);

	/** @brief The platform this compiler runs on (the default target of the artifacts). */
	Platform hostPlatform();

	/** @brief Whether an artifact for @p required runs on @p host (also checks a `libc: >=x.y` requirement). */
	bool compatible(const Platform& required, const Platform& host);

	/**
	 * @brief Whether @p artifact is for the device of @p target: CPU artifacts (no device-arch) only for CPU targets,
	 * device artifacts only for that device. A cubin runs on the same major architecture with an equal or higher minor
	 * one (sm_80 on sm_86, not on sm_90); PTX and libraries on equal or higher architectures.
	 */
	bool forDevice(const Artifact& artifact, const Platform& target);

} // namespace mdl::kernels

#endif
