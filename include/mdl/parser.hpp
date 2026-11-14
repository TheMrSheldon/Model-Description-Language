#ifndef MDL_PARSER_HPP
#define MDL_PARSER_HPP

#include "./analysis/index.hpp"
#include "./diagnostics.hpp"
#include "./parserv2/compilation_unit.hpp"

#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mdl::parser {
	/** @brief How a diagnostic is reported (like g++'s -Werror=x, -Wx, -Wno-x). */
	enum class DiagnosticLevel {
		error,	 ///< an error (the default for implicit float conversions)
		warning, ///< a warning; the compiler does what the diagnostic is about (e.g. inserts the conversion)
		ignored, ///< silently done
	};

	struct ParseOptions {
		/**
		 * @brief Directories that are searched for imported files (after the directory of the importing file).
		 */
		std::vector<std::filesystem::path> importDirs;
		/**
		 * @brief Directories that contain the standard library, i.e., the files of the `std` package (`import std.x;`
		 * loads `<dir>/x.dlm`). If empty, defaultStdlibDirs() is used.
		 */
		std::vector<std::filesystem::path> stdlibDirs;
		/**
		 * @brief The type that `float` stands for (float32, bfloat16 or float16); the explicit types (e.g. `float32`)
		 * are not affected.
		 */
		PrimitiveType defaultFloat = PrimitiveType::float32;
		/**
		 * @brief Implicit conversions between float types (`-W[no-|error=]float-widening`, `float-narrowing`):
		 * widening converts bfloat16/float16 to float32 (costs memory and time), narrowing converts float32 to a 16-bit
		 * type or between bfloat16 and float16 (loses precision). They apply to operands of different float types (both
		 * are converted to the wider type, float32 for bfloat16 with float16), to arguments (to the parameter's type),
		 * and to results (to the declared type). By default both are errors: conversions are written explicitly
		 * (`float32(x)`).
		 */
		DiagnosticLevel floatWidening = DiagnosticLevel::error;
		DiagnosticLevel floatNarrowing = DiagnosticLevel::error;
		/**
		 * @brief Code that is fine for some values of the hyperparameters but an error for others
		 * (`-W[no-|error=]permits-illegal-instantiation`):
		 * - sizes that are different compile-time expressions where equal sizes are needed, e.g. `W @ x` with
		 *   `W: float[input, output]` and `x: float[input]` (they may differ in an instance);
		 * - a member with `@conditional(c)` used where `c` may be false (not in the first branch of `c ? ... : ...`).
		 *
		 * A warning by default, an error in strict mode (`mdlc --strict`).
		 */
		DiagnosticLevel permitsIllegalInstantiation = DiagnosticLevel::warning;
		/** @brief Annotations that the compiler does not know (`-W[no-|error=]unknown-annotation`): a warning by default. */
		DiagnosticLevel unknownAnnotation = DiagnosticLevel::warning;
		/**
		 * @brief Definitions of the parsed file that are never used (`-W[no-|error=]unused`): local variables,
		 * parameters, loop variables, members of modules, imports (see analysis::unusedDefinitions). A warning by
		 * default.
		 */
		DiagnosticLevel unused = DiagnosticLevel::warning;
		/**
		 * @brief A local variable, parameter, or loop variable that hides a variable, parameter, loop variable, member,
		 * or hyperparameter of an enclosing scope (`-W[no-|error=]shadow`, as in g++). A warning by default.
		 */
		DiagnosticLevel shadow = DiagnosticLevel::warning;

		/**
		 * @brief Receives the diagnostics (errors and warnings) as they are found; if empty, parse() prints them to
		 * stderr (with a summary per file).
		 */
		DiagnosticSink diagnostics;
		/**
		 * @brief Reads a source file (e.g. from an editor's unsaved buffer); returns std::nullopt to read it from disk.
		 * Applies to the parsed file and its imports.
		 */
		std::function<std::optional<std::string>(const std::filesystem::path&)> readFile;
		/**
		 * @brief The directory of a program parsed from a stream (imports are resolved relative to it); the current
		 * directory if unset.
		 */
		std::optional<std::filesystem::path> sourceDir;
	};

	/** @brief The result of analyze(): the compilation unit also if there are errors, and all diagnostics. */
	struct AnalysisResult {
		CompilationUnit unit;
		std::vector<Diagnostic> diagnostics;
		bool ok = false; ///< whether there are no errors
		/** @brief Definitions, references, types, and documentation of the program (and its imports). */
		std::shared_ptr<const analysis::SemanticIndex> index;
	};

	/**
	 * @brief The default location(s) of the standard library: the paths in the `MDL_STDLIB_PATH` environment variable
	 * (separated by ':', ';' on Windows), the installation's `<prefix>/share/mdl/std`, and the source tree's `std/`.
	 */
	std::vector<std::filesystem::path> defaultStdlibDirs();

	/**
	 * @brief Parses a program into a compilation unit.
	 * @details Reports the same diagnostics as analyze() (also of modules with hyperparameters, checked with symbolic
	 * values). Imported definitions are visible to the program but are not part of CompilationUnit::definitions, i.e.,
	 * only the definitions of the parsed file itself are compiled and exported.
	 */
	std::optional<CompilationUnit>
	parse(std::istream& stream, std::string_view sourcename = "<stream>", const ParseOptions& options = {});
	std::optional<CompilationUnit> parse(const std::filesystem::path& file, const ParseOptions& options = {});

	/**
	 * @brief Like parse(), for tools (e.g. a language server): keeps the compilation unit also if there are errors
	 * (with error nodes where the code is wrong; a file with syntax errors is not translated), and returns the
	 * diagnostics instead of printing them (they also go to ParseOptions::diagnostics if set), and the semantic
	 * index. Modules with hyperparameters are checked without values, as in parse(): with symbolic hyperparameters
	 * (types like `float[..., dim]`; what depends on unknown values, e.g. `?:` or loop bounds, is checked
	 * conservatively).
	 */
	AnalysisResult
	analyze(std::istream& stream, std::string_view sourcename = "<stream>", const ParseOptions& options = {});
	AnalysisResult analyze(const std::filesystem::path& file, const ParseOptions& options = {});
} // namespace mdl::parser

#endif
