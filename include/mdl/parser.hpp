#ifndef MDL_PARSER_HPP
#define MDL_PARSER_HPP

#include "./parserv2/compilation_unit.hpp"

#include <filesystem>
#include <iostream>
#include <optional>
#include <vector>

namespace mdl::parser {
	/** @brief How a diagnostic is reported (like g++'s -Werror=x, -Wx, -Wno-x). */
	enum class Diagnostic {
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
		Diagnostic floatWidening = Diagnostic::error;
		Diagnostic floatNarrowing = Diagnostic::error;
		/** @brief Annotations that the compiler does not know (`-W[no-|error=]unknown-annotation`): a warning by default. */
		Diagnostic unknownAnnotation = Diagnostic::warning;
	};

	/**
	 * @brief The default location(s) of the standard library: the paths in the `MDL_STDLIB_PATH` environment variable
	 * (separated by ':'), followed by the location configured at build time.
	 */
	std::vector<std::filesystem::path> defaultStdlibDirs();

	/**
	 * @brief Parses a program into a compilation unit.
	 * @details Imported definitions are visible to the program but are not part of CompilationUnit::definitions, i.e.,
	 * only the definitions of the parsed file itself are compiled and exported.
	 */
	std::optional<CompilationUnit>
	parse(std::istream& stream, std::string_view sourcename = "<stream>", const ParseOptions& options = {});
	std::optional<CompilationUnit> parse(const std::filesystem::path& file, const ParseOptions& options = {});
} // namespace mdl::parser

#endif
