#ifndef MDL_PARSER_HPP
#define MDL_PARSER_HPP

#include "./parserv2/compilation_unit.hpp"

#include <filesystem>
#include <iostream>
#include <optional>
#include <vector>

namespace mdl::parser {
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
