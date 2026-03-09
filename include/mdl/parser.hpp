#ifndef MDL_PARSER_HPP
#define MDL_PARSER_HPP

#include "./parserv2/compilation_unit.hpp"

#include <filesystem>
#include <iostream>
#include <optional>

namespace mdl::parser {
	std::optional<CompilationUnit> parse(std::istream& stream, std::string_view sourcename = "<stream>");
	std::optional<CompilationUnit> parse(const std::filesystem::path& file);
} // namespace mdl::parser

#endif