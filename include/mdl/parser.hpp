#ifndef MDL_PARSER_HPP
#define MDL_PARSER_HPP

#include <mdl/parser/definitions.hpp>

#include <filesystem>
#include <iostream>
#include <optional>

namespace mdl::parser {
	std::optional<Program> parse(std::istream& stream, std::string_view sourcename = "<stream>");
	std::optional<Program> parse(const std::filesystem::path& file);
} // namespace mdl::parser

#endif