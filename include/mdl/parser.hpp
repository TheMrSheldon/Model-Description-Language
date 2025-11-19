#ifndef MDL_PARSER_HPP
#define MDL_PARSER_HPP

#include <mdl/parser/definitions.hpp>

#include <filesystem>
#include <iostream>

namespace mdl::parser {
	Program parse(std::istream& stream);
	Program parse(const std::filesystem::path& file);
} // namespace mdl::parser

#endif