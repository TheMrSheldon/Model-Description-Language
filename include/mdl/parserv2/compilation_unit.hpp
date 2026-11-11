#ifndef MDL_PARSER_COMPILATIONUNIT_HPP
#define MDL_PARSER_COMPILATIONUNIT_HPP

#include "../computegraphv2/datatype.hpp"
#include "../computegraphv2/graph.hpp"

#include <filesystem>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace mdl {
	/**
     * @brief Represents a compilation unit from the parser.
     */
	struct CompilationUnit {
		Computegraph graph;
		using Member = std::variant<ModuleType, AggregateType, FunctionDecl>;
		std::map<std::string, Member> definitions;
		/** @brief All source files that were translated (the main file and all imported files). */
		std::vector<std::filesystem::path> sources;
	};
} // namespace mdl

#endif