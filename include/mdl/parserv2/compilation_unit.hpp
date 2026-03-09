#ifndef MDL_PARSER_COMPILATIONUNIT_HPP
#define MDL_PARSER_COMPILATIONUNIT_HPP

#include "../computegraphv2/datatype.hpp"
#include "../computegraphv2/graph.hpp"

#include <map>
#include <string>
#include <variant>

namespace mdl {
	/**
     * @brief Represents a compilation unit from the parser.
     */
	struct CompilationUnit {
		Computegraph graph;
		using Member = std::variant<ModuleType, AggregateType, FunctionDecl>;
		std::map<std::string, Member> definitions;
	};
} // namespace mdl

#endif