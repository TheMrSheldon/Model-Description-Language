#ifndef MDL_PARSER_DEFINITIONS_HPP
#define MDL_PARSER_DEFINITIONS_HPP
#include <map>
#include <string>
#include <variant>

#include <mdl/computegraph.hpp>

using mdl::Computegraph;

// Not a reference because ANTLR wants to copy construct them :/
using ComputegraphRef = Computegraph*;

extern Computegraph empty;

struct Module;
struct Struct;
struct VarDesc {
	std::string name;
	const Computegraph::Node* node;
};
struct FuncDesc {
	std::string name;
	std::vector<VarDesc> arguments;
	Computegraph* graph;
};

struct Program {
	std::map<std::string, std::variant<Module, Struct>> definitions;
};

struct Module {
	std::string name;
	Computegraph graph;
	std::vector<VarDesc> arguments;
	std::vector<VarDesc> variables;
	std::vector<FuncDesc> functions;
};

struct Struct {
	std::string name;
};

struct Type {
	std::string name;
	std::vector<Computegraph::Node> args;
};
#endif