#include <mdl/parser.hpp>

#include "./parser/errorlistener.hpp"

#include <mdlLexer.h>
#include <mdlParser.h>

#include <fstream>

Program mdl::parser::parse(std::istream& stream, std::string_view sourcename) {
	antlr4::ANTLRInputStream input{stream};
	antlr4::mdlLexer lexer(&input);
	antlr4::CommonTokenStream tokens(&lexer);
	antlr4::mdlParser parser(&tokens);
	parser.removeErrorListeners();
	PrettyErrorListener errorlistener(std::string{sourcename});
	parser.addErrorListener(&errorlistener);
	return std::move(parser.program()->value);
}
Program mdl::parser::parse(const std::filesystem::path& file) {
	std::ifstream stream{file.c_str()};
	return parse(stream, std::filesystem::absolute(file).c_str());
}