#ifndef MDL_PARSER_ERRORLISTENER_HPP
#define MDL_PARSER_ERRORLISTENER_HPP

#include <antlr4-runtime.h>

#include <termcolor/termcolor.hpp>

#include <iostream>
#include <sstream>

using namespace antlr4;

class PrettyErrorListener : public BaseErrorListener {
private:
	enum class ErrorLevel { Error, Warning };

public:
	PrettyErrorListener(const std::string& filename) : file(filename) {}

	void syntaxError(
			Recognizer* recognizer, Token* offendingSymbol, size_t line, size_t charPositionInLine,
			const std::string& msg, std::exception_ptr e
	) override {
		auto& is = *offendingSymbol->getInputStream();
		size_t startOfLine = offendingSymbol->getStartIndex();
		size_t endOfLine = offendingSymbol->getStopIndex();
		while (is.getText(antlr4::misc::Interval{startOfLine - 1, startOfLine - 1}) != "\n")
			startOfLine -= 1;
		while (is.getText(antlr4::misc::Interval{endOfLine + 1, endOfLine + 1}) != "\n")
			endOfLine += 1;

		auto linestr = is.getText(antlr4::misc::Interval{startOfLine, endOfLine});
		printError(ErrorLevel::Error, linestr, line, charPositionInLine, offendingSymbol, msg);
	}

private:
	std::string file;

	void printError(
			ErrorLevel level, std::string srcLine, size_t line, size_t col, Token* offendingSymbol,
			const std::string& msg, const std::string& hint = ""
	) {
		switch (level) {
		case ErrorLevel::Error:
			std::cerr << termcolor::bold << termcolor::red << "error: " << termcolor::reset;
			break;
		case ErrorLevel::Warning:
			std::cerr << termcolor::bold << termcolor::yellow << "warning: " << termcolor::reset;
			break;
		}
		std::cerr << msg << "\n";
		std::cerr << "~~> " << termcolor::bold << file << ":" << line << ":" << (col + 1) << ": " << termcolor::reset
				  << std::endl;

		std::cerr << termcolor::bright_grey << " " << line << " | " << termcolor::reset << srcLine << "\n";

		// caret line
		for (size_t i = 0; i < col + std::to_string(line).size() + 4; ++i)
			std::cerr << (srcLine[i] == '\t' ? '\t' : ' ');

		std::cerr << termcolor::red;
		for (size_t i = col; i < offendingSymbol->getCharPositionInLine() + offendingSymbol->getText().size(); ++i)
			std::cerr << "^";
		std::cerr << termcolor::reset << "\n";

		if (!hint.empty()) {
			std::cerr << termcolor::blue << "hint: " << termcolor::reset << hint << std::endl;
		}
	}
};

#endif