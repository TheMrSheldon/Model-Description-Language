#ifndef MDL_PARSER_ERRORLISTENER_HPP
#define MDL_PARSER_ERRORLISTENER_HPP

#include <antlr4-runtime.h>

#include <termcolor/termcolor.hpp>

#include <iostream>
#include <sstream>

using namespace antlr4;

class PrettyErrorListener : public BaseErrorListener {
private:
	struct ContextLine {
		size_t linenr;
		std::string line;
		size_t beginMark;
		size_t markWidth;
	};

public:
	enum class ErrorLevel { Error, Warning };

	PrettyErrorListener(const std::string& filename) : file(filename) {}

	std::string getline(antlr4::CharStream& stream, int offset, size_t startOffset) {
		auto findLinefeed = [&stream](size_t offset, int dir) {
			while (stream.getText(antlr4::misc::Interval{offset, offset}) != "\n")
				offset += dir;
			return offset;
		};

		auto startOfLine = findLinefeed(startOffset, -1);
		auto dir = (offset <= 0) ? -1 : 1;
		for (size_t i = 0; i < offset * dir; ++i) {
			startOfLine = findLinefeed(startOfLine + dir, dir);
		}
		size_t endOfLine = findLinefeed(startOfLine + 1, 1);
		return stream.getText(antlr4::misc::Interval{startOfLine + 1, endOfLine - 1});
	}
	void syntaxError(
			Recognizer* recognizer, Token* offendingSymbol, size_t line, size_t charPositionInLine,
			const std::string& msg, std::exception_ptr e
	) override {
		printError(
				ErrorLevel::Error,
				{
						{.linenr = line - 1,
						 .line = getline(*offendingSymbol->getInputStream(), -1, offendingSymbol->getStartIndex())},
						{.linenr = line,
						 .line = getline(*offendingSymbol->getInputStream(), 0, offendingSymbol->getStartIndex()),
						 .beginMark = offendingSymbol->getCharPositionInLine(),
						 .markWidth = offendingSymbol->getText().size()},
						{.linenr = line + 1,
						 .line = getline(*offendingSymbol->getInputStream(), 1, offendingSymbol->getStartIndex())},
				},
				line, charPositionInLine, msg
		);
	}

	void semanticError(
			ErrorLevel level, Token* offendingSymbol, size_t line, size_t charPositionInLine, const std::string& msg
	) {
		auto linestr = getline(*offendingSymbol->getInputStream(), 0, offendingSymbol->getStartIndex());
		semanticError(
				level,
				{ContextLine{
						.linenr = line,
						.line = linestr,
						.beginMark = offendingSymbol->getCharPositionInLine(),
						.markWidth = offendingSymbol->getText().size()
				}},
				line, charPositionInLine, msg
		);
	}

	void semanticError(
			ErrorLevel level, std::vector<ContextLine> context, size_t offendingLine, size_t offendingCol,
			const std::string& msg
	) {
		printError(level, context, offendingLine, offendingCol, msg);
	}

private:
	std::string file;

	void printError(
			ErrorLevel level, std::vector<ContextLine> context, size_t offendingLine, size_t offendingCol,
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
		std::cerr << "~~> " << termcolor::bold << file << ":" << offendingLine << ":" << (offendingCol + 1) << ": "
				  << termcolor::reset << std::endl;

		for (auto& ctxt : context) {
			std::cerr << termcolor::bright_grey << " " << ctxt.linenr << " | " << termcolor::reset << ctxt.line << "\n";

			if (ctxt.markWidth != 0) {
				// caret line
				for (size_t i = 0; i < ctxt.beginMark + std::to_string(ctxt.linenr).size() + 4; ++i)
					std::cerr << (ctxt.line[i] == '\t' ? '\t' : ' ');

				std::cerr << termcolor::red;
				for (size_t i = 0; i < ctxt.markWidth; ++i)
					std::cerr << "^";
				std::cerr << termcolor::reset << "\n";
			}
		}

		if (!hint.empty()) {
			std::cerr << termcolor::blue << "hint: " << termcolor::reset << hint << std::endl;
		}
	}
};

#endif