#ifndef MDL_PARSER_ERRORLISTENER_HPP
#define MDL_PARSER_ERRORLISTENER_HPP

#include <antlr4-runtime.h>

#include <termcolor/termcolor.hpp>

#include <iostream>
#include <sstream>

using namespace antlr4;

// #include <windows.h>
#include <sys/ioctl.h>
#include <unistd.h>

/**
 * @brief Get the width of the terminal in characters
 * 
 * @return Width of the terminal in characters
 */
static int getTerminalWidth() {
#ifdef _WIN32
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	if (GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &csbi))
		return csbi.srWindow.Right - csbi.srWindow.Left + 1;
	return 80;
#else
	winsize w{};
	if (ioctl(STDERR_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0)
		return w.ws_col;
	return 80;
#endif
}

class PrettyErrorListener : public BaseErrorListener {
private:
	bool errorOccured;
	std::string file;

public:
	struct ContextLine {
		size_t linenr;
		std::string line;
		size_t beginMark;
		size_t markWidth;
	};

	/**
	 * @brief Represents the severity level of an error encountered during parsing.
	 */
	enum class ErrorLevel { Error, Warning };

	PrettyErrorListener(const std::string& filename) : errorOccured(false), file(filename) {}

	static std::string getline(antlr4::CharStream& stream, int offset, size_t startOffset) {
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
			ErrorLevel level, Token* offendingSymbol, size_t line, size_t charPositionInLine, const std::string& msg,
			const std::string& hint = ""
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
				line, charPositionInLine, msg, hint
		);
	}

	void semanticError(
			ErrorLevel level, std::vector<ContextLine> context, size_t offendingLine, size_t offendingCol,
			const std::string& msg, const std::string& hint = ""
	) {
		printError(level, context, offendingLine, offendingCol, msg, hint);
	}

	constexpr bool anyErrorOccured() const { return errorOccured; }

private:
	void printError(
			ErrorLevel level, std::vector<ContextLine> context, size_t offendingLine, size_t offendingCol,
			const std::string& msg, const std::string& hint = ""
	);
};

#endif