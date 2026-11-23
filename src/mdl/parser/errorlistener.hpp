#ifndef MDL_PARSER_ERRORLISTENER_HPP
#define MDL_PARSER_ERRORLISTENER_HPP

#include <antlr4-runtime.h>

#include <algorithm>

using namespace antlr4;

class PrettyErrorListener : public BaseErrorListener {
private:
	size_t numErrors;
	size_t numWarnings;
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

	PrettyErrorListener(const std::string& filename) : numErrors(0), numWarnings(0), file(filename) {}

	/**
	 * @brief Returns the line that is @p offset lines away from the line containing the character @p startOffset.
	 * @details Lines are delimited by '\n' or by the beginning/end of the stream; an empty string is returned if the
	 * requested line does not exist.
	 */
	static std::string getline(antlr4::CharStream& stream, ssize_t offset, size_t startOffset) {
		const ssize_t size = static_cast<ssize_t>(stream.size());
		auto isLinefeed = [&stream](ssize_t pos) {
			return stream.getText(antlr4::misc::Interval{static_cast<size_t>(pos), static_cast<size_t>(pos)}) == "\n";
		};
		// Position of the linefeed that precedes the line containing `pos` (-1 for the first line).
		auto lineStart = [&](ssize_t pos) {
			pos = std::min(pos, size) - 1;
			while (pos >= 0 && !isLinefeed(pos))
				--pos;
			return pos;
		};
		// Position of the linefeed that terminates the line starting after `start` (size for the last line).
		auto lineEnd = [&](ssize_t start) {
			ssize_t pos = start + 1;
			while (pos < size && !isLinefeed(pos))
				++pos;
			return pos;
		};

		ssize_t start = lineStart(static_cast<ssize_t>(startOffset) + 1);
		for (ssize_t i = 0; i < -offset; ++i) {
			if (start < 0)
				return "";
			start = lineStart(start);
		}
		for (ssize_t i = 0; i < offset; ++i) {
			start = lineEnd(start);
			if (start >= size)
				return "";
		}
		ssize_t end = lineEnd(start);
		if (end - 1 < start + 1)
			return "";
		return stream.getText(antlr4::misc::Interval{static_cast<size_t>(start + 1), static_cast<size_t>(end - 1)});
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

	constexpr size_t getNumErrors() const { return numErrors; }
	constexpr size_t getNumWarnings() const { return numWarnings; }

	void printSummary() const;

private:
	void printError(
			ErrorLevel level, std::vector<ContextLine> context, size_t offendingLine, size_t offendingCol,
			const std::string& msg, const std::string& hint = ""
	);
};

#endif