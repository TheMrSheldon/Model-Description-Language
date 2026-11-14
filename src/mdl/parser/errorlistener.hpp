#ifndef MDL_PARSER_ERRORLISTENER_HPP
#define MDL_PARSER_ERRORLISTENER_HPP

#include <mdl/diagnostics.hpp>

#include <antlr4-runtime.h>

#include <cstddef>

#include <algorithm>
#include <set>
#include <tuple>

using namespace antlr4;

/**
 * @brief Collects the diagnostics of one source file (syntax errors from ANTLR, semantic errors from the translation)
 * and passes them to a sink (see mdl::Diagnostic); the same diagnostic (position and message) is reported once, also
 * if the code is elaborated several times (e.g. per module instantiation or loop iteration).
 */
class PrettyErrorListener : public BaseErrorListener {
private:
	size_t numErrors = 0;
	size_t numWarnings = 0;
	std::string file;
	const mdl::DiagnosticSink* sink;
	std::set<std::tuple<size_t, size_t, std::string>> reported;

public:
	using ContextLine = mdl::Diagnostic::Line;

	/**
	 * @brief Represents the severity level of an error encountered during parsing.
	 */
	enum class ErrorLevel { Error, Warning };

	/** @param sink receives the diagnostics (must outlive the listener). */
	PrettyErrorListener(const std::string& filename, const mdl::DiagnosticSink& sink) : file(filename), sink(&sink) {}

	const std::string& getFile() const { return file; }

	/**
	 * @brief Returns the line that is @p offset lines away from the line containing the character @p startOffset.
	 * @details Lines are delimited by '\n' or by the beginning/end of the stream; an empty string is returned if the
	 * requested line does not exist.
	 */
	static std::string getline(antlr4::CharStream& stream, std::ptrdiff_t offset, size_t startOffset) {
		const std::ptrdiff_t size = static_cast<std::ptrdiff_t>(stream.size());
		auto isLinefeed = [&stream](std::ptrdiff_t pos) {
			return stream.getText(antlr4::misc::Interval{static_cast<size_t>(pos), static_cast<size_t>(pos)}) == "\n";
		};
		// Position of the linefeed that precedes the line containing `pos` (-1 for the first line).
		auto lineStart = [&](std::ptrdiff_t pos) {
			pos = std::min(pos, size) - 1;
			while (pos >= 0 && !isLinefeed(pos))
				--pos;
			return pos;
		};
		// Position of the linefeed that terminates the line starting after `start` (size for the last line).
		auto lineEnd = [&](std::ptrdiff_t start) {
			std::ptrdiff_t pos = start + 1;
			while (pos < size && !isLinefeed(pos))
				++pos;
			return pos;
		};

		std::ptrdiff_t start = lineStart(static_cast<std::ptrdiff_t>(startOffset) + 1);
		for (std::ptrdiff_t i = 0; i < -offset; ++i) {
			if (start < 0)
				return "";
			start = lineStart(start);
		}
		for (std::ptrdiff_t i = 0; i < offset; ++i) {
			start = lineEnd(start);
			if (start >= size)
				return "";
		}
		std::ptrdiff_t end = lineEnd(start);
		if (end - 1 < start + 1)
			return "";
		return stream.getText(antlr4::misc::Interval{static_cast<size_t>(start + 1), static_cast<size_t>(end - 1)});
	}
	void syntaxError(
			Recognizer* recognizer, Token* offendingSymbol, size_t line, size_t charPositionInLine,
			const std::string& msg, std::exception_ptr e
	) override {
		// The lexer reports unrecognized characters without a token: mark the character at the position.
		antlr4::CharStream* input = nullptr;
		size_t start = 0, width = 1, column = charPositionInLine;
		if (offendingSymbol != nullptr) {
			input = offendingSymbol->getInputStream();
			start = offendingSymbol->getStartIndex();
			width = std::max<size_t>(offendingSymbol->getText().size(), 1);
			column = offendingSymbol->getCharPositionInLine();
		} else if (auto* lexer = dynamic_cast<antlr4::Lexer*>(recognizer)) {
			input = lexer->getInputStream();
			start = lexer->tokenStartCharIndex;
		}
		if (input == nullptr) {
			report(ErrorLevel::Error, {}, line, charPositionInLine, msg);
			return;
		}
		report(ErrorLevel::Error,
			   {
					   {.number = line - 1, .text = getline(*input, -1, start)},
					   {.number = line, .text = getline(*input, 0, start), .markBegin = column, .markWidth = width},
					   {.number = line + 1, .text = getline(*input, 1, start)},
			   },
			   line, charPositionInLine, msg);
	}

	void semanticError(
			ErrorLevel level, Token* offendingSymbol, size_t line, size_t charPositionInLine, const std::string& msg,
			const std::string& hint = ""
	) {
		auto linestr = getline(*offendingSymbol->getInputStream(), 0, offendingSymbol->getStartIndex());
		semanticError(
				level,
				{ContextLine{
						.number = line,
						.text = linestr,
						.markBegin = offendingSymbol->getCharPositionInLine(),
						.markWidth = offendingSymbol->getText().size()
				}},
				line, charPositionInLine, msg, hint
		);
	}

	void semanticError(
			ErrorLevel level, std::vector<ContextLine> context, size_t offendingLine, size_t offendingCol,
			const std::string& msg, const std::string& hint = "", const std::string& flag = ""
	) {
		report(level, std::move(context), offendingLine, offendingCol, msg, hint, flag);
	}

	constexpr size_t getNumErrors() const { return numErrors; }
	constexpr size_t getNumWarnings() const { return numWarnings; }

private:
	void
	report(ErrorLevel level, std::vector<ContextLine> context, size_t offendingLine, size_t offendingCol,
		   const std::string& msg, const std::string& hint = "", const std::string& flag = "");
};

#endif