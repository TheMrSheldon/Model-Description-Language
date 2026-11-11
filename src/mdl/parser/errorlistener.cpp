#include "./errorlistener.hpp"

#include <termcolor/termcolor.hpp>

#include <cassert>
#include <format>
#include <iostream>
#include <sstream>

/**
 * @brief Get the width of the terminal in characters
 * 
 * @return Width of the terminal in characters
 */
static int getTerminalWidth();
#ifdef _WIN32
#include <windows.h>

static int getTerminalWidth() {
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	if (GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &csbi))
		return csbi.srWindow.Right - csbi.srWindow.Left + 1;
	return 80;
}
#else
#include <sys/ioctl.h>
#include <unistd.h>

static int getTerminalWidth() {
	winsize w{};
	if (ioctl(STDERR_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0)
		return w.ws_col;
	return 80;
}
#endif

/**
 * @brief Wrapper class for pretty printing error levels.
 */
struct FmtLevel {
	/**
     * @brief The error level.
     */
	PrettyErrorListener::ErrorLevel level;
};

/**
 * @brief Wrapper class for pretty printing file positions in error messages.
 */
struct FmtFilepos {
	/**
     * @brief The file path.
     */
	std::string_view file;
	/**
     * @brief The line number.
     */
	size_t line;
	/**
     * @brief The column number.
     */
	size_t col;
};

/**
 * @brief Wrapper class for pretty printing ellipses in error messages.
 */
struct FmtElipsis {
	/**
     * @brief The symbol to use for the ellipsis.
     */
	char symbol;
	/**
     * @brief Whether the ellipsis should be printed.
     */
	bool enabled;
};

/**
 * @brief Wrapper class for pretty printing hints in error messages.
 */
struct FmtHint {
	std::string_view hint;
};

static std::ostream& operator<<(std::ostream& os, const FmtLevel& fmt) {
	switch (fmt.level) {
	case PrettyErrorListener::ErrorLevel::Error:
		return os << termcolor::bold << termcolor::red << "error: " << termcolor::reset;
	case PrettyErrorListener::ErrorLevel::Warning:
		return os << termcolor::bold << termcolor::yellow << "warning: " << termcolor::reset;
	}
	return os;
}

static std::ostream& operator<<(std::ostream& os, const FmtFilepos& fmt) {
	return os << "~~> " << termcolor::bold << fmt.file << ":" << fmt.line << ":" << fmt.col + 1 << ": "
			  << termcolor::reset;
}

static std::ostream& operator<<(std::ostream& os, const FmtElipsis& fmt) {
	if (!fmt.enabled)
		return os;
	return os << termcolor::reverse << fmt.symbol << termcolor::reset;
}

static std::ostream& operator<<(std::ostream& os, const FmtHint& fmt) {
	return os << termcolor::blue << "hint: " << termcolor::reset << fmt.hint;
}

static std::ostream& operator<<(std::ostream& os, const PrettyErrorListener::ContextLine& ctxt) {
	auto termwidth = getTerminalWidth();
	std::string gutter = std::format(" {} | ", ctxt.linenr);
	size_t gutterWidth = gutter.size();
	assert(termwidth > gutterWidth);
	size_t usableWidth = termwidth - gutterWidth;

	size_t lineLen = ctxt.line.size();

	// Compute window
	size_t windowStart = (ctxt.beginMark > usableWidth / 2) ? (ctxt.beginMark - usableWidth / 2) : 0;
	size_t windowEnd = windowStart + usableWidth;

	if (windowEnd > lineLen) {
		windowEnd = lineLen;
		windowStart = (windowEnd > usableWidth) ? (windowEnd - usableWidth) : 0;
	}

	// Slice visible text
	std::string visible = ctxt.line.substr(windowStart, windowEnd - windowStart);

	// Ellipses if trimmed
	bool leftTrimmed = windowStart > 0;
	bool rightTrimmed = windowEnd < lineLen;

	// Print gutter and line content:
	// 123 | ...some code...
	// Optionally with ellipses if trimming was needed to fit the terminal:
	// 123 | <...some code...>
	std::cerr << termcolor::bright_grey << gutter << termcolor::reset
			  << FmtElipsis{.symbol = '<', .enabled = leftTrimmed}
			  << visible.substr(leftTrimmed, visible.size() - (leftTrimmed + rightTrimmed))
			  << FmtElipsis{.symbol = '>', .enabled = rightTrimmed} << "\n";

	if (ctxt.markWidth != 0) {
		size_t caretPos = ctxt.beginMark - windowStart + leftTrimmed;
		if (leftTrimmed && caretPos > 0)
			caretPos -= 1;

		size_t numCarets = std::min(ctxt.markWidth, usableWidth - caretPos);
		std::cerr << termcolor::bright_grey << std::string(gutterWidth - 2, ' ') << "| " << termcolor::reset
				  << std::string(caretPos, ' ') << termcolor::red << std::string(numCarets, '^') << termcolor::reset
				  << "\n";
	}
	return os;
}

void PrettyErrorListener::printError(
		ErrorLevel level, std::vector<ContextLine> context, size_t offendingLine, size_t offendingCol,
		const std::string& msg, const std::string& hint
) {
	if (level == ErrorLevel::Error)
		numErrors++;
	else if (level == ErrorLevel::Warning)
		numWarnings++;
	std::cerr << FmtLevel{level} << msg << "\n";
	std::cerr << FmtFilepos{file, offendingLine, offendingCol + 1} << std::endl;
	for (auto& ctxt : context)
		std::cerr << ctxt;
	if (!hint.empty())
		std::cerr << FmtHint{hint} << std::endl;
}

void PrettyErrorListener::printSummary() const {
	if (numErrors == 0 && numWarnings == 0)
		return;
	std::cerr << "Encountered ";
	if (numErrors > 0)
		std::cerr << termcolor::red << numErrors << " error(s)" << termcolor::reset;
	if (numErrors > 0 && numWarnings > 0)
		std::cerr << " and ";
	if (numWarnings > 0)
		std::cerr << termcolor::yellow << numWarnings << " warning(s)" << termcolor::reset;
	std::cerr << std::endl;
}