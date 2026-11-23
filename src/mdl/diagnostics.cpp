// Printing of diagnostics for a terminal (see mdl::printDiagnostic).
#ifdef _WIN32
// <windows.h> (also included by termcolor) must not define min and max (std::min/std::max below).
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif

#include <mdl/diagnostics.hpp>

#include <cstdint> // (termcolor uses uint8_t without including it)

#include <termcolor/termcolor.hpp>

#include <cstddef>
#include <format>
#include <string>
#include <string_view>

#include <support/Utf8.h>

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
	mdl::Severity level;
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
	case mdl::Severity::error:
		return os << termcolor::bold << termcolor::red << "error: " << termcolor::reset;
	case mdl::Severity::warning:
		return os << termcolor::bold << termcolor::yellow << "warning: " << termcolor::reset;
	case mdl::Severity::note:
		return os << termcolor::bold << "note: " << termcolor::reset;
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

static std::ostream& operator<<(std::ostream& os, const mdl::Diagnostic::Line& ctxt) {
	auto termwidth = getTerminalWidth();
	std::string gutter = std::format(" {} | ", ctxt.number);
	size_t gutterWidth = gutter.size();
	size_t usableWidth = static_cast<size_t>(termwidth) > gutterWidth + 10 ? termwidth - gutterWidth : 80;

	size_t lineLen = ctxt.text.size();

	// Compute window
	size_t windowStart = (ctxt.markBegin > usableWidth / 2) ? (ctxt.markBegin - usableWidth / 2) : 0;
	size_t windowEnd = windowStart + usableWidth;

	if (windowEnd > lineLen) {
		windowEnd = lineLen;
		windowStart = (windowEnd > usableWidth) ? (windowEnd - usableWidth) : 0;
	}

	// Slice visible text
	std::string visible = ctxt.text.substr(windowStart, windowEnd - windowStart);

	// Ellipses if trimmed
	bool leftTrimmed = windowStart > 0;
	bool rightTrimmed = windowEnd < lineLen;

	// Print gutter and line content:
	// 123 | ...some code...
	// Optionally with ellipses if trimming was needed to fit the terminal:
	// 123 | <...some code...>
	os << termcolor::bright_grey << gutter << termcolor::reset << FmtElipsis{.symbol = '<', .enabled = leftTrimmed}
	   << visible.substr(leftTrimmed, visible.size() - (leftTrimmed + rightTrimmed))
	   << FmtElipsis{.symbol = '>', .enabled = rightTrimmed} << "\n";

	if (ctxt.markWidth != 0) {
		size_t caretPos = ctxt.markBegin - windowStart + leftTrimmed;
		if (leftTrimmed && caretPos > 0)
			caretPos -= 1;

		size_t numCarets = std::min(ctxt.markWidth, usableWidth - caretPos);
		os << termcolor::bright_grey << std::string(gutterWidth - 2, ' ') << "| " << termcolor::reset
		   << std::string(caretPos, ' ') << termcolor::red << std::string(numCarets, '^') << termcolor::reset << "\n";
	}
	return os;
}

void mdl::printDiagnostic(std::ostream& os, const Diagnostic& d) {
	os << FmtLevel{d.severity} << d.message;
	if (!d.flag.empty())
		os << " [-W" << (d.severity == Severity::error ? "error=" : "") << d.flag << "]";
	os << "\n";
	os << FmtFilepos{d.file, d.range.begin.line, d.range.begin.column} << std::endl;
	for (const auto& line : d.context)
		os << line;
	if (!d.hint.empty())
		os << FmtHint{d.hint} << std::endl;
}

void mdl::printDiagnosticSummary(std::ostream& os, size_t errors, size_t warnings) {
	if (errors == 0 && warnings == 0)
		return;
	os << "Encountered ";
	if (errors > 0)
		os << termcolor::red << errors << " error(s)" << termcolor::reset;
	if (errors > 0 && warnings > 0)
		os << " and ";
	if (warnings > 0)
		os << termcolor::yellow << warnings << " warning(s)" << termcolor::reset;
	os << std::endl;
}

std::optional<mdl::Diagnostic> mdl::invalidUtf8(std::string_view text, const std::string& file) {
	if (text.starts_with("\xEF\xBB\xBF"))
		text.remove_prefix(3); // the byte order mark (skipped by the parser)
	size_t line = 1, column = 0, lineStart = 0;
	for (size_t i = 0; i < text.size();) {
		const auto [point, length] = antlrcpp::Utf8::decode(text.substr(i));
		if (point == 0xFFFD && text.substr(i, 3) != "\xEF\xBF\xBD") { // (not an encoded replacement character)
			const auto end = text.find('\n', i);
			const std::string lineText = antlrcpp::Utf8::lenientEncode(antlrcpp::Utf8::lenientDecode(
					text.substr(lineStart, end == std::string_view::npos ? end : end - lineStart)
			));
			return Diagnostic{
					.file = file,
					.range = {{line, column}, {line, column + 1}},
					.message = std::format(
							"Invalid UTF-8 (byte 0x{:02X}): source files must be UTF-8",
							static_cast<unsigned char>(text[i])
					),
					.context = {{.number = line, .text = lineText, .markBegin = column, .markWidth = 1}},
			};
		}
		if (point == U'\n') {
			++line;
			column = 0;
			lineStart = i + 1;
		} else
			++column;
		i += length;
	}
	return std::nullopt;
}
