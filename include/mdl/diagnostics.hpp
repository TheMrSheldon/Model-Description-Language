#ifndef MDL_DIAGNOSTICS_HPP
#define MDL_DIAGNOSTICS_HPP

#include <cstddef>
#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace mdl {
	enum class Severity { error, warning, note };

	/** @brief A position in a source file: the line (from 1) and the column (from 0, in Unicode code points). */
	struct SourcePosition {
		size_t line = 0;
		size_t column = 0;

		auto operator<=>(const SourcePosition&) const = default;
	};

	/** @brief A range in a source file; @p end is exclusive. */
	struct SourceRange {
		SourcePosition begin, end;

		auto operator<=>(const SourceRange&) const = default;
		bool contains(SourcePosition p) const { return begin <= p && p < end; }
	};

	/** @brief An error, warning, or note of the frontend, e.g. for a terminal, a language server, or a test. */
	struct Diagnostic {
		Severity severity = Severity::error;
		std::string file; ///< the path or source name (e.g. `<stream>`)
		SourceRange range;
		std::string message;
		std::string hint; ///< a suggestion how to fix it (may be empty)
		std::string flag; ///< the option that controls it (e.g. `float-widening` for `-Wfloat-widening`), if any

		/** @brief A source line around the range, for printing (the marked part is the range on its line). */
		struct Line {
			size_t number = 0;
			std::string text;
			size_t markBegin = 0, markWidth = 0;
		};
		std::vector<Line> context;

		/** @brief A text change of a fix: replaces @p range by @p text (an empty range inserts). */
		struct Edit {
			SourceRange range;
			std::string text;
		};
		/** @brief A way to fix the problem (e.g. a quick fix of an editor). */
		struct Fix {
			std::string title;
			std::vector<Edit> edits;
		};
		std::vector<Fix> fixes;
	};

	/** @brief Receives the diagnostics of a compilation as they are found. */
	using DiagnosticSink = std::function<void(const Diagnostic&)>;

	/**
	 * @brief Prints @p diagnostic for a terminal: the message, `~~> file:line:column:`, the source lines with the range
	 * marked, and the hint (with colors if @p os is a terminal).
	 */
	void printDiagnostic(std::ostream& os, const Diagnostic& diagnostic);

	/** @brief Prints `Encountered N error(s) and M warning(s)` (nothing if both are 0). */
	void printDiagnosticSummary(std::ostream& os, size_t errors, size_t warnings);
} // namespace mdl

#endif
