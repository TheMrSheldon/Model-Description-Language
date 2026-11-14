#include "./errorlistener.hpp"

void PrettyErrorListener::report(
		ErrorLevel level, std::vector<ContextLine> context, size_t offendingLine, size_t offendingCol,
		const std::string& msg, const std::string& hint, const std::string& flag,
		std::vector<mdl::Diagnostic::Fix> fixes, bool syntax
) {
	if (!reported.emplace(offendingLine, offendingCol, msg).second)
		return; // the same code elaborated again (e.g. another instantiation of a module)
	if (level == ErrorLevel::Error)
		numErrors++;
	else
		numWarnings++;
	mdl::Diagnostic d{
			.severity = level == ErrorLevel::Error ? mdl::Severity::error : mdl::Severity::warning,
			.file = file,
			.range = {{offendingLine, offendingCol}, {offendingLine, offendingCol}},
			.message = msg,
			.hint = hint,
			.flag = flag,
			.context = std::move(context),
			.fixes = std::move(fixes),
	};
	d.syntax = syntax;
	// The range: the marked part of the context (one line), else the position.
	for (const auto& line : d.context)
		if (line.markWidth > 0) {
			d.range = {{line.number, line.markBegin}, {line.number, line.markBegin + line.markWidth}};
			break;
		}
	if (*sink)
		(*sink)(d);
}
