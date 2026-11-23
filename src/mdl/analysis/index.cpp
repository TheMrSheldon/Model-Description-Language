#include <mdl/analysis/index.hpp>

#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>

using namespace mdl::analysis;

size_t SemanticIndex::define(Symbol symbol) {
	const auto key = std::tuple{symbol.file, symbol.range.begin.line, symbol.range.begin.column};
	if (auto it = symbolAt_.find(key); it != symbolAt_.end()) {
		symbols_[it->second].types.insert(symbol.types.begin(), symbol.types.end());
		reference(Reference{
				.file = symbol.file, .range = symbol.range, .symbol = it->second, .types = symbol.types, .members = {}
		});
		return it->second;
	}
	symbols_.push_back(std::move(symbol));
	symbolAt_.emplace(key, symbols_.size() - 1);
	// The declaration's name is also a reference (hover, go to definition).
	const auto& s = symbols_.back();
	reference(
			Reference{.file = s.file, .range = s.range, .symbol = symbols_.size() - 1, .types = s.types, .members = {}}
	);
	return symbols_.size() - 1;
}

void SemanticIndex::reference(Reference reference) {
	const auto key = std::tuple{reference.file, reference.range.begin.line, reference.range.begin.column};
	if (auto it = referenceAt_.find(key); it != referenceAt_.end()) {
		auto& existing = references_[it->second];
		existing.types.insert(reference.types.begin(), reference.types.end());
		existing.members.insert(reference.members.begin(), reference.members.end());
		existing.weights.insert(reference.weights.begin(), reference.weights.end());
		existing.results.insert(reference.results.begin(), reference.results.end());
		if (!existing.symbol)
			existing.symbol = reference.symbol;
		return;
	}
	references_.push_back(std::move(reference));
	referenceAt_.emplace(key, references_.size() - 1);
}

const Reference* SemanticIndex::referenceAt(std::string_view file, SourcePosition position) const {
	for (const auto& r : references_)
		if (r.file == file && r.range.begin.line == position.line && r.range.begin.column <= position.column &&
			position.column <= r.range.end.column)
			return &r;
	return nullptr;
}

std::vector<const Symbol*> SemanticIndex::visibleAt(std::string_view file, SourcePosition position) const {
	std::vector<const Symbol*> local, global;
	for (const auto& s : symbols_) {
		if (s.container.empty() &&
			(s.kind == SymbolKind::module || s.kind == SymbolKind::structure || s.kind == SymbolKind::function)) {
			global.push_back(&s);
			continue;
		}
		if (s.file != file || !s.scope.contains(position))
			continue;
		// Locals are visible after their declaration; members, hyperparameters and parameters in their whole scope.
		const bool local_ = s.kind == SymbolKind::variable || s.kind == SymbolKind::loopVariable;
		if (local_ && position < s.range.end)
			continue;
		local.push_back(&s);
	}
	// Innermost (latest starting scope) first.
	std::stable_sort(local.begin(), local.end(), [](const Symbol* a, const Symbol* b) {
		return a->scope.begin > b->scope.begin;
	});
	local.insert(local.end(), global.begin(), global.end());
	return local;
}

namespace {
	/** @brief The parts of a Doxygen comment (Markdown, without the inline conversions). */
	struct DocComment {
		std::string brief, details, returns, todo;
		std::vector<std::pair<std::string, std::string>> params;
		/** @brief `@note`, `@warning`, `@see`, `@deprecated` paragraphs: (title, text), in order. */
		std::vector<std::pair<std::string, std::string>> sections;
	};

	/** @brief The text of the lines of a comment without the comment markers. */
	std::vector<std::string> commentLines(std::string_view comment) {
		std::vector<std::string> lines;
		std::istringstream in{std::string{comment}};
		std::string line;
		while (std::getline(in, line)) {
			if (!line.empty() && line.back() == '\r')
				line.pop_back();
			size_t start = line.find_first_not_of(" \t");
			if (start == std::string::npos) {
				lines.emplace_back();
				continue;
			}
			std::string_view l{line};
			l.remove_prefix(start);
			for (std::string_view marker : {"/**", "/*!", "///", "//!", "/*"})
				if (l.substr(0, marker.size()) == marker) {
					l.remove_prefix(marker.size());
					break;
				}
			if (l.size() >= 2 && l.substr(l.size() - 2) == "*/")
				l.remove_suffix(2);
			if (!l.empty() && l.front() == '*')
				l.remove_prefix(1);
			if (!l.empty() && l.front() == ' ')
				l.remove_prefix(1);
			while (!l.empty() && (l.back() == ' ' || l.back() == '\t'))
				l.remove_suffix(1);
			lines.emplace_back(l);
		}
		// Commands in the middle of a line start a new paragraph (like in Doxygen): one line per command.
		std::vector<std::string> split;
		const std::regex command{R"(\s+(?=[@\\](brief|details|t?param|returns?|todo|note|warning|see|sa|deprecated)\b))"
		};
		for (const auto& l : lines) {
			std::sregex_token_iterator it{l.begin(), l.end(), command, -1}, end;
			for (; it != end; ++it)
				split.push_back(*it);
			if (l.empty())
				split.emplace_back();
		}
		return split;
	}

	DocComment parseDoc(std::string_view comment) {
		DocComment doc;
		std::string* current = &doc.details;
		auto append = [](std::string& to, const std::string& text) {
			if (text.empty()) {
				if (!to.empty() && to.substr(to.size() - 2) != "\n\n")
					to += "\n\n";
				return;
			}
			if (!to.empty() && to.back() != '\n')
				to += ' ';
			to += text;
		};
		bool code = false; // in `@code ... @endcode` (lines kept as they are)
		for (auto text : commentLines(comment)) {
			if (code) {
				const auto first = text.find_first_not_of(' ');
				if (first != std::string::npos &&
					(text.compare(first, 8, "@endcode") == 0 || text.compare(first, 8, "\\endcode") == 0)) {
					*current += "```\n\n";
					code = false;
				} else {
					*current += text + "\n";
				}
				continue;
			}
			auto command = [&](std::string_view name) {
				for (char lead : {'@', '\\'})
					if (text.size() > name.size() && text[0] == lead && text.compare(1, name.size(), name) == 0 &&
						(text.size() == name.size() + 1 || text[name.size() + 1] == ' ' || text[name.size() + 1] == '['
						)) {
						text = text.substr(name.size() + 1);
						// `@param[in] name`: the direction is not shown.
						if (!text.empty() && text[0] == '[')
							text = text.substr(std::min(text.size(), text.find(']') + 1));
						text.erase(0, text.find_first_not_of(' '));
						return true;
					}
				return false;
			};
			if (command("brief")) {
				current = &doc.brief;
			} else if (command("details")) {
				current = &doc.details;
			} else if (command("param") || command("tparam")) {
				std::string name = text.substr(0, text.find(' '));
				text = text.size() > name.size() ? text.substr(name.size() + 1) : "";
				doc.params.emplace_back(name, "");
				current = &doc.params.back().second;
			} else if (command("return") || command("returns")) {
				current = &doc.returns;
			} else if (command("todo")) {
				current = &doc.todo;
			} else if (const char* title = [&]() -> const char* {
						   for (auto [name, title] :
								{std::pair{"note", "Note"}, std::pair{"warning", "Warning"},
								 std::pair{"see", "See also"}, std::pair{"sa", "See also"},
								 std::pair{"deprecated", "Deprecated"}})
							   if (command(name))
								   return title;
						   return nullptr;
					   }()) {
				doc.sections.emplace_back(title, "");
				current = &doc.sections.back().second;
			} else if (command("code")) { // `@code` or `@code{.py}`: the following lines as a code block
				if (!current->empty() && !current->ends_with("\n\n"))
					*current += current->ends_with("\n") ? "\n" : "\n\n";
				*current += "```\n";
				code = true;
				continue;
			} else if (command("file") || command("ingroup") || command("defgroup") || command("addtogroup")) {
				continue;
			}
			append(*current, text);
		}
		return doc;
	}

	/** @brief Inline commands as Markdown: `\p x`, math (`\f$`, `\f(`, `\f[`, `\f{env}{`; also with `@`), `\cite`. */
	std::string inlineMarkdown(std::string s) {
		s = std::regex_replace(s, std::regex{R"([@\\]p\s+(\w+))"}, "`$1`");
		s = std::regex_replace(s, std::regex{R"([@\\]f\$(.*?)[@\\]f\$)"}, "$$$1$$");
		s = std::regex_replace(s, std::regex{R"([@\\]f\((.*?)[@\\]f\))"}, "$$$1$$");
		s = std::regex_replace(s, std::regex{R"([@\\]f\[(.*?)[@\\]f\])"}, "$$$$$1$$$$");
		s = std::regex_replace(
				s, std::regex{R"([@\\]f\{(\w+\*?)\}\{?(.*?)[@\\]f\})"}, "$$$$\\begin{$1}$2\\end{$1}$$$$"
		);
		s = std::regex_replace(s, std::regex{R"([@\\]cite\s+(\S+))"}, "[$1]");
		// Word styles: `\a`, `\e`, `\em` italic, `\b` bold, `\c` code.
		s = std::regex_replace(s, std::regex{R"([@\\](a|e|em)\s+(\w+))"}, "*$2*");
		s = std::regex_replace(s, std::regex{R"([@\\]b\s+(\w+))"}, "**$1**");
		s = std::regex_replace(s, std::regex{R"([@\\]c\s+(\w+))"}, "`$1`");
		while (!s.empty() && (s.back() == '\n' || s.back() == ' '))
			s.pop_back();
		return s;
	}
} // namespace

std::string mdl::analysis::docCommentToMarkdown(std::string_view comment) {
	const DocComment doc = parseDoc(comment);
	std::string md = inlineMarkdown(doc.brief);
	auto section = [&](const std::string& text) {
		if (text.empty())
			return;
		if (!md.empty())
			md += "\n\n";
		md += text;
	};
	section(inlineMarkdown(doc.details));
	if (!doc.params.empty()) {
		std::string list = "**Parameters**";
		for (const auto& [name, text] : doc.params)
			list += "\n- `" + name + "`: " + inlineMarkdown(text);
		section(list);
	}
	if (!doc.returns.empty())
		section("**Returns** " + inlineMarkdown(doc.returns));
	for (const auto& [title, text] : doc.sections)
		section("**" + title + "** " + inlineMarkdown(text));
	if (!doc.todo.empty())
		section("**Todo** " + inlineMarkdown(doc.todo));
	return md;
}

std::string mdl::analysis::docParameter(std::string_view comment, std::string_view name) {
	for (const auto& [param, text] : parseDoc(comment).params)
		if (param == name)
			return inlineMarkdown(text);
	return {};
}

std::string mdl::analysis::fileDocumentation(std::string_view text) {
	// The comments at the start of the file (before the first code), the first one with `@file`.
	size_t pos = 0;
	while (pos < text.size()) {
		while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
			++pos;
		if (text.compare(pos, 2, "//") == 0) {
			pos = text.find('\n', pos);
			continue;
		}
		if (text.compare(pos, 2, "/*") != 0)
			break;
		const size_t end = text.find("*/", pos + 2);
		if (end == std::string_view::npos)
			break;
		const std::string_view comment = text.substr(pos, end + 2 - pos);
		if (comment.find("@file") != std::string_view::npos || comment.find("\\file") != std::string_view::npos)
			return docCommentToMarkdown(comment);
		pos = end + 2;
	}
	return {};
}
