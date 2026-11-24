// mdl-doc: writes the API documentation of a package of `.dlm` files (e.g. the standard library) as Doxygen pages.
//
// Doxygen has no MDL parser, so instead of disguising MDL as C++ (an input filter), the pages are generated from the
// semantic index of the frontend: one page per file with its public definitions, their declarations as MDL code, and
// their doc comments as written (Doxygen interprets `@brief`, `@param`, `\cite`, math, ... itself). See
// docs/dev/technical/documentation.md.
#include <mdl/analysis/index.hpp>
#include <mdl/parser.hpp>
#include <mdl/version.hpp>

#include <CLI/CLI.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <ranges>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
using mdl::analysis::Symbol;
using mdl::analysis::SymbolKind;

namespace {
	/** @brief The Doxygen page ID of a dotted name: `std.nn` → `std_nn` (and `std_nn_Linear` for its definitions). */
	std::string pageId(std::string dotted) {
		std::ranges::replace(dotted, '.', '_');
		return dotted;
	}

	/** @brief A doc comment as Doxygen text: without comment markers and without `@file` (it is a page). */
	std::string docText(const std::string& comment) {
		std::istringstream in{mdl::analysis::stripCommentMarkers(comment)};
		std::string text, line;
		while (std::getline(in, line))
			if (!line.starts_with("@file") && !line.starts_with("\\file"))
				text += line + "\n";
		return text;
	}

	/**
	 * @brief The brief description in a doc comment: the `@brief` paragraph, else the first paragraph (up to a blank
	 * line or the next command at the start of a line).
	 */
	std::string brief(const std::string& comment) {
		std::istringstream in{docText(comment)};
		std::string line, result;
		bool explicitBrief = false;
		while (std::getline(in, line)) {
			line.erase(0, std::min(line.size(), line.find_first_not_of(' ')));
			const bool command = line.starts_with('@') || line.starts_with('\\');
			if (line.starts_with("@brief") || line.starts_with("\\brief")) {
				explicitBrief = true;
				result = line.substr(std::min(line.size(), size_t{7}));
				continue;
			}
			if ((explicitBrief || !result.empty()) && (line.empty() || command))
				break;
			if (!command && !line.empty())
				result += (result.empty() ? "" : " ") + line;
		}
		return result;
	}

	/**
	 * @brief @p text without the `@brief` command (its text stays, as the first paragraph): on a page, Doxygen would
	 * move it to the top of the page.
	 */
	std::string withoutBriefCommand(std::string text) {
		for (std::string_view command : {"@brief", "\\brief"})
			for (size_t at = text.find(command); at != std::string::npos; at = text.find(command, at))
				if (at == 0 || text[at - 1] == '\n')
					text.erase(at, command.size() + (at + command.size() < text.size() && text[at + command.size()] == ' '));
				else
					at += command.size();
		return text;
	}

	/** @brief @p text in a Markdown table cell (`|` escaped). */
	std::string tableCell(const std::string& text) {
		std::string cell;
		for (char c : text)
			cell += c == '|' ? std::string{"\\|"} : std::string(1, c);
		return cell;
	}

	struct Page {
		std::string dotted; ///< e.g. `std.nn`
		fs::path file;
		std::string path; ///< as shown, e.g. `std/nn.dlm`
		std::string text, sourceText;
		std::string brief; ///< of the file's `@file` comment
	};

	/** @brief A definition, by the position of its name: (file, line, column). */
	using DefinitionKey = std::tuple<std::string, size_t, size_t>;
	DefinitionKey keyOf(const Symbol& s) {
		return {fs::weakly_canonical(s.file).string(), s.range.begin.line, s.range.begin.column};
	}
	/** @brief The documentation links of the definitions: their cards, e.g. `std_nn.html#std_nn_Linear`. */
	using Links = std::map<DefinitionKey, std::string>;

	std::string escapeHtml(std::string_view text) {
		std::string html;
		for (char c : text)
			html += c == '&' ? "&amp;" : c == '<' ? "&lt;" : c == '>' ? "&gt;" : std::string(1, c);
		return html;
	}

	/**
	 * @brief The source page of a file: its lines, numbered, each with the anchor `l<number>` (for the "Definition at
	 * line ..." links of the cards), in Doxygen's code markup (raw HTML, so that the line anchors survive), marked as
	 * MDL for docs/doxygen/mdl-highlight.js.
	 */
	std::string renderSource(
			const Page& page, const std::string& source, const mdl::analysis::SemanticIndex& index, const Links& links,
			const std::map<fs::path, std::string>& pages
	) {
		// The names to link, per line: (column, length, target). A name of a documented definition links to its card,
		// one of another definition in the package (not a local variable or parameter) to its line in the source.
		std::map<size_t, std::vector<std::tuple<size_t, size_t, std::string, std::string>>> targets;
		for (const auto& r : index.references()) {
			std::error_code ec;
			if (!r.symbol || !fs::equivalent(r.file, page.file, ec) || r.range.begin.line != r.range.end.line)
				continue;
			const Symbol& s = index.symbols()[*r.symbol];
			std::string target;
			if (auto link = links.find(keyOf(s)); link != links.end()) {
				target = link->second;
			} else if (s.kind == SymbolKind::module || s.kind == SymbolKind::structure ||
					   s.kind == SymbolKind::function || s.kind == SymbolKind::method ||
					   s.kind == SymbolKind::member) {
				if (auto defining = pages.find(fs::weakly_canonical(s.file)); defining != pages.end())
					target = pageId(defining->second) + "_source.html#l" + std::to_string(s.range.begin.line);
			}
			if (!target.empty())
				targets[r.range.begin.line].emplace_back(
						r.range.begin.column, r.range.end.column - r.range.begin.column, target, s.name
				);
		}
		std::ostringstream out;
		out << "\\page " << pageId(page.dotted) << "_source " << page.path << "\n\n"
			<< "The source code of \\ref " << pageId(page.dotted) << ".\n\n"
			<< "\\htmlonly[block]<div class=\"mdl\"><div class=\"fragment\">";
		std::istringstream in{source};
		std::string line;
		for (int n = 1; std::getline(in, line); ++n) {
			if (!line.empty() && line.back() == '\r')
				line.pop_back();
			out << "<div class=\"line\" id=\"l" << n << "\"><span class=\"lineno\">" << n << "</span>";
			auto& names = targets[n];
			std::ranges::sort(names);
			size_t column = 0; // (the standard library is ASCII: columns are bytes)
			for (const auto& [begin, length, target, name] : names) {
				// Only names as written (not e.g. an operator that stands for a function).
				if (begin < column || begin + length > line.size() || line.compare(begin, length, name) != 0)
					continue;
				out << escapeHtml(line.substr(column, begin - column)) << "<a class=\"code\" href=\"" << target << "\">"
					<< escapeHtml(line.substr(begin, length)) << "</a>";
				column = begin + length;
			}
			out << escapeHtml(line.substr(column)) << "</div>\n";
		}
		out << "</div></div>\\endhtmlonly\n";
		return out.str();
	}

	/** @brief The page of @p file: its public definitions (and re-exports). */
	std::string render(
			const Page& page, const mdl::analysis::SemanticIndex& index, const std::map<fs::path, std::string>& pages,
			const std::string& fileComment, Links& links
	) {
		const std::string id = pageId(page.dotted);
		std::ostringstream out;
		out << "\\page " << id << " " << page.dotted << "\n\n" << docText(fileComment) << "\n\\tableofcontents\n\n";
		auto inFile = [&](const Symbol& s) {
			std::error_code ec;
			return fs::equivalent(s.file, page.file, ec);
		};
		// The declaration (without `public`: everything on the page is public).
		auto declaration = [](const Symbol& s) {
			std::string detail = s.detail;
			if (auto at = detail.find("public "); at != std::string::npos)
				detail.erase(at, 7);
			return detail;
		};
		// Overloads share a name: their anchors get a counter (`std_activations_softmax`, `..._softmax_2`).
		std::map<std::string, int> anchors;
		auto anchor = [&](const std::string& name) {
			const int n = ++anchors[name];
			return n == 1 ? name : name + "_" + std::to_string(n);
		};
		auto isPublic = [&](const Symbol& s, SymbolKind kind, const std::string& container) {
			return s.kind == kind && s.container == container && s.visibility == mdl::Visibility::Public && inFile(s);
		};
		// A definition as a Doxygen member card (styled by doxygen-awesome): the declaration (as MDL, see
		// docs/doxygen/mdl-highlight.js) and the doc comment. The card's tags are raw HTML (\htmlonly[block]): HTML tags
		// in the text would be closed and reopened by Doxygen at every paragraph.
		auto card = [&](const std::string& heading, const std::string& level, const std::string& anchorId,
						const Symbol& s) {
			links[keyOf(s)] = id + ".html#" + anchorId;
			out << level << " " << heading << " {#" << anchorId << "}\n\n"
				<< "\\htmlonly[block]<div class=\"memitem\"><div class=\"memproto mdl\">\\endhtmlonly\n```\n"
				<< declaration(s) << "\n```\n\\htmlonly[block]</div><div class=\"memdoc\">\\endhtmlonly\n\n"
				<< withoutBriefCommand(docText(s.comment)) << "\n"
				// "Definition at line N of FILE" like Doxygen's, with links into the source page.
				<< "\n\\htmlonly[block]<p class=\"definition\">Definition at line <a href=\"" << id << "_source.html#l"
				<< s.range.begin.line << "\">" << s.range.begin.line << "</a> of <a href=\"" << id << "_source.html\">"
				<< escapeHtml(page.path) << "</a>.</p>\\endhtmlonly\n";
		};
		auto endCard = [&] { out << "\n\\htmlonly[block]</div></div>\\endhtmlonly\n\n"; };
		auto section = [&](std::string_view title, SymbolKind kind) {
			std::vector<const Symbol*> symbols;
			for (const auto& s : index.symbols())
				if (isPublic(s, kind, ""))
					symbols.push_back(&s);
			if (symbols.empty())
				return;
			// An overview (name and brief description) before the details, like Doxygen's member lists.
			std::map<std::string, int> overview;
			out << "# " << title << "\n\n| Name | Description |\n|---|---|\n";
			for (const auto* s : symbols) {
				const int n = ++overview[s->name];
				const std::string target = id + "_" + s->name + (n == 1 ? "" : "_" + std::to_string(n));
				out << "| \\ref " << target << " \"" << s->name << "\" | " << tableCell(brief(s->comment)) << " |\n";
			}
			out << "\n";
			for (const auto* s : symbols) {
				card(s->name, "##", anchor(id + "_" + s->name), *s);
				// Of a module: its public members as a table (hyperparameters are documented with `@param`) ...
				bool members = false;
				for (const auto& m : index.symbols()) {
					if (!isPublic(m, SymbolKind::member, s->name))
						continue;
					if (!std::exchange(members, true))
						out << "<b>Members</b>\n\n| Member | Type | Description |\n|---|---|---|\n";
					std::string type = declaration(m);
					type = type.substr(type.find(':') + 1);
					type.erase(0, type.find_first_not_of(' '));
					out << "| <tt>" << m.name << "</tt> | <tt>" << tableCell(type) << "</tt> | "
						<< tableCell(brief(m.comment)) << " |\n";
				}
				endCard();
				// ... and its public methods as cards of their own.
				for (const auto& m : index.symbols())
					if (isPublic(m, SymbolKind::method, s->name)) {
						card(s->name + "." + m.name, "###", anchor(id + "_" + s->name + "_" + m.name), m);
						endCard();
					}
			}
		};
		section("Modules", SymbolKind::module);
		section("Structs", SymbolKind::structure);
		section("Functions", SymbolKind::function);

		// Re-exports (`public from .x import a, b;`): the names with a link to the page that defines them.
		bool any = false;
		for (const auto& i : index.imports()) {
			std::error_code ec;
			if (!i.reexported || i.names.empty() || !fs::equivalent(i.file, page.file, ec))
				continue;
			if (!std::exchange(any, true))
				out << "# Re-exports\n\n";
			std::string names;
			for (const auto& n : i.names)
				names += (names.empty() ? "" : ", ") + (n == "*" ? std::string{"all public names"} : "`" + n + "`");
			auto target = i.targets.empty() ? pages.end() : pages.find(fs::canonical(i.targets.front()));
			out << "- " << names << " of "
				<< (target != pages.end() ? "\\ref " + pageId(target->second) : "`" + i.name + "`") << "\n";
		}
		out << "\nThe source code: \\subpage " << id << "_source \"" << page.path << "\".\n";
		return out.str();
	}
} // namespace

int main(int argc, char* argv[]) {
	CLI::App app("mdl-doc -- writes the API documentation of .dlm files as Doxygen pages");
	app.set_version_flag("--version", std::string{mdl::version});
	fs::path root, output;
	std::string name, title;
	std::vector<fs::path> importDirs;
	app.add_option("root", root, "The package directory (e.g. std/).")->required()->check(CLI::ExistingDirectory);
	app.add_option("-o,--output", output, "The directory of the generated pages.")->required();
	app.add_option("--name", name, "The package name, the prefix of the page names (default: the directory name).");
	app.add_option("--title", title, "The title of the package's index page (default: the package name).");
	app.add_option("-i,--import-dir", importDirs, "Additional import directories.");
	CLI11_PARSE(app, argc, argv);
	root = fs::canonical(root);
	if (name.empty())
		name = root.filename().string();
	if (title.empty())
		title = name;

	std::vector<Page> pages;
	for (const auto& entry : fs::recursive_directory_iterator(root))
		if (entry.is_regular_file() && entry.path().extension() == ".dlm") {
			std::string dotted = name;
			for (const auto& part : fs::relative(entry.path(), root).replace_extension())
				dotted += "." + part.string();
			const std::string path = name + "/" + fs::relative(entry.path(), root).generic_string();
			pages.push_back({dotted, fs::canonical(entry.path()), path, {}, {}, {}});
		}
	std::ranges::sort(pages, {}, &Page::dotted);
	std::map<fs::path, std::string> dottedOf;
	for (const auto& p : pages)
		dottedOf[p.file] = p.dotted;

	bool ok = true;
	Links links;
	std::vector<std::pair<std::shared_ptr<const mdl::analysis::SemanticIndex>, std::string>> analyses;
	for (auto& page : pages) {
		auto result = mdl::parser::analyze(page.file, {.importDirs = importDirs});
		for (const auto& d : result.diagnostics)
			if (d.severity == mdl::Severity::error) {
				std::cerr << d.file << ":" << d.range.begin.line << ": error: " << d.message << "\n";
				ok = false;
			}
		std::ifstream in{page.file, std::ios::binary};
		const std::string source{std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
		page.text = render(page, *result.index, dottedOf, mdl::analysis::fileComment(source), links);
		page.brief = brief(mdl::analysis::fileComment(source));
		analyses.emplace_back(result.index, source);
	}
	// The source pages after all pages: they link to the cards of every file.
	for (size_t i = 0; i < pages.size(); ++i)
		pages[i].sourceText = renderSource(pages[i], analyses[i].second, *analyses[i].first, links, dottedOf);

	fs::create_directories(output);
	std::ofstream index{output / (name + ".md")};
	// The index: the files with their brief description (of the `@file` comment).
	index << "\\page " << pageId(name) << "_api " << title << "\n\n| File | Description |\n|---|---|\n";
	for (const auto& page : pages)
		index << "| \\ref " << pageId(page.dotted) << " \"" << page.dotted << "\" | " << tableCell(page.brief) << " |\n";
	index << "\n<div class=\"hidden-subpages\">\n\n";
	for (const auto& page : pages) {
		std::ofstream{output / (page.dotted + ".md")} << page.text;
		std::ofstream{output / (page.dotted + ".source.md")} << page.sourceText;
		index << "- \\subpage " << pageId(page.dotted) << "\n";
	}
	index << "\n</div>\n";
	return ok ? 0 : 1;
}
