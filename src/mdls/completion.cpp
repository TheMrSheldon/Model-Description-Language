// textDocument/completion, from the tokens before the cursor (the text need not parse) and the document's index:
// members after `x.`, named hyperparameters after `.` in `Module[..., .name=...]`, annotations after `@`, import paths
// after `import`, else the names in scope, keywords, and types. Nothing is proposed inside comments.
#include "./features.hpp"
#include "./tokens.hpp"

#include <mdlLexer.h>

#include <set>

using antlr4::mdlLexer;

namespace {
	using mdl::analysis::SymbolKind;

	/** @brief The protocol's CompletionItemKind of a definition. */
	int completionKind(SymbolKind kind) {
		switch (kind) {
		case SymbolKind::module:
			return 7; // Class
		case SymbolKind::structure:
			return 22; // Struct
		case SymbolKind::function:
			return 3; // Function
		case SymbolKind::method:
			return 2; // Method
		case SymbolKind::hyperparameter:
		case SymbolKind::templateParameter:
			return 25; // TypeParameter
		case SymbolKind::member:
			return 5; // Field
		default:
			return 6; // Variable
		}
	}

	using mdls::Token;
	using mdls::tokenize;

	class Completer {
	public:
		Completer(const mdls::Document& doc, const mdl::parser::ParseOptions& options) : doc(doc), options(options) {}

		mdls::json complete(mdl::SourcePosition cursor) {
			const auto tokens = tokenize(doc.text);
			// The tokens before the cursor; the word being typed (if any) is not part of the context.
			std::vector<const Token*> before;
			for (const auto& t : tokens) {
				if (t.begin >= cursor)
					break;
				if (t.end > cursor || (t.type == mdlLexer::LINECOMMENT && t.begin.line == cursor.line)) {
					if (t.type == mdlLexer::BLOCKCOMMENT || t.type == mdlLexer::LINECOMMENT)
						return items; // inside a comment
				}
				if (!t.comment())
					before.push_back(&t);
			}
			if (!before.empty() && before.back()->end == cursor &&
				(before.back()->identifier() || isWord(*before.back())))
				before.pop_back(); // the word being typed
			auto at = [&](size_t fromEnd) -> const Token* {
				return fromEnd < before.size() ? before[before.size() - 1 - fromEnd] : nullptr;
			};

			if (importPath(before))
				return items;
			if (einops(tokens, before, cursor))
				return items;
			if (at(0) != nullptr && at(0)->is("@")) {
				add("conditional", 14, "@conditional(c): the member exists only if c holds", "");
				return items;
			}
			if (at(0) != nullptr && at(0)->is(".")) {
				const Token* previous = at(1);
				if (previous != nullptr && (previous->is(",") || previous->is("[")))
					namedHyperparameters(before, cursor);
				else if (previous != nullptr && previous->identifier())
					members(*previous, cursor);
				return items;
			}
			names(cursor);
			return items;
		}

	private:
		const mdls::Document& doc;
		const mdl::parser::ParseOptions& options;
		mdls::json items = mdls::json::array();
		std::set<std::string> seen;

		static bool isWord(const Token& t) {
			return !t.text.empty() && std::isalpha(static_cast<unsigned char>(t.text[0]));
		}

		void add(const std::string& label, int kind, const std::string& detail, const std::string& documentation) {
			if (!seen.insert(label).second)
				return;
			mdls::json item{{"label", label}, {"kind", kind}};
			if (!detail.empty())
				item["detail"] = detail;
			if (!documentation.empty())
				item["documentation"] = {{"kind", "markdown"}, {"value", documentation}};
			items.push_back(std::move(item));
		}

		void add(const mdl::analysis::Symbol& s) { add(s.name, completionKind(s.kind), s.detail, s.documentation); }

		/**
		 * @brief Inside an einops pattern (`x[... (h d) -> |`): the axis names of the pattern; after `|` the names to
		 * bind (`h=`). Whether the cursor is in one: an open `[` whose contents (up to its `]`) have `->`.
		 */
		bool
		einops(const std::vector<Token>& tokens, const std::vector<const Token*>& before, mdl::SourcePosition cursor) {
			int depth = 0;
			const Token* open = nullptr;
			for (auto it = before.rbegin(); it != before.rend(); ++it) {
				if ((*it)->is("]") || (*it)->is(")"))
					++depth;
				else if ((*it)->is("(") && depth > 0)
					--depth;
				else if ((*it)->is("[") && depth-- == 0) {
					open = *it;
					break;
				} else if ((*it)->is("(")) {
					continue; // a group of axes `(h d)`
				}
			}
			if (open == nullptr)
				return false;
			std::vector<const Token*> pattern;
			bool arrow = false, binding = false;
			int d = 0;
			for (const auto& t : tokens) {
				if (t.begin <= open->begin || t.comment())
					continue;
				if (t.is("[") || t.is("("))
					++d;
				else if ((t.is("]") || t.is(")")) && d-- == 0)
					break;
				arrow |= t.is("->");
				if (t.is("|") && t.begin < cursor)
					binding = true;
				if (t.identifier() && !(t.begin <= cursor && cursor <= t.end))
					pattern.push_back(&t);
			}
			if (!arrow)
				return false;
			std::set<std::string> names;
			for (const auto* t : pattern)
				names.insert(t->text);
			if (doc.index != nullptr)
				for (const auto& a : doc.index->axes())
					if (a.file == doc.sourceName() && a.range.begin.line == doc.toIndex(open->begin).line)
						names.insert(a.name);
			for (const auto& name : names) {
				add(name, 25, binding ? "the size of the axis" : "an axis of the pattern", "");
				if (binding)
					items.back()["insertText"] = name + "=";
			}
			return true;
		}

		/** @brief `import a.b.`: the packages and files there (only the standard library is listed). */
		bool importPath(const std::vector<const Token*>& before) {
			size_t k = before.size();
			std::vector<std::string> path;
			while (k >= 2 && before[k - 1]->is(".") && before[k - 2]->identifier()) {
				path.insert(path.begin(), before[k - 2]->text);
				k -= 2;
			}
			if (k == 0 || !before[k - 1]->is("import"))
				return false;
			if (path.empty())
				add("std", 9, "the standard library", "");
			else if (path == std::vector<std::string>{"std"}) {
				for (const auto& dir :
					 options.stdlibDirs.empty() ? mdl::parser::defaultStdlibDirs() : options.stdlibDirs) {
					std::error_code ec;
					for (const auto& entry : std::filesystem::directory_iterator(dir, ec))
						if (entry.path().extension() == ".dlm")
							add(entry.path().stem().string(), 17, "std." + entry.path().stem().string(), "");
				}
			}
			return true;
		}

		/** @brief The definition of @p name visible at @p cursor (resolved like the frontend: innermost first). */
		const mdl::analysis::Symbol* visible(const std::string& name, mdl::SourcePosition cursor) const {
			for (const auto* s : doc.index->visibleAt(doc.sourceName(), doc.toIndex(cursor)))
				if (s->name == name)
					return s;
			return nullptr;
		}

		/** @brief `Module[4, 3, .`: the hyperparameters of the module of the innermost open `[`. */
		void namedHyperparameters(const std::vector<const Token*>& before, mdl::SourcePosition cursor) {
			if (doc.index == nullptr)
				return;
			int depth = 0;
			for (size_t k = before.size(); k-- > 0;) {
				if (before[k]->is("]") || before[k]->is(")"))
					++depth;
				else if ((before[k]->is("[") || before[k]->is("(")) && depth-- == 0) {
					if (!before[k]->is("[") || k == 0 || !before[k - 1]->identifier())
						return;
					const auto* module = visible(before[k - 1]->text, cursor);
					if (module == nullptr || module->kind != SymbolKind::module)
						return;
					for (const auto& s : doc.index->symbols())
						if (s.kind == SymbolKind::hyperparameter && s.container == module->name &&
							s.file == module->file)
							add(s);
					return;
				}
			}
		}

		/** @brief `x.`: the members of the value of @p name (its reference there, else its visible definition's). */
		void members(const Token& name, mdl::SourcePosition cursor) {
			if (doc.index == nullptr)
				return;
			const auto& index = *doc.index;
			const mdl::analysis::Reference* target = index.referenceAt(doc.sourceName(), doc.toIndex(name.begin));
			if (target != nullptr && target->range.end.column - target->range.begin.column != name.text.size())
				target = nullptr; // (a different name at that position of an earlier text)
			if (target == nullptr || target->members.empty())
				if (const auto* s = visible(name.text, cursor)) {
					const size_t id = static_cast<size_t>(s - index.symbols().data());
					for (const auto& r : index.references())
						if (r.symbol == id && !r.members.empty())
							target = &r;
				}
			if (target == nullptr)
				return;
			for (const auto& m : target->members) {
				// The member's definition (in the module of the value's type), for its kind and documentation.
				const mdl::analysis::Symbol* def = nullptr;
				for (const auto& s : index.symbols())
					if (s.name == m && (s.kind == SymbolKind::member || s.kind == SymbolKind::method) &&
						target->types.contains(s.container))
						def = &s;
				if (def != nullptr)
					add(*def);
				else
					add(m, m == "T" ? 2 : 5, "", "");
			}
		}

		void names(mdl::SourcePosition cursor) {
			if (doc.index != nullptr)
				for (const auto* s : doc.index->visibleAt(doc.sourceName(), doc.toIndex(cursor)))
					add(*s);
			for (const char* keyword :
				 {"auto", "return", "for", "in", "masked", "module", "struct", "import", "template", "array", "true",
				  "false"})
				add(keyword, 14, "", "");
			for (const char* type : {"float", "float32", "bfloat16", "float16", "int", "uint", "bool"})
				add(type, 25, "", "");
		}
	};
} // namespace

mdls::json mdls::completion(const Document& doc, const json& params, const mdl::parser::ParseOptions& options) {
	return Completer{doc, options}.complete(toSource(LineIndex{doc.text}, params.at("position")));
}
