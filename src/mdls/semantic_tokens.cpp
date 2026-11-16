#include "./semantic_tokens.hpp"
#include "./utf8.hpp"

#include <mdlLexer.h>
#include <mdlParser.h>

#include <algorithm>
#include <map>
#include <regex>
#include <set>

using antlr4::mdlLexer;
using antlr4::mdlParser;

namespace {
	enum Type : unsigned {
		Namespace,
		TypeName,
		Class,
		Struct,
		TypeParameter,
		Parameter,
		Variable,
		Property,
		Function,
		Method,
		Decorator,
		Axis,
		Keyword,   ///< (only in doc comments: the commands)
		Label,	   ///< (only in doc comments: targets of `\ref`, `\see`, keys of `\cite`)
		String,	   ///< (only in doc comments: math)
		Discarded, ///< `_`: a discarded value (loop variable, declaration, einops axis)
	};
	enum Modifier : unsigned { Declaration = 1, Readonly = 2, DefaultLibrary = 4, Documentation = 8 };

	const std::set<std::string> primitiveTypes{"float", "float32", "float16", "bfloat16", "int", "uint", "bool"};

	struct Name {
		unsigned type, modifiers;
	};

	class Classifier {
	public:
		std::vector<mdls::SemanticToken> tokens;

		void program(mdlParser::ProgramContext* ctx) {
			// Global definitions first: they are visible everywhere in the file.
			scopes.emplace_back();
			for (auto* m : ctx->module_def())
				if (m->IDENT())
					declare(m->IDENT()->getSymbol(), Class, false);
			for (auto* s : ctx->struct_def())
				if (s->IDENT())
					declare(s->IDENT()->getSymbol(), Struct, false);
			for (auto* f : ctx->funcdef())
				if (f->IDENT())
					declare(f->IDENT()->getSymbol(), Function, false);
			visitChildren(ctx);
		}

	private:
		std::vector<std::map<std::string, Name>> scopes;
		std::set<size_t> classified; ///< token indices already emitted

		void emit(antlr4::Token* t, unsigned type, unsigned modifiers = 0) {
			if (t == nullptr || t->getType() == antlr4::Token::EOF || !classified.insert(t->getTokenIndex()).second)
				return;
			tokens.push_back({t->getLine() - 1, t->getCharPositionInLine(), t->getText().size(), type, modifiers});
		}

		/** @brief Declares @p t in the innermost scope (and emits it as a declaration if @p emitIt). */
		void declare(antlr4::Token* t, unsigned type, bool emitIt = true, unsigned modifiers = 0) {
			if (t == nullptr || t->getText() == "_")
				return; // `_` is not bound (and is emitted with its terminal, see visit)
			scopes.back()[t->getText()] = {type, modifiers};
			if (emitIt)
				emit(t, type, modifiers | Declaration);
		}

		std::optional<Name> lookup(const std::string& name) const {
			for (auto it = scopes.rbegin(); it != scopes.rend(); ++it)
				if (auto found = it->find(name); found != it->end())
					return found->second;
			return std::nullopt;
		}

		struct Scope {
			Classifier& c;
			explicit Scope(Classifier& c) : c(c) { c.scopes.emplace_back(); }
			~Scope() { c.scopes.pop_back(); }
		};

		void visitChildren(antlr4::tree::ParseTree* tree) {
			for (auto* child : tree->children)
				visit(child);
		}

		void visit(antlr4::tree::ParseTree* tree) {
			if (tree == nullptr)
				return;
			if (auto* terminal = dynamic_cast<antlr4::tree::TerminalNode*>(tree)) {
				if (terminal->getSymbol()->getText() == "_")
					emit(terminal->getSymbol(), Discarded); // a discarded value: not a variable
				return; // keywords, literals, operators: lexical (the editor's grammar colors them)
			}
			if (auto* ctx = dynamic_cast<mdlParser::Import_stmtContext*>(tree)) {
				for (auto* part : ctx->path)
					emit(part, Namespace);
				visitChildren(ctx);
			} else if (auto* ctx = dynamic_cast<mdlParser::Module_defContext*>(tree)) {
				if (ctx->IDENT())
					emit(ctx->IDENT()->getSymbol(), Class, Declaration);
				Scope hyperparameters{*this};
				for (auto* p : ctx->param())
					param(p, TypeParameter);
				Scope members{*this};
				for (auto* v : ctx->vardecl())
					if (v->IDENT())
						declare(v->IDENT()->getSymbol(), Property, false);
				for (auto* f : ctx->funcdef())
					if (f->IDENT())
						declare(f->IDENT()->getSymbol(), Method, false);
				for (auto* child : ctx->children)
					if (dynamic_cast<mdlParser::VardeclContext*>(child) ||
						dynamic_cast<mdlParser::FuncdefContext*>(child) ||
						dynamic_cast<antlr4::tree::TerminalNode*>(child))
						visit(child);
			} else if (auto* ctx = dynamic_cast<mdlParser::Struct_defContext*>(tree)) {
				if (ctx->IDENT())
					emit(ctx->IDENT()->getSymbol(), Struct, Declaration);
				Scope members{*this};
				visitChildren(ctx);
			} else if (auto* ctx = dynamic_cast<mdlParser::VardeclContext*>(tree)) {
				for (auto* a : ctx->annotation())
					visit(a);
				if (ctx->IDENT())
					declare(ctx->IDENT()->getSymbol(), memberKind(), true);
				visit(ctx->type());
			} else if (auto* ctx = dynamic_cast<mdlParser::FuncdefContext*>(tree)) {
				const bool member = inModule();
				Scope function{*this};
				if (auto* t = ctx->templatespec())
					for (auto* v : t->vardecl()) {
						if (v->IDENT())
							declare(v->IDENT()->getSymbol(), TypeParameter);
						visit(v->type());
					}
				visit(ctx->rettype());
				if (ctx->IDENT())
					emit(ctx->IDENT()->getSymbol(), member ? Method : Function, Declaration);
				for (auto* p : ctx->param())
					param(p, Parameter);
				visit(ctx->expr());
				for (auto* s : ctx->statement())
					visit(s);
				visitChildren(ctx); // terminals (keywords)
			} else if (auto* ctx = dynamic_cast<mdlParser::DeclarationStatementContext*>(tree)) {
				visit(ctx->expr()); // the initializer cannot see the declared names
				visit(ctx->type());
				for (auto* n : ctx->names)
					declare(n, Variable);
				visitChildren(ctx);
			} else if (auto* ctx = dynamic_cast<mdlParser::AssignStatementContext*>(tree)) {
				reference(ctx->name, false);
				visitChildren(ctx);
			} else if (auto* ctx = dynamic_cast<mdlParser::SlotAssignStatementContext*>(tree)) {
				reference(ctx->name, false);
				visitChildren(ctx);
			} else if (auto* ctx = dynamic_cast<mdlParser::ForStatementContext*>(tree)) {
				visit(ctx->lo);
				visit(ctx->hi);
				Scope loop{*this};
				declare(ctx->var, Variable, true, Readonly);
				visitChildren(ctx);
			} else if (auto* ctx = dynamic_cast<mdlParser::ComprehensionTypeContext*>(tree)) {
				visit(ctx->lo);
				visit(ctx->hi);
				Scope loop{*this};
				declare(ctx->var, Variable, true, Readonly);
				visitChildren(ctx);
			} else if (auto* ctx = dynamic_cast<mdlParser::TypeSpecContext*>(tree)) {
				if (auto* t = ctx->typename_) {
					if (primitiveTypes.contains(t->getText()))
						emit(t, TypeName, DefaultLibrary);
					else if (auto name = lookup(t->getText()))
						emit(t, name->type, name->modifiers);
					else
						emit(t, Class);
				}
				visitChildren(ctx);
			} else if (auto* ctx = dynamic_cast<mdlParser::Type_argContext*>(tree)) {
				if (ctx->name)
					emit(ctx->name, TypeParameter);
				visitChildren(ctx);
			} else if (auto* ctx = dynamic_cast<mdlParser::AnnotationContext*>(tree)) {
				emit(ctx->start, Decorator);
				emit(ctx->name, Decorator);
				visitChildren(ctx);
			} else if (auto* ctx = dynamic_cast<mdlParser::VarAccessExpressionContext*>(tree)) {
				if (ctx->IDENT())
					reference(ctx->IDENT()->getSymbol(), isCalled(ctx));
			} else if (auto* ctx = dynamic_cast<mdlParser::AccessExpressionContext*>(tree)) {
				visit(ctx->expr());
				if (ctx->IDENT())
					emit(ctx->IDENT()->getSymbol(), isCalled(ctx) ? Method : Property);
				visitChildren(ctx);
			} else if (auto* ctx = dynamic_cast<mdlParser::EinopsContext*>(tree)) {
				for (auto* child : ctx->children) {
					if (auto* spec = dynamic_cast<mdlParser::Einops_spec_fullContext*>(child)) {
						for (auto* atom : spec->einops_atom())
							if (atom->IDENT())
								emit(atom->IDENT()->getSymbol(), Axis);
						visitChildren(spec);
					} else if (auto* kv = dynamic_cast<mdlParser::KeyvalueContext*>(child)) {
						if (kv->IDENT())
							emit(kv->IDENT()->getSymbol(), Axis);
						visit(kv->expr());
						visitChildren(kv);
					} else {
						visit(child);
					}
				}
			} else {
				visitChildren(tree);
			}
		}

		void param(mdlParser::ParamContext* p, unsigned type) {
			visit(p->type());
			visit(p->default_);
			if (p->IDENT())
				declare(p->IDENT()->getSymbol(), type);
			visitChildren(p);
		}

		void reference(antlr4::Token* t, bool called) {
			if (t == nullptr)
				return;
			if (auto name = lookup(t->getText()))
				emit(t, called && name->type == Property ? Method : name->type, name->modifiers);
			else if (primitiveTypes.contains(t->getText()))
				emit(t, TypeName, DefaultLibrary); // float32(x)
			else
				emit(t, called ? Function : Variable, called ? DefaultLibrary : 0); // imported
		}

		/** @brief Whether @p ctx is the function of a call (`f(x)`, `inst.member(x)`). */
		static bool isCalled(antlr4::ParserRuleContext* ctx) {
			auto* call = dynamic_cast<mdlParser::MethodcallExpressionContext*>(ctx->parent);
			return call != nullptr && call->func == ctx;
		}

		bool inModule() const {
			// The member scope of a module is the second scope on top of its hyperparameter scope.
			for (auto it = scopes.rbegin(); it != scopes.rend(); ++it)
				for (const auto& [_, n] : *it)
					if (n.type == Property || n.type == Method)
						return true;
			return false;
		}

		unsigned memberKind() const { return Property; }
	};
} // namespace

namespace {
	/**
	 * @brief The Doxygen commands of a doc comment (`/** *\/`, `///`) and their arguments: commands as keywords,
	 * parameter names (`@param x`, `\p x`) as parameters, targets of `\ref`, `\see`, `\sa` and keys of `\cite` as
	 * labels, math as strings (all with the modifier `documentation`).
	 */
	void docCommentTokens(const antlr4::Token& token, std::vector<mdls::SemanticToken>& out) {
		const std::string text = token.getText();
		if (!(text.starts_with("/**") || text.starts_with("/*!") || text.starts_with("///") || text.starts_with("//!")))
			return;
		// Positions: line and code-point column of each byte offset.
		auto points = mdls::utf8::codePoints;
		auto position = [&](size_t offset) {
			const size_t lineStart = text.rfind('\n', offset == 0 ? 0 : offset - 1);
			const size_t newlines = std::count(text.begin(), text.begin() + offset, '\n');
			const size_t line = token.getLine() - 1 + newlines;
			const size_t column =
					newlines == 0 ? token.getCharPositionInLine() + points(std::string_view{text}.substr(0, offset))
								  : points(std::string_view{text}.substr(lineStart + 1, offset - lineStart - 1));
			return std::pair{line, column};
		};
		auto emit = [&](size_t offset, size_t length, unsigned type) {
			// (one line at a time: tokens do not span lines)
			while (length > 0) {
				const size_t newline = text.find('\n', offset);
				const size_t piece = newline == std::string::npos ? length : std::min(length, newline - offset);
				const auto [line, column] = position(offset);
				if (piece > 0)
					out.push_back(
							{line, column, points(std::string_view{text}.substr(offset, piece)), type, Documentation}
					);
				if (piece == length)
					break;
				length -= piece + 1;
				offset = newline + 1;
			}
		};
		// The word after offset @p from (skipping spaces); its offset and length (0 if none).
		auto word = [&](size_t from) -> std::pair<size_t, size_t> {
			size_t begin = text.find_first_not_of(" \t", from);
			if (begin == std::string::npos || text[begin] == '\n')
				return {from, 0};
			size_t end = begin;
			while (end < text.size() && (std::isalnum(static_cast<unsigned char>(text[end])) || text[end] == '_' ||
										 text[end] == '.' || text[end] == ':' || text[end] == '-'))
				++end;
			while (end > begin && (text[end - 1] == '.' || text[end - 1] == ':'))
				--end; // (sentence punctuation)
			return {begin, end - begin};
		};
		const std::regex command{R"([@\\](f[$(\[{]|[a-zA-Z]+))"};
		for (std::sregex_iterator it{text.begin(), text.end(), command}, end; it != end; ++it) {
			const size_t at = static_cast<size_t>(it->position());
			if (at > 0 && (std::isalnum(static_cast<unsigned char>(text[at - 1])) || text[at - 1] == '\\'))
				continue; // (e.g. an e-mail address)
			const std::string name = (*it)[1];
			if (name.starts_with("f")) {
				// Math up to the matching end (`\f$`, `\f)`, `\f]`, `\f}`).
				if (name.size() == 2) {
					const char close = name[1] == '$' ? '$' : name[1] == '(' ? ')' : name[1] == '[' ? ']' : '}';
					size_t stop = at + 3;
					while ((stop = text.find(std::string{"f"} + close, stop)) != std::string::npos &&
						   !(stop > 0 && (text[stop - 1] == '\\' || text[stop - 1] == '@')))
						++stop;
					if (stop != std::string::npos) {
						emit(at, stop + 2 - at, String);
						continue;
					}
				}
			}
			emit(at, it->length(), Keyword);
			const size_t after = at + static_cast<size_t>(it->length());
			if (name == "param" || name == "tparam") {
				size_t from = after;
				if (from < text.size() && text[from] == '[') // `@param[in]`
					from = text.find(']', from) == std::string::npos ? from : text.find(']', from) + 1;
				if (auto [begin, length] = word(from); length > 0)
					emit(begin, length, Parameter);
			} else if (name == "p") {
				if (auto [begin, length] = word(after); length > 0)
					emit(begin, length, Parameter);
			} else if (name == "ref" || name == "cite" || name == "see" || name == "sa") {
				if (auto [begin, length] = word(after); length > 0)
					emit(begin, length, Label);
			}
		}
	}
} // namespace

const std::vector<std::string>& mdls::semanticTokenTypes() {
	static const std::vector<std::string> types{
			"namespace", "type",   "class",		"struct", "typeParameter", "parameter", "variable", "property",
			"function",	 "method", "decorator", "axis",	  "keyword",	   "label",		"string",	"discarded",
	};
	return types;
}

const std::vector<std::string>& mdls::semanticTokenModifiers() {
	static const std::vector<std::string> modifiers{"declaration", "readonly", "defaultLibrary", "documentation"};
	return modifiers;
}

std::vector<mdls::SemanticToken> mdls::semanticTokens(std::string_view text) {
	antlr4::ANTLRInputStream input;
	input.load(std::string{text}, /*lenient=*/true); // (like the parser, see mdl::invalidUtf8)
	mdlLexer lexer{&input};
	lexer.removeErrorListeners();
	antlr4::CommonTokenStream stream{&lexer};
	mdlParser parser{&stream};
	parser.removeErrorListeners();
	Classifier classifier;
	classifier.program(parser.program());
	auto tokens = std::move(classifier.tokens);
	for (auto* t : stream.getTokens())
		if (t->getType() == mdlLexer::BLOCKCOMMENT || t->getType() == mdlLexer::LINECOMMENT)
			docCommentTokens(*t, tokens);
	std::sort(tokens.begin(), tokens.end(), [](const auto& a, const auto& b) {
		return std::pair{a.line, a.column} < std::pair{b.line, b.column};
	});
	return tokens;
}

mdls::json mdls::encodeSemanticTokens(const std::vector<SemanticToken>& tokens, const LineIndex& lines) {
	json data = json::array();
	size_t prevLine = 0, prevColumn = 0;
	for (const auto& t : tokens) {
		const size_t column = lines.toUtf16(t.line, t.column);
		const size_t length = lines.utf16Length(t.line, t.column, t.length);
		data.push_back(t.line - prevLine);
		data.push_back(t.line == prevLine ? column - prevColumn : column);
		data.push_back(length);
		data.push_back(t.type);
		data.push_back(t.modifiers);
		prevLine = t.line;
		prevColumn = column;
	}
	return json{{"data", data}};
}
