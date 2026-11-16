// The formatter of `.dlm` files (see include/mdl/format.hpp).
//
// It works on the token stream (comments and whitespace are on ANTLR's hidden channel) and uses the parse tree only to
// classify tokens: where a line must break (definitions, members, statements, blocks), where a source line break is
// kept (after a template header or an annotation), and how a token is spaced (unary minus, annotations, `?:`, named
// hyperparameters, einops). Lines that are too long are wrapped afterwards; consecutive trailing comments are aligned.
#include <mdl/format.hpp>

#include "../parser/errorlistener.hpp"

#include <mdlLexer.h>
#include <mdlParser.h>

#include <algorithm>
#include <cctype>
#include <format>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <unordered_map>

using antlr4::mdlLexer;
using antlr4::mdlParser;

namespace {
	/** @brief A token that is printed (code or comment), with the whitespace that preceded it in the source. */
	struct Tok {
		size_t index; ///< in the token stream
		size_t type;
		std::string text;
		size_t column;			   ///< in the source (code points)
		size_t newlinesBefore = 0; ///< line breaks in the whitespace before it
		bool comment = false, lineComment = false;
	};

	/** @brief What the parse tree says about a token. */
	struct Props {
		bool breakBefore = false;	  ///< a line break before it is required
		bool keepBreakBefore = false; ///< a line break before it in the source is kept
		bool blockOpen = false;		  ///< `{` of a block: the lines after it are indented
		bool blockClose = false;	  ///< `}` of a block
		bool unary = false;			  ///< unary minus
		bool annotationAt = false;	  ///< `@` of an annotation
		bool conditional = false;	  ///< `?` or `:` of `c ? a : b`
		bool tight = false;			  ///< `=` without spaces (`.bias=false`, einops `h=heads`)
		bool einops = false;		  ///< inside an einops pattern
		bool templateAngle = false;	  ///< `<` or `>` of `template<...>`
	};

	bool isComment(size_t type) { return type == mdlLexer::BLOCKCOMMENT || type == mdlLexer::LINECOMMENT; }

	std::string rstrip(std::string s) {
		while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r'))
			s.pop_back();
		return s;
	}

	class Formatter {
	public:
		Formatter(const mdl::format::FormatOptions& options, antlr4::CommonTokenStream& stream) : options(options) {
			for (auto* t : stream.getTokens()) {
				if (t->getType() == mdlLexer::WS) {
					pendingNewlines += t->getText() == "\n" ? 1 : 0;
					continue;
				}
				if (t->getType() == antlr4::Token::EOF)
					break;
				Tok tok{
						.index = t->getTokenIndex(),
						.type = t->getType(),
						.text = t->getText(),
						.column = t->getCharPositionInLine(),
						.newlinesBefore = pendingNewlines,
						.comment = isComment(t->getType()),
						.lineComment = t->getType() == mdlLexer::LINECOMMENT,
				};
				if (tok.lineComment)
					tok.text = rstrip(tok.text);
				position[tok.index] = toks.size();
				toks.push_back(std::move(tok));
				pendingNewlines = 0;
			}
		}

		void classify(antlr4::tree::ParseTree* tree) {
			using namespace antlr4::tree;
			if (auto* ctx = dynamic_cast<mdlParser::ProgramContext*>(tree)) {
				for (auto* child : ctx->children)
					if (auto* rule = dynamic_cast<antlr4::ParserRuleContext*>(child))
						props(rule->start).breakBefore = true;
			} else if (auto* ctx = dynamic_cast<mdlParser::Struct_defContext*>(tree)) {
				block(ctx, ctx->vardecl());
			} else if (auto* ctx = dynamic_cast<mdlParser::Module_defContext*>(tree)) {
				std::vector<antlr4::ParserRuleContext*> members;
				for (auto* child : ctx->children)
					if (dynamic_cast<mdlParser::VardeclContext*>(child) ||
						dynamic_cast<mdlParser::FuncdefContext*>(child))
						members.push_back(static_cast<antlr4::ParserRuleContext*>(child));
				block(ctx, members);
			} else if (auto* ctx = dynamic_cast<mdlParser::FuncdefContext*>(tree)) {
				if (ctx->templatespec() != nullptr)
					keepBreakAfter(ctx->templatespec()->stop);
				if (!ctx->statement().empty() || ctx->expr() == nullptr)
					block(ctx, ctx->statement());
			} else if (auto* ctx = dynamic_cast<mdlParser::ForStatementContext*>(tree)) {
				block(ctx, ctx->statement());
			} else if (auto* ctx = dynamic_cast<mdlParser::AnonTypeContext*>(tree)) {
				// An inline struct type is a block if it is written as one (a line break after `{`), else one line.
				auto* open = terminal(ctx, "{");
				auto* next = ctx->vardecl().empty() ? terminal(ctx, "}") : ctx->vardecl().front()->start;
				if (open != nullptr && next != nullptr && at(next).newlinesBefore > 0)
					block(ctx, ctx->vardecl());
			} else if (auto* ctx = dynamic_cast<mdlParser::TemplatespecContext*>(tree)) {
				for (auto* child : ctx->children)
					if (auto* t = dynamic_cast<TerminalNode*>(child); t && (t->getText() == "<" || t->getText() == ">"))
						props(t->getSymbol()).templateAngle = true;
			} else if (auto* ctx = dynamic_cast<mdlParser::AnnotationContext*>(tree)) {
				props(ctx->start).annotationAt = true;
				keepBreakAfter(ctx->stop);
			} else if (auto* ctx = dynamic_cast<mdlParser::NegateExpressionContext*>(tree)) {
				props(ctx->start).unary = true;
			} else if (auto* ctx = dynamic_cast<mdlParser::ConditionalExpressionContext*>(tree)) {
				for (auto* child : ctx->children)
					if (auto* t = dynamic_cast<TerminalNode*>(child))
						props(t->getSymbol()).conditional = true;
			} else if (auto* ctx = dynamic_cast<mdlParser::Type_argContext*>(tree)) {
				if (ctx->name != nullptr)
					if (auto* eq = terminal(ctx, "="))
						props(eq).tight = true;
			} else if (auto* ctx = dynamic_cast<mdlParser::KeyvalueContext*>(tree)) {
				if (auto* eq = terminal(ctx, "="))
					props(eq).tight = true;
			} else if (auto* ctx = dynamic_cast<mdlParser::EinopsContext*>(tree)) {
				// The pattern (`... (h d) -> h ... d`) and the separators; the values of the bindings are expressions.
				for (auto* child : ctx->children) {
					if (auto* t = dynamic_cast<TerminalNode*>(child))
						props(t->getSymbol()).einops = true;
					else if (auto* spec = dynamic_cast<mdlParser::Einops_spec_fullContext*>(child))
						for (size_t i = spec->start->getTokenIndex(); i <= spec->stop->getTokenIndex(); ++i)
							if (position.contains(i))
								properties[i].einops = true;
				}
			}
			for (size_t i = 0; i < tree->children.size(); ++i)
				classify(tree->children[i]);
		}

		std::string run() {
			buildLines();
			std::vector<std::string> out;
			for (auto& line : lines)
				wrap(line, out);
			alignTrailingComments(out);
			std::string text;
			for (auto& l : out)
				text += rstrip(l) + "\n";
			// No blank lines at the end.
			while (text.size() >= 2 && text[text.size() - 1] == '\n' && text[text.size() - 2] == '\n')
				text.pop_back();
			return text == "\n" ? "" : text;
		}

	private:
		const mdl::format::FormatOptions& options;
		std::vector<Tok> toks;
		std::unordered_map<size_t, size_t> position; ///< token index -> index in toks
		std::unordered_map<size_t, Props> properties;
		size_t pendingNewlines = 0;

		/** @brief A line: its indentation and tokens (indices into toks); empty for a blank line. */
		struct Line {
			size_t indent = 0;
			std::vector<size_t> tokens;
			bool continuation = false; ///< a statement continued from the previous line
		};
		std::vector<Line> lines;

		Props& props(antlr4::Token* t) { return properties[t->getTokenIndex()]; }
		const Props& props(size_t tokIndex) { return properties[toks[tokIndex].index]; }
		const Tok& at(antlr4::Token* t) { return toks[position.at(t->getTokenIndex())]; }

		static antlr4::Token* terminal(antlr4::ParserRuleContext* ctx, std::string_view text) {
			for (auto* child : ctx->children)
				if (auto* t = dynamic_cast<antlr4::tree::TerminalNode*>(child); t && t->getText() == text)
					return t->getSymbol();
			return nullptr;
		}

		/** @brief Keeps a source line break after @p t (before the next printed token). */
		void keepBreakAfter(antlr4::Token* t) {
			auto it = position.find(t->getTokenIndex());
			if (it != position.end() && it->second + 1 < toks.size())
				properties[toks[it->second + 1].index].keepBreakBefore = true;
		}

		/** @brief A block `{ items }` of @p ctx: each item on its own line, indented; `{}` if it is empty. */
		template <typename Items>
		void block(antlr4::ParserRuleContext* ctx, const Items& items) {
			auto* open = terminal(ctx, "{");
			antlr4::Token* close = nullptr;
			for (auto it = ctx->children.rbegin(); it != ctx->children.rend() && close == nullptr; ++it)
				if (auto* t = dynamic_cast<antlr4::tree::TerminalNode*>(*it); t && t->getText() == "}")
					close = t->getSymbol();
			if (open == nullptr || close == nullptr)
				return;
			props(open).blockOpen = true;
			props(close).blockClose = true;
			for (auto* item : items)
				props(item->start).breakBefore = true;
			// Comments inside an otherwise empty block also need the break before `}`.
			if (position.at(close->getTokenIndex()) != position.at(open->getTokenIndex()) + 1)
				props(close).breakBefore = true;
		}

		/** @brief Whether a space separates @p a and @p b on one line. */
		bool spaced(size_t a, size_t b) {
			const Tok &P = toks[a], &C = toks[b];
			const Props &pp = props(a), &cp = props(b);
			const std::string &p = P.text, &c = C.text;
			if (C.comment)
				return !(p == "(" || p == "[");
			if (P.comment)
				return !(c == ")" || c == "]" || c == "," || c == ";");
			if (pp.einops && cp.einops) {
				if (c == ")" || c == "," || p == "(" || pp.tight || cp.tight || p == "/")
					return false;
				return true;
			}
			if (p == "," || p == ";")
				return true;
			if (c == ")" || c == "]" || c == "," || c == ";")
				return false;
			if (p == "(" || p == "[")
				return false;
			if (pp.templateAngle && p == "<")
				return false;
			if (cp.templateAngle)
				return false;
			if (c == "}" && p == "{")
				return false;
			if (c == "." || p == "." || c == ".." || p == "..")
				return false;
			const bool word = P.type == mdlLexer::IDENT;
			if (c == "(")
				return !(word || p == ")" || p == "]");
			if (c == "[")
				return !(word || p == ")" || p == "]" || p == "array");
			if (pp.unary || pp.annotationAt)
				return false;
			if (c == ":")
				return cp.conditional;
			if (pp.tight || cp.tight)
				return false;
			return true;
		}

		/** @brief Whether the token at @p i (a comment) is followed by an item that starts a line at block level. */
		bool beforeItem(size_t i) {
			for (size_t j = i + 1; j < toks.size(); ++j) {
				if (toks[j].comment)
					continue;
				const auto& pr = props(j);
				return pr.breakBefore || pr.keepBreakBefore || pr.blockClose;
			}
			return true;
		}

		void buildLines() {
			size_t level = 0;
			bool afterOpen = true; // at the start of the file or a block (no blank lines)
			for (size_t i = 0; i < toks.size(); ++i) {
				const Tok& tok = toks[i];
				const Props& pr = props(i);
				if (pr.blockClose && level > 0)
					--level;
				bool newline = lines.empty();
				bool continuation = false;
				if (!newline) {
					const Tok& prev = toks[i - 1];
					if (pr.breakBefore || prev.lineComment || (pr.keepBreakBefore && tok.newlinesBefore > 0) ||
						(tok.comment && tok.newlinesBefore > 0) || (prev.comment && tok.newlinesBefore > 0)) {
						newline = true;
						// A line inside a statement (e.g. after a comment within an expression) is a continuation.
						continuation =
								!(pr.breakBefore || pr.keepBreakBefore || pr.blockClose ||
								  (tok.comment && beforeItem(i)));
					}
				}
				if (newline) {
					if (!lines.empty() && !afterOpen && !pr.blockClose && tok.newlinesBefore >= 2)
						for (size_t k = 0; k < std::min<size_t>(tok.newlinesBefore - 1, options.maxEmptyLines); ++k)
							lines.push_back(Line{});
					lines.push_back(Line{.indent = level, .tokens = {}, .continuation = continuation});
				}
				lines.back().tokens.push_back(i);
				afterOpen = false;
				if (pr.blockOpen) {
					++level;
					afterOpen = true;
				}
			}
		}

		std::string indentation(size_t level, bool continuation) const {
			return std::string(level * options.indentWidth + (continuation ? 2 * options.indentWidth : 0), ' ');
		}

		/**
		 * @brief @p line (starting with @p prefix, e.g. indentation and `// `) broken between words so that the lines
		 * fit the column limit (with ReflowComments; a single long word stays on its line).
		 */
		std::string reflow(const std::string& line, size_t prefixLength) {
			if (!options.reflowComments || line.size() <= options.columnLimit || prefixLength >= line.size())
				return line;
			const std::string prefix = line.substr(0, prefixLength);
			std::istringstream words{line.substr(prefixLength)};
			std::string result, current = prefix, word;
			bool empty = true;
			while (words >> word) {
				if (!empty && current.size() + 1 + word.size() > options.columnLimit) {
					result += current + "\n";
					current = prefix;
					empty = true;
				}
				current += (empty ? "" : " ") + word;
				empty = false;
			}
			return result + current;
		}

		/** @brief The length of the comment prefix of @p line: indentation, the marker (`//`, `*`), and a space. */
		static size_t commentPrefix(const std::string& line) {
			size_t i = line.find_first_not_of(" \t");
			if (i == std::string::npos)
				return line.size();
			if (line.compare(i, 2, "//") == 0) {
				i += 2;
				while (i < line.size() && (line[i] == '/' || line[i] == '!'))
					++i;
			} else if (line.compare(i, 3, "/**") == 0 || line.compare(i, 3, "/*!") == 0) {
				i += 3;
			} else if (line.compare(i, 2, "/*") == 0) {
				i += 2;
			} else if (line[i] == '*' && line.compare(i, 2, "*/") != 0) {
				i += 1;
			}
			while (i < line.size() && line[i] == ' ')
				++i;
			return i;
		}

		/** @brief The text of a comment token at the start of a line with indentation @p indent (re-indented). */
		std::string commentText(const Tok& tok, const std::string& indent) {
			if (tok.lineComment)
				return reflow(indent + tok.text, indent.size() + commentPrefix(tok.text)).substr(indent.size());
			if (tok.text.find('\n') == std::string::npos)
				return tok.text;
			// Continuation lines of a block comment keep their indentation relative to the comment's first line.
			std::string result;
			std::istringstream in{tok.text};
			std::string line;
			bool first = true;
			while (std::getline(in, line)) {
				line = rstrip(line);
				if (first) {
					result = line;
					first = false;
					continue;
				}
				size_t strip = 0;
				while (strip < tok.column && strip < line.size() && (line[strip] == ' ' || line[strip] == '\t'))
					++strip;
				if (line.size() <= strip) {
					result += "\n";
					continue;
				}
				const std::string full = indent + line.substr(strip);
				result += "\n" + reflow(full, commentPrefix(full));
			}
			return result;
		}

		/** @brief The text of @p tokens on one line (from @p from to @p to, exclusive). */
		std::string join(const std::vector<size_t>& tokens, size_t from, size_t to, const std::string& indent) {
			std::string s;
			for (size_t k = from; k < to; ++k) {
				const Tok& tok = toks[tokens[k]];
				if (k > from && spaced(tokens[k - 1], tokens[k]))
					s += ' ';
				s += k == from && tok.comment ? commentText(tok, indent) : tok.text;
			}
			return s;
		}

		/** @brief Appends @p line, wrapped at the column limit (see wrapTokens); a trailing comment stays at its end. */
		void wrap(const Line& line, std::vector<std::string>& out) {
			if (line.tokens.empty()) {
				out.emplace_back();
				return;
			}
			const std::string indent = indentation(line.indent, line.continuation);
			std::vector<size_t> tokens = line.tokens;
			std::string trailing;
			if (tokens.size() > 1 && toks[tokens.back()].lineComment) {
				trailing = "  " + toks[tokens.back()].text;
				tokens.pop_back();
			}
			auto wrapped = wrapTokens(tokens, indent, indentation(line.indent, true));
			wrapped.back() += trailing;
			out.insert(out.end(), wrapped.begin(), wrapped.end());
		}

		/**
		 * @brief @p tokens as lines of at most the column limit where possible: a function body after `=>` goes to the
		 * next line (indented one level); else the line breaks after the commas of its outermost list, else before its binary operators;
		 * continuation lines get @p contIndent.
		 */
		std::vector<std::string>
		wrapTokens(const std::vector<size_t>& tokens, const std::string& indent, const std::string& contIndent) {
			const size_t end = tokens.size();
			std::string text = indent + join(tokens, 0, end, indent);
			if (text.size() <= options.columnLimit || end < 2)
				return {text};
			// Bracket depth of each token on the line.
			std::vector<int> depth(end);
			int d = 0;
			for (size_t k = 0; k < end; ++k) {
				const auto& t = toks[tokens[k]].text;
				if (t == ")" || t == "]" || (t == ">" && props(tokens[k]).templateAngle))
					--d;
				depth[k] = d;
				if (t == "(" || t == "[" || (t == "<" && props(tokens[k]).templateAngle))
					++d;
			}
			for (size_t k = 1; k < end; ++k)
				if (toks[tokens[k - 1]].text == "=>" && depth[k - 1] == 0) {
					auto lines = wrapTokens({tokens.begin(), tokens.begin() + k}, indent, contIndent);
					// The body one level deeper than the function (like a block body).
					auto body = wrapTokens(
							{tokens.begin() + k, tokens.end()}, indent + std::string(options.indentWidth, ' '),
							contIndent
					);
					lines.insert(lines.end(), body.begin(), body.end());
					return lines;
				}
			auto candidates = [&](auto isCandidate) {
				std::vector<size_t> result; // break before these positions
				int best = INT32_MAX;
				for (size_t k = 1; k < end; ++k)
					if (isCandidate(k))
						best = std::min(best, depth[k]);
				for (size_t k = 1; k < end; ++k)
					if (isCandidate(k) && depth[k] == best)
						result.push_back(k);
				return result;
			};
			auto breaks = candidates([&](size_t k) {
				return toks[tokens[k - 1]].text == "," && !props(tokens[k - 1]).einops;
			});
			if (breaks.empty())
				breaks = candidates([&](size_t k) {
					const auto& t = toks[tokens[k]].text;
					const auto& pr = props(tokens[k]);
					return !pr.einops && !pr.unary && !pr.annotationAt &&
						   (t == "+" || t == "-" || t == "*" || t == "/" || t == "@" || pr.conditional);
				});
			if (breaks.empty())
				return {text};
			// Greedy: the pieces between the break positions, as many per line as fit.
			breaks.push_back(end);
			std::vector<std::string> lines;
			std::string current = indent;
			size_t pieceStart = 0;
			bool lineEmpty = true;
			for (size_t b : breaks) {
				std::string piece = join(tokens, pieceStart, b, indent);
				std::string candidate =
						lineEmpty ? current + piece
								  : current + (spaced(tokens[pieceStart - 1], tokens[pieceStart]) ? " " : "") + piece;
				if (!lineEmpty && candidate.size() > options.columnLimit) {
					lines.push_back(current);
					current = contIndent + piece;
				} else {
					current = std::move(candidate);
				}
				lineEmpty = false;
				pieceStart = b;
			}
			lines.push_back(current);
			return lines;
		}

		/** @brief Aligns the trailing `//` comments of consecutive lines. */
		static void alignTrailingComments(std::vector<std::string>& out) {
			auto trailing = [](const std::string& l) -> size_t {
				// The position of a trailing line comment (after code), or npos.
				size_t first = l.find_first_not_of(' ');
				if (first == std::string::npos || l.compare(first, 2, "//") == 0 || l.compare(first, 2, "/*") == 0 ||
					l.compare(first, 1, "*") == 0)
					return std::string::npos;
				size_t pos = l.find("  //");
				return pos == std::string::npos ? std::string::npos : pos;
			};
			for (size_t i = 0; i < out.size();) {
				size_t j = i;
				size_t column = 0;
				while (j < out.size() && trailing(out[j]) != std::string::npos) {
					column = std::max(column, trailing(out[j]));
					++j;
				}
				for (size_t k = i; k < j; ++k) {
					size_t pos = trailing(out[k]);
					std::string code = out[k].substr(0, pos), comment = out[k].substr(pos + 2);
					out[k] = code + std::string(column - pos + 2, ' ') + comment;
				}
				i = j > i ? j : i + 1;
			}
		}
	};

	/** @brief The printed tokens of @p source (type and text; whitespace inside comments ignored), for the check. */
	std::vector<std::pair<size_t, std::string>> significantTokens(std::string_view source) {
		antlr4::ANTLRInputStream input{std::string{source}};
		mdlLexer lexer{&input};
		lexer.removeErrorListeners();
		antlr4::CommonTokenStream stream{&lexer};
		stream.fill();
		// Comments are compared without whitespace and comment markers, and consecutive line comments as one (reflowing
		// moves words between lines and adds markers).
		std::vector<std::pair<size_t, std::string>> result;
		for (auto* t : stream.getTokens()) {
			if (t->getType() == mdlLexer::WS || t->getType() == antlr4::Token::EOF)
				continue;
			std::string text = t->getText();
			if (isComment(t->getType())) {
				std::erase_if(text, [](unsigned char ch) { return std::isspace(ch) != 0 || ch == '*' || ch == '/'; });
				if (t->getType() == mdlLexer::LINECOMMENT && !result.empty() &&
					result.back().first == mdlLexer::LINECOMMENT) {
					result.back().second += text;
					continue;
				}
			}
			result.emplace_back(t->getType(), std::move(text));
		}
		return result;
	}
	/** @brief The definitions of a program whose names are checked (see mdl::format::NamingOptions). */
	class NamingChecker {
	public:
		NamingChecker(const mdl::format::NamingOptions& naming, std::string file, std::vector<mdl::Diagnostic>& out)
				: naming(naming), file(std::move(file)), out(out) {}

		void visit(antlr4::tree::ParseTree* tree, bool inModule = false) {
			using N = mdl::format::NamingOptions;
			if (auto* ctx = dynamic_cast<mdlParser::Module_defContext*>(tree)) {
				check(ctx->IDENT(), &N::module, "module");
				for (auto* p : ctx->param())
					check(p->IDENT(), &N::hyperparameter, "hyperparameter");
				for (auto* v : ctx->vardecl())
					check(v->IDENT(), &N::moduleMember, "module member");
				for (auto* f : ctx->funcdef())
					visit(f, true);
				return;
			}
			if (auto* ctx = dynamic_cast<mdlParser::Struct_defContext*>(tree)) {
				check(ctx->IDENT(), &N::structure, "struct");
				for (auto* v : ctx->vardecl())
					check(v->IDENT(), &N::structMember, "struct member");
			} else if (auto* ctx = dynamic_cast<mdlParser::AnonTypeContext*>(tree)) {
				for (auto* v : ctx->vardecl())
					check(v->IDENT(), &N::structMember, "struct member");
			} else if (auto* ctx = dynamic_cast<mdlParser::FuncdefContext*>(tree)) {
				check(ctx->IDENT(), &N::function, inModule ? "member function" : "function");
				if (auto* t = ctx->templatespec())
					for (auto* v : t->vardecl())
						check(v->IDENT(), &N::templateParameter, "template parameter");
				for (auto* p : ctx->param())
					check(p->IDENT(), &N::parameter, "parameter");
			} else if (auto* ctx = dynamic_cast<mdlParser::DeclarationStatementContext*>(tree)) {
				for (auto* n : ctx->names)
					check(n, &N::variable, "variable");
			} else if (auto* ctx = dynamic_cast<mdlParser::ForStatementContext*>(tree)) {
				check(ctx->var, &N::variable, "loop variable");
			} else if (auto* ctx = dynamic_cast<mdlParser::ComprehensionTypeContext*>(tree)) {
				check(ctx->var, &N::variable, "loop variable");
			}
			for (auto* child : tree->children)
				visit(child, inModule);
		}

	private:
		const mdl::format::NamingOptions& naming;
		std::string file;
		std::vector<mdl::Diagnostic>& out;

		void
		check(antlr4::tree::TerminalNode* node, mdl::format::Naming mdl::format::NamingOptions::* kind,
			  const char* what) {
			if (node != nullptr)
				check(node->getSymbol(), kind, what);
		}

		void check(antlr4::Token* t, mdl::format::Naming mdl::format::NamingOptions::* kind, const char* what) {
			const auto convention = naming.*kind;
			if (t == nullptr || t->getText() == "_" || mdl::format::follows(t->getText(), convention))
				return;
			const size_t line = t->getLine(), column = t->getCharPositionInLine();
			out.push_back(mdl::Diagnostic{
					.severity = mdl::Severity::warning,
					.file = file,
					.range = {{line, column}, {line, column + t->getText().size()}},
					.message = std::format(
							"The {} name {} is not {}", what, t->getText(), mdl::format::namingName(convention)
					),
					.hint = std::format("rename it to `{}`", mdl::format::convert(t->getText(), convention)),
					.flag = "naming",
			});
		}
	};
} // namespace

bool mdl::format::isSnakeCase(std::string_view name) {
	static const std::regex re{"[a-z][a-z0-9]*(_[a-z0-9]+)*"};
	return std::regex_match(name.begin(), name.end(), re);
}
bool mdl::format::isCamelCase(std::string_view name) {
	static const std::regex re{"[a-z][a-zA-Z0-9]*"};
	return std::regex_match(name.begin(), name.end(), re);
}
bool mdl::format::isPascalCase(std::string_view name) {
	static const std::regex re{"[A-Z][a-zA-Z0-9]*"};
	return std::regex_match(name.begin(), name.end(), re);
}
bool mdl::format::isAllCaps(std::string_view name) {
	static const std::regex re{"[A-Z][A-Z0-9]*(_[A-Z0-9]+)*"};
	return std::regex_match(name.begin(), name.end(), re);
}
bool mdl::format::isLowercase(std::string_view name) {
	static const std::regex re{"[a-z][a-z0-9]*"};
	return std::regex_match(name.begin(), name.end(), re);
}

bool mdl::format::follows(std::string_view name, Naming naming) {
	switch (naming) {
	case Naming::any:
		return true;
	case Naming::snakeCase:
		return isSnakeCase(name);
	case Naming::camelCase:
		return isCamelCase(name);
	case Naming::pascalCase:
		return isPascalCase(name);
	case Naming::allCaps:
		return isAllCaps(name);
	case Naming::lowercase:
		return isLowercase(name);
	}
	return true;
}

std::string_view mdl::format::namingName(Naming naming) {
	switch (naming) {
	case Naming::any:
		return "any";
	case Naming::snakeCase:
		return "snake_case";
	case Naming::camelCase:
		return "camelCase";
	case Naming::pascalCase:
		return "PascalCase";
	case Naming::allCaps:
		return "ALL_CAPS";
	case Naming::lowercase:
		return "lowercase";
	}
	return "any";
}

std::string mdl::format::convert(std::string_view name, Naming naming) {
	// Words: split at `_` and before an upper-case letter that follows a lower-case letter or digit, or that starts a
	// word after an acronym (`HTTPServer` -> HTTP, Server).
	std::vector<std::string> words;
	std::string word;
	auto flush = [&] {
		if (!word.empty())
			words.push_back(std::exchange(word, {}));
	};
	for (size_t i = 0; i < name.size(); ++i) {
		const char c = name[i];
		if (c == '_') {
			flush();
			continue;
		}
		const bool upper = std::isupper(static_cast<unsigned char>(c)) != 0;
		if (upper && !word.empty()) {
			const char prev = word.back();
			const bool nextLower = i + 1 < name.size() && std::islower(static_cast<unsigned char>(name[i + 1]));
			if (!std::isupper(static_cast<unsigned char>(prev)) || nextLower)
				flush();
		}
		word += c;
	}
	flush();
	auto lower = [](std::string w) {
		for (auto& c : w)
			c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
		return w;
	};
	auto capitalized = [&](std::string w) {
		w = lower(std::move(w));
		if (!w.empty())
			w[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(w[0])));
		return w;
	};
	std::string result;
	for (size_t i = 0; i < words.size(); ++i) {
		switch (naming) {
		case Naming::any:
			return std::string{name};
		case Naming::snakeCase:
			result += (i > 0 ? "_" : "") + lower(words[i]);
			break;
		case Naming::allCaps: {
			std::string w = words[i];
			for (auto& c : w)
				c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
			result += (i > 0 ? "_" : "") + w;
			break;
		}
		case Naming::camelCase:
			result += i == 0 ? lower(words[i]) : capitalized(words[i]);
			break;
		case Naming::pascalCase:
			result += capitalized(words[i]);
			break;
		case Naming::lowercase:
			result += lower(words[i]);
			break;
		}
	}
	return result;
}

std::vector<mdl::Diagnostic>
mdl::format::checkNaming(std::string_view source, const NamingOptions& naming, std::string_view sourcename) {
	std::vector<Diagnostic> result;
	if (invalidUtf8(source, std::string{sourcename}))
		return result;
	antlr4::ANTLRInputStream input{std::string{source}};
	mdlLexer lexer{&input};
	lexer.removeErrorListeners();
	antlr4::CommonTokenStream stream{&lexer};
	mdlParser parser{&stream};
	parser.removeErrorListeners();
	auto* program = parser.program();
	if (parser.getNumberOfSyntaxErrors() > 0)
		return result;
	NamingChecker{naming, std::string{sourcename}, result}.visit(program);
	return result;
}

mdl::format::FormatResult
mdl::format::format(std::string_view source, const FormatOptions& options, std::string_view sourcename) {
	FormatResult result;
	DiagnosticSink sink = [&](const Diagnostic& d) { result.diagnostics.push_back(d); };
	PrettyErrorListener errors{std::string{sourcename}, sink};
	// Invalid UTF-8 is reported and the text is left unchanged (decoding it would replace bytes).
	if (auto invalid = invalidUtf8(source, std::string{sourcename})) {
		result.text = std::string{source};
		result.diagnostics.push_back(std::move(*invalid));
		return result;
	}
	antlr4::ANTLRInputStream input{std::string{source}};
	mdlLexer lexer{&input};
	lexer.removeErrorListeners();
	lexer.addErrorListener(&errors);
	antlr4::CommonTokenStream stream{&lexer};
	mdlParser parser{&stream};
	parser.removeErrorListeners();
	parser.addErrorListener(&errors);
	auto* program = parser.program();
	if (errors.getNumErrors() > 0) {
		result.text = std::string{source};
		return result;
	}
	NamingChecker{options.naming, std::string{sourcename}, result.diagnostics}.visit(program);
	Formatter formatter{options, stream};
	formatter.classify(program);
	std::string text = formatter.run();
	if (significantTokens(text) != significantTokens(source)) {
		result.text = std::string{source};
		result.diagnostics.push_back(Diagnostic{
				.file = std::string{sourcename}, .message = "internal error: formatting would change the code"
		});
		return result;
	}
	result.text = std::move(text);
	result.ok = true;
	return result;
}

std::expected<mdl::format::FormatOptions, std::string> mdl::format::loadStyle(const std::filesystem::path& file) {
	std::ifstream in{file};
	if (!in)
		return std::unexpected("cannot read " + file.string());
	FormatOptions options;
	const std::map<std::string, unsigned FormatOptions::*> keys{
			{"IndentWidth", &FormatOptions::indentWidth},
			{"ColumnLimit", &FormatOptions::columnLimit},
			{"MaxEmptyLines", &FormatOptions::maxEmptyLines},
	};
	const std::map<std::string, Naming NamingOptions::*> namingKeys{
			{"ModuleNaming", &NamingOptions::module},
			{"StructNaming", &NamingOptions::structure},
			{"StructMemberNaming", &NamingOptions::structMember},
			{"ModuleMemberNaming", &NamingOptions::moduleMember},
			{"FunctionNaming", &NamingOptions::function},
			{"ParameterNaming", &NamingOptions::parameter},
			{"HyperparameterNaming", &NamingOptions::hyperparameter},
			{"TemplateParameterNaming", &NamingOptions::templateParameter},
			{"VariableNaming", &NamingOptions::variable},
	};
	std::string line;
	for (size_t number = 1; std::getline(in, line); ++number) {
		if (auto hash = line.find('#'); hash != std::string::npos)
			line.erase(hash);
		line = rstrip(line);
		const size_t first = line.find_first_not_of(" \t");
		if (first == std::string::npos || line.substr(first) == "---")
			continue;
		const auto colon = line.find(':');
		auto where = [&] { return file.string() + ":" + std::to_string(number) + ": "; };
		if (colon == std::string::npos)
			return std::unexpected(where() + "expected `Key: value`");
		std::string key = line.substr(first, colon - first);
		std::string value = line.substr(colon + 1);
		key = rstrip(key);
		value.erase(0, value.find_first_not_of(" \t"));
		if (key == "ReflowComments") {
			if (value != "true" && value != "false")
				return std::unexpected(where() + "invalid value for ReflowComments: " + value + " (true or false)");
			options.reflowComments = value == "true";
			continue;
		}
		if (auto n = namingKeys.find(key); n != namingKeys.end()) {
			bool known = false;
			for (auto convention :
				 {Naming::any, Naming::snakeCase, Naming::camelCase, Naming::pascalCase, Naming::allCaps,
				  Naming::lowercase})
				if (value == namingName(convention)) {
					options.naming.*(n->second) = convention;
					known = true;
				}
			if (!known)
				return std::unexpected(
						where() + "invalid value for " + key + ": " + value +
						" (snake_case, camelCase, PascalCase, ALL_CAPS, lowercase, any)"
				);
			continue;
		}
		auto it = keys.find(key);
		if (it == keys.end())
			return std::unexpected(
					where() + "unknown key " + key +
					" (IndentWidth, ColumnLimit, MaxEmptyLines, ReflowComments, <Kind>Naming)"
			);
		try {
			size_t used = 0;
			const unsigned long v = std::stoul(value, &used);
			if (used != value.size() || (key != "MaxEmptyLines" && v == 0) || v > 1000)
				throw std::invalid_argument(value);
			options.*(it->second) = static_cast<unsigned>(v);
		} catch (const std::exception&) {
			return std::unexpected(where() + "invalid value for " + key + ": " + value);
		}
	}
	return options;
}

std::optional<std::filesystem::path> mdl::format::findStyleFile(const std::filesystem::path& source) {
	std::error_code ec;
	auto dir = std::filesystem::absolute(source, ec).parent_path();
	while (!dir.empty()) {
		auto candidate = dir / ".mdl-format";
		if (std::filesystem::is_regular_file(candidate, ec))
			return candidate;
		if (dir == dir.parent_path())
			break;
		dir = dir.parent_path();
	}
	return std::nullopt;
}
