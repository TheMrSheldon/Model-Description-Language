#include "./tokens.hpp"

#include <mdlLexer.h>

#include <algorithm>

using antlr4::mdlLexer;

bool mdls::Token::identifier() const { return type == mdlLexer::IDENT; }
bool mdls::Token::comment() const { return type == mdlLexer::BLOCKCOMMENT || type == mdlLexer::LINECOMMENT; }

std::vector<mdls::Token> mdls::tokenize(const std::string& text) {
	antlr4::ANTLRInputStream input{text};
	mdlLexer lexer{&input};
	lexer.removeErrorListeners();
	antlr4::CommonTokenStream stream{&lexer};
	stream.fill();
	// Columns in code points: count the UTF-8 lead bytes.
	auto points = [](std::string_view v) {
		return static_cast<size_t>(std::count_if(v.begin(), v.end(), [](char c) { return (c & 0xC0) != 0x80; }));
	};
	std::vector<Token> tokens;
	for (auto* t : stream.getTokens()) {
		if (t->getType() == mdlLexer::WS || t->getType() == antlr4::Token::EOF)
			continue;
		const std::string s = t->getText();
		const size_t newlines = std::count(s.begin(), s.end(), '\n');
		const size_t line = t->getLine(), column = t->getCharPositionInLine();
		const mdl::SourcePosition end =
				newlines == 0
						? mdl::SourcePosition{line, column + points(s)}
						: mdl::SourcePosition{line + newlines, points(std::string_view{s}.substr(s.rfind('\n') + 1))};
		tokens.push_back({t->getType(), s, {line, column}, end});
	}
	return tokens;
}
