#include "./partial.hpp"

#include "./tokens.hpp"

#include <mdlLexer.h>
#include <mdlParser.h>

#include <vector>

namespace {
	/** @brief Whether @p text parses as a program. */
	bool parses(const std::string& text) {
		antlr4::ANTLRInputStream input{text};
		antlr4::mdlLexer lexer{&input};
		lexer.removeErrorListeners();
		antlr4::CommonTokenStream stream{&lexer};
		antlr4::mdlParser parser{&stream};
		parser.removeErrorListeners();
		parser.program();
		return parser.getNumberOfSyntaxErrors() == 0 && lexer.getNumberOfSyntaxErrors() == 0;
	}

	/** @brief The byte offsets of the lines' starts and code points, to convert token positions to offsets. */
	class Offsets {
	public:
		explicit Offsets(const std::string& text) : text(text) {
			starts.push_back(0);
			for (size_t i = 0; i < text.size(); ++i)
				if (text[i] == '\n')
					starts.push_back(i + 1);
		}
		size_t of(mdl::SourcePosition p) const {
			size_t offset = p.line - 1 < starts.size() ? starts[p.line - 1] : text.size();
			for (size_t points = 0; offset < text.size() && points < p.column; ++points) {
				++offset;
				while (offset < text.size() && (text[offset] & 0xC0) == 0x80)
					++offset;
			}
			return offset;
		}

	private:
		const std::string& text;
		std::vector<size_t> starts;
	};

	void blank(std::string& text, size_t begin, size_t end) {
		for (size_t i = begin; i < end && i < text.size(); ++i)
			if (text[i] != '\n')
				text[i] = ' ';
	}

	/** @brief Spans of tokens [first, last] that end with `;` or a `}` at depth @p depth (+1 inside the span). */
	std::vector<std::pair<size_t, size_t>>
	items(const std::vector<mdls::Token>& tokens, size_t from, size_t to, int depth) {
		std::vector<std::pair<size_t, size_t>> result;
		int d = depth;
		size_t first = from;
		for (size_t i = from; i < to; ++i) {
			const auto& t = tokens[i];
			if (t.is("{") || t.is("[") || t.is("("))
				++d;
			else if (t.is("}") || t.is("]") || t.is(")"))
				--d;
			const bool end = d == depth && (t.is(";") || t.is("}"));
			if (end || d < depth) {
				result.emplace_back(first, i);
				first = i + 1;
				d = std::max(d, depth);
			}
		}
		if (first < to)
			result.emplace_back(first, to - 1);
		return result;
	}
} // namespace

std::string mdls::blankBrokenDefinitions(const std::string& text) {
	std::vector<Token> tokens;
	for (auto& t : tokenize(text))
		if (!t.comment())
			tokens.push_back(std::move(t));
	const Offsets offsets{text};
	std::string result = text;
	auto source = [&](size_t first, size_t last) {
		const size_t begin = offsets.of(tokens[first].begin), end = offsets.of(tokens[last].end);
		return std::pair{begin, end};
	};
	for (const auto& [first, last] : items(tokens, 0, tokens.size(), 0)) {
		auto [begin, end] = source(first, last);
		if (parses(result.substr(begin, end - begin)))
			continue;
		// A module: blank its members that do not parse (in a module of their own), then check it again.
		if (tokens[first].is("module")) {
			size_t open = first;
			while (open <= last && !tokens[open].is("{"))
				++open;
			if (open < last) {
				for (const auto& [f, l] : items(tokens, open + 1, last, 1)) {
					if (f > l || (f == l && tokens[f].is("}")))
						continue;
					auto [b, e] = source(f, l);
					if (!parses("module __M[] {\n" + result.substr(b, e - b) + "\n}"))
						blank(result, b, e);
				}
				if (parses(result.substr(begin, end - begin)))
					continue;
			}
		}
		blank(result, begin, end);
	}
	return result;
}
