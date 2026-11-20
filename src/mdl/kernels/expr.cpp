// The expression language of kernel packages: parsing and type checking (see include/mdl/kernels/expr.hpp).
#include <mdl/kernels/expr.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <string_view>

namespace mdl::kernels::expr {

	std::string str(Type type) {
		switch (type) {
		case Type::integer:
			return "an integer";
		case Type::boolean:
			return "a condition";
		case Type::dtype:
			return "an element type";
		case Type::shape:
			return "a shape";
		case Type::pointer:
			return "a pointer";
		case Type::real:
			return "a number";
		}
		return "";
	}

	namespace {
		constexpr std::array kDtypes{"float32", "float16", "bfloat16", "float64", "int8",	"int16",
									 "int32",	"int64",   "uint8",	   "uint16",  "uint32", "bool"};

		struct Token {
			enum class Kind { end, integer, name, capture, symbol } kind;
			std::string text;
			int64_t value = 0;
			size_t column = 0; ///< from 1
		};

		struct Error {
			std::string message;
		};

		class Parser {
		public:
			Parser(const std::string& text, const std::vector<std::string>& captures) : text(text), captures(captures) {
				tokenize();
			}

			NodePtr parseAll() {
				auto node = orExpr();
				if (peek().kind != Token::Kind::end)
					fail(peek(), std::format("unexpected '{}'", peek().text));
				return node;
			}

		private:
			const std::string& text;
			const std::vector<std::string>& captures;
			std::vector<Token> tokens;
			size_t pos = 0;

			[[noreturn]] static void fail(const Token& at, const std::string& message) {
				throw Error{std::format("column {}: {}", at.column, message)};
			}

			void tokenize() {
				size_t i = 0;
				while (i < text.size()) {
					const char c = text[i];
					if (std::isspace(static_cast<unsigned char>(c))) {
						++i;
						continue;
					}
					Token t{.kind = Token::Kind::symbol, .text = {}, .value = 0, .column = i + 1};
					if (std::isdigit(static_cast<unsigned char>(c))) {
						size_t j = i;
						while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j])))
							++j;
						t.kind = Token::Kind::integer;
						t.text = text.substr(i, j - i);
						t.value = std::stoll(t.text);
						i = j;
					} else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '$') {
						size_t j = i + (c == '$' ? 1 : 0);
						while (j < text.size() && (std::isalnum(static_cast<unsigned char>(text[j])) || text[j] == '_'))
							++j;
						t.kind = c == '$' ? Token::Kind::capture : Token::Kind::name;
						t.text = text.substr(c == '$' ? i + 1 : i, j - i - (c == '$' ? 1 : 0));
						if (t.kind == Token::Kind::capture && t.text.empty())
							fail(t, "'$' must be followed by the name of a capture");
						i = j;
					} else {
						static constexpr std::array two{"==", "!=", "<=", ">=", "//"};
						const auto pair = text.substr(i, 2);
						if (std::ranges::find(two, pair) != two.end()) {
							t.text = pair;
							i += 2;
						} else if (std::string_view{"+-*%<>()[],:."}.find(c) != std::string_view::npos) {
							t.text = std::string(1, c);
							++i;
						} else {
							fail(t, std::format("unexpected character '{}'", c));
						}
					}
					tokens.push_back(std::move(t));
				}
				tokens.push_back(Token{.kind = Token::Kind::end, .text = "end", .value = 0, .column = text.size() + 1});
			}

			const Token& peek() const { return tokens[pos]; }
			const Token& next() { return tokens[pos++]; }
			bool accept(std::string_view symbol) {
				if (peek().kind == Token::Kind::symbol && peek().text == symbol) {
					++pos;
					return true;
				}
				return false;
			}
			bool acceptName(std::string_view name) {
				if (peek().kind == Token::Kind::name && peek().text == name) {
					++pos;
					return true;
				}
				return false;
			}
			void expect(std::string_view symbol) {
				if (!accept(symbol))
					fail(peek(), std::format("expected '{}'", symbol));
			}

			static NodePtr make(Node node) { return std::make_shared<const Node>(std::move(node)); }

			void require(const NodePtr& n, Type type, const Token& at, std::string_view what) {
				if (n->type != type)
					fail(at, std::format("{} must be {}, not {}", what, str(type), str(n->type)));
			}

			NodePtr orExpr() {
				auto lhs = andExpr();
				while (true) {
					const auto& at = peek();
					if (!acceptName("or"))
						return lhs;
					auto rhs = andExpr();
					require(lhs, Type::boolean, at, "an operand of 'or'");
					require(rhs, Type::boolean, at, "an operand of 'or'");
					lhs = make({.kind = Node::Kind::binary, .type = Type::boolean, .op = "or", .args = {lhs, rhs}});
				}
			}

			NodePtr andExpr() {
				auto lhs = notExpr();
				while (true) {
					const auto& at = peek();
					if (!acceptName("and"))
						return lhs;
					auto rhs = notExpr();
					require(lhs, Type::boolean, at, "an operand of 'and'");
					require(rhs, Type::boolean, at, "an operand of 'and'");
					lhs = make({.kind = Node::Kind::binary, .type = Type::boolean, .op = "and", .args = {lhs, rhs}});
				}
			}

			NodePtr notExpr() {
				const auto& at = peek();
				if (acceptName("not")) {
					auto operand = notExpr();
					require(operand, Type::boolean, at, "the operand of 'not'");
					return make({.kind = Node::Kind::unary, .type = Type::boolean, .op = "not", .args = {operand}});
				}
				return comparison();
			}

			NodePtr comparison() {
				auto lhs = sum();
				const auto& at = peek();
				if (acceptName("in")) {
					auto list = postfix();
					if (list->type != Type::shape && list->kind != Node::Kind::list)
						fail(at, "'in' needs a list, e.g. [float32, bfloat16], or a shape");
					const Type element = list->type == Type::shape ? Type::integer : Type::dtype;
					require(lhs, element, at, "the left operand of 'in'");
					return make({.kind = Node::Kind::in, .type = Type::boolean, .args = {lhs, list}});
				}
				for (const char* op : {"==", "!=", "<=", ">=", "<", ">"}) {
					if (!accept(op))
						continue;
					auto rhs = sum();
					const bool equality = std::string_view{op} == "==" || std::string_view{op} == "!=";
					if (lhs->type != rhs->type)
						fail(at, std::format("'{}' compares {} with {}", op, str(lhs->type), str(rhs->type)));
					if (!equality)
						require(lhs, Type::integer, at, std::format("an operand of '{}'", op));
					else if (lhs->type == Type::boolean || lhs->type == Type::pointer)
						fail(at, std::format("'{}' cannot compare {}", op, str(lhs->type)));
					return make({.kind = Node::Kind::binary, .type = Type::boolean, .op = op, .args = {lhs, rhs}});
				}
				return lhs;
			}

			NodePtr arithmetic(NodePtr lhs, const std::string& op, NodePtr rhs, const Token& at) {
				require(lhs, Type::integer, at, std::format("an operand of '{}'", op));
				require(rhs, Type::integer, at, std::format("an operand of '{}'", op));
				return make({.kind = Node::Kind::binary, .type = Type::integer, .op = op, .args = {lhs, rhs}});
			}

			NodePtr sum() {
				auto lhs = term();
				while (true) {
					const auto& at = peek();
					if (accept("+"))
						lhs = arithmetic(lhs, "+", term(), at);
					else if (accept("-"))
						lhs = arithmetic(lhs, "-", term(), at);
					else
						return lhs;
				}
			}

			NodePtr term() {
				auto lhs = unary();
				while (true) {
					const auto& at = peek();
					if (accept("*"))
						lhs = arithmetic(lhs, "*", unary(), at);
					else if (accept("//"))
						lhs = arithmetic(lhs, "//", unary(), at);
					else if (accept("%"))
						lhs = arithmetic(lhs, "%", unary(), at);
					else
						return lhs;
				}
			}

			NodePtr unary() {
				const auto& at = peek();
				if (accept("-")) {
					auto operand = unary();
					require(operand, Type::integer, at, "the operand of '-'");
					return make({.kind = Node::Kind::unary, .type = Type::integer, .op = "-", .args = {operand}});
				}
				return postfix();
			}

			/** @brief An integer literal, possibly negative (slice bounds). */
			std::optional<int64_t> literal() {
				const bool negative = accept("-");
				if (peek().kind != Token::Kind::integer)
					return negative ? (fail(peek(), "expected an integer"), std::nullopt) : std::nullopt;
				const auto v = next().value;
				return negative ? -v : v;
			}

			NodePtr postfix() {
				auto node = primary();
				while (true) {
					const auto& at = peek();
					if (!accept("["))
						return node;
					require(node, Type::shape, at, "an indexed value");
					// A slice `[a:b]` (literal bounds) or an index `[i]`.
					const size_t save = pos;
					auto from = literal();
					if (accept(":")) {
						auto to = literal();
						expect("]");
						node =
								make({.kind = Node::Kind::slice,
									  .type = Type::shape,
									  .args = {node},
									  .from = from.value_or(0),
									  .to = to.value_or(0),
									  .hasFrom = from.has_value(),
									  .hasTo = to.has_value()});
						continue;
					}
					pos = save;
					auto index = sum();
					require(index, Type::integer, at, "an index");
					expect("]");
					node = make({.kind = Node::Kind::index, .type = Type::integer, .args = {node, index}});
				}
			}

			NodePtr primary() {
				const auto& t = peek();
				switch (t.kind) {
				case Token::Kind::integer:
					next();
					return make({.kind = Node::Kind::integer, .type = Type::integer, .value = t.value});
				case Token::Kind::capture:
					return access();
				case Token::Kind::name: {
					next();
					if (std::ranges::find(kDtypes, t.text) != kDtypes.end())
						return make({.kind = Node::Kind::dtypeName, .type = Type::dtype, .name = t.text});
					if (t.text == "prod" || t.text == "len" || t.text == "strides") {
						expect("(");
						auto arg = orExpr();
						expect(")");
						require(arg, Type::shape, t, std::format("the argument of {}", t.text));
						const Type type = t.text == "strides" ? Type::shape : Type::integer;
						return make({.kind = Node::Kind::call, .type = type, .name = t.text, .args = {arg}});
					}
					if (t.text == "permute" || t.text == "unpermute") { // (list, permutation)
						expect("(");
						auto list = orExpr();
						expect(",");
						auto perm = orExpr();
						expect(")");
						require(list, Type::shape, t, std::format("the first argument of {}", t.text));
						require(perm, Type::shape, t, std::format("the permutation of {}", t.text));
						return make({.kind = Node::Kind::call, .type = Type::shape, .name = t.text, .args = {list, perm}});
					}
					fail(t, std::format("unknown name '{}' (captures are written $name)", t.text));
				}
				case Token::Kind::symbol:
					if (accept("(")) {
						auto inner = orExpr();
						expect(")");
						return inner;
					}
					if (accept("["))
						return list(t);
					fail(t, std::format("unexpected '{}'", t.text));
				case Token::Kind::end:
					fail(t, "unexpected end of the expression");
				}
				fail(t, "unexpected token");
			}

			NodePtr list(const Token& at) {
				Node node{.kind = Node::Kind::list, .type = Type::shape};
				if (!accept("]")) {
					do
						node.args.push_back(orExpr());
					while (accept(","));
					expect("]");
				}
				const bool dtypes = !node.args.empty() && node.args[0]->type == Type::dtype;
				for (const auto& element : node.args)
					require(element, dtypes ? Type::dtype : Type::integer, at, "an element of a list");
				node.type = dtypes ? Type::dtype : Type::shape;
				return make(std::move(node));
			}

			/** @brief `out`, or `out0`, `out1`, ... (the results of an operator with several). */
			static bool isOutput(const std::string& name) {
				return name == "out" || (name.size() > 3 && name.starts_with("out") &&
										 std::all_of(name.begin() + 3, name.end(), [](char c) {
											 return std::isdigit(static_cast<unsigned char>(c));
										 }));
			}

			NodePtr access() {
				const auto& t = next();
				if (!isOutput(t.text) && std::ranges::find(captures, t.text) == captures.end())
					fail(t, std::format("unknown capture '${}'", t.text));
				Node node{.kind = Node::Kind::access, .type = Type::integer, .name = t.text};
				constexpr const char* kFields = "shape, ndim, dtype, data, is_tensor, values, value, real, attr.<name>";
				if (!accept("."))
					fail(peek(), std::format("'${}' needs a field: {}", t.text, kFields));
				const auto& field = next();
				if (field.kind != Token::Kind::name)
					fail(field, std::format("expected a field: {}", kFields));
				node.field = field.text;
				if (field.text == "shape" || field.text == "values")
					node.type = Type::shape;
				else if (field.text == "ndim" || field.text == "value")
					node.type = Type::integer;
				else if (field.text == "dtype")
					node.type = Type::dtype;
				else if (field.text == "data")
					node.type = Type::pointer;
				else if (field.text == "is_tensor")
					node.type = Type::boolean;
				else if (field.text == "real")
					node.type = Type::real;
				else if (field.text == "attr") {
					expect(".");
					const auto& name = next();
					if (name.kind != Token::Kind::name)
						fail(name, "expected the name of an attribute");
					node.field = "attr." + name.text;
					node.type = Type::real; // (for float arguments)
				} else
					fail(field, std::format("unknown field '{}' ({})", field.text, kFields));
				return make(std::move(node));
			}
		};
	} // namespace

	std::expected<NodePtr, std::string> parse(const std::string& text, const std::vector<std::string>& captures) {
		try {
			return Parser{text, captures}.parseAll();
		} catch (const Error& e) {
			return std::unexpected(e.message);
		}
	}

	std::string str(const Node& node) {
		switch (node.kind) {
		case Node::Kind::integer:
			return std::to_string(node.value);
		case Node::Kind::dtypeName:
			return node.name;
		case Node::Kind::list: {
			std::string s = "[";
			for (size_t i = 0; i < node.args.size(); ++i)
				s += (i > 0 ? ", " : "") + str(*node.args[i]);
			return s + "]";
		}
		case Node::Kind::access:
			return "$" + node.name + "." + node.field;
		case Node::Kind::unary:
			return node.op == "not" ? "not " + str(*node.args[0]) : "-" + str(*node.args[0]);
		case Node::Kind::binary:
			return "(" + str(*node.args[0]) + " " + node.op + " " + str(*node.args[1]) + ")";
		case Node::Kind::in:
			return str(*node.args[0]) + " in " + str(*node.args[1]);
		case Node::Kind::index:
			return str(*node.args[0]) + "[" + str(*node.args[1]) + "]";
		case Node::Kind::slice:
			return str(*node.args[0]) + "[" + (node.hasFrom ? std::to_string(node.from) : "") + ":" +
				   (node.hasTo ? std::to_string(node.to) : "") + "]";
		case Node::Kind::call: {
			std::string s = node.name + "(";
			for (size_t i = 0; i < node.args.size(); ++i)
				s += (i > 0 ? ", " : "") + str(*node.args[i]);
			return s + ")";
		}
		}
		return "";
	}

} // namespace mdl::kernels::expr
