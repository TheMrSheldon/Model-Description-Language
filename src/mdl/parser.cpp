#include <mdl/parser.hpp>

#include "./parser/errorlistener.hpp"

#include <mdl/analysis/index.hpp>
#include <mdl/computegraphv2/function.hpp>
#include <mdl/computegraphv2/object.hpp>
#include <mdl/namespace.hpp>
#include <mdl/passes/shapes.hpp>
#include <mdl/utils/einops.hpp>
#include <mdl/utils/platform.hpp>
#include <mdl/utils/utils.hpp>

#include <mdlBaseVisitor.h>
#include <mdlLexer.h>
#include <mdlParser.h>

#include <cassert>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

namespace mdl {
	class TranslatingVisitor;

	/** @brief A parsed source file; kept alive so that module templates can be expanded after parsing. */
	struct ParsedFile {
		std::unique_ptr<PrettyErrorListener> errors; // destroyed last (the parser refers to it)
		std::unique_ptr<antlr4::ANTLRInputStream> input;
		std::unique_ptr<antlr4::mdlLexer> lexer;
		std::unique_ptr<antlr4::CommonTokenStream> tokens;
		std::unique_ptr<antlr4::mdlParser> parser;
	};

	/** @brief The definition of a module, from which instances are expanded with concrete hyperparameter values. */
	struct ModuleTemplate {
		std::string name;
		antlr4::mdlParser::Module_defContext* ctx;
		Namespace* scope;			 ///< The namespace the module is defined in
		PrettyErrorListener* errors; ///< Error listener of the defining file
	};

	/** @brief The definition of a function, from which it is elaborated again for other argument types. */
	struct FunctionTemplate {
		antlr4::mdlParser::FuncdefContext* ctx;
		Namespace* scope;			 ///< The namespace the function is defined in (persistent)
		std::string weightScope;	 ///< The weight scope at the definition (TranslatingVisitor::scope)
		PrettyErrorListener* errors; ///< Error listener of the defining file
	};

	/**
	 * @brief The namespaces of one translated file: its own definitions (`local`) and what its imports bind
	 * (`imports`, the parent of `local`; its parent is the prelude, or the builtins for the files of `std`).
	 */
	struct FileScope {
		std::filesystem::path path;		  ///< canonical path (for a source without file: its name)
		std::filesystem::path packageDir; ///< identifies the package (see docs/dev/spec/language.md, Visibility)
		bool isStd = false;				  ///< a file of the standard library (no implicit prelude)
		Namespace imports;
		Namespace local;
	};

	/**
	 * @brief The value of a name bound by `import a.b;`: the namespace of a file, or of a directory without index
	 * file (its files and subdirectories as namespaces).
	 */
	struct NamespaceRef {
		const FileScope* file = nullptr;
		std::filesystem::path dir; ///< for a directory
		std::filesystem::path packageDir;
		bool isStd = false;
	};

	/**
	 * @brief The stack of namespaces in which names are looked up. A frame refers to a namespace that lives elsewhere
	 * (a file's, an instance's) or owns a temporary one (a function body, a loop).
	 */
	class NamespaceStack {
	public:
		void push(Namespace ns) {
			auto owned = std::make_unique<Namespace>(std::move(ns));
			Namespace* raw = owned.get();
			frames.push_back({raw, std::move(owned)});
		}
		void pushRef(Namespace& ns) { frames.push_back({&ns, nullptr}); }
		void pop() { frames.pop_back(); }
		Namespace& top() { return *frames.back().ns; }
		Namespace& bottom() { return *frames.front().ns; }

	private:
		struct Frame {
			Namespace* ns;
			std::unique_ptr<Namespace> owned;
		};
		std::vector<Frame> frames;
	};

	struct parser::FrontendState {
		/** @brief Receives every diagnostic (also of later instantiations); destroyed last (the listeners refer to it). */
		DiagnosticSink sink;
		std::vector<Diagnostic> diagnostics; ///< all diagnostics so far
		bool printing = false;				 ///< whether the diagnostics are printed to stderr (no sink in the options)
		std::unique_ptr<analysis::SemanticIndex> index; ///< for tools (parser::analyze); null for the compiler
		Computegraph graph;
		std::vector<std::unique_ptr<ParsedFile>> files;
		std::deque<FunctionDecl> derivedFunctions;	 ///< e.g. gradient functions (stable references)
		std::unique_ptr<TranslatingVisitor> visitor; // destroyed first
	};

	class TranslatingVisitor : public antlr4::mdlBaseVisitor {
	private:
		parser::FrontendState& state;
		parser::ParseOptions options;
		/** @brief Definitions of the main file. */
		std::map<std::string, CompilationUnit::Member> definitions;
		/** @brief All translated files. */
		std::vector<std::filesystem::path> sources;
		/** @brief Expanded module instances by template and hyperparameter values. */
		std::map<std::pair<const ModuleTemplate*, std::string>, ModuleType> instances;
		/** @brief The errors (line, column, message) of modules checked with symbolic values (see genericErrors). */
		std::map<const ModuleTemplate*, std::set<std::tuple<size_t, size_t, std::string>>> genericErrorsOf;
		/**
		 * @brief After instantiate() failed: the error that only this instance has (not the module for all values,
		 * e.g. a member that @conditional removes), for the place where it is instantiated; empty if none.
		 */
		std::string instanceError;
		/** @brief The error listener of the file that is currently translated. */
		PrettyErrorListener* errorlistener = nullptr;
		/** @brief Whether a module is being elaborated with symbolic hyperparameter values. */
		bool symbolicMode = false;
		/** @brief A member with `@conditional(c)` and a symbolic `c` (see checkConditionalMember). */
		struct ConditionalMember {
			std::string condition;	///< `c` as written
			const Namespace* scope; ///< of the module instance (a local of the same name hides the member)
		};
		/** @brief (module, member) -> its condition. */
		std::map<std::pair<std::string, std::string>, ConditionalMember> conditionalMembers;
		/** @brief The symbolic conditions that hold where the code is elaborated (the first branch of `c ? a : b`). */
		std::vector<std::string> assumedConditions;
		/** @brief The token streams of the translated files (by source name), e.g. for doc comments. */
		std::map<std::string, antlr4::CommonTokenStream*> tokenStreams;
		/** @brief The module being instantiated (for the index), empty outside of modules. */
		std::string currentModule;

		// ── Semantic index (only for tools: parser::analyze) ──────────────────────────────────────────────────────

		analysis::SemanticIndex* index() const { return state.index.get(); }
		std::string currentFile() const { return errorlistener != nullptr ? errorlistener->getFile() : std::string{}; }

		static SourceRange rangeOf(antlr4::Token* t) {
			const size_t line = t->getLine(), column = t->getCharPositionInLine();
			return {{line, column}, {line, column + t->getText().size()}};
		}

		static SourceRange rangeOf(antlr4::ParserRuleContext* ctx) {
			if (ctx == nullptr || ctx->start == nullptr || ctx->stop == nullptr)
				return {};
			return {rangeOf(ctx->start).begin, rangeOf(ctx->stop).end};
		}

		/** @brief The source text of @p ctx with whitespace runs collapsed (e.g. a declaration for hovers). */
		std::string sourceText(antlr4::ParserRuleContext* ctx, antlr4::Token* last = nullptr) {
			auto it = tokenStreams.find(currentFile());
			if (it == tokenStreams.end() || ctx == nullptr || ctx->start == nullptr)
				return {};
			auto* stop = last != nullptr ? last : ctx->stop;
			std::string result;
			for (size_t i = ctx->start->getTokenIndex(); stop != nullptr && i <= stop->getTokenIndex(); ++i) {
				auto* t = it->second->get(i);
				if (t->getChannel() != antlr4::Token::DEFAULT_CHANNEL) {
					if (!result.empty() && result.back() != ' ')
						result += ' ';
					continue;
				}
				result += t->getText();
			}
			while (!result.empty() && result.back() == ' ')
				result.pop_back();
			return result;
		}

		/** @brief The doc comment before @p ctx (`/** ... *\/` or `///` lines) as Markdown, if any. */
		std::string documentation(antlr4::ParserRuleContext* ctx) {
			const std::string comment = docComment(ctx);
			return comment.empty() ? std::string{} : analysis::docCommentToMarkdown(comment);
		}

		/** @brief The doc comment before @p ctx as written (empty if there is none). */
		std::string docComment(antlr4::ParserRuleContext* ctx) {
			auto it = tokenStreams.find(currentFile());
			if (it == tokenStreams.end() || ctx == nullptr || ctx->start == nullptr)
				return {};
			auto hidden = it->second->getHiddenTokensToLeft(ctx->start->getTokenIndex());
			std::vector<std::string> comments;
			size_t newlines = 0;
			for (auto t = hidden.rbegin(); t != hidden.rend(); ++t) {
				const std::string text = (*t)->getText();
				if ((*t)->getType() == antlr4::mdlLexer::WS) {
					newlines += text == "\n";
					if (newlines > 1)
						break; // a blank line separates the comment from the definition
					continue;
				}
				newlines = 0;
				const bool doc = text.starts_with("/**") || text.starts_with("/*!") || text.starts_with("///");
				if (!doc)
					break;
				comments.insert(comments.begin(), text);
				if (text.starts_with("/*"))
					break;
			}
			std::string joined;
			for (const auto& c : comments)
				joined += c + "\n";
			return joined;
		}

		/**
		 * @brief Gives the definition of the parameter @p name (a hyperparameter, template parameter, or function
		 * parameter) its `@param` (or `@tparam`) documentation from the doc comment of @p owner (the module or function).
		 */
		void documentParameter(antlr4::Token* name, antlr4::ParserRuleContext* owner) {
			if (index() == nullptr || name == nullptr)
				return;
			auto id = index()->find(currentFile(), rangeOf(name).begin);
			if (!id || !index()->symbol(*id).documentation.empty())
				return;
			index()->symbol(*id).documentation = analysis::docParameter(docComment(owner), name->getText());
		}

		/** @brief The type of @p type for hovers, e.g. `float32[..., dim]`, `Linear`, `struct`. */
		static std::string typeString(const DataType& type) {
			if (const auto* bound = std::get_if<BoundModuleType>(&type))
				return bound->definition.templ ? bound->definition.templ->name : "module";
			if (const auto* m = std::get_if<ModuleType>(&type))
				return m->templ ? m->templ->name : "module";
			if (std::holds_alternative<AggregateType>(type))
				return "struct";
			if (const auto* array = std::get_if<ArrayType>(&type))
				return std::format("array[{}; {}]", typeString(*array->element), array->size);
			return shapes::str(type);
		}

		/** @brief The type of the value @p object for hovers. */
		static std::string objectType(const Object& object) {
			return std::visit(
					overloaded{
							[](const Node* node) -> std::string {
								if (const auto* c = std::get_if<ConstNode>(node))
									if (const auto* i = std::get_if<int>(&c->value))
										return std::format("{} = {}", typeString(getDataType(*node)), *i);
								return typeString(getDataType(*node));
							},
							[](const FunctionDecl& f) -> std::string {
								return f.name.empty() ? "function" : "function " + f.name;
							},
							[](const AggregateInstance&) -> std::string { return "struct"; },
							[](const ArrayInstance& a) -> std::string {
								std::string element =
										a.elements.empty() ? std::string{"?"} : objectType(a.elements.front());
								return a.representative ? std::format("array of {}", element)
														: std::format("array of {} {}", a.elements.size(), element);
							},
							[](const auto&) -> std::string { return {}; },
					},
					object
			);
		}

		/** @brief What `.` can access on the value @p object (module and struct members, tensor members). */
		static std::set<std::string> objectMembers(const Object& object) {
			std::set<std::string> members;
			if (const auto* agg = std::get_if<AggregateInstance>(&object)) {
				for (const auto& [name, _] : agg->getType().members)
					members.insert(name);
			} else if (const auto* node = std::get_if<const Node*>(&object)) {
				const auto& type = getDataType(**node);
				if (const auto* bound = std::get_if<BoundModuleType>(&type)) {
					if (const auto& t = bound->definition.templ) {
						for (auto* v : t->ctx->vardecl())
							if (v->IDENT())
								members.insert(v->IDENT()->getText());
						for (auto* f : t->ctx->funcdef())
							if (f->IDENT())
								members.insert(f->IDENT()->getText());
					}
				} else if (const auto* tensor = std::get_if<TensorType>(&type)) {
					members.insert("T");
					if (tensor->masked) {
						members.insert("data");
						members.insert("mask");
					}
				}
			}
			return members;
		}

		/** @brief The visibility written on the definition @p ctx (a module, struct, function, or member). */
		Visibility declaredVisibility(antlr4::ParserRuleContext* ctx) {
			mdlParser::VisibilityContext* v = nullptr;
			if (auto* m = dynamic_cast<mdlParser::Module_defContext*>(ctx))
				v = m->visibility();
			else if (auto* st = dynamic_cast<mdlParser::Struct_defContext*>(ctx))
				v = st->visibility();
			else if (auto* f = dynamic_cast<mdlParser::FuncdefContext*>(ctx))
				v = f->visibility();
			else if (auto* d = dynamic_cast<mdlParser::VardeclContext*>(ctx))
				v = d->visibility();
			return visibilityOf(v);
		}

		/** @brief Records a definition in the index (if any); returns its index. */
		std::optional<size_t>
		define(antlr4::Token* name, analysis::SymbolKind kind, const std::string& detail,
			   antlr4::ParserRuleContext* docCtx, SourceRange scope, std::set<std::string> types = {},
			   const std::string& container = {}) {
			if (index() == nullptr || name == nullptr)
				return std::nullopt;
			return index()->define(analysis::Symbol{
					.name = name->getText(),
					.kind = kind,
					.file = currentFile(),
					.range = rangeOf(name),
					.container = container,
					.detail = detail,
					.documentation = documentation(docCtx),
					.comment = docComment(docCtx),
					.visibility = declaredVisibility(docCtx),
					.scope = scope,
					.extent = kind == analysis::SymbolKind::module || kind == analysis::SymbolKind::structure ||
											  kind == analysis::SymbolKind::function ||
											  kind == analysis::SymbolKind::method
									  ? rangeOf(docCtx)
									  : SourceRange{},
					.types = std::move(types),
			});
		}

		/** @brief define() and binds the name to the definition in the current namespace. */
		void defineHere(
				antlr4::Token* name, analysis::SymbolKind kind, const std::string& detail,
				antlr4::ParserRuleContext* docCtx, SourceRange scope, std::set<std::string> types = {},
				const std::string& container = {}
		) {
			if (auto id = define(name, kind, detail, docCtx, scope, std::move(types), container)) {
				reportShadowing(name, kind);
				currentNamespace().setSymbol(name->getText(), *id);
			}
		}

		/** @brief Reports a local definition @p name that hides one of an enclosing scope (`-Wshadow`). */
		void reportShadowing(antlr4::Token* name, analysis::SymbolKind kind) {
			using K = analysis::SymbolKind;
			auto local = [](K k) {
				return k == K::variable || k == K::parameter || k == K::loopVariable || k == K::templateParameter;
			};
			if (options.shadow == parser::DiagnosticLevel::ignored || !local(kind) || errorlistener == nullptr)
				return;
			auto outer = currentNamespace().outerSymbol(name->getText());
			if (!outer)
				return;
			const auto& hidden = index()->symbols()[*outer];
			const char* what = hidden.kind == K::variable			 ? "the variable"
							   : hidden.kind == K::parameter		 ? "the parameter"
							   : hidden.kind == K::loopVariable		 ? "the loop variable"
							   : hidden.kind == K::templateParameter ? "the template parameter"
							   : hidden.kind == K::member			 ? "the member"
							   : hidden.kind == K::hyperparameter	 ? "the hyperparameter"
																	 : nullptr;
			if (what == nullptr)
				return; // functions, modules, structs: not reported
			const bool error = options.shadow == parser::DiagnosticLevel::error;
			errorlistener->semanticError(
					error ? PrettyErrorListener::ErrorLevel::Error : PrettyErrorListener::ErrorLevel::Warning,
					tokenContext(name), name->getLine(), name->getCharPositionInLine(),
					std::format(
							"{} shadows {} {} (line {})", name->getText(), what, hidden.name, hidden.range.begin.line
					),
					"", "shadow"
			);
		}

		/**
		 * @brief Reports a definition of the file that hides an explicitly imported name (`-Wshadow`); prelude names are
		 * hidden silently.
		 */
		void reportImportShadowing(antlr4::Token* name) {
			if (options.shadow == parser::DiagnosticLevel::ignored || errorlistener == nullptr ||
				currentScope == nullptr || &currentNamespace() != &currentScope->local ||
				!currentScope->imports.definesLocally(name->getText()))
				return;
			const bool error = options.shadow == parser::DiagnosticLevel::error;
			errorlistener->semanticError(
					error ? PrettyErrorListener::ErrorLevel::Error : PrettyErrorListener::ErrorLevel::Warning,
					tokenContext(name), name->getLine(), name->getCharPositionInLine(),
					std::format("{} shadows the imported {}", name->getText(), name->getText()), "", "shadow"
			);
		}

		/** @brief Records a use of @p name (bound in the current namespace) with the value @p value. */
		void referenceHere(antlr4::Token* name, const std::optional<Object>& value) {
			if (index() == nullptr || name == nullptr)
				return;
			analysis::Reference r{
					.file = currentFile(), .range = rangeOf(name), .symbol = currentNamespace().symbol(name->getText())
			};
			if (value) {
				if (auto type = objectType(*value); !type.empty())
					r.types.insert(type);
				r.members = objectMembers(*value);
				if (auto path = weightPath(*value))
					r.weights.insert(*path);
			}
			index()->reference(std::move(r));
		}

		/** @brief The full path of the weight @p value (e.g. `layers.3.weightQuery.W`), if it is one. */
		static std::optional<std::string> weightPath(const Object& value) {
			const auto* node = std::get_if<const Node*>(&value);
			const auto* weight = node ? std::get_if<ParameterNode>(*node) : nullptr;
			if (weight == nullptr || weight->name.empty() || std::holds_alternative<BoundModuleType>(weight->type))
				return std::nullopt;
			return weight->name;
		}

		/** @brief A key of the structure of @p type (to find the struct a value of the type belongs to). */
		static std::string aggregateKey(const AggregateType& type) {
			std::string key = "{";
			for (const auto& [name, member] : type.members) {
				key += name + ":";
				if (const auto* nested = std::get_if<AggregateType>(&member))
					key += aggregateKey(*nested);
				else
					key += typeString(member);
				key += ";";
			}
			return key + "}";
		}

		/** @brief The struct (or `struct.member` for inline struct types) of each struct type (for the index). */
		std::map<std::string, std::string> structContainers;

		/** @brief Defines the members of the struct type @p ctxMembers in the index (recursively for inline structs). */
		void defineStructMembers(
				const std::vector<mdlParser::VardeclContext*>& vars, const AggregateType& type,
				const std::string& container, antlr4::ParserRuleContext* scope
		) {
			structContainers.try_emplace(aggregateKey(type), container);
			for (auto* var : vars) {
				if (!var->IDENT() || !type.members.contains(var->IDENT()->getText()))
					continue;
				const auto& member = type.members.at(var->IDENT()->getText());
				define(var->IDENT()->getSymbol(), analysis::SymbolKind::member, sourceText(var), var, rangeOf(scope),
					   {typeString(member)}, container);
				const auto* nested = std::get_if<AggregateType>(&member);
				auto* anon = dynamic_cast<mdlParser::AnonTypeContext*>(var->type());
				if (nested != nullptr && anon != nullptr)
					defineStructMembers(anon->vardecl(), *nested, container + "." + var->IDENT()->getText(), scope);
			}
		}

		/** @brief The range of the whole file of @p ctx (the scope of global definitions). */
		static SourceRange fileScope() { return {{1, 0}, {std::numeric_limits<size_t>::max(), 0}}; }

		/** @brief The innermost function body or loop around @p ctx (the scope of a local variable). */
		static antlr4::ParserRuleContext* enclosingBlock(antlr4::ParserRuleContext* ctx) {
			for (auto* p = dynamic_cast<antlr4::ParserRuleContext*>(ctx->parent); p != nullptr;
				 p = dynamic_cast<antlr4::ParserRuleContext*>(p->parent))
				if (dynamic_cast<mdlParser::FuncdefContext*>(p) || dynamic_cast<mdlParser::ForStatementContext*>(p))
					return p;
			return ctx;
		}

		/** @brief @p text without a leading `public`/`private` (for signatures shown in tools). */
		static std::string withoutVisibility(std::string text, mdlParser::VisibilityContext* visibility) {
			if (visibility != nullptr && text.starts_with(visibility->getText())) {
				text.erase(0, visibility->getText().size());
				text.erase(0, text.find_first_not_of(" \t\r\n"));
			}
			return text;
		}

		/** @brief The signature of a function as written (without its body and visibility). */
		std::string signature(mdlParser::FuncdefContext* ctx) {
			antlr4::Token* close = nullptr;
			for (auto* child : ctx->children)
				if (auto* t = dynamic_cast<antlr4::tree::TerminalNode*>(child); t && t->getText() == ")")
					close = t->getSymbol();
			return withoutVisibility(sourceText(ctx, close), ctx->visibility());
		}
		/** @brief Directory of the file that is currently translated (first search path for its imports). */
		std::filesystem::path currentDir;
		/** @brief Whether definitions of the current file are added to the compilation unit (false for imports). */
		bool recordDefinitions = true;
		/** @brief Canonical paths of all files that have been (or are being) translated. */
		std::set<std::filesystem::path> translatedFiles;
		size_t numErrors = 0;

		std::stack<mdl::Computegraph*> graphstack;
		NamespaceStack nsstack;
		/** @brief The scopes of the translated files (stable addresses), by canonical path. */
		std::deque<FileScope> fileScopes;
		std::map<std::filesystem::path, FileScope*> scopeOfFile;
		/** @brief The scope of the file that is currently translated, and of the compiled (main) file. */
		FileScope* currentScope = nullptr;
		FileScope* mainScope = nullptr;
		/** @brief The compiled file's directory: an import root, and the directory of its package. */
		std::filesystem::path mainDir;
		/** @brief The names of std.prelude (the parent of the files' import namespaces, below the builtins). */
		Namespace prelude;
		bool preludeLoaded = false;
		/** @brief The namespaces of module instances (kept so that their functions can be specialized later). */
		std::deque<mdl::Namespace> instanceScopes;
		/** @brief While specializing a function: which parameters take masked tensors (see specialize()). */
		std::optional<std::vector<bool>> maskedParams;
		/** @brief While specializing a function: the values of its template parameters (inferred at the call). */
		std::map<std::string, int> templateValues;
		/** @brief Specialized functions by definition and argument pattern. */
		std::map<std::pair<const FunctionTemplate*, std::string>, const FunctionDecl*> specializations;
		/** @brief Name of the module whose definition is currently parsed (scope of its weights); empty outside. */
		std::string scope;

		/** @brief The result of an expression that evaluates to multiple values (a call of a multi-output function). */
		struct MultiValue {
			std::vector<const Node*> nodes;
		};

		using ModuleParam = std::pair<std::string, mdl::DataType>;
		using VarDecl = std::pair<std::string, mdl::DataType>;

		mdl::Computegraph& graph() { return *graphstack.top(); }
		void pushGraph(mdl::Computegraph* graph) { graphstack.push(graph); }
		void popGraph() { graphstack.pop(); }

		mdl::Namespace& currentNamespace() { return nsstack.top(); }
		void pushNamespace() { nsstack.push(nsstack.top().createSubnamespace()); }
		void popNamespace() { nsstack.pop(); }

		// Convert any expression result (const Node*, Object, or MultiValue) to Object.
		Object asObject(std::any value, antlr4::Token* tok = nullptr) {
			if (auto* node = std::any_cast<const Node*>(&value))
				return Object{*node};
			if (auto* obj = std::any_cast<Object>(&value))
				return *obj;
			if (auto* multi = std::any_cast<MultiValue>(&value)) {
				if (tok != nullptr)
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
							std::format(
									"Expected a single value but the expression yields {} values; use `auto a, b = "
									"...;`",
									multi->nodes.size()
							)
					);
				return Object{&graph().addErrorNode()};
			}
			std::abort();
		}

		// Extract const Node* from an expression result, emitting an error if it is not a node.
		const Node* asNode(std::any value, antlr4::Token* tok) {
			auto obj = asObject(std::move(value), tok);
			return std::visit(
					overloaded{
							[](const Node* node) { return node; },
							[&](const FunctionDecl&) -> const Node* {
								errorlistener->semanticError(
										PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(),
										tok->getCharPositionInLine(), "Expected a value, got a function"
								);
								return &graph().addErrorNode();
							},
							[&](const AggregateInstance&) -> const Node* {
								errorlistener->semanticError(
										PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(),
										tok->getCharPositionInLine(), "Expected a scalar/tensor value, got a struct"
								);
								return &graph().addErrorNode();
							},
							[&](const ModuleInstance&) -> const Node* {
								errorlistener->semanticError(
										PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(),
										tok->getCharPositionInLine(), "Expected a scalar/tensor value, got a module"
								);
								return &graph().addErrorNode();
							},
							[&](const ArrayInstance&) -> const Node* {
								errorlistener->semanticError(
										PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(),
										tok->getCharPositionInLine(), "Expected a scalar/tensor value, got an array"
								);
								return &graph().addErrorNode();
							}
					},
					obj
			);
		}

		// Build an AggregateInstance for namespace member access, creating one
		// node per field via addNode (these nodes are NOT added to any leaves list).
		using AddNodeFn = std::function<const Node&(DataType, std::string)>;
		Object makeAggregateObject(const AggregateType& agg, const std::string& prefix, const AddNodeFn& addNode) {
			std::map<std::string, Object> values;
			for (auto& [memberName, memberType] : agg.members) {
				const std::string memberPath = prefix + "." + memberName;
				if (auto* subAgg = std::get_if<AggregateType>(&memberType))
					values[memberName] = makeAggregateObject(*subAgg, memberPath, addNode);
				else
					values[memberName] = Object{&addNode(memberType, memberPath)};
			}
			return Object{AggregateInstance::create(agg, std::move(values)).value()};
		}

		/** @brief Reports a redefinition and returns true if @p name is already defined in the current scope. */
		bool checkRedefinition(antlr4::tree::TerminalNode* ident) {
			if (!currentNamespace().definesLocally(ident->getText()))
				return false;
			auto* tok = ident->getSymbol();
			errorlistener->semanticError(
					PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
					std::format("Redefinition of {}", ident->getText())
			);
			return true;
		}

		// ── Imports (see docs/dev/spec/language.md, Imports and visibility) ───────────────────────────────────────────

		static bool isSource(const std::filesystem::path& p) {
			return std::filesystem::is_regular_file(p) && (p.extension() == ".dlm" || p.extension() == ".mdl");
		}

		/** @brief The source file `path.dlm` (or `.mdl`), if it exists. */
		static std::optional<std::filesystem::path> sourceFile(const std::filesystem::path& path) {
			for (const char* ext : {".dlm", ".mdl"}) {
				auto file = path;
				file += ext;
				if (isSource(file))
					return file;
			}
			return std::nullopt;
		}

		static std::filesystem::path canonical(const std::filesystem::path& p) {
			std::error_code ec;
			auto c = std::filesystem::weakly_canonical(p, ec);
			return ec ? p : c;
		}

		std::vector<std::filesystem::path> stdlibDirs() const {
			return options.stdlibDirs.empty() ? parser::defaultStdlibDirs() : options.stdlibDirs;
		}

		/** @brief The package directory that identifies the package `std` (any of the standard library directories). */
		static std::filesystem::path stdPackage() { return "<std>"; }

		/** @brief Whether @p p is @p dir or inside it. */
		static bool isWithin(const std::filesystem::path& p, const std::filesystem::path& dir) {
			auto rel = canonical(p).lexically_relative(canonical(dir));
			return !rel.empty() && *rel.begin() != "..";
		}

		/** @brief What an import path refers to: a file or a directory, with its package. */
		struct ImportTarget {
			std::filesystem::path path;
			bool directory = false;
			std::filesystem::path packageDir;
			bool isStd = false;
		};

		/**
		 * @brief Resolves an import path. A relative path (leading dots) starts at the importing file's directory; an
		 * absolute one is searched in the compiled file's directory, the import directories, and (`std`) the standard
		 * library. @p dotted receives the path as written.
		 */
		std::optional<ImportTarget> resolveImportPath(mdlParser::Import_pathContext* ctx, std::string& dotted) {
			size_t dots = 0;
			for (auto* d : ctx->dots)
				dots += d->getText().size();
			std::filesystem::path relative;
			for (auto* part : ctx->path) {
				relative /= part->getText();
				dotted += (dotted.empty() ? "" : ".") + part->getText();
			}
			dotted = std::string(dots, '.') + dotted;
			auto found = [&](const std::filesystem::path& base, const std::filesystem::path& rel
						 ) -> std::optional<std::pair<std::filesystem::path, bool>> {
				if (!rel.empty())
					if (auto file = sourceFile(base / rel))
						return std::pair{*file, false};
				auto dir = rel.empty() ? base : base / rel;
				if (std::filesystem::is_directory(dir))
					return std::pair{dir, true};
				return std::nullopt;
			};
			if (dots > 0) {
				auto base = currentDir;
				for (size_t i = 1; i < dots; ++i)
					base = base.parent_path();
				auto hit = found(base, relative);
				if (!hit)
					return std::nullopt;
				// Within the importing file's package: the same package; otherwise a package of its own.
				const bool within = currentScope != nullptr &&
									(currentScope->isStd || isWithin(hit->first, currentScope->packageDir));
				return ImportTarget{
						.path = hit->first,
						.directory = hit->second,
						.packageDir = within ? currentScope->packageDir : packageOfLoose(hit->first, hit->second),
						.isStd = within && currentScope->isStd
				};
			}
			if (!relative.empty() && *relative.begin() == "std") {
				std::filesystem::path rest;
				for (auto it = std::next(relative.begin()); it != relative.end(); ++it)
					rest /= *it;
				for (const auto& dir : stdlibDirs())
					if (auto hit = found(dir, rest))
						return ImportTarget{hit->first, hit->second, stdPackage(), true};
			}
			std::vector<std::filesystem::path> roots{mainDir};
			roots.insert(roots.end(), options.importDirs.begin(), options.importDirs.end());
			for (const auto& root : roots) {
				auto hit = found(root, relative);
				if (!hit)
					continue;
				// The compiled program's directory is one package; below another root, a top-level directory is a
				// package (a file directly in the root is one of its own).
				std::filesystem::path package = root == mainDir ? mainDir : root / *relative.begin();
				if (root != mainDir && std::distance(relative.begin(), relative.end()) == 1 && !hit->second)
					package = hit->first;
				return ImportTarget{hit->first, hit->second, package, false};
			}
			return std::nullopt;
		}

		/** @brief The package of a file outside the importer's package (reached by `..`): its directory. */
		static std::filesystem::path packageOfLoose(const std::filesystem::path& path, bool directory) {
			return directory ? path : path.parent_path();
		}

		/** @brief The visibility written before a definition or import (package if none). */
		static Visibility visibilityOf(mdlParser::VisibilityContext* ctx) {
			if (ctx == nullptr)
				return Visibility::Package;
			return ctx->getText() == "public" ? Visibility::Public : Visibility::Private;
		}

		static const char* str(Visibility v) {
			return v == Visibility::Private ? "private" : v == Visibility::Package ? "package-visible" : "public";
		}

		/** @brief Whether a name of @p scope with visibility @p v is visible in the file @p requester. */
		static bool visibleFrom(Visibility v, const FileScope& scope, const FileScope* requester) {
			switch (v) {
			case Visibility::Private:
				return requester == &scope;
			case Visibility::Package:
				return requester != nullptr && requester->packageDir == scope.packageDir;
			case Visibility::Public:
				return true;
			}
			return false;
		}

		/** @brief A name of a file's namespace (a definition or an import), as seen from another file. */
		struct ScopeName {
			const std::any* object = nullptr; ///< an object (Object or NamespaceRef)
			const DataType* type = nullptr;	  ///< a module or struct
			std::string origin;
			std::optional<size_t> symbol;
		};

		/**
		 * @brief The name @p name of @p scope if the file @p requester may use it: a definition of the file, or a name
		 * it re-exports (`public` imports). @p why receives the reason if the name exists but is not visible.
		 */
		std::optional<ScopeName>
		nameIn(const FileScope& scope, const std::string& name, const FileScope* requester,
			   std::string* why = nullptr) const {
			for (const Namespace* ns : {&scope.local, &scope.imports}) {
				const auto* object = ns->localObject(name);
				const auto* type = ns->localType(name);
				if (object == nullptr && type == nullptr)
					continue;
				const auto* entry = ns->entry(name);
				const Visibility v = entry != nullptr		? entry->visibility
									 : ns == &scope.imports ? Visibility::Private
															: Visibility::Package;
				if (!visibleFrom(v, scope, requester)) {
					if (why != nullptr)
						*why = std::format("{} is {} in {}", name, str(v), scope.path.filename().string());
					continue;
				}
				return ScopeName{
						.object = object,
						.type = type,
						.origin = entry != nullptr && !entry->origin.empty() ? entry->origin
																			 : scope.path.string() + "#" + name,
						.symbol = ns->localSymbol(name)
				};
			}
			return std::nullopt;
		}

		/** @brief The names of @p scope that @p requester may use (for `from x import *`). */
		std::vector<std::string> namesIn(const FileScope& scope, const FileScope* requester) const {
			std::vector<std::string> names;
			for (const Namespace* ns : {&scope.local, &scope.imports})
				for (const auto& name : ns->localNames())
					if (nameIn(scope, name, requester) && std::ranges::find(names, name) == names.end())
						names.push_back(name);
			return names;
		}

		/** @brief The files that re-export @p name publicly (for the hint when it is not visible). */
		std::vector<std::string> reexportersOf(const std::string& origin, const std::string& name) const {
			std::vector<std::string> files;
			for (const auto& scope : fileScopes)
				if (const auto* entry = scope.imports.entry(name);
					entry && entry->visibility == Visibility::Public && entry->origin == origin)
					files.push_back(scope.path.filename().string());
			return files;
		}

		/** @brief The namespace of an import target (translating the file, or the directory's index file). */
		std::optional<NamespaceRef> namespaceOfTarget(const ImportTarget& target) {
			if (!target.directory) {
				const FileScope* file = translateFile(target.path, false, target.packageDir, target.isStd);
				return file != nullptr ? std::optional<NamespaceRef>{NamespaceRef{.file = file}} : std::nullopt;
			}
			if (auto index = sourceFile(target.path / target.path.filename())) {
				const FileScope* file = translateFile(*index, false, target.packageDir, target.isStd);
				return file != nullptr ? std::optional<NamespaceRef>{NamespaceRef{.file = file}} : std::nullopt;
			}
			return NamespaceRef{.dir = target.path, .packageDir = target.packageDir, .isStd = target.isStd};
		}

		/** @brief A member of a namespace: a name of a file, or a file or subdirectory of a directory. */
		struct Member {
			std::optional<NamespaceRef> ns;
			std::optional<ScopeName> name;
			std::string error;
		};

		Member memberOf(const NamespaceRef& ref, const std::string& name) {
			if (ref.file != nullptr) {
				std::string why;
				if (auto n = nameIn(*ref.file, name, currentScope, &why)) {
					if (n->object != nullptr)
						if (const auto* nested = std::any_cast<NamespaceRef>(n->object))
							return Member{.ns = *nested};
					return Member{.name = *n};
				}
				if (why.empty())
					return Member{.error = std::format("{} has no {}", ref.file->path.filename().string(), name)};
				if (auto files = reexportersOf(ref.file->path.string() + "#" + name, name); !files.empty())
					why += std::format(" (it is re-exported by {})", files.front());
				return Member{.error = why};
			}
			ImportTarget target{.packageDir = ref.packageDir, .isStd = ref.isStd};
			if (auto file = sourceFile(ref.dir / name))
				target.path = *file;
			else if (std::filesystem::is_directory(ref.dir / name))
				target = {ref.dir / name, true, ref.packageDir, ref.isStd};
			else
				return Member{.error = std::format("the directory {} has no {}", ref.dir.filename().string(), name)};
			if (auto ns = namespaceOfTarget(target))
				return Member{.ns = *ns};
			return Member{.error = std::format("cannot import {}", name)};
		}

		void importError(antlr4::Token* tok, const std::string& message) {
			errorlistener->semanticError(
					PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(), message
			);
		}

		/**
		 * @brief Binds @p name in the current file's imports. An explicit import of a name that is already imported
		 * from another definition is an error; a `*` import (@p explicitImport false) keeps the earlier name.
		 */
		void bindImport(
				const std::string& name, const std::any* object, const DataType* type, const std::string& origin,
				Visibility visibility, std::optional<size_t> symbol, antlr4::Token* tok, bool explicitImport
		) {
			auto& imports = currentScope->imports;
			if (imports.localObject(name) != nullptr || imports.localType(name) != nullptr) {
				const auto* entry = imports.entry(name);
				if (entry != nullptr && entry->origin == origin) {
					if (visibility == Visibility::Public)
						imports.setEntry(name, {visibility, origin});
					return;
				}
				if (explicitImport)
					importError(tok, std::format("{} is already imported (from another definition)", name));
				return;
			}
			if (object != nullptr)
				imports.putAny(name, *object);
			if (type != nullptr)
				imports.putType(name, DataType{*type});
			imports.setEntry(name, {visibility, origin});
			if (symbol)
				imports.setSymbol(name, *symbol);
		}

		virtual std::any visitImportModule(mdlParser::ImportModuleContext* ctx) override {
			std::string dotted;
			auto target = resolveImportPath(ctx->import_path(), dotted);
			if (!target) {
				importError(ctx->start, std::format("Cannot find {}", dotted));
				return {};
			}
			auto ns = namespaceOfTarget(*target);
			if (!ns)
				return {};
			const auto& path = ctx->import_path()->path;
			if (path.empty() && ctx->alias == nullptr) {
				importError(ctx->start, "`import .` needs a name: `import . as name;` or `from . import x;`");
				return {};
			}
			const std::string name = ctx->alias != nullptr ? ctx->alias->getText() : path.back()->getText();
			const std::any value = *ns;
			bindImport(
					name, &value, nullptr, "namespace:" + (ns->file ? ns->file->path : ns->dir).string(),
					visibilityOf(ctx->visibility()), std::nullopt, ctx->start, true
			);
			if (index() != nullptr)
				indexImport(
						ctx->import_path(), ctx->visibility() != nullptr && ctx->visibility()->getText() == "public"
				);
			return {};
		}

		virtual std::any visitImportNames(mdlParser::ImportNamesContext* ctx) override {
			std::string dotted;
			auto target = resolveImportPath(ctx->import_path(), dotted);
			if (!target) {
				importError(ctx->start, std::format("Cannot find {}", dotted));
				return {};
			}
			auto ns = namespaceOfTarget(*target);
			if (!ns)
				return {};
			if (index() != nullptr) {
				std::vector<std::string> names;
				if (ctx->star != nullptr)
					names.emplace_back("*");
				for (auto* n : ctx->names)
					names.push_back((n->alias != nullptr ? n->alias : n->name)->getText());
				indexImport(
						ctx->import_path(), ctx->visibility() != nullptr && ctx->visibility()->getText() == "public",
						std::move(names)
				);
			}
			const Visibility visibility = visibilityOf(ctx->visibility());
			auto bindMember = [&](const std::string& name, const std::string& as, antlr4::Token* tok,
								  bool explicitImport) {
				Member m = memberOf(*ns, name);
				if (m.ns) {
					const std::any value = *m.ns;
					bindImport(
							as, &value, nullptr, "namespace:" + (m.ns->file ? m.ns->file->path : m.ns->dir).string(),
							visibility, std::nullopt, tok, explicitImport
					);
				} else if (m.name) {
					bindImport(
							as, m.name->object, m.name->type, m.name->origin, visibility, m.name->symbol, tok,
							explicitImport
					);
				} else if (explicitImport) {
					importError(tok, std::format("Cannot import {} from {}: {}", name, dotted, m.error));
					// (Uses of the name are not reported again: it stands for an error.)
					if (!currentScope->imports.definesLocally(as)) {
						currentScope->imports.put<Object>(as, Object{&graph().addErrorNode()});
						currentScope->imports.setEntry(as, {Visibility::Private, "error:" + as});
					}
				}
			};
			if (ctx->star != nullptr) {
				std::vector<std::string> names;
				if (ns->file != nullptr) {
					names = namesIn(*ns->file, currentScope);
				} else {
					for (const auto& entry : std::filesystem::directory_iterator(ns->dir))
						if (isSource(entry.path()) || entry.is_directory())
							names.push_back(entry.path().stem().string());
					std::ranges::sort(names);
				}
				for (const auto& name : names)
					bindMember(name, name, ctx->star, false);
				return {};
			}
			for (auto* n : ctx->names) {
				const std::string name = n->name->getText();
				bindMember(name, n->alias != nullptr ? n->alias->getText() : name, n->name, true);
				if (index() != nullptr && currentScope->imports.localSymbol(n->alias ? n->alias->getText() : name))
					index()->reference(analysis::Reference{
							.file = currentFile(),
							.range = rangeOf(n->name),
							.symbol = currentScope->imports.localSymbol(n->alias ? n->alias->getText() : name),
					});
			}
			return {};
		}

		/** @brief Loads std.prelude (once) and binds its public names in the prelude namespace. */
		void ensurePrelude() {
			if (preludeLoaded)
				return;
			preludeLoaded = true;
			for (const auto& dir : stdlibDirs()) {
				auto file = sourceFile(dir / "prelude");
				if (!file)
					continue;
				const FileScope* scope = translateFile(*file, false, stdPackage(), true);
				if (scope == nullptr)
					return;
				for (const auto& name : namesIn(*scope, nullptr)) {
					auto n = nameIn(*scope, name, nullptr);
					if (n->object != nullptr)
						prelude.putAny(name, *n->object);
					if (n->type != nullptr)
						prelude.putType(name, DataType{*n->type});
					if (n->symbol)
						prelude.setSymbol(name, *n->symbol);
				}
				return;
			}
		}

		/** @brief Records every part of an import path with the file(s) it stands for (for hover and go to definition). */
		void indexImport(mdlParser::Import_pathContext* ctx, bool reexported, std::vector<std::string> names = {}) {
			auto read = [](const std::filesystem::path& file) {
				std::ifstream in{file, std::ios::binary};
				return std::string{std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
			};
			// Resolve each prefix of the path (`std`, `std.nn`) like the whole path.
			std::string dotted;
			for (size_t k = 0; k < ctx->path.size(); ++k) {
				auto* part = ctx->path[k];
				std::filesystem::path relative;
				size_t dots = 0;
				for (auto* d : ctx->dots)
					dots += d->getText().size();
				for (size_t i = 0; i <= k; ++i)
					relative /= ctx->path[i]->getText();
				dotted += (k == 0 ? std::string(dots, '.') : ".") + part->getText();
				std::vector<std::filesystem::path> files;
				std::filesystem::path base = currentDir;
				for (size_t i = 1; i < dots; ++i)
					base = base.parent_path();
				std::vector<std::filesystem::path> roots;
				if (dots > 0)
					roots = {base};
				else {
					roots = {mainDir};
					roots.insert(roots.end(), options.importDirs.begin(), options.importDirs.end());
					if (*relative.begin() == "std") {
						std::filesystem::path rest;
						for (auto it = std::next(relative.begin()); it != relative.end(); ++it)
							rest /= *it;
						for (const auto& dir : stdlibDirs()) {
							if (rest.empty()) {
								if (std::filesystem::is_directory(dir))
									for (const auto& entry : std::filesystem::directory_iterator(dir))
										if (isSource(entry.path()))
											files.push_back(entry.path());
							} else if (auto file = sourceFile(dir / rest))
								files.push_back(*file);
							else if (std::filesystem::is_directory(dir / rest))
								for (const auto& entry : std::filesystem::directory_iterator(dir / rest))
									if (isSource(entry.path()))
										files.push_back(entry.path());
							if (!files.empty())
								break;
						}
					}
				}
				if (files.empty())
					for (const auto& root : roots) {
						if (auto file = sourceFile(root / relative)) {
							files.push_back(*file);
							break;
						}
						if (std::filesystem::is_directory(root / relative)) {
							for (const auto& entry : std::filesystem::directory_iterator(root / relative))
								if (isSource(entry.path()))
									files.push_back(entry.path());
							break;
						}
					}
				std::ranges::sort(files);
				if (files.empty())
					continue;
				analysis::ImportPath path{.file = currentFile(), .range = rangeOf(part), .name = dotted};
				path.reexported = reexported;
				if (k + 1 == ctx->path.size())
					path.names = names;
				const bool package = files.size() > 1 || std::filesystem::path{files.front()}.stem() != part->getText();
				for (const auto& file : files) {
					// (the name translateFile gives the file: not canonical, a symbolic link in the path stays, so that
					// the uses of its definitions match the import, e.g. for -Wunused)
					path.targets.push_back(std::filesystem::absolute(file).string());
					auto doc = analysis::fileDocumentation(
							options.readFile && options.readFile(file) ? *options.readFile(file) : read(file)
					);
					if (!package) {
						path.documentation = doc;
						continue;
					}
					// A package: its files with the first paragraph of their documentation.
					path.documentation += std::format("- `{}.{}`", dotted, file.stem().string());
					if (auto brief = doc.substr(0, doc.find("\n\n")); !brief.empty())
						path.documentation += ": " + brief;
					path.documentation += "\n";
				}
				index()->addImport(std::move(path));
			}
		}

	public:
		/** @brief Also check the modules with hyperparameters of the main file symbolically (see analyzeGeneric()). */

		TranslatingVisitor(parser::FrontendState& state, parser::ParseOptions options)
				: state(state), options(std::move(options)) {
			pushGraph(&state.graph);
			nsstack.push(mdl::Namespace::baseNamespace());
			prelude = nsstack.bottom().createSubnamespace();
		}

		size_t getNumErrors() const { return numErrors; }
		std::map<std::string, CompilationUnit::Member>& getDefinitions() { return definitions; }
		std::vector<std::filesystem::path>& getSources() { return sources; }

		/** @brief Instantiates the module named @p name (see CompilationUnit::instantiate). */
		std::expected<std::reference_wrapper<const ModuleType>, std::string>
		instantiateByName(const std::string& name, const std::map<std::string, Object>& values) {
			auto type = (mainScope != nullptr ? mainScope->local : nsstack.top()).getType(name);
			const auto* mod = type ? std::get_if<ModuleType>(&type.value().get()) : nullptr;
			if (mod == nullptr || mod->templ == nullptr)
				return std::unexpected(std::format("{} is not a module", name));
			// Given values by name; the others take their defaults.
			std::vector<std::pair<std::string, Object>> named(values.begin(), values.end());
			auto ordered = completeHyperparameters(*mod, {}, std::move(named));
			if (!ordered)
				return std::unexpected(ordered.error().message);
			return instantiate(*mod, *ordered);
		}

		/**
		 * @brief Translates the program read from @p stream into the compilation unit.
		 * @param main whether the definitions are added to CompilationUnit::definitions (false for imported files).
		 */
		/**
		 * @brief The scope of a file (created on first use). The compiled file's directory is its package (and an
		 * import root).
		 */
		FileScope&
		scopeFor(const std::filesystem::path& path, const std::filesystem::path& packageDir, bool isStd, bool main) {
			if (auto it = scopeOfFile.find(path); it != scopeOfFile.end())
				return *it->second;
			if (!isStd)
				ensurePrelude();
			auto& scope = fileScopes.emplace_back();
			scope.path = path;
			scope.packageDir = packageDir;
			scope.isStd = isStd;
			scope.imports = (isStd ? nsstack.bottom() : prelude).createSubnamespace();
			scope.local = scope.imports.createSubnamespace();
			scopeOfFile.emplace(path, &scope);
			if (main)
				mainScope = &scope;
			return scope;
		}

		void translate(
				std::istream& stream, const std::string& sourcename, const std::filesystem::path& dir, bool main,
				FileScope* scope = nullptr
		) {
			if (main && mainDir.empty())
				mainDir = dir;
			if (scope == nullptr)
				scope = &scopeFor(sourcename, dir, false, main);
			nsstack.pushRef(scope->local);
			auto* outerScope = std::exchange(currentScope, scope);
			// The parse tree stays alive (owned by the frontend state) so that module templates defined in this file
			// can be expanded later.
			auto& file = *state.files.emplace_back(std::make_unique<ParsedFile>());
			file.errors = std::make_unique<PrettyErrorListener>(sourcename, state.sink);
			// Invalid UTF-8 is an error; the file is still decoded (leniently: replacement characters) and analyzed.
			const std::string text{std::istreambuf_iterator<char>{stream}, {}};
			if (auto invalid = invalidUtf8(text, sourcename))
				file.errors->semanticError(
						PrettyErrorListener::ErrorLevel::Error, invalid->context, invalid->range.begin.line,
						invalid->range.begin.column, invalid->message
				);
			file.input = std::make_unique<antlr4::ANTLRInputStream>();
			file.input->load(text, /*lenient=*/true);
			file.lexer = std::make_unique<antlr4::mdlLexer>(file.input.get());
			file.tokens = std::make_unique<antlr4::CommonTokenStream>(file.lexer.get());
			file.parser = std::make_unique<antlr4::mdlParser>(file.tokens.get());
			file.lexer->removeErrorListeners();
			file.lexer->addErrorListener(file.errors.get());
			file.parser->removeErrorListeners();
			file.parser->addErrorListener(file.errors.get());

			tokenStreams[sourcename] = file.tokens.get();
			auto* outerErrors = std::exchange(errorlistener, file.errors.get());
			auto outerDir = std::exchange(currentDir, dir);
			bool outerRecord = std::exchange(recordDefinitions, main);
			const size_t errorsBefore = file.errors->getNumErrors();
			auto* program = file.parser->program();
			// A tree with syntax errors has missing children (null contexts): only report the syntax errors.
			if (file.parser->getNumberOfSyntaxErrors() == 0) {
				program->accept(this);
				if (main)
					reportUnused(sourcename, *file.input);
			}
			errorlistener = outerErrors;
			currentDir = std::move(outerDir);
			recordDefinitions = outerRecord;
			currentScope = outerScope;
			nsstack.pop();

			if (state.printing)
				printDiagnosticSummary(std::cerr, file.errors->getNumErrors(), file.errors->getNumWarnings());
			numErrors += file.errors->getNumErrors() - errorsBefore;
		}

		/** @brief Reports the definitions of the file @p sourcename that are never used (`-Wunused`). */
		void reportUnused(const std::string& sourcename, antlr4::CharStream& input) {
			if (index() == nullptr || options.unused == parser::DiagnosticLevel::ignored)
				return;
			const auto level = options.unused == parser::DiagnosticLevel::error
									   ? PrettyErrorListener::ErrorLevel::Error
									   : PrettyErrorListener::ErrorLevel::Warning;
			const std::string text = input.toString();
			for (const auto& u : analysis::unusedDefinitions(*index(), sourcename)) {
				// The source line (the index's lines count from 1).
				size_t start = 0;
				for (size_t line = 1; line < u.range.begin.line && start != std::string::npos; ++line)
					if ((start = text.find('\n', start)) != std::string::npos)
						++start;
				if (start == std::string::npos)
					continue;
				const std::string line = text.substr(start, text.find('\n', start) - start);
				const size_t width = u.range.end.line == u.range.begin.line && u.range.end.column > u.range.begin.column
											 ? u.range.end.column - u.range.begin.column
											 : 1;
				errorlistener->semanticError(
						level,
						{{.number = u.range.begin.line,
						  .text = line,
						  .markBegin = u.range.begin.column,
						  .markWidth = width}},
						u.range.begin.line, u.range.begin.column, u.message, "", "unused"
				);
			}
		}

		/**
		 * @brief Translates @p file once (its scope on later calls, also while it is being translated: a cycle sees
		 * the names defined so far). The compiled file (@p main) is its own package's root.
		 */
		FileScope* translateFile(
				const std::filesystem::path& file, bool main, std::filesystem::path packageDir = {}, bool isStd = false
		) {
			const auto path = canonical(file);
			if (main) {
				mainDir = path.parent_path();
				packageDir = mainDir;
				// A file of the standard library is in the package `std` (without the prelude, which imports it).
				for (const auto& dir : stdlibDirs())
					if (isWithin(path, dir)) {
						packageDir = stdPackage();
						isStd = true;
					}
			}
			if (!translatedFiles.insert(path).second)
				return scopeOfFile.contains(path) ? scopeOfFile.at(path) : nullptr;
			auto& scope = scopeFor(path, packageDir.empty() ? path.parent_path() : packageDir, isStd, main);
			sources.push_back(path);
			// The contents: from the options' reader (e.g. an editor's unsaved buffer), else from the file.
			std::optional<std::string> text = options.readFile ? options.readFile(file) : std::nullopt;
			std::ifstream fileStream;
			std::istringstream textStream;
			if (text)
				textStream.str(std::move(*text));
			else
				fileStream.open(file);
			std::istream& stream = text ? static_cast<std::istream&>(textStream) : fileStream;
			if (!stream) {
				state.sink(Diagnostic{.file = file.string(), .message = "cannot read " + file.string()});
				++numErrors;
				return nullptr;
			}
			translate(stream, std::filesystem::absolute(file).string(), file.parent_path(), main, &scope);
			return &scope;
		}

		virtual std::any visitProgram(mdlParser::ProgramContext* ctx) override {
			// Imports are processed first so that their names are visible to the whole file: explicit names before
			// `*` imports (which do not override them).
			for (auto* import : ctx->import_stmt())
				if (auto* names = dynamic_cast<mdlParser::ImportNamesContext*>(import); !names || !names->star)
					import->accept(this);
			for (auto* import : ctx->import_stmt())
				if (auto* names = dynamic_cast<mdlParser::ImportNamesContext*>(import); names && names->star)
					import->accept(this);
			auto record = [&](std::string name, CompilationUnit::Member member) {
				if (recordDefinitions)
					definitions.try_emplace(std::move(name), std::move(member));
			};
			// Parse Structs first (no matter where they are in the file)
			// Does this make sense? Only if Structs can not have instances of modules as members.
			for (auto&& def : ctx->struct_def()) {
				auto [name, s] = std::any_cast<std::pair<std::string, AggregateType>>(def->accept(this));
				record(std::move(name), std::move(s));
			}
			// Parse func definitions
			for (auto&& def : ctx->funcdef()) {
				auto [name, f] = std::any_cast<std::pair<std::string, FunctionDecl>>(def->accept(this));
				record(std::move(name), std::move(f));
			}
			// Parse Modules third (no matter where they are in the file)
			std::vector<ModuleType> generic;
			for (auto&& def : ctx->module_def()) {
				auto [name, m] = std::any_cast<std::pair<std::string, ModuleType>>(def->accept(this));
				if (recordDefinitions && !m.instantiated && !m.typeargs.empty())
					generic.push_back(m);
				record(std::move(name), std::move(m));
			}
			for (const auto& m : generic)
				analyzeGeneric(m);
			// The visibility of the file's definitions (an overload set: the widest of its functions).
			auto entry = [&](antlr4::tree::TerminalNode* ident, mdlParser::VisibilityContext* v) {
				if (ident == nullptr || currentScope == nullptr)
					return;
				const std::string name = ident->getText();
				Visibility visibility = visibilityOf(v);
				if (const auto* e = currentScope->local.entry(name))
					visibility = std::max(visibility, e->visibility);
				currentScope->local.setEntry(name, {visibility, currentScope->path.string() + "#" + name});
			};
			for (auto* def : ctx->struct_def())
				entry(def->IDENT(), def->visibility());
			for (auto* def : ctx->funcdef())
				entry(def->IDENT(), def->visibility());
			for (auto* def : ctx->module_def())
				entry(def->IDENT(), def->visibility());
			return {};
		}

		/**
		 * @brief A symbolic value of a hyperparameter of type @p type named @p name: a hyperparameter node for numbers
		 * and booleans, structs and arrays of them (members `conf.dims.model`, elements `sizes[2]`).
		 */
		std::optional<Object> symbolicValue(const DataType& type, const std::string& name) {
			if (std::holds_alternative<PrimitiveType>(type))
				return Object{&graph().addHyperparameterNode(type, name)};
			if (const auto* agg = std::get_if<AggregateType>(&type)) {
				std::map<std::string, Object> members;
				for (const auto& [member, memberType] : agg->members) {
					auto value = symbolicValue(memberType, name + "." + member);
					if (!value)
						return std::nullopt;
					members.emplace(member, std::move(*value));
				}
				auto instance = AggregateInstance::create(*agg, std::move(members));
				return instance ? std::optional<Object>{Object{std::move(*instance)}} : std::nullopt;
			}
			if (const auto* array = std::get_if<ArrayType>(&type)) {
				ArrayInstance instance;
				for (unsigned i = 0; i < array->size; ++i) {
					auto element = symbolicValue(*array->element, std::format("{}[{}]", name, i));
					if (!element)
						return std::nullopt;
					instance.elements.push_back(std::move(*element));
				}
				return Object{std::move(instance)};
			}
			return std::nullopt;
		}

		/**
		 * @brief Checks a module with hyperparameters without values (for tools, see parser::analyze): the module is
		 * instantiated with symbolic values, so that types read e.g. `float[..., dim]`. What depends on unknown values
		 * is checked conservatively: both branches of `?:`, members with `@conditional`, and one iteration of loops and
		 * one instance of arrays of unknown size; only definite mismatches are errors.
		 */
		void analyzeGeneric(const ModuleType& def) {
			std::vector<Object> values;
			for (const auto& [pname, hn] : def.typeargs) {
				auto value = symbolicValue(hn.get().type, pname);
				if (!value)
					return;
				values.push_back(std::move(*value));
			}
			const bool outer = std::exchange(symbolicMode, true);
			(void)instantiate(def, values);
			symbolicMode = outer;
		}

		/** @brief Reports the annotations of a declaration where they do not apply (e.g. members of structs). */
		void rejectAnnotations(antlr4::mdlParser::VardeclContext* var, const std::string& where) {
			for (auto* a : var->annotation()) {
				if (!memberAnnotations().contains(a->name->getText())) {
					reportUnknownAnnotation(a);
					continue;
				}
				auto* tok = a->name;
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
						std::format("@{} applies to members of modules, not to {}", a->name->getText(), where)
				);
			}
		}

		void updateTypeWithVarDecl(
				AggregateType& type, const std::string& tname, std::vector<antlr4::mdlParser::VardeclContext*> vars
		) {
			for (auto& var : vars) {
				rejectAnnotations(var, "members of structs");
				auto [name, dtype] = std::any_cast<VarDecl>(var->accept(this));
				auto [_, inserted] = type.members.insert_or_assign(name, dtype);
				if (!inserted) {
					auto offendingToken = var->IDENT()->getSymbol();
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Warning, offendingToken, offendingToken->getLine(),
							offendingToken->getCharPositionInLine(),
							std::format(
									"Member {} was declared multiple times in {}. I will ignore the second "
									"declaration.",
									var->IDENT()->getText(), tname
							)
					);
				}
			}
		}

		virtual std::any visitStruct_def(mdlParser::Struct_defContext* ctx) override {
			AggregateType s;
			auto name = ctx->IDENT()->getText();
			updateTypeWithVarDecl(s, name, ctx->vardecl());
			if (!checkRedefinition(ctx->IDENT())) {
				reportImportShadowing(ctx->IDENT()->getSymbol());
				currentNamespace().putType(name, DataType{s});
				defineHere(
						ctx->IDENT()->getSymbol(), analysis::SymbolKind::structure, "struct " + name, ctx, fileScope()
				);
			}
			if (index() != nullptr)
				defineStructMembers(ctx->vardecl(), s, name, ctx);
			return std::make_pair(name, s);
		}

		/**
		 * @brief Registers a module template. Modules without hyperparameters are expanded immediately; all others
		 * when they are instantiated (see instantiate()).
		 */
		virtual std::any visitModule_def(mdlParser::Module_defContext* ctx) override {
			ModuleType m;
			auto name = ctx->IDENT()->getText();
			for (auto& param : ctx->param()) {
				const auto [pname, dtype] = std::any_cast<ModuleParam>(param->accept(this));
				// The hyperparameter nodes only document the types; the bodies are expanded with concrete values.
				const Node& node = graph().addHyperparameterNode(dtype, name + "." + pname);
				m.typeargs.emplace_back(pname, std::cref(std::get<HyperparameterNode>(node)));
			}
			m.templ = std::make_shared<const ModuleTemplate>(
					ModuleTemplate{.name = name, .ctx = ctx, .scope = &currentNamespace(), .errors = errorlistener}
			);
			if (index() != nullptr) {
				antlr4::Token* close = nullptr; // the `]` after the hyperparameters
				for (auto* child : ctx->children)
					if (auto* t = dynamic_cast<antlr4::tree::TerminalNode*>(child); t && t->getText() == "]" && !close)
						close = t->getSymbol();
				if (auto id =
							define(ctx->IDENT()->getSymbol(), analysis::SymbolKind::module,
								   withoutVisibility(sourceText(ctx, close), ctx->visibility()), ctx, fileScope()))
					currentNamespace().setSymbol(name, *id);
				for (size_t i = 0; i < ctx->param().size() && i < m.typeargs.size(); ++i) {
					define(ctx->param()[i]->IDENT()->getSymbol(), analysis::SymbolKind::hyperparameter,
						   sourceText(ctx->param()[i]), nullptr, rangeOf(ctx),
						   {typeString(m.typeargs[i].second.get().type)}, name);
					documentParameter(ctx->param()[i]->IDENT()->getSymbol(), ctx);
				}
			}
			if (m.typeargs.empty())
				if (auto instance = instantiate(m, {}))
					m = instance.value().get();
			if (!checkRedefinition(ctx->IDENT())) {
				reportImportShadowing(ctx->IDENT()->getSymbol());
				currentNamespace().putType(name, DataType{m});
			}
			return std::make_pair(name, m);
		}

		/** @brief Evaluates a compile-time integer or floating-point expression over constants. */
		static std::optional<Atomic> foldConstant(const Node& node) {
			using FIDs = builtins::FunctionIDs;
			if (const auto* cn = std::get_if<ConstNode>(&node)) {
				if (std::holds_alternative<std::vector<int>>(cn->value))
					return std::nullopt;
				return cn->value;
			}
			const auto* inv = std::get_if<InvocationNode>(&node);
			if (inv == nullptr || inv->arguments.size() != 2)
				return std::nullopt;
			auto lhs = foldConstant(inv->arguments[0].get());
			auto rhs = foldConstant(inv->arguments[1].get());
			if (!lhs || !rhs)
				return std::nullopt;
			if (std::holds_alternative<int>(*lhs) && std::holds_alternative<int>(*rhs)) {
				int a = std::get<int>(*lhs), b = std::get<int>(*rhs);
				switch (inv->id) {
				case FIDs::TENSOR_ADD:
					return Atomic{a + b};
				case FIDs::TENSOR_SUB:
					return Atomic{a - b};
				case FIDs::TENSOR_PROD:
					return Atomic{a * b};
				case FIDs::TENSOR_DIV:
					return b != 0 ? std::optional<Atomic>{Atomic{a / b}} : std::nullopt;
				default:
					return std::nullopt;
				}
			}
			auto toDouble = [](const Atomic& v) {
				return std::holds_alternative<int>(v) ? static_cast<double>(std::get<int>(v)) : std::get<double>(v);
			};
			double a = toDouble(*lhs), b = toDouble(*rhs);
			switch (inv->id) {
			case FIDs::TENSOR_ADD:
				return Atomic{a + b};
			case FIDs::TENSOR_SUB:
				return Atomic{a - b};
			case FIDs::TENSOR_PROD:
				return Atomic{a * b};
			case FIDs::TENSOR_DIV:
				return Atomic{a / b};
			default:
				return std::nullopt;
			}
		}

		/**
		 * @brief In the symbolic analysis of a module without hyperparameter values (see analyzeGeneric()): whether
		 * @p node is a value that depends on the unknown hyperparameters.
		 */
		bool isSymbolic(const Node& node) const {
			return symbolicMode && !std::holds_alternative<ErrorNode>(node) && !foldConstant(node);
		}

		/** @brief constantInt()'s result for a symbolic value without fallback. */
		static constexpr int unknownInt = std::numeric_limits<int>::min();

		/**
		 * @brief Evaluates a compile-time integer (e.g. a loop bound or an index); reports an error otherwise. In the
		 * symbolic analysis, a value that depends on the hyperparameters yields @p symbolic (unknownInt by default).
		 */
		std::optional<int> constantInt(mdlParser::ExprContext* expr, int symbolic = unknownInt) {
			const Node* node = asNode(expr->accept(this), expr->start);
			if (std::holds_alternative<ErrorNode>(*node))
				return std::nullopt;
			if (auto value = foldConstant(*node); value && std::holds_alternative<int>(*value))
				return std::get<int>(*value);
			if (isSymbolic(*node))
				return symbolic;
			errorlistener->semanticError(
					PrettyErrorListener::ErrorLevel::Error, expr->start, expr->start->getLine(),
					expr->start->getCharPositionInLine(), "Expected a compile-time integer"
			);
			return std::nullopt;
		}

		/** @brief A hyperparameter value for messages, e.g. `768`, `d` (symbolic), `{...}`, `[8, 16]`. */
		static std::string valueName(const Object& value) {
			if (const auto* node = std::get_if<const Node*>(&value)) {
				if (auto v = foldConstant(**node))
					return std::visit(
							overloaded{
									[](const std::vector<int>&) { return std::string{"[...]"}; },
									[](auto x) { return std::format("{}", x); },
							},
							*v
					);
				return shapes::Dim::of(**node).str();
			}
			if (const auto* array = std::get_if<ArrayInstance>(&value)) {
				std::string s = "[";
				for (size_t i = 0; i < array->elements.size(); ++i)
					s += (i > 0 ? ", " : "") + valueName(array->elements[i]);
				return s + "]";
			}
			return "{...}";
		}

		/** @brief A string identifying a (concrete) hyperparameter value; used to cache module instances. */
		static std::string valueKey(const Object& value) {
			return std::visit(
					overloaded{
							[](const Node* node) -> std::string {
								if (auto v = foldConstant(*node))
									return std::visit(
											overloaded{
													[](const std::vector<int>& xs) {
														std::string s = "(";
														for (int x : xs)
															s += std::to_string(x) + ",";
														return s + ")";
													},
													[](auto x) { return std::format("{}", x); },
											},
											*v
									);
								// A symbolic value (interned: the same expression is the same node).
								return std::format("#{}", static_cast<const void*>(node));
							},
							[](const AggregateInstance& agg) -> std::string {
								std::string key = "{";
								for (const auto& member : agg.getMembers())
									key += valueKey(member) + ",";
								return key + "}";
							},
							[](const ArrayInstance& array) -> std::string {
								std::string key = "[";
								for (const auto& element : array.elements)
									key += valueKey(element) + ",";
								return key + "]";
							},
							[](const auto&) -> std::string { return "?"; },
					},
					value
			);
		}

		/** @brief Flattens a (concrete) hyperparameter value into (path, number) pairs (see ModuleType). */
		static void flattenHyperparameter(
				const std::string& path, const Object& value,
				std::vector<std::pair<std::string, std::variant<int64_t, double>>>& out
		) {
			if (const auto* node = std::get_if<const Node*>(&value)) {
				if (auto v = foldConstant(**node)) {
					if (const auto* i = std::get_if<int>(&*v))
						out.emplace_back(path, int64_t{*i});
					else if (const auto* d = std::get_if<double>(&*v))
						out.emplace_back(path, *d);
				}
			} else if (const auto* agg = std::get_if<AggregateInstance>(&value)) {
				size_t i = 0;
				for (const auto& [name, _] : agg->getType().members)
					flattenHyperparameter(path + "." + name, agg->getMembers()[i++], out);
			} else if (const auto* array = std::get_if<ArrayInstance>(&value)) {
				for (size_t i = 0; i < array->elements.size(); ++i)
					flattenHyperparameter(path + "." + std::to_string(i), array->elements[i], out);
			}
		}

		/** @brief The compile-time value of a hyperparameter expression (numbers are folded to constants). */
		std::optional<Object> hyperparameterValue(mdlParser::ExprContext* expr) {
			Object value = asObject(expr->accept(this), expr->start);
			std::function<bool(Object&)> fold = [&](Object& v) -> bool {
				if (const auto* node = std::get_if<const Node*>(&v)) {
					if (std::holds_alternative<ErrorNode>(**node))
						return false;
					auto folded = foldConstant(**node);
					if (!folded && isSymbolic(**node))
						return true; // symbolic analysis: the instance gets the symbolic value
					if (!folded) {
						errorlistener->semanticError(
								PrettyErrorListener::ErrorLevel::Error, expr->start, expr->start->getLine(),
								expr->start->getCharPositionInLine(),
								"Module hyperparameters must be compile-time constants"
						);
						return false;
					}
					v = Object{&graph().addConstNode(Atomic{*folded})};
				} else if (auto* array = std::get_if<ArrayInstance>(&v)) {
					for (auto& element : array->elements)
						if (!fold(element))
							return false;
				}
				return true;
			};
			if (!fold(value))
				return std::nullopt;
			return value;
		}

		/**
		 * @brief The values of all hyperparameters of @p def (in declaration order): the positional ones, then the
		 * named ones (`.name=value`), then the defaults of the others (evaluated with the earlier hyperparameters in
		 * scope). Checks the arrays' sizes. Errors in default expressions are reported at them; the others are
		 * returned.
		 */
		/** @brief Why hyperparameters are invalid (see completeHyperparameters). */
		struct HyperparameterError {
			std::string message;
			std::string hint = {};
			std::optional<size_t> positional = std::nullopt; ///< the positional value at fault, if any
		};

		/** @brief A positional hyperparameter value and its source text (for hints). */
		struct PositionalValue {
			Object value;
			std::string text;
		};

		std::expected<std::vector<Object>, HyperparameterError> completeHyperparameters(
				const ModuleType& def, std::vector<PositionalValue> positional,
				std::vector<std::pair<std::string, Object>> named
		) {
			const auto& params = def.typeargs;
			const std::string module = def.templ ? def.templ->name : "module";
			auto error = [](std::string message, std::string hint = {}, std::optional<size_t> at = std::nullopt) {
				return std::unexpected(HyperparameterError{std::move(message), std::move(hint), at});
			};
			// Positional values fill the required hyperparameters (in declaration order); the ones with a default value
			// are only set by name (so that adding or reordering them does not change the meaning of instantiations).
			auto* ctx = def.templ ? def.templ->ctx : nullptr;
			auto hasDefault = [&](size_t i) {
				return ctx != nullptr && i < ctx->param().size() && ctx->param()[i]->default_ != nullptr;
			};
			std::vector<std::optional<Object>> values(params.size());
			size_t next = 0;
			for (size_t k = 0; k < positional.size(); ++k) {
				while (next < params.size() && hasDefault(next))
					++next;
				if (next == params.size()) {
					// The first optional hyperparameter that the value would have meant (in declaration order).
					for (size_t i = 0; i < params.size(); ++i)
						if (hasDefault(i) && !values[i]) {
							const bool named_ = std::any_of(named.begin(), named.end(), [&](const auto& n) {
								return n.first == params[i].first;
							});
							if (named_)
								continue;
							return error(
									std::format(
											"{} of {} has a default value and is only set by name", params[i].first,
											module
									),
									std::format("write `.{}={}`", params[i].first, positional[k].text), k
							);
						}
					return error(
							std::format(
									"{} has {} required hyperparameter(s) but got {} positional value(s)", module,
									std::count_if(
											params.begin(), params.end(),
											[&](const auto& p) {
												return !hasDefault(static_cast<size_t>(&p - params.data()));
											}
									),
									positional.size()
							),
							"", k
					);
				}
				values[next++] = std::move(positional[k].value);
			}
			for (auto& [name, value] : named) {
				auto it = std::find_if(params.begin(), params.end(), [&](const auto& p) { return p.first == name; });
				if (it == params.end())
					return error(std::format("{} has no hyperparameter {}", module, name));
				auto& slot = values[static_cast<size_t>(it - params.begin())];
				if (slot)
					return error(std::format("The hyperparameter {} of {} is given twice", name, module));
				slot = std::move(value);
			}
			// Defaults: evaluated in the module's defining scope with the earlier hyperparameters bound.
			std::vector<Object> result;
			for (size_t i = 0; i < params.size(); ++i) {
				if (!values[i]) {
					auto* param = ctx != nullptr && i < ctx->param().size() ? ctx->param()[i] : nullptr;
					if (param == nullptr || param->default_ == nullptr)
						return error(std::format("No value for the hyperparameter {} of {}", params[i].first, module));
					auto* outerErrors = std::exchange(errorlistener, def.templ->errors);
					nsstack.push(def.templ->scope->createSubnamespace());
					for (size_t k = 0; k < i; ++k)
						currentNamespace().put(params[k].first, Object{result[k]});
					values[i] = hyperparameterValue(param->default_);
					nsstack.pop();
					errorlistener = outerErrors;
					if (!values[i])
						return error(std::format(
								"The default of the hyperparameter {} of {} is invalid", params[i].first, module
						));
				}
				// Arrays: the declared size.
				const DataType& type = params[i].second.get().type;
				if (const auto* arrayType = std::get_if<ArrayType>(&type)) {
					const auto* array = std::get_if<ArrayInstance>(&*values[i]);
					if (array == nullptr || array->elements.size() != arrayType->size)
						return error(std::format(
								"The hyperparameter {} of {} is an array of {} values (array[...; {}]){}",
								params[i].first, module, arrayType->size, arrayType->size,
								array ? std::format(", but {} are given", array->elements.size()) : ""
						));
				}
				result.push_back(std::move(*values[i]));
			}
			return result;
		}

		/**
		 * @brief Expands the module @p def with the given hyperparameter values (in the order of its typeargs).
		 * @details Each distinct instantiation is expanded once. The members are elaborated in the namespace the
		 * module was defined in, with the hyperparameters bound to their values; weights are registered in a scope
		 * that identifies the instantiation.
		 */
		std::expected<std::reference_wrapper<const ModuleType>, std::string>
		instantiate(const ModuleType& def, const std::vector<Object>& values) {
			const auto& t = *def.templ;
			std::string key = t.name + "[";
			for (size_t i = 0; i < values.size(); ++i)
				key += (i > 0 ? "," : "") + valueKey(values[i]);
			key += "]";
			if (auto it = instances.find({&t, key}); it != instances.end())
				return std::cref(it->second);

			ModuleType m;
			m.typeargs = def.typeargs;
			m.templ = def.templ;
			m.instantiated = true;
			m.instanceName = t.name;
			if (!values.empty()) {
				m.instanceName += "[";
				for (size_t i = 0; i < values.size(); ++i) {
					std::string v = valueName(values[i]);
					if (i < def.typeargs.size() && (v == "0" || v == "1")) {
						const auto* p = std::get_if<PrimitiveType>(&def.typeargs[i].second.get().type);
						if (p != nullptr && *p == PrimitiveType::boolean)
							v = v == "1" ? "true" : "false";
					}
					m.instanceName += (i > 0 ? ", " : "") + v;
				}
				m.instanceName += "]";
			}
			for (size_t i = 0; i < values.size() && i < def.typeargs.size(); ++i)
				flattenHyperparameter(def.typeargs[i].first, values[i], m.hyperparameterValues);

			const size_t errorsBefore = t.errors->getNumErrors();
			const size_t encounteredBefore = t.errors->getNumEncountered();
			const std::string outerScope = std::exchange(scope, key);
			auto* outerErrors = std::exchange(errorlistener, t.errors);
			nsstack.push(t.scope->createSubnamespace());
			const std::string outerModule = std::exchange(currentModule, t.name);
			for (size_t i = 0; i < values.size() && i < def.typeargs.size(); ++i) {
				currentNamespace().put(def.typeargs[i].first, Object{values[i]});
				if (i < t.ctx->param().size())
					defineHere(
							t.ctx->param()[i]->IDENT()->getSymbol(), analysis::SymbolKind::hyperparameter,
							sourceText(t.ctx->param()[i]), nullptr, rangeOf(t.ctx), {objectType(values[i])}, t.name
					);
			}

			for (auto* var : t.ctx->vardecl())
				declareMember(m, var);
			for (auto& func : t.ctx->funcdef()) {
				if (func->IDENT())
					defineHere(
							func->IDENT()->getSymbol(), analysis::SymbolKind::method, signature(func), func,
							rangeOf(t.ctx), {}, t.name
					);
				auto [name, decl] = std::any_cast<std::pair<std::string, FunctionDecl>>(func->accept(this));
				if (auto it = m.functions.find(name); it != m.functions.end())
					it->second.overloads.push_back(std::move(decl)); // (duplicates were reported by registerFunction)
				else
					m.functions.emplace(std::move(name), std::move(decl));
			}

			// The member functions can be specialized later (e.g. for masked tensors): keep the instance's namespace.
			Namespace& persistent = instanceScopes.emplace_back(nsstack.top());
			for (auto& [_, f] : m.functions)
				for (FunctionDecl* d : [&] {
						 std::vector<FunctionDecl*> all{&f};
						 for (auto& o : f.overloads)
							 all.push_back(&o);
						 return all;
					 }())
					if (d->templ != nullptr && d->templ->scope == &nsstack.top())
						d->templ->scope = &persistent;
			nsstack.pop();
			errorlistener = outerErrors;
			scope = outerScope;
			currentModule = outerModule;
			numErrors += t.errors->getNumErrors() - errorsBefore;
			if (auto all = t.errors->errorsSince(encounteredBefore); !all.empty()) {
				// Errors that only this instance has (e.g. a member that @conditional removes) are reported again
				// where it is instantiated (also if they were reported before, by another use of the instance).
				std::vector<mdl::Diagnostic> errors(all.begin(), all.end());
				const auto& generic = genericErrors(def);
				std::erase_if(errors, [&](const mdl::Diagnostic& e) {
					return generic.contains({0, 0, ""}) ||
						   generic.contains({e.range.begin.line, e.range.begin.column, e.message});
				});
				instanceError.clear();
				if (!errors.empty()) {
					const auto& first = errors.front();
					instanceError = std::format(
							"{} has an error: {} ({}:{}:{}){}", m.instanceName, first.message,
							std::filesystem::path(first.file).filename().string(), first.range.begin.line,
							first.range.begin.column + 1,
							errors.size() > 1 ? std::format(" and {} more", errors.size() - 1) : ""
					);
				}
				return std::unexpected(std::format("{} error(s) while instantiating {}", all.size(), key));
			}
			if (index() != nullptr)
				recordLayout(t, m, key);
			return std::cref(instances.emplace(std::pair{&t, key}, std::move(m)).first->second);
		}

		/**
		 * @brief The errors of the module @p def for all values of its hyperparameters: found by checking it with
		 * symbolic values (once, without reporting). All errors of a module without hyperparameters are generic.
		 */
		const std::set<std::tuple<size_t, size_t, std::string>>& genericErrors(const ModuleType& def) {
			const ModuleTemplate* t = def.templ.get();
			if (auto it = genericErrorsOf.find(t); it != genericErrorsOf.end())
				return it->second;
			auto& result = genericErrorsOf[t]; // (empty while it is computed: recursive instantiations)
			std::vector<Object> values;
			for (const auto& [pname, hn] : def.typeargs)
				if (auto value = symbolicValue(hn.get().type, pname))
					values.push_back(std::move(*value));
			if (def.typeargs.empty() || values.size() != def.typeargs.size()) {
				result.emplace(0, 0, ""); // marks "all generic" (see below)
				return result;
			}
			const size_t before = t->errors->getNumEncountered();
			const bool wasMuted = t->errors->setMuted(true);
			const bool outer = std::exchange(symbolicMode, true);
			(void)instantiate(def, values);
			symbolicMode = outer;
			t->errors->setMuted(wasMuted);
			for (const auto& e : t->errors->errorsSince(before))
				result.emplace(e.range.begin.line, e.range.begin.column, e.message);
			return result;
		}

		/** @brief Records the weights of the instance @p m of the module @p t in the index (parameter counts). */
		void recordLayout(const ModuleTemplate& t, const ModuleType& m, const std::string& key) {
			auto module = index()->find(t.errors->getFile(), rangeOf(t.ctx->IDENT()->getSymbol()).begin);
			if (!module)
				return;
			const std::string& instance = m.instanceName;
			analysis::ModuleLayout layout{.module = *module, .instance = instance, .weights = {}};
			for (const auto& [path, weight] : m.variables) {
				const auto& type = weight.get().type;
				analysis::ModuleLayout::Weight w{
						.path = path, .type = typeString(type), .count = std::nullopt, .formula = {}
				};
				if (const auto* bound = std::get_if<BoundModuleType>(&type)) {
					w.submodule = true;
					w.instance = bound->definition.instanceName;
					if (const auto& sub = bound->definition.templ)
						w.module = index()->find(sub->errors->getFile(), rangeOf(sub->ctx->IDENT()->getSymbol()).begin);
				} else if (const auto* tensor = std::get_if<TensorType>(&type)) {
					int64_t count = 1;
					bool constant = true;
					for (const auto& dim : tensor->shape) {
						std::string factor;
						if (const auto* u = std::get_if<unsigned>(&dim)) {
							count *= *u;
							factor = std::to_string(*u);
						} else if (const auto* n = std::get_if<const Node*>(&dim)) {
							auto v = foldConstant(**n);
							if (v && std::holds_alternative<int>(*v)) {
								count *= std::get<int>(*v);
								factor = std::to_string(std::get<int>(*v));
							} else {
								constant = false;
								factor = shapes::Dim::of(**n).str();
								if (factor.find(' ') != std::string::npos)
									factor = "(" + factor + ")";
							}
						} else {
							constant = false;
							factor = "?";
						}
						w.formula += (w.formula.empty() ? "" : " * ") + factor;
					}
					if (constant)
						w.count = count;
				} else {
					w.count = 1;
					w.formula = "1";
				}
				layout.weights.push_back(std::move(w));
			}
			index()->addLayout(std::move(layout));
		}

		/** @brief A member that an annotation left out of an instance (e.g. `@conditional(bias)` with bias false). */
		struct OmittedMember {
			std::string reason; ///< e.g. "@conditional(bias) is false"
		};

		/**
		 * @brief The handler of a member annotation `@name(args...)`: whether the member exists (false: left out, with
		 * the reason). Handlers report their own errors (e.g. wrong arguments). Further annotations are registered in
		 * memberAnnotations().
		 */
		using MemberAnnotation =
				std::function<std::optional<std::string>(TranslatingVisitor&, mdlParser::AnnotationContext*)>;

		static const std::map<std::string, MemberAnnotation>& memberAnnotations() {
			static const std::map<std::string, MemberAnnotation> handlers{
					// @conditional(c): the member exists only if the compile-time condition c holds.
					{"conditional",
					 [](TranslatingVisitor& v, mdlParser::AnnotationContext* a) -> std::optional<std::string> {
						 auto* tok = a->name;
						 if (a->args.size() != 1) {
							 v.errorlistener->semanticError(
									 PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(),
									 tok->getCharPositionInLine(),
									 "@conditional takes one argument: the condition (e.g. `@conditional(bias)`)"
							 );
							 return std::nullopt;
						 }
						 const Node* cond = v.asNode(a->args[0]->accept(&v), a->args[0]->start);
						 auto value = foldConstant(*cond);
						 const auto* i = value ? std::get_if<int>(&*value) : nullptr;
						 if (i == nullptr && v.isSymbolic(*cond)) {
							 // Symbolic analysis: the member may exist; its uses are checked (see checkConditionalMember).
							 if (auto* var = dynamic_cast<mdlParser::VardeclContext*>(a->parent))
								 v.conditionalMembers[{v.currentModule, var->IDENT()->getText()}] = {
										 a->args[0]->getText(), &v.currentNamespace()
								 };
							 return std::nullopt;
						 }
						 if (i == nullptr) {
							 if (!std::holds_alternative<ErrorNode>(*cond))
								 v.errorlistener->semanticError(
										 PrettyErrorListener::ErrorLevel::Error, a->args[0]->start,
										 a->args[0]->start->getLine(), a->args[0]->start->getCharPositionInLine(),
										 "The condition of @conditional must be a compile-time value (e.g. a "
										 "hyperparameter)"
								 );
							 return std::nullopt;
						 }
						 if (*i != 0)
							 return std::nullopt;
						 return std::format("@conditional({}) is false", a->args[0]->getText());
					 }},
			};
			return handlers;
		}

		/**
		 * @brief Applies the annotations of a member; std::nullopt if the member exists, else the reason why it is left
		 * out. Unknown annotations are reported as configured (ParseOptions::unknownAnnotation).
		 */
		std::optional<std::string> applyAnnotations(mdlParser::VardeclContext* var) {
			std::optional<std::string> omitted;
			for (auto* a : var->annotation()) {
				const std::string name = a->name->getText();
				auto it = memberAnnotations().find(name);
				if (it == memberAnnotations().end()) {
					reportUnknownAnnotation(a);
					continue;
				}
				if (auto reason = it->second(*this, a); reason && !omitted)
					omitted = reason;
			}
			return omitted;
		}

		/** @brief A fix that replaces @p range by the code of a hint ``write `code` `` (none for other hints). */
		static std::vector<Diagnostic::Fix> hintFix(const std::string& hint, SourceRange range) {
			const auto open = hint.find('`'), close = hint.rfind('`');
			if (!hint.starts_with("write `") || close == open)
				return {};
			const std::string code = hint.substr(open + 1, close - open - 1);
			return {{.title = "Write `" + code + "`", .edits = {{range, code}}}};
		}

		/** @brief The source line of @p tok with the token marked (for diagnostics). */
		std::vector<PrettyErrorListener::ContextLine> tokenContext(antlr4::Token* tok) const {
			return {
					{.number = tok->getLine(),
					 .text = PrettyErrorListener::getline(*tok->getInputStream(), 0, tok->getStartIndex()),
					 .markBegin = tok->getCharPositionInLine(),
					 .markWidth = tok->getText().size()}
			};
		}

		/**
		 * @brief Reports a possible mismatch (`-Wpermits-illegal-instantiation`, an error with `--strict`): sizes that are different
		 * compile-time expressions (e.g. the hyperparameters `input` and `output`), which may differ in an instance.
		 */
		void reportPossibleMismatch(antlr4::Token* tok, const std::string& message) {
			if (options.permitsIllegalInstantiation == parser::DiagnosticLevel::ignored)
				return;
			const bool error = options.permitsIllegalInstantiation == parser::DiagnosticLevel::error;
			errorlistener->semanticError(
					error ? PrettyErrorListener::ErrorLevel::Error : PrettyErrorListener::ErrorLevel::Warning,
					tokenContext(tok), tok->getLine(), tok->getCharPositionInLine(), message, "",
					"permits-illegal-instantiation"
			);
		}

		/**
		 * @brief Reports a use of a member with `@conditional(c)` (symbolic `c`) where `c` is not known to hold
		 * (`-Wpermits-illegal-instantiation`): the instances where `c` is false have an error there.
		 */
		void checkConditionalMember(antlr4::Token* tok) {
			auto it = conditionalMembers.find({currentModule, tok->getText()});
			if (it == conditionalMembers.end() ||
				options.permitsIllegalInstantiation == parser::DiagnosticLevel::ignored ||
				std::ranges::contains(assumedConditions, it->second.condition) ||
				currentNamespace().definedIn(tok->getText()) != it->second.scope)
				return;
			const std::string& condition = it->second.condition;
			const bool error = options.permitsIllegalInstantiation == parser::DiagnosticLevel::error;
			errorlistener->semanticError(
					error ? PrettyErrorListener::ErrorLevel::Error : PrettyErrorListener::ErrorLevel::Warning,
					tokenContext(tok), tok->getLine(), tok->getCharPositionInLine(),
					std::format(
							"{} does not exist in instances where {} is false (@conditional({}))", tok->getText(),
							condition, condition
					),
					std::format("use it in `{} ? ... : ...`", condition), "permits-illegal-instantiation"
			);
		}

		void reportUnknownAnnotation(mdlParser::AnnotationContext* a) {
			if (options.unknownAnnotation == parser::DiagnosticLevel::ignored)
				return;
			const bool error = options.unknownAnnotation == parser::DiagnosticLevel::error;
			auto* tok = a->name;
			errorlistener->semanticError(
					error ? PrettyErrorListener::ErrorLevel::Error : PrettyErrorListener::ErrorLevel::Warning,
					tokenContext(tok), tok->getLine(), tok->getCharPositionInLine(),
					std::format("Unknown annotation @{}", a->name->getText()), "", "unknown-annotation"
			);
		}

		/** @brief Declares a member variable (weight, submodule, or array of them) of a module being expanded. */
		void declareMember(ModuleType& m, mdlParser::VardeclContext* var) {
			const auto vname = var->IDENT()->getText();
			auto* moduleCtx = m.templ ? m.templ->ctx : nullptr;
			// The definition of the member in the index (with the type of its value).
			auto record = [&](std::set<std::string> types) {
				defineHere(
						var->IDENT()->getSymbol(), analysis::SymbolKind::member, sourceText(var), var,
						rangeOf(moduleCtx), std::move(types), currentModule
				);
			};
			if (auto reason = applyAnnotations(var)) {
				currentNamespace().put(vname, OmittedMember{*reason});
				record({"left out: " + *reason});
				return;
			}
			auto addVariable = [&](const std::string& path, const DataType& dtype) -> const Node& {
				// Weights are registered in the instance's scope so that all functions of the module share them.
				const Node& node = graph().getOrAddParameterNode(scope, path, dtype);
				m.variables.emplace_back(path, std::cref(std::get<ParameterNode>(node)));
				return node;
			};

			if (auto* comp = dynamic_cast<mdlParser::ComprehensionTypeContext*>(var->type())) {
				// `name: [T for i in lo..hi]` declares the members name.lo, ..., name.(hi-1).
				auto lo = constantInt(comp->lo, 0), hi = constantInt(comp->hi);
				ArrayInstance array;
				if (lo && hi == unknownInt) { // symbolic analysis: one instance stands for all
					hi = *lo + 1;
					array.representative = true;
				}
				for (int i = lo.value_or(0); lo && hi && i < *hi; ++i) {
					pushNamespace();
					if (!isDiscard(comp->var)) {
						currentNamespace().put(comp->var->getText(), Object{&graph().addConstNode(Atomic{i})});
						defineHere(
								comp->var, analysis::SymbolKind::loopVariable,
								comp->var->getText() + " in " + sourceText(comp->lo) + ".." + sourceText(comp->hi),
								nullptr, rangeOf(comp), {"int"}
						);
					}
					auto dtype = std::any_cast<DataType>(comp->element->accept(this));
					popNamespace();
					array.elements.push_back(Object{&addVariable(vname + "." + std::to_string(i), dtype)});
				}
				currentNamespace().put(vname, Object{std::move(array)});
				if (auto value = currentNamespace().get<Object>(vname))
					record({objectType(value->get())});
				return;
			}

			const auto [_, dtype] = std::any_cast<VarDecl>(var->accept(this));
			const Node& node = addVariable(vname, dtype);
			if (auto* agg = std::get_if<AggregateType>(&dtype))
				currentNamespace()
						.put(vname, makeAggregateObject(*agg, vname, [&](DataType t, std::string path) -> const Node& {
								 return graph().getOrAddParameterNode(scope, path, std::move(t));
							 }));
			else
				currentNamespace().put(vname, Object{&node});
			record({typeString(dtype)});
		}

		virtual std::any visitParam(mdlParser::ParamContext* ctx) override {
			// (Default values are evaluated when a module is instantiated, see completeHyperparameters.)
			return std::make_pair(ctx->IDENT()->getText(), std::any_cast<mdl::DataType>(ctx->type()->accept(this)));
		}

		/** @brief The definition of the type that the last qualified type (`nn.Linear`) named. */
		std::optional<size_t> qualifiedTypeSymbol;

		virtual std::any visitTypeSpec(mdlParser::TypeSpecContext* ctx) override {
			qualifiedTypeSymbol.reset();
			auto dtype = std::any_cast<DataType>(unqualifiedTypeSpec(ctx));
			if (index() != nullptr && ctx->typename_ != nullptr)
				index()->reference(analysis::Reference{
						.file = currentFile(),
						.range = rangeOf(ctx->typename_),
						.symbol = ctx->qualifier.empty() ? currentNamespace().symbol(ctx->typename_->getText())
														 : qualifiedTypeSymbol,
						.types = {typeString(dtype)},
						.members = {},
				});
			if (ctx->masked != nullptr) {
				if (auto* tensor = std::get_if<TensorType>(&dtype)) {
					tensor->masked = true;
				} else {
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error, ctx->masked, ctx->masked->getLine(),
							ctx->masked->getCharPositionInLine(), "Only tensors can be masked"
					);
				}
			}
			return dtype;
		}

		std::any unqualifiedTypeSpec(mdlParser::TypeSpecContext* ctx) {
			// `float` is the configured default float type (see ParseOptions::defaultFloat).
			const DataType defaultFloat{options.defaultFloat};
			std::optional<std::reference_wrapper<const DataType>> dtype;
			if (!ctx->qualifier.empty()) {
				// `nn.Linear[...]`: the type of an imported namespace.
				std::optional<NamespaceRef> ns;
				const auto* first = currentNamespace().find<NamespaceRef>(ctx->qualifier.front()->getText());
				if (first == nullptr) {
					importError(
							ctx->qualifier.front(),
							std::format("{} is not a namespace", ctx->qualifier.front()->getText())
					);
					return mdl::DataType{mdl::PrimitiveType::error};
				}
				ns = *first;
				indexNamespace(ctx->qualifier.front(), *ns);
				std::string error;
				for (size_t i = 1; i < ctx->qualifier.size() && ns; ++i) {
					Member m = memberOf(*ns, ctx->qualifier[i]->getText());
					ns = m.ns;
					error = m.error;
					if (ns)
						indexNamespace(ctx->qualifier[i], *ns);
				}
				Member m = ns ? memberOf(*ns, ctx->typename_->getText()) : Member{.error = error};
				if (!m.name || m.name->type == nullptr) {
					importError(
							ctx->typename_, !m.error.empty()
													? m.error
													: std::format("{} does not name a type", ctx->typename_->getText())
					);
					return mdl::DataType{mdl::PrimitiveType::error};
				}
				dtype = std::cref(*m.name->type);
				qualifiedTypeSymbol = m.name->symbol;
			} else
				dtype = ctx->typename_->getText() == "float"
								? std::optional<std::reference_wrapper<const DataType>>{std::cref(defaultFloat)}
								: currentNamespace().getType(ctx->typename_->getText());
			if (!dtype.has_value()) {
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, ctx->typename_, ctx->typename_->getLine(),
						ctx->typename_->getCharPositionInLine(),
						std::format("{} does not name a type", ctx->typename_->getText())
				);
				return mdl::DataType{mdl::PrimitiveType::error};
			} else {
				if (const auto* mod = std::get_if<ModuleType>(&dtype.value().get()); mod && mod->templ) {
					// Module instantiation: the arguments must be compile-time values (positional, then named).
					auto* tok = ctx->typename_;
					std::vector<PositionalValue> positional;
					std::vector<mdlParser::Type_argContext*> positionalArgs;
					std::vector<std::pair<std::string, Object>> named;
					for (auto* arg : ctx->type_arg()) {
						if (arg->expr() == nullptr) {
							errorlistener->semanticError(
									PrettyErrorListener::ErrorLevel::Error, arg->start, arg->start->getLine(),
									arg->start->getCharPositionInLine(), "`...` is not a module hyperparameter"
							);
							return DataType{PrimitiveType::error};
						}
						auto value = hyperparameterValue(arg->expr());
						if (!value)
							return DataType{PrimitiveType::error};
						if (arg->name != nullptr) {
							named.emplace_back(arg->name->getText(), std::move(*value));
						} else if (!named.empty()) {
							errorlistener->semanticError(
									PrettyErrorListener::ErrorLevel::Error, arg->start, arg->start->getLine(),
									arg->start->getCharPositionInLine(),
									"Positional hyperparameters must come before named ones (`.name=value`)"
							);
							return DataType{PrimitiveType::error};
						} else {
							positional.push_back({std::move(*value), arg->expr()->getText()});
							positionalArgs.push_back(arg);
						}
					}
					auto values = completeHyperparameters(*mod, std::move(positional), std::move(named));
					if (!values) {
						const auto& e = values.error();
						if (e.positional) { // marks the value at fault
							auto* arg = positionalArgs[*e.positional];
							errorlistener->semanticError(
									PrettyErrorListener::ErrorLevel::Error,
									{{.number = arg->start->getLine(),
									  .text = errorlistener->getline(
											  *arg->start->getInputStream(), 0, arg->start->getStartIndex()
									  ),
									  .markBegin = arg->start->getCharPositionInLine(),
									  .markWidth = arg->stop->getStopIndex() - arg->start->getStartIndex() + 1}},
									arg->start->getLine(), arg->start->getCharPositionInLine(), e.message, e.hint, "",
									hintFix(e.hint, rangeOf(arg))
							);
						} else {
							errorlistener->semanticError(
									PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(),
									tok->getCharPositionInLine(), e.message, e.hint
							);
						}
						return DataType{PrimitiveType::error};
					}
					auto instance = instantiate(*mod, *values);
					if (!instance && !instanceError.empty()) {
						errorlistener->semanticError(
								PrettyErrorListener::ErrorLevel::Error,
								{{.number = ctx->start->getLine(),
								  .text = errorlistener->getline(
										  *ctx->start->getInputStream(), 0, ctx->start->getStartIndex()
								  ),
								  .markBegin = ctx->start->getCharPositionInLine(),
								  .markWidth = ctx->stop->getStopIndex() - ctx->start->getStartIndex() + 1}},
								ctx->start->getLine(), ctx->start->getCharPositionInLine(), instanceError
						);
					}
					if (!instance)
						return DataType{PrimitiveType::error};
					return DataType{BoundModuleType{.definition = instance.value().get(), .hyperparamValues = {}}};
				}
				if (ctx->type_arg().size() != 0) {
					std::vector<TypeArg> args;
					for (auto arg : ctx->type_arg()) {
						args.emplace_back(std::any_cast<TypeArg>(arg->accept(this)));
					}
					auto ta = setTypeArgs(dtype.value().get(), args);
					if (!ta.has_value()) {
						auto firsttok = ctx->type_arg().empty() ? ctx->typename_ : ctx->type_arg().front()->start;
						auto lasttok = ctx->type_arg().empty() ? ctx->stop : ctx->type_arg().back()->stop;
						errorlistener->semanticError(
								PrettyErrorListener::ErrorLevel::Error,
								{{.number = ctx->typename_->getLine(),
								  .text = errorlistener->getline(
										  *firsttok->getInputStream(), 0, firsttok->getStartIndex()
								  ),
								  .markBegin = firsttok->getCharPositionInLine(),
								  .markWidth = lasttok->getStopIndex() - firsttok->getStartIndex() + 1}},
								ctx->typename_->getLine(), firsttok->getCharPositionInLine(), ta.error()
						);
						return dtype.value().get(); /** Or return some "error" datatype here? */
					}
					return ta.value();
				}
				return dtype.value().get();
			}
		}

		virtual std::any visitAnonType(mdlParser::AnonTypeContext* ctx) override {
			auto type = AggregateType{};
			updateTypeWithVarDecl(type, "anonymous type", ctx->vardecl());
			return DataType{type};
		}

		virtual std::any visitType_arg(mdlParser::Type_argContext* ctx) override {
			if (ctx->elipsis != nullptr) {
				return TypeArg{mdl::Elipsis{}};
			} else if (ctx->expr() != nullptr) {
				const auto& node = *asNode(ctx->expr()->accept(this), ctx->expr()->start);
				if (!mdl::isCompiletimeEvalable(node)) {
					auto firsttok = ctx->expr()->start;
					auto lasttok = ctx->expr()->stop;
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error,
							{{.number = firsttok->getLine(),
							  .text = errorlistener->getline(*firsttok->getInputStream(), 0, firsttok->getStartIndex()),
							  .markBegin = firsttok->getCharPositionInLine(),
							  .markWidth = lasttok->getStopIndex() - firsttok->getStartIndex() + 1}},
							firsttok->getLine(), firsttok->getCharPositionInLine(),
							"Parameters must be compile time constant"
					);
					return TypeArg{&node}; /** return this node anyway for now **/
				} else {
					return TypeArg{&node};
				}
			} else {
				assert(false); // this case should not exist
			}
			std::unreachable();
		}

		virtual std::any visitConstant(mdlParser::ConstantContext* ctx) override {
			if (ctx->INTEGER() != nullptr)
				return mdl::Atomic{std::atoi(ctx->INTEGER()->getText().c_str())};
			if (ctx->FLOAT() != nullptr)
				return mdl::Atomic{std::atof(ctx->FLOAT()->getText().c_str())};
			if (ctx->BOOL() != nullptr)
				return mdl::Atomic(ctx->BOOL()->getText() == "true");
			assert(false); // Invalid case
			std::unreachable();
		}

		virtual std::any visitVardecl(mdlParser::VardeclContext* ctx) override {
			return std::make_pair(ctx->IDENT()->getText(), std::any_cast<mdl::DataType>(ctx->type()->accept(this)));
		}

		virtual std::any visitFuncdef(mdlParser::FuncdefContext* ctx) override {
			const auto specialization = std::exchange(maskedParams, std::nullopt);
			const auto boundTemplate = std::exchange(templateValues, {});
			std::vector<std::reference_wrapper<const Node>> outputs;
			std::vector<std::pair<std::string, std::reference_wrapper<const HyperparameterNode>>> tparams;
			std::vector<std::pair<std::string, std::reference_wrapper<const InputNode>>> params;
			pushNamespace();
			if (ctx->templatespec() != nullptr) {
				for (auto decl : ctx->templatespec()->vardecl()) {
					rejectAnnotations(decl, "template parameters");
					auto [name, dtype] = std::any_cast<VarDecl>(decl->accept(this));
					if (auto value = boundTemplate.find(name); value != boundTemplate.end()) {
						// Specialized for a call: the template parameter is a constant in the body.
						currentNamespace().put(name, Object{&graph().addConstNode(Atomic{value->second})});
						defineHere(
								decl->IDENT()->getSymbol(), analysis::SymbolKind::templateParameter, sourceText(decl),
								nullptr, rangeOf(ctx), {std::format("{} = {}", typeString(dtype), value->second)}
						);
						documentParameter(decl->IDENT()->getSymbol(), ctx);
						continue;
					}
					defineHere(
							decl->IDENT()->getSymbol(), analysis::SymbolKind::templateParameter, sourceText(decl),
							nullptr, rangeOf(ctx), {typeString(dtype)}
					);
					documentParameter(decl->IDENT()->getSymbol(), ctx);
					auto& p = currentNamespace().put(name, Object{&graph().addHyperparameterNode(dtype, name)});
					tparams.emplace_back(name, std::get<mdl::HyperparameterNode>(*std::get<const mdl::Node*>(p)));
				}
			}
			for (size_t i = 0; i < ctx->param().size(); ++i) {
				if (auto* d = ctx->param()[i]->default_) {
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error, d->start, d->start->getLine(),
							d->start->getCharPositionInLine(),
							"Default values are only supported for module hyperparameters"
					);
				}
				auto [name, dtype] = std::any_cast<VarDecl>(ctx->param()[i]->accept(this));
				if (auto* tensor = std::get_if<TensorType>(&dtype);
					tensor && specialization && i < specialization->size())
					tensor->masked |= (*specialization)[i];
				auto& p = currentNamespace().put(name, Object{&graph().addInputNode(dtype, name)});
				defineHere(
						ctx->param()[i]->IDENT()->getSymbol(), analysis::SymbolKind::parameter,
						sourceText(ctx->param()[i]), nullptr, rangeOf(ctx), {typeString(dtype)}
				);
				documentParameter(ctx->param()[i]->IDENT()->getSymbol(), ctx);
				params.emplace_back(name, std::get<mdl::InputNode>(*std::get<const mdl::Node*>(p)));
			}
			std::vector<std::variant<PrimitiveType, TensorType>> returnTypes;
			for (auto* type : ctx->rettype()->type()) {
				auto dtype = std::any_cast<DataType>(type->accept(this));
				if (auto* prim = std::get_if<PrimitiveType>(&dtype)) {
					returnTypes.emplace_back(*prim);
				} else if (auto* tensor = std::get_if<TensorType>(&dtype)) {
					returnTypes.emplace_back(*tensor);
				} else {
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error, type->start, type->start->getLine(),
							type->start->getCharPositionInLine(), "Functions can only return tensors or scalars"
					);
					returnTypes.emplace_back(PrimitiveType::error);
				}
			}
			if (ctx->expr() != nullptr) {
				outputs.push_back(std::cref(*asNode(ctx->expr()->accept(this), ctx->expr()->start)));
			} else {
				for (auto stmt : ctx->statement()) {
					auto result = stmt->accept(this);
					// Return statements yield the returned nodes; all other statements yield std::nullopt.
					if (auto* returned = std::any_cast<MultiValue>(&result)) {
						if (!outputs.empty()) {
							errorlistener->semanticError(
									PrettyErrorListener::ErrorLevel::Error, stmt->start, stmt->start->getLine(),
									stmt->start->getCharPositionInLine(),
									std::format("Function {} has multiple return statements", ctx->IDENT()->getText())
							);
							continue;
						}
						for (const Node* node : returned->nodes)
							outputs.push_back(std::cref(*node));
					}
				}
				if (outputs.empty()) {
					auto* tok = ctx->IDENT()->getSymbol();
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
							std::format("Function {} has no return statement", ctx->IDENT()->getText())
					);
					outputs.push_back(std::cref(graph().addErrorNode()));
				}
			}
			popNamespace();
			FunctionDecl decl{
					.name = ctx->IDENT()->getText(),
					.templateParameters = std::move(tparams),
					.parameters = std::move(params),
					.outputs = std::move(outputs),
					.returnTypes = std::move(returnTypes),
			};
			decl.visibility = visibilityOf(ctx->visibility());
			if (decl.returnTypes.size() != decl.outputs.size()) {
				auto* tok = ctx->IDENT()->getSymbol();
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
						std::format(
								"{} declares {} return value(s) but returns {}", decl.name, decl.returnTypes.size(),
								decl.outputs.size()
						)
				);
			}
			// The results must have the declared types (results of another float type are converted, if allowed).
			for (size_t i = 0; i < std::min(decl.returnTypes.size(), decl.outputs.size()); ++i) {
				const DataType declared = std::visit([](const auto& t) -> DataType { return t; }, decl.returnTypes[i]);
				if (const auto* t = std::get_if<TensorType>(&declared); t != nullptr && isFloat(t->type)) {
					const Node* converted = implicitConversion(
							decl.outputs[i].get(), t->type, ctx->IDENT()->getSymbol(),
							std::format("Result {} of {}", i + 1, decl.name)
					);
					if (converted == nullptr)
						continue; // reported
					decl.outputs[i] = std::cref(*converted);
				}
				if (!shapes::matches(declared, getDataType(decl.outputs[i].get()))) {
					auto* tok = ctx->IDENT()->getSymbol();
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
							std::format(
									"{} returns {}, but its declared return type is {}", decl.name,
									shapes::str(getDataType(decl.outputs[i].get())), shapes::str(declared)
							)
					);
				} else if (!shapes::matches(declared, getDataType(decl.outputs[i].get()), true, /*strict=*/true)) {
					reportPossibleMismatch(
							ctx->IDENT()->getSymbol(),
							std::format(
									"{} returns {}, but its declared return type is {} (the sizes may differ)",
									decl.name, shapes::str(getDataType(decl.outputs[i].get())), shapes::str(declared)
							)
					);
				}
			}
			if (specialization)
				return std::make_pair(ctx->IDENT()->getText(), std::move(decl)); // the caller sets the template
			decl.templ = std::make_shared<FunctionTemplate>(FunctionTemplate{
					.ctx = ctx, .scope = &currentNamespace(), .weightScope = scope, .errors = errorlistener
			});
			registerFunction(ctx->IDENT(), decl);
			return std::make_pair(ctx->IDENT()->getText(), std::move(decl));
		}

		/** @brief Which parameters of @p decl take masked tensors (the overload's signature). */
		static std::vector<bool> maskedSignature(const FunctionDecl& decl) {
			std::vector<bool> signature;
			for (const auto& [_, p] : decl.parameters) {
				const auto* t = std::get_if<TensorType>(&p.get().type);
				signature.push_back(t != nullptr && t->masked);
			}
			return signature;
		}

		/** @brief Defines the function @p decl in the current namespace, as an overload if the name is a function. */
		void registerFunction(antlr4::tree::TerminalNode* ident, const FunctionDecl& decl) {
			const std::string name = ident->getText();
			if (decl.templ != nullptr)
				define(ident->getSymbol(), analysis::SymbolKind::function, signature(decl.templ->ctx), decl.templ->ctx,
					   fileScope());
			auto bind = [&] {
				if (index() != nullptr)
					if (auto id = index()->find(currentFile(), rangeOf(ident->getSymbol()).begin))
						currentNamespace().setSymbol(name, *id);
			};
			if (!currentNamespace().definesLocally(name)) {
				reportImportShadowing(ident->getSymbol());
				currentNamespace().put<mdl::Object>(name, Object{decl});
				bind();
				return;
			}
			auto existing = currentNamespace().get<Object>(name);
			const auto* function = existing ? std::get_if<FunctionDecl>(&existing->get()) : nullptr;
			if (function == nullptr) {
				checkRedefinition(ident);
				return;
			}
			auto sameSignature = [&](const FunctionDecl& other) {
				return other.parameters.size() == decl.parameters.size() &&
					   maskedSignature(other) == maskedSignature(decl);
			};
			bool duplicate = sameSignature(*function);
			for (const auto& o : function->overloads)
				duplicate |= sameSignature(o);
			if (duplicate) {
				checkRedefinition(ident);
				return;
			}
			FunctionDecl merged = *function;
			merged.overloads.push_back(decl);
			currentNamespace().assign(name, Object{std::move(merged)});
			bind();
		}

		/**
		 * @brief The function to call for arguments of the given types: the best matching overload of @p decl,
		 * specialized for the argument types if they differ from its parameter types (a masked tensor passed to a plain
		 * parameter). Without masked arguments, this is @p decl itself (or its matching overload).
		 * @return nullptr if no overload takes these arguments (e.g. a plain tensor for a masked parameter).
		 */
		const FunctionDecl* resolveCall(const FunctionDecl& decl, const std::vector<DataType>& argtypes) {
			auto masked = [](const DataType& t) {
				const auto* tensor = std::get_if<TensorType>(&t);
				return tensor != nullptr && tensor->masked;
			};
			const FunctionDecl* best = nullptr;
			int bestScore = -1;
			std::vector<const FunctionDecl*> candidates{&decl};
			for (const auto& o : decl.overloads)
				candidates.push_back(&o);
			for (const auto* c : candidates) {
				if (c->parameters.size() != argtypes.size())
					continue;
				int score = 0;
				bool viable = true;
				for (size_t i = 0; i < argtypes.size(); ++i) {
					const bool param = masked(c->parameters[i].second.get().type), arg = masked(argtypes[i]);
					if (param && !arg)
						viable = false; // a masked parameter needs a masked tensor
					score += (param == arg) ? 2 : 1;
				}
				if (viable && score > bestScore) {
					best = c;
					bestScore = score;
				}
			}
			if (best != nullptr) {
				auto values = inferTemplateValues(*best, argtypes);
				if (!values.empty() && best->templ != nullptr) {
					std::vector<bool> pattern;
					for (size_t i = 0; i < best->parameters.size(); ++i)
						pattern.push_back(masked(best->parameters[i].second.get().type) || masked(argtypes[i]));
					return &specialize(*best, pattern, values);
				}
			}
			if (best == nullptr) {
				// Let the arity check report wrong argument counts; otherwise no overload takes the arguments.
				const bool arityMatches = std::any_of(candidates.begin(), candidates.end(), [&](const FunctionDecl* c) {
					return c->parameters.size() == argtypes.size();
				});
				if (arityMatches)
					return nullptr;
				best = &decl;
			}
			std::vector<bool> pattern;
			bool differs = false;
			for (size_t i = 0; i < best->parameters.size() && i < argtypes.size(); ++i) {
				const bool param = masked(best->parameters[i].second.get().type), arg = masked(argtypes[i]);
				pattern.push_back(param || arg);
				differs |= arg && !param;
			}
			if (!differs || best->templ == nullptr)
				return best;
			return &specialize(*best, pattern);
		}

		/**
		 * @brief The values of the template parameters of @p decl that the argument types determine: a parameter
		 * dimension that is a template parameter (e.g. N in `x: float[..., N]`) takes the size of the argument's
		 * dimension (dimensions before `...` are matched from the front, those after it from the back).
		 */
		static std::map<std::string, int>
		inferTemplateValues(const FunctionDecl& decl, const std::vector<DataType>& argtypes) {
			std::map<std::string, int> values;
			if (decl.templateParameters.empty())
				return values;
			auto templateParameter = [&](const TensorType::Shape::value_type& dim) -> const std::string* {
				const auto* node = std::get_if<const Node*>(&dim);
				const auto* hp = node ? std::get_if<HyperparameterNode>(*node) : nullptr;
				for (const auto& [name, tp] : decl.templateParameters)
					if (hp == &tp.get())
						return &name;
				return nullptr;
			};
			auto size = [](const TensorType::Shape::value_type& dim) -> std::optional<int> {
				if (const auto* u = std::get_if<unsigned>(&dim))
					return static_cast<int>(*u);
				if (const auto* node = std::get_if<const Node*>(&dim))
					if (auto v = foldConstant(**node); v && std::holds_alternative<int>(*v))
						return std::get<int>(*v);
				return std::nullopt;
			};
			for (size_t i = 0; i < decl.parameters.size() && i < argtypes.size(); ++i) {
				const auto* p = std::get_if<TensorType>(&decl.parameters[i].second.get().type);
				const auto* a = std::get_if<TensorType>(&argtypes[i]);
				if (p == nullptr || a == nullptr)
					continue;
				auto bind = [&](const auto& pdim, const auto& adim) {
					if (const auto* name = templateParameter(pdim))
						if (auto v = size(adim))
							values.emplace(*name, *v);
				};
				auto ellipsis = std::find_if(p->shape.begin(), p->shape.end(), [](const auto& d) {
					return std::holds_alternative<Elipsis>(d);
				});
				const size_t before = static_cast<size_t>(ellipsis - p->shape.begin());
				for (size_t k = 0; k < before && k < a->shape.size(); ++k)
					bind(p->shape[k], a->shape[k]);
				if (ellipsis != p->shape.end())
					for (size_t k = 1; k + before < p->shape.size() && k <= a->shape.size(); ++k)
						bind(p->shape[p->shape.size() - k], a->shape[a->shape.size() - k]);
			}
			return values;
		}

		/**
		 * @brief Elaborates @p decl again with the parameters marked in @p maskedPattern taking masked tensors and the
		 * template parameters bound to @p values.
		 */
		const FunctionDecl& specialize(
				const FunctionDecl& decl, const std::vector<bool>& maskedPattern,
				const std::map<std::string, int>& values = {}
		) {
			const auto& t = *decl.templ;
			std::string key;
			for (bool m : maskedPattern)
				key += m ? 'm' : '-';
			for (const auto& [name, v] : values)
				key += ";" + name + "=" + std::to_string(v);
			if (auto it = specializations.find({&t, key}); it != specializations.end())
				return *it->second;

			auto* outerErrors = std::exchange(errorlistener, t.errors);
			const std::string outerScope = std::exchange(scope, t.weightScope);
			nsstack.push(t.scope->createSubnamespace());
			maskedParams = maskedPattern;
			templateValues = values;
			auto [name, specialized] = std::any_cast<std::pair<std::string, FunctionDecl>>(t.ctx->accept(this));
			nsstack.pop();
			scope = outerScope;
			errorlistener = outerErrors;

			specialized.templ = decl.templ;
			state.derivedFunctions.push_back(std::move(specialized));
			const FunctionDecl& result = state.derivedFunctions.back();
			specializations.emplace(std::pair{&t, key}, &result);
			return result;
		}

	public:
		/** @brief See CompilationUnit::specialize. */
		const FunctionDecl& specializeMasked(const FunctionDecl& decl, const std::vector<bool>& maskedPattern) {
			if (decl.templ == nullptr)
				return decl;
			std::vector<DataType> argtypes;
			for (size_t i = 0; i < decl.parameters.size(); ++i) {
				DataType t = decl.parameters[i].second.get().type;
				if (auto* tensor = std::get_if<TensorType>(&t); tensor && i < maskedPattern.size())
					tensor->masked |= maskedPattern[i];
				argtypes.push_back(std::move(t));
			}
			const FunctionDecl* resolved = resolveCall(decl, argtypes);
			return resolved != nullptr ? *resolved : decl;
		}

	public:
		virtual std::any visitVarAccessExpression(mdlParser::VarAccessExpressionContext* ctx) override {
			if (const auto* tensor = currentNamespace().find<SlottedTensor>(ctx->IDENT()->getText())) {
				const Node* whole = stacked(*tensor, ctx->IDENT()->getText(), ctx->IDENT()->getSymbol());
				referenceHere(ctx->IDENT()->getSymbol(), Object{whole});
				return whole;
			}
			if (const auto* omitted = currentNamespace().find<OmittedMember>(ctx->IDENT()->getText())) {
				auto* sym = ctx->IDENT()->getSymbol();
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, sym, sym->getLine(), sym->getCharPositionInLine(),
						std::format("{} does not exist in this instance ({})", ctx->IDENT()->getText(), omitted->reason)
				);
				return static_cast<const Node*>(&graph().addErrorNode());
			}
			if (symbolicMode)
				checkConditionalMember(ctx->IDENT()->getSymbol());
			auto value = currentNamespace().get<Object>(ctx->IDENT()->getText());
			if (!value.has_value() && currentNamespace().find<NamespaceRef>(ctx->IDENT()->getText())) {
				importError(
						ctx->IDENT()->getSymbol(), std::format(
														   "{} is a namespace: use one of its names ({}.name)",
														   ctx->IDENT()->getText(), ctx->IDENT()->getText()
												   )
				);
				return static_cast<const Node*>(&graph().addErrorNode());
			}
			if (!value.has_value()) {
				auto* sym = ctx->IDENT()->getSymbol();
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, sym, sym->getLine(), sym->getCharPositionInLine(),
						std::format("Variable {} not defined", ctx->IDENT()->getText())
				);
				return &graph().addErrorNode();
			}
			referenceHere(ctx->IDENT()->getSymbol(), value->get());
			return value.value().get(); // Returns Object as std::any
		}
		virtual std::any visitConstantExpression(mdlParser::ConstantExpressionContext* ctx) override {
			return &graph().addConstNode(std::any_cast<mdl::Atomic>(ctx->constant()->accept(this)));
		}
		virtual std::any visitParenExpression(mdlParser::ParenExpressionContext* ctx) override {
			return ctx->expr()->accept(this);
		}

		/**
		 * @brief The namespace that @p e denotes (`nn` bound by `import std.nn;`, `std.nn` of `import std;`), if any.
		 * @p error receives why a member of a namespace is not available.
		 */
		std::optional<NamespaceRef> namespaceOf(mdlParser::ExprContext* e, std::string* error = nullptr) {
			if (auto* var = dynamic_cast<mdlParser::VarAccessExpressionContext*>(e)) {
				if (var->IDENT() == nullptr)
					return std::nullopt;
				const std::string name = var->IDENT()->getText();
				// (A local variable of the same name hides the namespace.)
				if (currentNamespace().get<Object>(name))
					return std::nullopt;
				if (const auto* ref = currentNamespace().find<NamespaceRef>(name)) {
					indexNamespace(var->IDENT()->getSymbol(), *ref);
					return *ref;
				}
				return std::nullopt;
			}
			if (auto* access = dynamic_cast<mdlParser::AccessExpressionContext*>(e)) {
				auto base = namespaceOf(access->expr(), error);
				if (!base || access->IDENT() == nullptr)
					return std::nullopt;
				Member m = memberOf(*base, access->IDENT()->getText());
				if (!m.error.empty() && error != nullptr)
					*error = m.error;
				if (m.ns)
					indexNamespace(access->IDENT()->getSymbol(), *m.ns);
				return m.ns;
			}
			return std::nullopt;
		}

		/** @brief The names of a namespace that the current file may use (files' names, or a directory's entries). */
		std::vector<std::string> namesOf(const NamespaceRef& ns) const {
			if (ns.file != nullptr)
				return namesIn(*ns.file, currentScope);
			std::vector<std::string> names;
			std::error_code ec;
			for (const auto& entry : std::filesystem::directory_iterator(ns.dir, ec))
				if (isSource(entry.path()) || entry.is_directory())
					names.push_back(entry.path().stem().string());
			std::ranges::sort(names);
			return names;
		}

		/** @brief Records a use of a namespace (`nn` in `nn.Linear`) with its names (for completion after `nn.`). */
		void indexNamespace(antlr4::Token* tok, const NamespaceRef& ns) {
			if (index() == nullptr || tok == nullptr ||
				!indexedNamespaces.insert({currentFile(), tok->getTokenIndex()}).second)
				return;
			const auto names = namesOf(ns);
			index()->reference(analysis::Reference{
					.file = currentFile(),
					.range = rangeOf(tok),
					.types = {"namespace"},
					.members = {names.begin(), names.end()}
			});
		}
		std::set<std::pair<std::string, size_t>> indexedNamespaces;

		/** @brief `ns.name` for a namespace `ns`: the definition (a value), or an error. */
		std::any namespaceMember(mdlParser::AccessExpressionContext* ctx, const NamespaceRef& ns) {
			auto* tok = ctx->IDENT()->getSymbol();
			Member m = memberOf(ns, ctx->IDENT()->getText());
			const Object* object = m.name && m.name->object ? std::any_cast<Object>(m.name->object) : nullptr;
			if (object == nullptr) {
				importError(
						tok, !m.error.empty() ? m.error
							 : m.ns			  ? std::format("{} is a namespace, not a value", ctx->IDENT()->getText())
							 : m.name		  ? std::format("{} is a type, not a value", ctx->IDENT()->getText())
											  : std::format("{} is not defined", ctx->IDENT()->getText())
				);
				return static_cast<const Node*>(&graph().addErrorNode());
			}
			if (index() != nullptr)
				index()->reference(analysis::Reference{
						.file = currentFile(),
						.range = rangeOf(tok),
						.symbol = m.name->symbol,
						.types = {objectType(*object)}
				});
			return *object;
		}

		virtual std::any visitAccessExpression(mdlParser::AccessExpressionContext* ctx) override {
			if (ctx->IDENT() != nullptr)
				if (auto ns = namespaceOf(ctx->expr()))
					return namespaceMember(ctx, *ns);
			std::optional<Object> base;
			auto result = accessExpression(ctx, base);
			if (index() != nullptr && ctx->IDENT() != nullptr) {
				analysis::Reference r{.file = currentFile(), .range = rangeOf(ctx->IDENT()->getSymbol())};
				// The member's definition: in the struct of the value, or in the module of the instance.
				if (const auto* agg = base ? std::get_if<AggregateInstance>(&*base) : nullptr)
					if (auto it = structContainers.find(aggregateKey(agg->getType())); it != structContainers.end())
						for (size_t i = 0; i < index()->symbols().size(); ++i) {
							const auto& s = index()->symbols()[i];
							if (s.kind == analysis::SymbolKind::member && s.container == it->second &&
								s.name == ctx->IDENT()->getText())
								r.symbol = i;
						}
				const auto* node = base ? std::get_if<const Node*>(&*base) : nullptr;
				const auto* bound = node ? std::get_if<BoundModuleType>(&getDataType(**node)) : nullptr;
				if (bound != nullptr && bound->definition.templ) {
					const auto& t = *bound->definition.templ;
					auto locate = [&](antlr4::Token* tok) {
						if (tok != nullptr && tok->getText() == ctx->IDENT()->getText())
							r.symbol = index()->find(t.errors->getFile(), rangeOf(tok).begin);
					};
					for (auto* v : t.ctx->vardecl())
						locate(v->IDENT() ? v->IDENT()->getSymbol() : nullptr);
					for (auto* f : t.ctx->funcdef())
						locate(f->IDENT() ? f->IDENT()->getSymbol() : nullptr);
				}
				try {
					Object value = std::any_cast<Object>(result);
					r.types.insert(objectType(value));
					r.members = objectMembers(value);
					if (auto path = weightPath(value))
						r.weights.insert(*path);
				} catch (const std::bad_any_cast&) {
					if (const auto* n = std::any_cast<const Node*>(&result);
						n && !std::holds_alternative<ErrorNode>(**n)) {
						r.types.insert(objectType(Object{*n}));
						r.members = objectMembers(Object{*n});
						if (auto path = weightPath(Object{*n}))
							r.weights.insert(*path);
					}
				}
				r.types.erase(std::string{});
				index()->reference(std::move(r));
			}
			return result;
		}

		/**
		 * @brief The member @p name (a weight, a submodule, an array of them, or a struct-typed weight) of the submodule
		 * @p receiver of type @p bmt, e.g. `proj.W`: the weight of the caller's scope that the submodule's functions
		 * use (`proj.W`, see Computegraph::addModuleInvocationNode). std::nullopt if there is no such member.
		 */
		std::optional<Object>
		submoduleMember(const Node& receiver, const BoundModuleType& bmt, const std::string& name) {
			const auto* instance = std::get_if<ParameterNode>(&receiver);
			if (instance == nullptr || !bmt.hyperparamValues.empty())
				return std::nullopt; // (instances have their types expanded: no hyperparameters to substitute)
			const std::string prefix = instance->name + ".";
			auto weight = [&](const std::string& path, DataType type) -> const Node& {
				return graph().getOrAddParameterNode(scope, prefix + path, std::move(type));
			};
			ArrayInstance array; // the elements `name.0`, `name.1`, ... of an array of members
			for (const auto& [path, w] : bmt.definition.variables) {
				const auto& type = w.get().type;
				if (path == name) {
					if (const auto* agg = std::get_if<AggregateType>(&type))
						return makeAggregateObject(*agg, prefix + path, [&](DataType t, std::string p) -> const Node& {
							return graph().getOrAddParameterNode(scope, p, std::move(t));
						});
					return Object{&weight(path, type)};
				}
				if (path.starts_with(name + ".") && path.find('.', name.size() + 1) == std::string::npos)
					array.elements.push_back(Object{&weight(path, type)});
			}
			if (array.elements.empty())
				return std::nullopt;
			array.representative = symbolicMode && array.elements.size() == 1;
			return Object{std::move(array)};
		}

		/** @brief The file scope whose namespace is @p ns (where a template was defined); nullptr if none. */
		const FileScope* scopeOfNamespace(const Namespace* ns) const {
			for (const auto& scope : fileScopes)
				if (&scope.local == ns)
					return &scope;
			return nullptr;
		}

		/**
		 * @brief Why the member @p name of a module instance is not accessible here (`instance.name`), if it is not: a
		 * private member is only used inside its module, a package-visible one only in the module's package.
		 */
		std::optional<std::string> memberVisibilityProblem(const BoundModuleType& bmt, const std::string& name) const {
			const auto& t = bmt.definition.templ;
			if (t == nullptr || t->ctx == nullptr)
				return std::nullopt;
			mdlParser::VisibilityContext* written = nullptr;
			bool found = false;
			for (auto* v : t->ctx->vardecl())
				if (v->IDENT() && v->IDENT()->getText() == name)
					written = v->visibility(), found = true;
			for (auto* f : t->ctx->funcdef())
				if (f->IDENT() && f->IDENT()->getText() == name)
					written = f->visibility(), found = true;
			if (!found)
				return std::nullopt; // (not a member: reported elsewhere)
			const Visibility v = visibilityOf(written);
			if (v == Visibility::Private)
				return std::format("{} is a private member of {}", name, t->name);
			const FileScope* defining = scopeOfNamespace(t->scope);
			if (v == Visibility::Package && defining != nullptr && !visibleFrom(v, *defining, currentScope))
				return std::format(
						"{} is a package-visible member of {} (in another package: {})", name, t->name,
						defining->path.filename().string()
				);
			return std::nullopt;
		}

		std::any accessExpression(mdlParser::AccessExpressionContext* ctx, std::optional<Object>& base) {
			/** \todo this function needs refactoring; e.g., merge the handling of AggregateInstance, ModuleInstance, TensorType, ... to simple getMember lookups **/
			auto baseObj = asObject(ctx->expr()->accept(this));
			base = baseObj;
			auto memberName = ctx->IDENT()->getText();
			auto* memberTok = ctx->IDENT()->getSymbol();

			// Struct/aggregate member access.
			if (auto* agg = std::get_if<AggregateInstance>(&baseObj)) {
				auto mem = agg->getMember(memberName);
				if (!mem.has_value()) {
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error, memberTok, memberTok->getLine(),
							memberTok->getCharPositionInLine(), std::format("No member '{}' in struct", memberName)
					);
					return &graph().addErrorNode();
				}
				return mem.value().get();
			} else if (auto* mod = std::get_if<ModuleInstance>(&baseObj)) {
				auto mem = mod->getMember(memberName);
				if (!mem.has_value()) {
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error, memberTok, memberTok->getLine(),
							memberTok->getCharPositionInLine(), std::format("No member '{}' in struct", memberName)
					);
					return &graph().addErrorNode();
				}
				return mem.value().get();
			} else if (auto* nodePtr =
							   std::get_if<const Node*>(&baseObj)) { // Module function access: node of BoundModuleType.
				if (isErroneous(**nodePtr)) {
					// Don't show too many errors; if we have <error node>.IDENT, we will just propagate the error.
					return &graph().addErrorNode();
				} else if (auto* bmt = std::get_if<BoundModuleType>(&mdl::getDataType(**nodePtr))) {
					if (auto problem = memberVisibilityProblem(*bmt, memberName)) {
						importError(memberTok, *problem);
						return static_cast<const Node*>(&graph().addErrorNode());
					}
					/** \todo bind the instance as the first argument **/
					/** \todo merge with ModuleInstance implementation above **/
					auto& funcs = bmt->definition.functions;
					auto it = funcs.find(memberName);
					if (it == funcs.end()) {
						if (auto member = submoduleMember(**nodePtr, *bmt, memberName))
							return *member;
						errorlistener->semanticError(
								PrettyErrorListener::ErrorLevel::Error, memberTok, memberTok->getLine(),
								memberTok->getCharPositionInLine(),
								std::format(
										"No member '{}' in {}", memberName,
										bmt->definition.instanceName.empty() ? "module" : bmt->definition.instanceName
								)
						);
						return &graph().addErrorNode();
					}
					return Object{it->second};
				} else if (auto* tensor = std::get_if<TensorType>(&mdl::getDataType(**nodePtr))) {
					// Masked tensors: `x.mask` (1.0 valid, 0.0 masked) and `x.data` (the values, zeros where masked).
					if (memberName == "mask" || memberName == "data") {
						if (!tensor->masked) {
							errorlistener->semanticError(
									PrettyErrorListener::ErrorLevel::Error, memberTok, memberTok->getLine(),
									memberTok->getCharPositionInLine(),
									std::format("`.{}` of a tensor that is not masked", memberName)
							);
							return &graph().addErrorNode();
						}
						const auto id =
								memberName == "mask" ? builtins::FunctionIDs::MASK_OF : builtins::FunctionIDs::UNMASK;
						return &graph().addInvocationNode(id, {std::cref(**nodePtr)});
					}
					auto member = mdl::getMember(mdl::Object{*nodePtr}, memberName, {mdl::getDataType(**nodePtr)});
					if (!member.has_value()) {
						errorlistener->semanticError(
								PrettyErrorListener::ErrorLevel::Error, memberTok, memberTok->getLine(),
								memberTok->getCharPositionInLine(), std::format("No member '{}' for tensor", memberName)
						);
						return &graph().addErrorNode();
					}
					// Bind self to the receiver; the caller invokes the returned 0-param FunctionDecl with ().
					return Object{mdl::bind(graph(), member.value().get(), **nodePtr)};
				}
			}

			auto* firsttok = ctx->start;
			auto* lasttok = ctx->stop;
			errorlistener->semanticError(
					PrettyErrorListener::ErrorLevel::Error,
					{{.number = firsttok->getLine(),
					  .text = errorlistener->getline(*firsttok->getInputStream(), 0, firsttok->getStartIndex()),
					  .markBegin = firsttok->getCharPositionInLine(),
					  .markWidth = lasttok->getStopIndex() - firsttok->getStartIndex() + 1}},
					firsttok->getLine(), firsttok->getCharPositionInLine(),
					"Member access is only supported on struct instances and module objects"
			);
			return &graph().addErrorNode();
		}

		// ---------------------------------------------------------------
		// Einops helpers
		// ---------------------------------------------------------------

		static mdl::einops::Atom parseEinopsAtom(mdlParser::Einops_atomContext* ctx) {
			using mdl::einops::Atom;
			using mdl::einops::AtomKind;
			std::string text = ctx->getText();
			if (text == "...")
				return Atom{AtomKind::Ellipsis, {}, 0};
			if (text == "_")
				return Atom{AtomKind::Anon, {}, 0};
			// Chunk: starts with '/'
			if (!text.empty() && text[0] == '/') {
				int64_t val = std::atoll(text.c_str() + 1);
				return Atom{AtomKind::Chunk, {}, val};
			}
			if (ctx->IDENT() != nullptr)
				return Atom{AtomKind::Named, ctx->IDENT()->getText(), 0};
			if (ctx->INTEGER() != nullptr) {
				int64_t val = std::atoll(ctx->INTEGER()->getText().c_str());
				return Atom{AtomKind::Literal, {}, val};
			}
			// Fallback (shouldn't happen)
			return Atom{AtomKind::Named, text, 0};
		}

		static mdl::einops::Spec parseEinopsSpec(mdlParser::Einops_spec_fullContext* specCtx) {
			using mdl::einops::Spec;
			using mdl::einops::SpecItem;
			Spec spec;
			bool inGroup = false;
			SpecItem currentGroup{.is_group = true, .atoms = {}};

			for (auto* child : specCtx->children) {
				auto* term = dynamic_cast<antlr4::tree::TerminalNode*>(child);
				if (term) {
					std::string t = term->getText();
					if (t == "(") {
						inGroup = true;
						currentGroup.atoms.clear();
					} else if (t == ")") {
						inGroup = false;
						spec.push_back(std::move(currentGroup));
						currentGroup = {.is_group = true, .atoms = {}};
					}
					continue;
				}
				auto* atomCtx = dynamic_cast<mdlParser::Einops_atomContext*>(child);
				if (!atomCtx)
					continue;
				auto atom = parseEinopsAtom(atomCtx);
				if (inGroup) {
					currentGroup.atoms.push_back(atom);
				} else {
					spec.push_back(SpecItem{.is_group = false, .atoms = {atom}});
				}
			}
			return spec;
		}

		virtual std::any visitEinopsExpression(mdlParser::EinopsExpressionContext* ctx) override {
			// Evaluate the tensor operand.
			const Node* inputNode = asNode(ctx->expr()->accept(this), ctx->expr()->start);
			if (std::holds_alternative<ErrorNode>(*inputNode))
				return &graph().addErrorNode();

			auto* einopsCtx = ctx->einops();
			auto lhs = parseEinopsSpec(einopsCtx->einops_spec_full(0));
			// (Without output axes, e.g. `[1 -> ]`, the result is a scalar.)
			auto rhs = einopsCtx->einops_spec_full().size() > 1 ? parseEinopsSpec(einopsCtx->einops_spec_full(1))
																: mdl::einops::Spec{};

			// Collect key=value bindings.
			std::map<std::string, const Node*> bindings;
			for (auto* kv : einopsCtx->keyvalue()) {
				std::string key = kv->IDENT()->getText();
				const Node* valNode = asNode(kv->expr()->accept(this), kv->expr()->start);
				if (std::holds_alternative<ErrorNode>(*valNode))
					return &graph().addErrorNode();
				bindings[key] = valNode;
			}

			auto result = mdl::einops::apply(graph(), *inputNode, lhs, rhs, bindings);
			if (result && index() != nullptr) {
				// The sizes of the axes (for hovers and inlay hints).
				const auto sizes = mdl::einops::axisSizes(*inputNode, lhs, rhs, bindings);
				auto record = [&](antlr4::Token* tok) {
					if (auto it = sizes.find(tok->getText()); it != sizes.end())
						index()->addAxis(
								{.file = currentFile(),
								 .range = rangeOf(tok),
								 .name = tok->getText(),
								 .sizes = {it->second}}
						);
				};
				for (auto* spec : einopsCtx->einops_spec_full())
					for (auto* atom : spec->einops_atom())
						if (atom->IDENT())
							record(atom->IDENT()->getSymbol());
				for (auto* kv : einopsCtx->keyvalue())
					record(kv->IDENT()->getSymbol());
			}
			if (!result) {
				auto* tok = ctx->start;
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
						std::format("einops error: {}", result.error())
				);
				return &graph().addErrorNode();
			}
			return static_cast<const Node*>(&result->get());
		}

		/**
		 * @brief A tensor declared without initializer (`float[n, ..., d] hs;`): its n slots along the first dimension
		 * are assigned one by one (`hs[i] = h;`, compile-time indices), e.g. the output of every layer in a loop.
		 * @details Reading a slot yields the assigned node; using the tensor as a whole stacks the slots (a STACK node).
		 * Every slot that is read must have been assigned (checked when compiling).
		 */
		struct SlottedTensor {
			TensorType type;
			std::vector<const Node*> slots; ///< nullptr: not assigned yet
			bool representative = false;	///< symbolic analysis: the only slot stands for all
		};

		/** @brief The type of a slot of @p t (the declared type without its first dimension). */
		static TensorType slotType(const SlottedTensor& t) {
			TensorType type = t.type;
			type.shape.erase(type.shape.begin());
			return type;
		}

		const Node* slotError(antlr4::Token* tok, const std::string& message) {
			errorlistener->semanticError(
					PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(), message
			);
			return &graph().addErrorNode();
		}

		/** @brief The slot @p index of @p t (negative: from the end), which must be assigned. */
		const Node* slot(const SlottedTensor& t, const std::string& name, int index, antlr4::Token* tok) {
			const int n = static_cast<int>(t.slots.size());
			const int i = t.representative ? 0 : index < 0 ? index + n : index;
			if (i < 0 || i >= n)
				return slotError(tok, std::format("Index {} is out of bounds ({} has {} slots)", index, name, n));
			if (t.slots[i] == nullptr)
				return slotError(tok, std::format("{}[{}] is used before it is assigned", name, i));
			return t.slots[i];
		}

		/** @brief @p t as a whole: its slots stacked (all must be assigned). */
		const Node* stacked(const SlottedTensor& t, const std::string& name, antlr4::Token* tok) {
			std::vector<Computegraph::NodeRef> parts;
			for (size_t i = 0; i < t.slots.size(); ++i) {
				if (t.slots[i] == nullptr)
					return slotError(
							tok,
							std::format("{} is used before all its slots are assigned ({}[{}] is not)", name, name, i)
					);
				if (std::holds_alternative<ErrorNode>(*t.slots[i]))
					return t.slots[i]; // (reported)
				parts.push_back(std::cref(*t.slots[i]));
			}
			return &graph().addInvocationNode(builtins::FunctionIDs::STACK, std::move(parts));
		}

		/** @brief The float element type of a tensor node, if it has one. */
		static std::optional<PrimitiveType> floatTensorType(const Node& node) {
			const auto* t = std::get_if<TensorType>(&getDataType(node));
			return t != nullptr && isFloat(t->type) ? std::optional{t->type} : std::nullopt;
		}

		/**
		 * @brief Converts @p x to the float type @p target where the language allows an implicit conversion (see
		 * ParseOptions::floatWidening and floatNarrowing); reports it at @p tok as configured.
		 * @return the (converted) value, or nullptr if the conversion is an error (reported).
		 */
		const Node* implicitConversion(
				const Node& x, PrimitiveType target, antlr4::Token* tok, const std::string& what,
				antlr4::ParserRuleContext* operand = nullptr
		) {
			const auto from = floatTensorType(x);
			if (!from || *from == target)
				return &x;
			const bool widening = target == PrimitiveType::float32;
			const auto level = widening ? options.floatWidening : options.floatNarrowing;
			const std::string flag = widening ? "float-widening" : "float-narrowing";
			if (level != parser::DiagnosticLevel::ignored) {
				const bool error = level == parser::DiagnosticLevel::error;
				// The fix: the operand wrapped in the conversion (`float32(x)`).
				std::vector<Diagnostic::Fix> fixes;
				if (operand != nullptr && operand->start != nullptr && operand->stop != nullptr) {
					const auto range = rangeOf(operand);
					fixes.push_back(Diagnostic::Fix{
							.title = std::format("Convert to {}", std::to_string(target)),
							.edits =
									{{{range.begin, range.begin}, std::to_string(target) + "("},
									 {{range.end, range.end}, ")"}},
					});
				}
				errorlistener->semanticError(
						error ? PrettyErrorListener::ErrorLevel::Error : PrettyErrorListener::ErrorLevel::Warning,
						tokenContext(tok), tok->getLine(), tok->getCharPositionInLine(),
						std::format(
								"{}: implicit {} from {} to {}", what, widening ? "widening" : "narrowing",
								std::to_string(*from), std::to_string(target)
						),
						error ? std::format("convert explicitly, e.g. `{}(x)`", std::to_string(target)) : "", flag,
						std::move(fixes)
				);
				if (error)
					return nullptr;
			}
			return &graph().addInvocationNode(
					builtins::FunctionIDs::CAST,
					{std::cref(x), std::cref(graph().addConstNode(Atomic{static_cast<int>(target)}))}
			);
		}

		/** @brief @p type with float32 as its float element type (to check only the shape against a builtin). */
		static DataType withoutFloatType(DataType type) {
			if (auto* t = std::get_if<TensorType>(&type); t != nullptr && isFloat(t->type))
				t->type = PrimitiveType::float32;
			return type;
		}

		/** @brief The float type named @p name (`float` is the configured default), if it names one. */
		std::optional<PrimitiveType> floatTypeNamed(const std::string& name) const {
			if (name == "float")
				return options.defaultFloat;
			if (name == "float32")
				return PrimitiveType::float32;
			if (name == "float16")
				return PrimitiveType::float16;
			if (name == "bfloat16")
				return PrimitiveType::bfloat16;
			return std::nullopt;
		}

		/**
		 * @brief An explicit conversion `T(x)` to the float type T (conversions between float types are never implicit:
		 * narrowing loses precision, widening costs memory and time).
		 */
		std::optional<const Node*> conversion(mdlParser::MethodcallExpressionContext* ctx) {
			auto* name = dynamic_cast<mdlParser::VarAccessExpressionContext*>(ctx->func);
			if (name == nullptr)
				return std::nullopt;
			auto target = floatTypeNamed(name->IDENT()->getText());
			if (!target)
				return std::nullopt;
			if (ctx->args.size() != 1) {
				auto* tok = ctx->start;
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
						std::format("{}(x) converts one value", name->IDENT()->getText())
				);
				return &graph().addErrorNode();
			}
			const Node* x = asNode(ctx->args[0]->accept(this), ctx->args[0]->start);
			if (std::holds_alternative<ErrorNode>(*x))
				return x;
			const auto* tensor = std::get_if<TensorType>(&getDataType(*x));
			if (tensor == nullptr || tensor->type == *target)
				return x; // scalars take the type of the tensors they are combined with
			return &graph().addInvocationNode(
					builtins::FunctionIDs::CAST,
					{std::cref(*x), std::cref(graph().addConstNode(Atomic{static_cast<int>(*target)}))}
			);
		}

		virtual std::any visitMethodcallExpression(mdlParser::MethodcallExpressionContext* ctx) override {
			auto result = methodcallExpression(ctx);
			// The type of the result at the callee's name (for hovers).
			antlr4::Token* name = nullptr;
			if (auto* var = dynamic_cast<mdlParser::VarAccessExpressionContext*>(ctx->func); var && var->IDENT())
				name = var->IDENT()->getSymbol();
			else if (auto* access = dynamic_cast<mdlParser::AccessExpressionContext*>(ctx->func);
					 access && access->IDENT())
				name = access->IDENT()->getSymbol();
			if (index() != nullptr && name != nullptr) {
				std::string type;
				if (const auto* node = std::any_cast<const Node*>(&result)) {
					if (!std::holds_alternative<ErrorNode>(**node))
						type = typeString(getDataType(**node));
				} else if (const auto* multi = std::any_cast<MultiValue>(&result)) {
					for (const auto* n : multi->nodes)
						type += (type.empty() ? "(" : ", ") + typeString(getDataType(*n));
					type += ")";
				}
				if (!type.empty())
					index()->reference(analysis::Reference{
							.file = currentFile(), .range = rangeOf(name), .symbol = std::nullopt, .results = {type}
					});
			}
			return result;
		}

		/** @brief Whether @p node is an error (already reported): an error node, or a value of erroneous type. */
		static bool isErroneous(const Node& node) {
			if (std::holds_alternative<ErrorNode>(node))
				return true;
			const auto* p = std::get_if<PrimitiveType>(&getDataType(node));
			return p != nullptr && *p == PrimitiveType::error;
		}

		std::any methodcallExpression(mdlParser::MethodcallExpressionContext* ctx) {
			if (auto converted = conversion(ctx))
				return *converted;
			// After an error in the callee (already reported), the arguments are still checked (and their uses
			// recorded, e.g. for -Wunused).
			auto failed = [&]() -> std::any {
				for (auto* arg : ctx->args)
					arg->accept(this);
				return static_cast<const Node*>(&graph().addErrorNode());
			};
			// `instance.member(...)`: the member function is inlined for the instance (its weights), like
			// `instance(...)`. Without this, the function would reference the module's unqualified weights.
			const Node* receiver = nullptr;
			if (auto* access = dynamic_cast<mdlParser::AccessExpressionContext*>(ctx->func);
				access != nullptr && !namespaceOf(access->expr())) {
				auto baseObj = asObject(access->expr()->accept(this), access->expr()->start);
				if (auto* base = std::get_if<const Node*>(&baseObj)) {
					if (isErroneous(**base))
						return failed(); // already reported
					if (std::holds_alternative<BoundModuleType>(mdl::getDataType(**base)))
						receiver = *base;
				}
			}
			auto calleeObj = asObject(ctx->func->accept(this), ctx->func->start);
			if (std::holds_alternative<const Node*>(calleeObj) && isErroneous(*std::get<const Node*>(calleeObj)))
				return failed();

			// Evaluate argument expressions.
			std::vector<std::reference_wrapper<const Node>> args;
			bool hasError = false;
			for (size_t i = 0; i < ctx->args.size(); ++i) {
				const Node* argNode = asNode(ctx->args[i]->accept(this), ctx->args[i]->start);
				if (std::holds_alternative<ErrorNode>(*argNode))
					hasError = true;
				args.emplace_back(std::cref(*argNode));
			}
			if (hasError)
				return &graph().addErrorNode();

			// Collect argument types for overload resolution.
			std::vector<DataType> argtypes;
			for (auto& ref : args)
				argtypes.push_back(mdl::getDataType(ref.get()));

			// Resolve operator() on the callee object.
			auto decl = getMember(calleeObj, "operator()", argtypes);
			// (A member whose type is erroneous, e.g. an invalid module instantiation, was reported already.)
			if (const auto* callee = std::get_if<const Node*>(&calleeObj); !decl.has_value() && callee) {
				const auto* p = std::get_if<PrimitiveType>(&mdl::getDataType(**callee));
				if (p != nullptr && *p == PrimitiveType::error)
					return &graph().addErrorNode();
			}
			if (!decl.has_value()) {
				auto* tok = ctx->func->start;
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
						"Object is not callable with these argument types"
				);
				return &graph().addErrorNode();
			}
			/** \todo For member function calls (obj.func(...)), obj should be passed as the first ("this") argument.
			 *  This is not yet implemented. */

			// The overload that matches the argument types, specialized for them if needed (e.g. masked tensors).
			const FunctionDecl* resolved = resolveCall(decl.value().get(), argtypes);
			if (resolved == nullptr) {
				auto* tok = ctx->func->start;
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
						"No overload takes these arguments (a `masked` parameter needs a masked tensor)"
				);
				return &graph().addErrorNode();
			}
			const FunctionDecl& callee = *resolved;
			// The arguments must have the shapes of the parameters (definite mismatches only, see shapes::matches).
			// (The builtins are generic in the float type: their parameters' float32 is a placeholder.)
			const bool builtin = callee.name.starts_with("__builtin_");
			// Arguments of another float type than their parameter: converted, if allowed.
			for (size_t i = 0; !builtin && i < std::min(args.size(), callee.parameters.size()); ++i) {
				const auto* param = std::get_if<TensorType>(&callee.parameters[i].second.get().type);
				if (param == nullptr || !isFloat(param->type) || !floatTensorType(args[i].get()) ||
					*floatTensorType(args[i].get()) == param->type)
					continue;
				const Node* converted = implicitConversion(
						args[i].get(), param->type, ctx->args[i]->start,
						std::format("Argument {} of {}", i + 1, callee.name), ctx->args[i]
				);
				if (converted == nullptr)
					return &graph().addErrorNode();
				args[i] = std::cref(*converted);
				argtypes[i] = getDataType(*converted);
			}
			for (size_t i = 0; i < std::min(args.size(), callee.parameters.size()); ++i)
				if (!shapes::matches(
							callee.parameters[i].second.get().type,
							builtin ? withoutFloatType(argtypes[i]) : argtypes[i], false
					)) {
					auto* tok = ctx->args[i]->start;
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
							std::format(
									"Argument {} of {} is {}, but the parameter {} is {}", i + 1, callee.name,
									shapes::str(argtypes[i]), callee.parameters[i].first,
									shapes::str(callee.parameters[i].second.get().type)
							)
					);
					return &graph().addErrorNode();
				} else if (!builtin && callee.templateParameters.empty() &&
						   !shapes::matches(
								   callee.parameters[i].second.get().type, argtypes[i], false, /*strict=*/true
						   )) {
					reportPossibleMismatch(
							ctx->args[i]->start,
							std::format(
									"Argument {} of {} is {}, but the parameter {} is {} (the sizes may differ)", i + 1,
									callee.name, shapes::str(argtypes[i]), callee.parameters[i].first,
									shapes::str(callee.parameters[i].second.get().type)
							)
					);
				}

			// If the callee is a module instance node, use addModuleInvocationNode so that each instance (weightQuery,
			// weightKey, ...) gets its own weights (with fully qualified paths, e.g. `weightQuery.W`) instead of sharing
			// the module type's nodes.
			auto resultOrErr = [&]() {
				if (receiver != nullptr) {
					std::string instanceName;
					if (const auto* pn = std::get_if<ParameterNode>(receiver))
						instanceName = pn->name;
					return graph().addModuleInvocationNode(
							std::get<BoundModuleType>(mdl::getDataType(*receiver)), scope, instanceName, callee,
							std::move(args)
					);
				}
				if (auto* calleeNodePtr = std::get_if<const Node*>(&calleeObj)) {
					if (!std::holds_alternative<ErrorNode>(**calleeNodePtr)) {
						if (auto* bmt = std::get_if<BoundModuleType>(&mdl::getDataType(**calleeNodePtr))) {
							std::string instanceName;
							if (const auto* pn = std::get_if<ParameterNode>(*calleeNodePtr))
								instanceName = pn->name;
							return graph().addModuleInvocationNode(*bmt, scope, instanceName, callee, std::move(args));
						}
					}
				}
				// A free function `f` with a custom gradient `f_gradient`: record the call for the gradient pass.
				std::vector<const Node*> inputs;
				for (const auto& arg : args)
					inputs.push_back(&arg.get());
				auto outputs = graph().addInvocationNode(callee, std::move(args));
				// A builtin applied to operands that do not fit (e.g. float tensors of different types).
				if (builtin && outputs && outputs->size() == 1)
					if (const auto* inv = std::get_if<InvocationNode>(&outputs->front().get()))
						if (auto problem = shapes::check(inv->id, inv->arguments)) {
							auto* tok = ctx->func->start;
							errorlistener->semanticError(
									PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(),
									tok->getCharPositionInLine(), std::format("{}: {}", callee.name, *problem)
							);
						} else if (auto possible = shapes::check(inv->id, inv->arguments, /*strict=*/true)) {
							reportPossibleMismatch(ctx->func->start, std::format("{}: {}", callee.name, *possible));
						}
				if (outputs) {
					// (In the callee's file: its gradient need not be visible at the call.)
					const Namespace& defining = callee.templ != nullptr ? *callee.templ->scope : currentNamespace();
					auto gradient = defining.get<Object>(callee.name + "_gradient");
					if (const auto* decl = gradient ? std::get_if<FunctionDecl>(&gradient->get()) : nullptr) {
						ModuleCall call{
								.scope = scope,
								.path = {},
								.module = nullptr,
								.hyperparameters = {},
								.function = callee.name,
								.inputs = std::move(inputs),
								.outputs = {},
								.customGradient = true,
								.gradient = decl
						};
						for (const auto& output : *outputs)
							call.outputs.push_back(&output.get());
						graph().addModuleCall(std::move(call));
					}
				}
				return outputs;
			}();
			if (!resultOrErr.has_value()) {
				auto* tok = ctx->func->start;
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
						resultOrErr.error()
				);
				return &graph().addErrorNode();
			}
			const auto& results = resultOrErr.value();
			if (results.size() == 1)
				return &results.front().get();
			MultiValue multi;
			for (const auto& result : results)
				multi.nodes.push_back(&result.get());
			return multi;
		}

		virtual std::any visitArithmeticExpression(mdlParser::ArithmeticExpressionContext* ctx) override {
			const auto* left = asNode(ctx->left->accept(this), ctx->left->start);
			const auto* right = asNode(ctx->right->accept(this), ctx->right->stop);
			if (std::holds_alternative<ErrorNode>(*left) || std::holds_alternative<ErrorNode>(*right)) {
				// If one of the arguments is an error node (invalid subexpression), we won't find the operator to call.
				// In this case, don't print another error, just propagate the error node so we do not spam the user
				// with unhelpful error messages.
				/** \todo in the future, we could try to infer despite the error nodes to recover from errors for the
				 *  analysis and give further insightful hints. */
				return &graph().addErrorNode();
			}
			// Operands of different float types: both to the wider type (float32 for bfloat16 with float16), if allowed.
			if (auto l = floatTensorType(*left), r = floatTensorType(*right); l && r && *l != *r) {
				const std::string what = std::format("Operator {}", ctx->bop->getText());
				left = implicitConversion(*left, PrimitiveType::float32, ctx->bop, what, ctx->left);
				right = implicitConversion(*right, PrimitiveType::float32, ctx->bop, what, ctx->right);
				if (left == nullptr || right == nullptr)
					return &graph().addErrorNode();
			}
			auto decl = getMember(
					mdl::getDataType(*left), std::format("operator{}", ctx->bop->getText()), {mdl::getDataType(*right)}
			);
			if (!decl.has_value()) { // Function not found!
				auto firsttok = ctx->start;
				auto lasttok = ctx->stop;
				/** \todo describe data types **/
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error,
						{{.number = firsttok->getLine(),
						  .text = errorlistener->getline(*firsttok->getInputStream(), 0, firsttok->getStartIndex()),
						  .markBegin = firsttok->getCharPositionInLine(),
						  .markWidth = lasttok->getStopIndex() - firsttok->getStartIndex() + 1}},
						firsttok->getLine(), firsttok->getCharPositionInLine(),
						std::format("Operator {} not supported on {}", ctx->bop->getText(), "TODO")
				);
				return &graph().addErrorNode();
			}
			const Node& result = graph().addInvocationNode(decl.value(), {std::cref(*left), std::cref(*right)})
										 .value()
										 .front()
										 .get();
			if (index() != nullptr) // the type of the result at the operator (for hovers)
				index()->reference(analysis::Reference{
						.file = currentFile(),
						.range = rangeOf(ctx->bop),
						.symbol = std::nullopt,
						.results = {typeString(getDataType(result))}
				});
			if (const auto* inv = std::get_if<InvocationNode>(&result))
				if (auto problem = shapes::check(inv->id, inv->arguments)) {
					auto* tok = ctx->bop;
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
							std::format("Operator {}: {}", tok->getText(), *problem)
					);
				} else if (auto possible = shapes::check(inv->id, inv->arguments, /*strict=*/true)) {
					reportPossibleMismatch(ctx->bop, std::format("Operator {}: {}", ctx->bop->getText(), *possible));
				}
			return &result;
		}

		virtual std::any visitNegateExpression(mdlParser::NegateExpressionContext* ctx) override {
			const auto* operand = asNode(ctx->expr()->accept(this), ctx->expr()->start);
			if (std::holds_alternative<ErrorNode>(*operand))
				return operand;
			// Fold negated literals so that, e.g., `-1` stays a compile-time constant.
			if (const auto* cn = std::get_if<ConstNode>(operand)) {
				if (const auto* i = std::get_if<int>(&cn->value))
					return &graph().addConstNode(Atomic{-*i});
				if (const auto* d = std::get_if<double>(&cn->value))
					return &graph().addConstNode(Atomic{-*d});
			}
			const Node& zero = graph().addConstNode(Atomic{0});
			return &graph().addInvocationNode(
					builtins::FunctionIDs::TENSOR_SUB, {std::cref(zero), std::cref(*operand)}
			);
		}

		virtual std::any visitConditionalExpression(mdlParser::ConditionalExpressionContext* ctx) override {
			// A compile-time choice: only the chosen branch is elaborated (it may use members that only exist then).
			const Node* cond = asNode(ctx->cond->accept(this), ctx->cond->start);
			if (std::holds_alternative<ErrorNode>(*cond))
				return static_cast<const Node*>(cond);
			auto value = foldConstant(*cond);
			const auto* i = value ? std::get_if<int>(&*value) : nullptr;
			const auto* d = value ? std::get_if<double>(&*value) : nullptr;
			if (i == nullptr && d == nullptr && isSymbolic(*cond)) {
				// Symbolic analysis: both branches are checked; the first one is the value. The first branch may
				// use the members that exist if the condition holds (@conditional).
				assumedConditions.push_back(ctx->cond->getText());
				auto result = ctx->then->accept(this);
				assumedConditions.pop_back();
				ctx->otherwise->accept(this);
				return result;
			}
			if (i == nullptr && d == nullptr) {
				auto* tok = ctx->cond->start;
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
						"The condition of `?:` must be a compile-time value (e.g. a hyperparameter)"
				);
				return static_cast<const Node*>(&graph().addErrorNode());
			}
			const bool chosen = i != nullptr ? *i != 0 : *d != 0.0;
			return (chosen ? ctx->then : ctx->otherwise)->accept(this);
		}

		virtual std::any visitArrayLiteralExpression(mdlParser::ArrayLiteralExpressionContext* ctx) override {
			ArrayInstance array;
			for (auto* element : ctx->elements)
				array.elements.push_back(asObject(element->accept(this), element->start));
			return Object{std::move(array)};
		}

		virtual std::any visitArrayType(mdlParser::ArrayTypeContext* ctx) override {
			auto element = std::any_cast<DataType>(ctx->element->accept(this));
			auto size = constantInt(ctx->size);
			if (size && *size < 0) {
				auto* tok = ctx->size->start;
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
						"The size of an array must not be negative"
				);
			}
			return DataType{ArrayType{
					.element = std::make_shared<const DataType>(std::move(element)),
					.size = static_cast<unsigned>(std::max(size.value_or(0), 0))
			}};
		}

		virtual std::any visitExpressionStatement(mdlParser::ExpressionStatementContext* ctx) override {
			ctx->expr()->accept(this);
			return std::nullopt;
		}

		/** @brief Whether @p name is `_`: a discarded value (not bound, no type, no members). */
		static bool isDiscard(const antlr4::Token* name) { return name != nullptr && name->getText() == "_"; }

		virtual std::any visitDiscardExpression(mdlParser::DiscardExpressionContext* ctx) override {
			auto* tok = ctx->start;
			errorlistener->semanticError(
					PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
					"`_` is a discarded value: it has no value, type, or members"
			);
			return static_cast<const Node*>(&graph().addErrorNode());
		}

		virtual std::any visitDeclarationStatement(mdlParser::DeclarationStatementContext* ctx) override {
			for (auto* name : ctx->names)
				if (isDiscard(name) && (ctx->type() != nullptr || ctx->expr() == nullptr)) {
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error, name, name->getLine(),
							name->getCharPositionInLine(),
							"`_` discards a value: it has no type (write `auto _ = ...;`)"
					);
					return std::nullopt;
				}
			if (ctx->expr() == nullptr) { // a tensor whose slots are assigned one by one
				auto* tok = ctx->names[0];
				if (ctx->type() == nullptr) {
					slotError(
							tok, "A declaration without initializer needs a tensor type (e.g. `float[n, ..., d] hs;`)"
					);
					return std::nullopt;
				}
				auto dtype = std::any_cast<DataType>(ctx->type()->accept(this));
				const auto* tensor = std::get_if<TensorType>(&dtype);
				std::optional<int> n;
				bool representative = false;
				if (tensor != nullptr && !tensor->shape.empty()) {
					if (const auto* u = std::get_if<unsigned>(&tensor->shape.front()))
						n = static_cast<int>(*u);
					else if (const auto* node = std::get_if<const Node*>(&tensor->shape.front())) {
						if (auto v = foldConstant(**node); v && std::holds_alternative<int>(*v))
							n = std::get<int>(*v);
						else if (isSymbolic(**node)) {
							n = 1; // symbolic analysis: one slot stands for all
							representative = true;
						}
					}
				}
				if (!n || *n <= 0) {
					slotError(
							tok, "A declaration without initializer needs a tensor type whose first dimension is a "
								 "compile-time size (the number of slots, e.g. `float[n, ..., d] hs;`)"
					);
					return std::nullopt;
				}
				for (auto* name : ctx->names) {
					currentNamespace().put(
							name->getText(),
							SlottedTensor{
									.type = *tensor,
									.slots = std::vector<const Node*>(*n, nullptr),
									.representative = representative
							}
					);
					defineHere(
							name, analysis::SymbolKind::variable, sourceText(ctx), nullptr,
							rangeOf(enclosingBlock(ctx)), {typeString(*tensor)}
					);
				}
				return std::nullopt;
			}
			/** \todo check datatype with ctx->type() **/
			auto value = ctx->expr()->accept(this);
			if (ctx->names.size() == 1 && isDiscard(ctx->names[0]))
				return std::nullopt; // evaluated (and checked), then discarded
			if (ctx->names.size() == 1) {
				const auto& object = currentNamespace().put(
						ctx->names[0]->getText(), asObject(std::move(value), ctx->expr()->start)
				);
				defineHere(
						ctx->names[0], analysis::SymbolKind::variable, sourceText(ctx), nullptr,
						rangeOf(enclosingBlock(ctx)), {objectType(object)}
				);
				return std::nullopt;
			}
			// Destructuring: `auto a, b = f(x);`
			std::vector<const Node*> nodes;
			if (auto* multi = std::any_cast<MultiValue>(&value))
				nodes = multi->nodes;
			else
				nodes.push_back(asNode(std::move(value), ctx->expr()->start));
			if (nodes.size() != ctx->names.size()) {
				auto* tok = ctx->expr()->start;
				if (!(nodes.size() == 1 && std::holds_alternative<ErrorNode>(*nodes[0])))
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
							std::format(
									"Expected {} values but the expression yields {}", ctx->names.size(), nodes.size()
							)
					);
				nodes.assign(ctx->names.size(), &graph().addErrorNode());
			}
			for (size_t i = 0; i < nodes.size(); ++i) {
				if (isDiscard(ctx->names[i]))
					continue;
				currentNamespace().put(ctx->names[i]->getText(), Object{nodes[i]});
				defineHere(
						ctx->names[i], analysis::SymbolKind::variable, sourceText(ctx), nullptr,
						rangeOf(enclosingBlock(ctx)), {objectType(Object{nodes[i]})}
				);
			}
			return std::nullopt;
		}

		virtual std::any visitAssignStatement(mdlParser::AssignStatementContext* ctx) override {
			// Weights, submodules, hyperparameters, and loop variables are read-only (only variables are rebound).
			if (auto id = currentNamespace().symbol(ctx->name->getText()); id && index() != nullptr) {
				const auto& s = index()->symbols()[*id];
				const char* what = s.kind == analysis::SymbolKind::member			? "a member of the module"
								   : s.kind == analysis::SymbolKind::hyperparameter ? "a hyperparameter"
								   : s.kind == analysis::SymbolKind::loopVariable	? "a loop variable"
																					: nullptr;
				if (what != nullptr) {
					ctx->expr()->accept(this); // (still checked)
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error, ctx->name, ctx->name->getLine(),
							ctx->name->getCharPositionInLine(),
							std::format("{} is {} and cannot be assigned", ctx->name->getText(), what),
							std::format("declare a variable: `auto {}2 = ...;`", ctx->name->getText())
					);
					return std::nullopt;
				}
			}
			auto value = asObject(ctx->expr()->accept(this), ctx->expr()->start);
			referenceHere(ctx->name, value);
			if (!currentNamespace().assign(ctx->name->getText(), std::move(value)))
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, ctx->name, ctx->name->getLine(),
						ctx->name->getCharPositionInLine(),
						std::format("Variable {} is not defined", ctx->name->getText()),
						std::format("declare it with `auto {} = ...;`", ctx->name->getText())
				);
			return std::nullopt;
		}

		virtual std::any visitSlotAssignStatement(mdlParser::SlotAssignStatementContext* ctx) override {
			const std::string name = ctx->name->getText();
			auto* tensor = currentNamespace().find<SlottedTensor>(name);
			if (tensor == nullptr) {
				slotError(
						ctx->name, currentNamespace().get<Object>(name)
										   ? std::format(
													 "`{}[i] = ...` assigns a slot of a tensor declared without "
													 "initializer (e.g. `float[n, ..., d] {};`)",
													 name, name
											 )
										   : std::format("Variable {} is not defined", name)
				);
				return std::nullopt;
			}
			auto index = constantInt(ctx->index, 0);
			const Node* value = asNode(ctx->value->accept(this), ctx->value->start);
			if (!index)
				return std::nullopt;
			const int n = static_cast<int>(tensor->slots.size());
			const int i = tensor->representative ? 0 : *index < 0 ? *index + n : *index;
			if (i < 0 || i >= n) {
				slotError(
						ctx->index->start, std::format("Index {} is out of bounds ({} has {} slots)", *index, name, n)
				);
				return std::nullopt;
			}
			// An erroneous value (already reported) counts as assigned, so that its uses are not reported again.
			if (std::holds_alternative<ErrorNode>(*value)) {
				tensor->slots[i] = value;
				return std::nullopt;
			}
			const TensorType type = slotType(*tensor);
			if (isFloat(type.type))
				value = implicitConversion(
						*value, type.type, ctx->value->start, std::format("Slot {}[{}]", name, i), ctx->value
				);
			if (value == nullptr) {
				tensor->slots[i] = &graph().addErrorNode(); // (reported)
				return std::nullopt;
			}
			if (shapes::matches(type, getDataType(*value)) &&
				!shapes::matches(type, getDataType(*value), true, /*strict=*/true))
				reportPossibleMismatch(
						ctx->value->start,
						std::format(
								"{}[{}] is assigned {}, but the slots of {} are {} (the sizes may differ)", name, i,
								shapes::str(getDataType(*value)), name, shapes::str(type)
						)
				);
			if (!shapes::matches(type, getDataType(*value))) {
				tensor->slots[i] = slotError(
						ctx->value->start, std::format(
												   "{}[{}] is assigned {}, but the slots of {} are {}", name, i,
												   shapes::str(getDataType(*value)), name, shapes::str(type)
										   )
				);
				return std::nullopt;
			}
			tensor->slots[i] = value;
			return std::nullopt;
		}

		virtual std::any visitForStatement(mdlParser::ForStatementContext* ctx) override {
			auto lo = constantInt(ctx->lo, 0), hi = constantInt(ctx->hi);
			if (!lo || !hi)
				return std::nullopt;
			if (*hi == unknownInt)
				hi = *lo + 1; // symbolic analysis: one iteration stands for all
			for (int i = *lo; i < *hi; ++i) {
				pushNamespace();
				if (!isDiscard(ctx->var)) {
					currentNamespace().put(ctx->var->getText(), Object{&graph().addConstNode(Atomic{i})});
					defineHere(
							ctx->var, analysis::SymbolKind::loopVariable,
							ctx->var->getText() + " in " + sourceText(ctx->lo) + ".." + sourceText(ctx->hi), nullptr,
							rangeOf(ctx), {"int"}
					);
				}
				for (auto* stmt : ctx->statement()) {
					auto result = stmt->accept(this);
					if (std::any_cast<MultiValue>(&result) != nullptr)
						errorlistener->semanticError(
								PrettyErrorListener::ErrorLevel::Error, stmt->start, stmt->start->getLine(),
								stmt->start->getCharPositionInLine(), "Return statements are not allowed in loops"
						);
				}
				popNamespace();
			}
			return std::nullopt;
		}

		virtual std::any visitIndexExpression(mdlParser::IndexExpressionContext* ctx) override {
			// A slot of a tensor declared without initializer: the assigned node.
			if (auto* var = dynamic_cast<mdlParser::VarAccessExpressionContext*>(ctx->base);
				var && ctx->indices.size() == 1)
				if (const auto* tensor = currentNamespace().find<SlottedTensor>(var->IDENT()->getText()))
					if (auto* value = dynamic_cast<mdlParser::IndexValueContext*>(ctx->indices[0])) {
						auto index = constantInt(value->expr(), 0);
						if (!index)
							return static_cast<const Node*>(&graph().addErrorNode());
						return slot(*tensor, var->IDENT()->getText(), *index, value->start);
					}
			auto base = asObject(ctx->base->accept(this), ctx->base->start);
			if (const auto* node = std::get_if<const Node*>(&base); node && std::holds_alternative<ErrorNode>(**node))
				return &graph().addErrorNode();
			auto error = [&](antlr4::Token* tok, const std::string& message) -> std::any {
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
						message
				);
				return &graph().addErrorNode();
			};

			if (const auto* array = std::get_if<ArrayInstance>(&base)) {
				auto* value = ctx->indices.size() == 1 ? dynamic_cast<mdlParser::IndexValueContext*>(ctx->indices[0])
													   : nullptr;
				if (value == nullptr)
					return error(ctx->start, "An array takes one index");
				auto index = constantInt(value->expr(), 0);
				if (!index)
					return &graph().addErrorNode();
				if (array->representative && !array->elements.empty())
					return array->elements.front();
				if (*index < 0 || *index >= static_cast<int>(array->elements.size()))
					return error(
							value->start,
							std::format(
									"Index {} is out of bounds (array of {} elements)", *index, array->elements.size()
							)
					);
				return array->elements[*index];
			}

			// Tensors: integers select (and remove) an axis, `:` keeps it, `...` stands for the axes not listed.
			const auto* node = std::get_if<const Node*>(&base);
			if (node == nullptr || !std::holds_alternative<TensorType>(mdl::getDataType(**node)))
				return error(ctx->base->start, "Only arrays and tensors can be indexed");

			// An integer tensor as the index selects rows: `table[ids]` is [..ids.., rest of table] (like NumPy).
			for (size_t i = 0; i < ctx->indices.size(); ++i) {
				auto* value = dynamic_cast<mdlParser::IndexValueContext*>(ctx->indices[i]);
				if (value == nullptr)
					continue;
				const Node* index = asNode(value->expr()->accept(this), value->expr()->start);
				const auto* indexType = std::get_if<TensorType>(&getDataType(*index));
				if (indexType == nullptr)
					continue; // a compile-time integer (below)
				if (indexType->type != PrimitiveType::integer && indexType->type != PrimitiveType::uint)
					return error(
							value->start, std::format(
												  "A tensor index must be an integer tensor, but it is {}",
												  shapes::str(getDataType(*index))
										  )
					);
				if (ctx->indices.size() != 1)
					return error(
							value->start, "An integer tensor index selects rows: it must be the only index "
										  "(`table[ids]`)"
					);
				return &graph().addInvocationNode(builtins::FunctionIDs::TAKE, {std::cref(**node), std::cref(*index)});
			}
			std::vector<std::pair<int, int>> leading, trailing; // (axis, index); trailing axes count from the end
			std::optional<size_t> ellipsis;
			for (size_t i = 0; i < ctx->indices.size(); ++i) {
				if (dynamic_cast<mdlParser::IndexEllipsisContext*>(ctx->indices[i])) {
					if (ellipsis)
						return error(ctx->indices[i]->start, "An index has at most one `...`");
					ellipsis = i;
				}
			}
			const size_t count = ctx->indices.size();
			for (size_t i = 0; i < count; ++i) {
				auto* value = dynamic_cast<mdlParser::IndexValueContext*>(ctx->indices[i]);
				if (value == nullptr)
					continue; // `:` or `...`
				auto index = constantInt(value->expr(), 0);
				if (!index)
					return &graph().addErrorNode();
				if (ellipsis && i > *ellipsis)
					trailing.emplace_back(static_cast<int>(i) - static_cast<int>(count), *index);
				else
					leading.emplace_back(static_cast<int>(i), *index);
			}
			// Remove the leading axes from the back and the trailing ones from the front, so that the axis numbers of
			// the remaining selections stay valid.
			// (Inlined from the builtin function, so that the result gets its type from the type rules.)
			auto builtin = currentNamespace().get<Object>("__builtin_select");
			const auto* selectDecl = builtin ? std::get_if<FunctionDecl>(&builtin->get()) : nullptr;
			if (selectDecl == nullptr)
				return error(ctx->start, "internal error: __builtin_select is not defined");
			const Node* result = *node;
			auto select = [&](int axis, int index) {
				auto outputs = graph().addInvocationNode(
						*selectDecl, {std::cref(*result), std::cref(graph().addConstNode(Atomic{axis})),
									  std::cref(graph().addConstNode(Atomic{index}))}
				);
				result = outputs ? &outputs->front().get() : &graph().addErrorNode();
			};
			for (auto it = leading.rbegin(); it != leading.rend(); ++it)
				select(it->first, it->second);
			for (const auto& [axis, index] : trailing)
				select(axis, index);
			return result;
		}

		virtual std::any visitReturnStatement(mdlParser::ReturnStatementContext* ctx) override {
			MultiValue result;
			for (auto* value : ctx->values)
				result.nodes.push_back(asNode(value->accept(this), value->start));
			return result;
		}
	};
} // namespace mdl

std::vector<std::filesystem::path> mdl::parser::defaultStdlibDirs() {
	std::vector<std::filesystem::path> dirs;
#if defined(_WIN32)
	constexpr char separator = ';';
#else
	constexpr char separator = ':';
#endif
	if (const char* env = std::getenv("MDL_STDLIB_PATH")) {
		std::string_view paths{env};
		while (!paths.empty()) {
			auto sep = paths.find(separator);
			if (auto dir = paths.substr(0, sep); !dir.empty())
				dirs.emplace_back(dir);
			paths = (sep == std::string_view::npos) ? std::string_view{} : paths.substr(sep + 1);
		}
	}
	// An installation: <prefix>/bin/<tool> and <prefix>/share/mdl/std.
	if (auto exe = utils::executablePath(); !exe.empty()) {
		std::error_code ec;
		auto installed = exe.parent_path().parent_path() / "share" / "mdl" / "std";
		if (std::filesystem::is_directory(installed, ec))
			dirs.push_back(std::move(installed));
	}
#ifdef MDL_DEFAULT_STDLIB_DIR
	dirs.emplace_back(MDL_DEFAULT_STDLIB_DIR); // the source tree (running from the build tree)
#endif
	return dirs;
}

namespace {
	/**
	 * @brief A frontend state whose diagnostics go to @p sink (stderr with a summary per file if empty). The compiler
	 * and tools run the same checks (also of generic modules, and of unused definitions, which need the index).
	 */
	std::shared_ptr<mdl::parser::FrontendState>
	makeState(const mdl::parser::ParseOptions& options, mdl::DiagnosticSink sink) {
		mdl::shapes::setDefaultFloat(options.defaultFloat);
		auto state = std::make_shared<mdl::parser::FrontendState>();
		state->index = std::make_unique<mdl::analysis::SemanticIndex>();
		state->printing = !sink;
		state->sink = [s = state.get(), forward = std::move(sink)](const mdl::Diagnostic& d) {
			s->diagnostics.push_back(d);
			if (forward)
				forward(d);
			else
				mdl::printDiagnostic(std::cerr, d);
		};
		state->visitor = std::make_unique<mdl::TranslatingVisitor>(*state, options);
		return state;
	}

	mdl::CompilationUnit makeUnit(std::shared_ptr<mdl::parser::FrontendState> state) {
		auto& visitor = *state->visitor;
		mdl::CompilationUnit unit;
		unit.definitions = std::move(visitor.getDefinitions());
		unit.sources = std::move(visitor.getSources());
		unit.state = std::move(state);
		return unit;
	}

	std::optional<mdl::CompilationUnit> finish(std::shared_ptr<mdl::parser::FrontendState> state) {
		if (state->visitor->getNumErrors() > 0)
			return std::nullopt;
		return makeUnit(std::move(state));
	}

	mdl::parser::AnalysisResult finishAnalysis(std::shared_ptr<mdl::parser::FrontendState> state) {
		const bool ok = state->visitor->getNumErrors() == 0;
		auto diagnostics = state->diagnostics;
		auto index = std::shared_ptr<const mdl::analysis::SemanticIndex>(state, state->index.get());
		return {.unit = makeUnit(std::move(state)), .diagnostics = std::move(diagnostics), .ok = ok, .index = index};
	}

	std::shared_ptr<mdl::parser::FrontendState> translateStream(
			std::istream& stream, std::string_view sourcename, const mdl::parser::ParseOptions& options,
			mdl::DiagnosticSink sink
	) {
		auto state = makeState(options, std::move(sink));
		state->visitor->translate(
				stream, std::string{sourcename}, options.sourceDir.value_or(std::filesystem::current_path()),
				/*main=*/true
		);
		return state;
	}

	std::shared_ptr<mdl::parser::FrontendState> translatePath(
			const std::filesystem::path& file, const mdl::parser::ParseOptions& options, mdl::DiagnosticSink sink
	) {
		auto state = makeState(options, std::move(sink));
		state->visitor->translateFile(file, /*main=*/true);
		return state;
	}

	/** @brief Collects the diagnostics without printing them (and forwards them to the options' sink, if any). */
	mdl::DiagnosticSink quiet(const mdl::parser::ParseOptions& options) {
		return options.diagnostics ? options.diagnostics : [](const mdl::Diagnostic&) {};
	}
} // namespace

std::optional<mdl::CompilationUnit>
mdl::parser::parse(std::istream& stream, std::string_view sourcename, const ParseOptions& options) {
	return finish(translateStream(stream, sourcename, options, options.diagnostics));
}

std::optional<mdl::CompilationUnit> mdl::parser::parse(const std::filesystem::path& file, const ParseOptions& options) {
	return finish(translatePath(file, options, options.diagnostics));
}

mdl::parser::AnalysisResult
mdl::parser::analyze(std::istream& stream, std::string_view sourcename, const ParseOptions& options) {
	return finishAnalysis(translateStream(stream, sourcename, options, quiet(options)));
}

mdl::parser::AnalysisResult mdl::parser::analyze(const std::filesystem::path& file, const ParseOptions& options) {
	return finishAnalysis(translatePath(file, options, quiet(options)));
}

mdl::Computegraph& mdl::CompilationUnit::graph() const { return state->graph; }

const mdl::FunctionDecl& mdl::CompilationUnit::keep(FunctionDecl decl) const {
	return state->derivedFunctions.emplace_back(std::move(decl));
}

const mdl::FunctionDecl&
mdl::CompilationUnit::specializeMasked(const FunctionDecl& decl, const std::vector<bool>& maskedParams) const {
	return state->visitor->specializeMasked(decl, maskedParams);
}

std::expected<std::reference_wrapper<const mdl::ModuleType>, std::string>
mdl::CompilationUnit::instantiate(const std::string& module, const std::map<std::string, Object>& values) const {
	return state->visitor->instantiateByName(module, values);
}
