#include <mdl/parser.hpp>

#include "./parser/errorlistener.hpp"

#include <mdl/computegraphv2/function.hpp>
#include <mdl/computegraphv2/object.hpp>
#include <mdl/namespace.hpp>
#include <mdl/passes/shapes.hpp>
#include <mdl/utils/einops.hpp>

#include <mdlBaseVisitor.h>
#include <mdlLexer.h>
#include <mdlParser.h>

#include <cassert>
#include <format>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <set>
#include <string_view>
#include <functional>
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

	struct parser::FrontendState {
		Computegraph graph;
		std::vector<std::unique_ptr<ParsedFile>> files;
		std::deque<FunctionDecl> derivedFunctions; ///< e.g. gradient functions (stable references)
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
		/** @brief The error listener of the file that is currently translated. */
		PrettyErrorListener* errorlistener = nullptr;
		/** @brief Directory of the file that is currently translated (first search path for its imports). */
		std::filesystem::path currentDir;
		/** @brief Whether definitions of the current file are added to the compilation unit (false for imports). */
		bool recordDefinitions = true;
		/** @brief Canonical paths of all files that have been (or are being) translated. */
		std::set<std::filesystem::path> translatedFiles;
		size_t numErrors = 0;

		std::stack<mdl::Computegraph*> graphstack;
		std::stack<mdl::Namespace> nsstack;
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
									"Expected a single value but the expression yields {} values; use `auto a, b = ...;`",
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

		/**
		 * @brief Searches the file or directory that `import a.b;` refers to.
		 * @details Searched are the directory of the importing file and the import directories. Imports of the
		 * standard library (`std` or `std.x`) are additionally resolved in the standard library directories, which
		 * contain the files of the `std` package itself.
		 */
		std::vector<std::filesystem::path> resolveImport(const std::filesystem::path& relative) const {
			std::vector<std::pair<std::filesystem::path, std::filesystem::path>> candidates; // (dir, relative path)
			candidates.emplace_back(currentDir, relative);
			for (const auto& dir : options.importDirs)
				candidates.emplace_back(dir, relative);
			if (!relative.empty() && *relative.begin() == "std") {
				std::filesystem::path rest;
				for (auto it = std::next(relative.begin()); it != relative.end(); ++it)
					rest /= *it;
				for (const auto& dir : options.stdlibDirs.empty() ? parser::defaultStdlibDirs() : options.stdlibDirs)
					candidates.emplace_back(dir, rest);
			}
			auto isSource = [](const std::filesystem::path& p) {
				return std::filesystem::is_regular_file(p) && (p.extension() == ".dlm" || p.extension() == ".mdl");
			};
			for (const auto& [dir, rel] : candidates) {
				if (!rel.empty()) {
					for (const char* ext : {".dlm", ".mdl"}) {
						auto file = dir / rel;
						file += ext;
						if (isSource(file))
							return {file};
					}
				}
				auto path = rel.empty() ? dir : dir / rel;
				if (std::filesystem::is_directory(path)) {
					std::vector<std::filesystem::path> files;
					for (const auto& entry : std::filesystem::directory_iterator(path))
						if (isSource(entry.path()))
							files.push_back(entry.path());
					std::sort(files.begin(), files.end());
					if (!files.empty())
						return files;
				}
			}
			return {};
		}

		void importDefinitions(mdlParser::Import_stmtContext* ctx) {
			std::filesystem::path relative;
			std::string dotted;
			for (auto* part : ctx->path) {
				relative /= part->getText();
				dotted += (dotted.empty() ? "" : ".") + part->getText();
			}
			auto files = resolveImport(relative);
			if (files.empty()) {
				auto* tok = ctx->start;
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
						std::format("Cannot find {} (searched for {}.dlm or a directory {})", dotted,
									relative.string(), relative.string())
				);
				return;
			}
			for (const auto& file : files)
				translateFile(file, /*main=*/false);
		}

	public:
		TranslatingVisitor(parser::FrontendState& state, parser::ParseOptions options)
				: state(state), options(std::move(options)) {
			pushGraph(&state.graph);
			nsstack.push(mdl::Namespace::baseNamespace());
		}

		size_t getNumErrors() const { return numErrors; }
		std::map<std::string, CompilationUnit::Member>& getDefinitions() { return definitions; }
		std::vector<std::filesystem::path>& getSources() { return sources; }

		/** @brief Instantiates the module named @p name (see CompilationUnit::instantiate). */
		std::expected<std::reference_wrapper<const ModuleType>, std::string>
		instantiateByName(const std::string& name, const std::map<std::string, Object>& values) {
			auto type = nsstack.top().getType(name);
			const auto* mod = type ? std::get_if<ModuleType>(&type.value().get()) : nullptr;
			if (mod == nullptr || mod->templ == nullptr)
				return std::unexpected(std::format("{} is not a module", name));
			std::vector<Object> ordered;
			for (const auto& [pname, _] : mod->typeargs) {
				auto it = values.find(pname);
				if (it == values.end())
					return std::unexpected(std::format("no value for hyperparameter {}.{}", name, pname));
				ordered.push_back(it->second);
			}
			return instantiate(*mod, ordered);
		}

		/**
		 * @brief Translates the program read from @p stream into the compilation unit.
		 * @param main whether the definitions are added to CompilationUnit::definitions (false for imported files).
		 */
		void translate(std::istream& stream, const std::string& sourcename, const std::filesystem::path& dir, bool main) {
			// The parse tree stays alive (owned by the frontend state) so that module templates defined in this file
			// can be expanded later.
			auto& file = *state.files.emplace_back(std::make_unique<ParsedFile>());
			file.errors = std::make_unique<PrettyErrorListener>(sourcename);
			file.input = std::make_unique<antlr4::ANTLRInputStream>(stream);
			file.lexer = std::make_unique<antlr4::mdlLexer>(file.input.get());
			file.tokens = std::make_unique<antlr4::CommonTokenStream>(file.lexer.get());
			file.parser = std::make_unique<antlr4::mdlParser>(file.tokens.get());
			file.parser->removeErrorListeners();
			file.parser->addErrorListener(file.errors.get());

			auto* outerErrors = std::exchange(errorlistener, file.errors.get());
			auto outerDir = std::exchange(currentDir, dir);
			bool outerRecord = std::exchange(recordDefinitions, main);
			const size_t errorsBefore = file.errors->getNumErrors();
			auto* program = file.parser->program();
			// A tree with syntax errors has missing children (null contexts): only report the syntax errors.
			if (file.parser->getNumberOfSyntaxErrors() == 0)
				program->accept(this);
			errorlistener = outerErrors;
			currentDir = std::move(outerDir);
			recordDefinitions = outerRecord;

			file.errors->printSummary();
			numErrors += file.errors->getNumErrors() - errorsBefore;
		}

		void translateFile(const std::filesystem::path& file, bool main) {
			std::error_code ec;
			auto canonical = std::filesystem::weakly_canonical(file, ec);
			if (!translatedFiles.insert(ec ? file : canonical).second)
				return; // Already imported (or currently being imported).
			sources.push_back(ec ? file : canonical);
			std::ifstream stream{file};
			if (!stream) {
				std::cerr << "error: cannot read " << file.string() << std::endl;
				++numErrors;
				return;
			}
			translate(stream, std::filesystem::absolute(file).string(), file.parent_path(), main);
		}

		virtual std::any visitProgram(mdlParser::ProgramContext* ctx) override {
			// Imports are processed first so that their definitions are visible to the whole file.
			for (auto* import : ctx->import_stmt())
				importDefinitions(import);
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
			for (auto&& def : ctx->module_def()) {
				auto [name, m] = std::any_cast<std::pair<std::string, ModuleType>>(def->accept(this));
				record(std::move(name), std::move(m));
			}
			return {};
		}

		void updateTypeWithVarDecl(
				AggregateType& type, const std::string& tname, std::vector<antlr4::mdlParser::VardeclContext*> vars
		) {
			for (auto& var : vars) {
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
			if (!checkRedefinition(ctx->IDENT()))
				currentNamespace().putType(name, DataType{s});
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
			if (m.typeargs.empty())
				if (auto instance = instantiate(m, {}))
					m = instance.value().get();
			if (!checkRedefinition(ctx->IDENT()))
				currentNamespace().putType(name, DataType{m});
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

		/** @brief Evaluates a compile-time integer (e.g. a loop bound or an index); reports an error otherwise. */
		std::optional<int> constantInt(mdlParser::ExprContext* expr) {
			const Node* node = asNode(expr->accept(this), expr->start);
			if (std::holds_alternative<ErrorNode>(*node))
				return std::nullopt;
			if (auto value = foldConstant(*node); value && std::holds_alternative<int>(*value))
				return std::get<int>(*value);
			errorlistener->semanticError(
					PrettyErrorListener::ErrorLevel::Error, expr->start, expr->start->getLine(),
					expr->start->getCharPositionInLine(), "Expected a compile-time integer"
			);
			return std::nullopt;
		}

		/** @brief A string identifying a (concrete) hyperparameter value; used to cache module instances. */
		static std::string valueKey(const Object& value) {
			return std::visit(
					overloaded{
							[](const Node* node) -> std::string {
								if (auto v = foldConstant(*node))
									return std::visit([](auto x) { return std::format("{}", x); }, *v);
								return "?";
							},
							[](const AggregateInstance& agg) -> std::string {
								std::string key = "{";
								for (const auto& member : agg.getMembers())
									key += valueKey(member) + ",";
								return key + "}";
							},
							[](const auto&) -> std::string { return "?"; },
					},
					value
			);
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

			const size_t errorsBefore = t.errors->getNumErrors();
			const std::string outerScope = std::exchange(scope, key);
			auto* outerErrors = std::exchange(errorlistener, t.errors);
			nsstack.push(t.scope->createSubnamespace());
			for (size_t i = 0; i < values.size() && i < def.typeargs.size(); ++i)
				currentNamespace().put(def.typeargs[i].first, Object{values[i]});

			for (auto* var : t.ctx->vardecl())
				declareMember(m, var);
			for (auto& func : t.ctx->funcdef()) {
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
			const size_t newErrors = t.errors->getNumErrors() - errorsBefore;
			numErrors += newErrors;
			if (newErrors > 0)
				return std::unexpected(std::format("{} error(s) while instantiating {}", newErrors, key));
			return std::cref(instances.emplace(std::pair{&t, key}, std::move(m)).first->second);
		}

		/** @brief Declares a member variable (weight, submodule, or array of them) of a module being expanded. */
		void declareMember(ModuleType& m, mdlParser::VardeclContext* var) {
			const auto vname = var->IDENT()->getText();
			auto addVariable = [&](const std::string& path, const DataType& dtype) -> const Node& {
				// Weights are registered in the instance's scope so that all functions of the module share them.
				const Node& node = graph().getOrAddParameterNode(scope, path, dtype);
				m.variables.emplace_back(path, std::cref(std::get<ParameterNode>(node)));
				return node;
			};

			if (auto* comp = dynamic_cast<mdlParser::ComprehensionTypeContext*>(var->type())) {
				// `name: [T for i in lo..hi]` declares the members name.lo, ..., name.(hi-1).
				auto lo = constantInt(comp->lo), hi = constantInt(comp->hi);
				ArrayInstance array;
				for (int i = lo.value_or(0); lo && hi && i < *hi; ++i) {
					pushNamespace();
					currentNamespace().put(comp->var->getText(), Object{&graph().addConstNode(Atomic{i})});
					auto dtype = std::any_cast<DataType>(comp->element->accept(this));
					popNamespace();
					array.elements.push_back(Object{&addVariable(vname + "." + std::to_string(i), dtype)});
				}
				currentNamespace().put(vname, Object{std::move(array)});
				return;
			}

			const auto [_, dtype] = std::any_cast<VarDecl>(var->accept(this));
			const Node& node = addVariable(vname, dtype);
			if (auto* agg = std::get_if<AggregateType>(&dtype))
				currentNamespace().put(
						vname, makeAggregateObject(*agg, vname, [&](DataType t, std::string path) -> const Node& {
							return graph().getOrAddParameterNode(scope, path, std::move(t));
						})
				);
			else
				currentNamespace().put(vname, Object{&node});
		}

		virtual std::any visitParam(mdlParser::ParamContext* ctx) override {
			if (ctx->default_ != nullptr) {
				auto firsttok = ctx->default_->start;
				auto lasttok = ctx->default_->stop;
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error,
						{{.linenr = firsttok->getLine(),
						  .line = errorlistener->getline(*firsttok->getInputStream(), 0, firsttok->getStartIndex()),
						  .beginMark = firsttok->getCharPositionInLine(),
						  .markWidth = lasttok->getStopIndex() - firsttok->getStartIndex() + 1}},
						firsttok->getLine(), firsttok->getCharPositionInLine(),
						"Default values have not yet been implemented"
				);
			}
			return std::make_pair(ctx->IDENT()->getText(), std::any_cast<mdl::DataType>(ctx->type()->accept(this)));
		}

		virtual std::any visitTypeSpec(mdlParser::TypeSpecContext* ctx) override {
			auto dtype = std::any_cast<DataType>(unqualifiedTypeSpec(ctx));
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
			auto dtype = currentNamespace().getType(ctx->typename_->getText());
			if (!dtype.has_value()) {
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, ctx->typename_, ctx->typename_->getLine(),
						ctx->typename_->getCharPositionInLine(),
						std::format("{} does not name a type", ctx->typename_->getText())
				);
				return mdl::DataType{mdl::PrimitiveType::error};
			} else {
				if (const auto* mod = std::get_if<ModuleType>(&dtype.value().get()); mod && mod->templ) {
					// Module instantiation: the arguments must be compile-time values.
					auto* tok = ctx->typename_;
					if (ctx->type_arg().size() != mod->typeargs.size()) {
						errorlistener->semanticError(
								PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
								std::format(
										"{} expects {} hyperparameter(s) but got {}", tok->getText(), mod->typeargs.size(),
										ctx->type_arg().size()
								)
						);
						return DataType{PrimitiveType::error};
					}
					std::vector<Object> values;
					for (auto* arg : ctx->type_arg()) {
						if (arg->expr() == nullptr) {
							errorlistener->semanticError(
									PrettyErrorListener::ErrorLevel::Error, arg->start, arg->start->getLine(),
									arg->start->getCharPositionInLine(), "`...` is not a module hyperparameter"
							);
							return DataType{PrimitiveType::error};
						}
						Object value = asObject(arg->expr()->accept(this), arg->expr()->start);
						if (const auto* node = std::get_if<const Node*>(&value)) {
							auto folded = foldConstant(**node);
							if (!folded) {
								if (!std::holds_alternative<ErrorNode>(**node))
									errorlistener->semanticError(
											PrettyErrorListener::ErrorLevel::Error, arg->start, arg->start->getLine(),
											arg->start->getCharPositionInLine(),
											"Module hyperparameters must be compile-time constants"
									);
								return DataType{PrimitiveType::error};
							}
							value = Object{&graph().addConstNode(Atomic{*folded})};
						}
						values.push_back(std::move(value));
					}
					auto instance = instantiate(*mod, values);
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
								{{.linenr = ctx->typename_->getLine(),
								  .line = errorlistener->getline(
										  *firsttok->getInputStream(), 0, firsttok->getStartIndex()
								  ),
								  .beginMark = firsttok->getCharPositionInLine(),
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
							{{.linenr = firsttok->getLine(),
							  .line = errorlistener->getline(*firsttok->getInputStream(), 0, firsttok->getStartIndex()),
							  .beginMark = firsttok->getCharPositionInLine(),
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
			assert(false);
		}

		virtual std::any visitConstant(mdlParser::ConstantContext* ctx) override {
			if (ctx->INTEGER() != nullptr)
				return mdl::Atomic{std::atoi(ctx->INTEGER()->getText().c_str())};
			if (ctx->FLOAT() != nullptr)
				return mdl::Atomic{std::atof(ctx->FLOAT()->getText().c_str())};
			if (ctx->BOOL() != nullptr)
				return mdl::Atomic(ctx->BOOL()->getText() == "true");
			assert(false); // Invalid case
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
					auto [name, dtype] = std::any_cast<VarDecl>(decl->accept(this));
					if (auto value = boundTemplate.find(name); value != boundTemplate.end()) {
						// Specialized for a call: the template parameter is a constant in the body.
						currentNamespace().put(name, Object{&graph().addConstNode(Atomic{value->second})});
						continue;
					}
					auto& p = currentNamespace().put(name, Object{&graph().addHyperparameterNode(dtype, name)});
					tparams.emplace_back(name, std::get<mdl::HyperparameterNode>(*std::get<const mdl::Node*>(p)));
				}
			}
			for (size_t i = 0; i < ctx->param().size(); ++i) {
				auto [name, dtype] = std::any_cast<VarDecl>(ctx->param()[i]->accept(this));
				if (auto* tensor = std::get_if<TensorType>(&dtype); tensor && specialization && i < specialization->size())
					tensor->masked |= (*specialization)[i];
				auto& p = currentNamespace().put(name, Object{&graph().addInputNode(dtype, name)});
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
			// The results must have the declared types.
			for (size_t i = 0; i < std::min(decl.returnTypes.size(), decl.outputs.size()); ++i) {
				const DataType declared = std::visit([](const auto& t) -> DataType { return t; }, decl.returnTypes[i]);
				if (!shapes::matches(declared, getDataType(decl.outputs[i].get()))) {
					auto* tok = ctx->IDENT()->getSymbol();
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
							std::format(
									"{} returns {}, but its declared return type is {}", decl.name,
									shapes::str(getDataType(decl.outputs[i].get())), shapes::str(declared)
							)
					);
				}
			}
			if (specialization)
				return std::make_pair(ctx->IDENT()->getText(), std::move(decl)); // the caller sets the template
			decl.templ = std::make_shared<FunctionTemplate>(
					FunctionTemplate{.ctx = ctx, .scope = &currentNamespace(), .weightScope = scope, .errors = errorlistener}
			);
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
			if (!currentNamespace().definesLocally(name)) {
				currentNamespace().put<mdl::Object>(name, Object{decl});
				return;
			}
			auto existing = currentNamespace().get<Object>(name);
			const auto* function = existing ? std::get_if<FunctionDecl>(&existing->get()) : nullptr;
			if (function == nullptr) {
				checkRedefinition(ident);
				return;
			}
			auto sameSignature = [&](const FunctionDecl& other) {
				return other.parameters.size() == decl.parameters.size() && maskedSignature(other) == maskedSignature(decl);
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
		static std::map<std::string, int> inferTemplateValues(const FunctionDecl& decl, const std::vector<DataType>& argtypes) {
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
				const FunctionDecl& decl, const std::vector<bool>& maskedPattern, const std::map<std::string, int>& values = {}
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
			auto value = currentNamespace().get<Object>(ctx->IDENT()->getText());
			if (!value.has_value()) {
				auto* sym = ctx->IDENT()->getSymbol();
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, sym, sym->getLine(), sym->getCharPositionInLine(),
						std::format("Variable {} not defined", ctx->IDENT()->getText())
				);
				return &graph().addErrorNode();
			}
			return value.value().get(); // Returns Object as std::any
		}
		virtual std::any visitConstantExpression(mdlParser::ConstantExpressionContext* ctx) override {
			return &graph().addConstNode(std::any_cast<mdl::Atomic>(ctx->constant()->accept(this)));
		}
		virtual std::any visitParenExpression(mdlParser::ParenExpressionContext* ctx) override {
			return ctx->expr()->accept(this);
		}

		virtual std::any visitAccessExpression(mdlParser::AccessExpressionContext* ctx) override {
			/** \todo this function needs refactoring; e.g., merge the handling of AggregateInstance, ModuleInstance, TensorType, ... to simple getMember lookups **/
			auto baseObj = asObject(ctx->expr()->accept(this));
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
			} else if (
					auto* nodePtr = std::get_if<const Node*>(&baseObj)
			) { // Module function access: node of BoundModuleType.
				if (std::holds_alternative<mdl::ErrorNode>(**nodePtr)) {
					// Don't show too many errors; if we have <error node>.IDENT, we will just propagate the error.
					return &graph().addErrorNode();
				} else if (auto* bmt = std::get_if<BoundModuleType>(&mdl::getDataType(**nodePtr))) {
					/** \todo bind the instance as the first argument **/
					/** \todo merge with ModuleInstance implementation above **/
					auto& funcs = bmt->definition.functions;
					auto it = funcs.find(memberName);
					if (it == funcs.end()) {
						errorlistener->semanticError(
								PrettyErrorListener::ErrorLevel::Error, memberTok, memberTok->getLine(),
								memberTok->getCharPositionInLine(),
								std::format("No function '{}' in module", memberName)
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
						const auto id = memberName == "mask" ? builtins::FunctionIDs::MASK_OF : builtins::FunctionIDs::UNMASK;
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
					{{.linenr = firsttok->getLine(),
					  .line = errorlistener->getline(*firsttok->getInputStream(), 0, firsttok->getStartIndex()),
					  .beginMark = firsttok->getCharPositionInLine(),
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
			auto rhs = parseEinopsSpec(einopsCtx->einops_spec_full(1));

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

		virtual std::any visitMethodcallExpression(mdlParser::MethodcallExpressionContext* ctx) override {
			// `instance.member(...)`: the member function is inlined for the instance (its weights), like
			// `instance(...)`. Without this, the function would reference the module's unqualified weights.
			const Node* receiver = nullptr;
			if (auto* access = dynamic_cast<mdlParser::AccessExpressionContext*>(ctx->func)) {
				auto baseObj = asObject(access->expr()->accept(this), access->expr()->start);
				if (auto* base = std::get_if<const Node*>(&baseObj)) {
					if (std::holds_alternative<ErrorNode>(**base))
						return &graph().addErrorNode(); // already reported
					if (std::holds_alternative<BoundModuleType>(mdl::getDataType(**base)))
						receiver = *base;
				}
			}
			auto calleeObj = asObject(ctx->func->accept(this), ctx->func->start);
			if (std::holds_alternative<const Node*>(calleeObj) &&
				std::holds_alternative<ErrorNode>(*std::get<const Node*>(calleeObj)))
				return &graph().addErrorNode();

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
			for (size_t i = 0; i < std::min(args.size(), callee.parameters.size()); ++i)
				if (!shapes::matches(callee.parameters[i].second.get().type, argtypes[i], false)) {
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
							return graph().addModuleInvocationNode(
									*bmt, scope, instanceName, callee, std::move(args)
							);
						}
					}
				}
				// A free function `f` with a custom gradient `f_gradient`: record the call for the gradient pass.
				std::vector<const Node*> inputs;
				for (const auto& arg : args)
					inputs.push_back(&arg.get());
				auto outputs = graph().addInvocationNode(callee, std::move(args));
				if (outputs) {
					auto gradient = currentNamespace().get<Object>(callee.name + "_gradient");
					if (const auto* decl = gradient ? std::get_if<FunctionDecl>(&gradient->get()) : nullptr) {
						ModuleCall call{.scope = scope, .path = {}, .module = nullptr, .hyperparameters = {},
										.function = callee.name, .inputs = std::move(inputs), .outputs = {},
										.customGradient = true, .gradient = decl};
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
			auto decl = getMember(
					mdl::getDataType(*left), std::format("operator{}", ctx->bop->getText()), {mdl::getDataType(*right)}
			);
			if (!decl.has_value()) { // Function not found!
				auto firsttok = ctx->start;
				auto lasttok = ctx->stop;
				/** \todo describe data types **/
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error,
						{{.linenr = firsttok->getLine(),
						  .line = errorlistener->getline(*firsttok->getInputStream(), 0, firsttok->getStartIndex()),
						  .beginMark = firsttok->getCharPositionInLine(),
						  .markWidth = lasttok->getStopIndex() - firsttok->getStartIndex() + 1}},
						firsttok->getLine(), firsttok->getCharPositionInLine(),
						std::format("Operator {} not supported on {}", ctx->bop->getText(), "TODO")
				);
				return &graph().addErrorNode();
			}
			const Node& result =
					graph().addInvocationNode(decl.value(), {std::cref(*left), std::cref(*right)}).value().front().get();
			if (const auto* inv = std::get_if<InvocationNode>(&result))
				if (auto problem = shapes::check(inv->id, inv->arguments)) {
					auto* tok = ctx->bop;
					errorlistener->semanticError(
							PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
							std::format("Operator {}: {}", tok->getText(), *problem)
					);
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

		virtual std::any visitSwitchExpression(mdlParser::SwitchExpressionContext* ctx) override {
			auto firsttok = ctx->start;
			auto lasttok = ctx->stop;
			std::vector<PrettyErrorListener::ContextLine> context;
			for (size_t i = firsttok->getLine(); i <= lasttok->getLine(); ++i) {
				context.push_back(
						{.linenr = i,
						 .line = errorlistener->getline(
								 *firsttok->getInputStream(), i - firsttok->getLine(), firsttok->getStartIndex()
						 )}
				);
			}
			errorlistener->semanticError(
					PrettyErrorListener::ErrorLevel::Error, context, firsttok->getLine(),
					firsttok->getCharPositionInLine(), "Switch expressions are not yet implemented"
			);
			return &graph().addErrorNode();
		}

		virtual std::any visitExpressionStatement(mdlParser::ExpressionStatementContext* ctx) override {
			ctx->expr()->accept(this);
			return std::nullopt;
		}

		virtual std::any visitDeclarationStatement(mdlParser::DeclarationStatementContext* ctx) override {
			/** \todo check datatype with ctx->type() **/
			auto value = ctx->expr()->accept(this);
			if (ctx->names.size() == 1) {
				currentNamespace().put(ctx->names[0]->getText(), asObject(std::move(value), ctx->expr()->start));
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
							std::format("Expected {} values but the expression yields {}", ctx->names.size(), nodes.size())
					);
				nodes.assign(ctx->names.size(), &graph().addErrorNode());
			}
			for (size_t i = 0; i < nodes.size(); ++i)
				currentNamespace().put(ctx->names[i]->getText(), Object{nodes[i]});
			return std::nullopt;
		}

		virtual std::any visitAssignStatement(mdlParser::AssignStatementContext* ctx) override {
			auto value = asObject(ctx->expr()->accept(this), ctx->expr()->start);
			if (!currentNamespace().assign(ctx->name->getText(), std::move(value)))
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, ctx->name, ctx->name->getLine(),
						ctx->name->getCharPositionInLine(),
						std::format("Variable {} is not defined (declare it with `auto {} = ...;`)", ctx->name->getText(),
									ctx->name->getText())
				);
			return std::nullopt;
		}

		virtual std::any visitForStatement(mdlParser::ForStatementContext* ctx) override {
			auto lo = constantInt(ctx->lo), hi = constantInt(ctx->hi);
			if (!lo || !hi)
				return std::nullopt;
			for (int i = *lo; i < *hi; ++i) {
				pushNamespace();
				currentNamespace().put(ctx->var->getText(), Object{&graph().addConstNode(Atomic{i})});
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
			auto base = asObject(ctx->base->accept(this), ctx->base->start);
			if (const auto* node = std::get_if<const Node*>(&base); node && std::holds_alternative<ErrorNode>(**node))
				return &graph().addErrorNode();
			auto error = [&](antlr4::Token* tok, const std::string& message) -> std::any {
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(), message
				);
				return &graph().addErrorNode();
			};

			if (const auto* array = std::get_if<ArrayInstance>(&base)) {
				auto* value = ctx->indices.size() == 1 ? dynamic_cast<mdlParser::IndexValueContext*>(ctx->indices[0]) : nullptr;
				if (value == nullptr)
					return error(ctx->start, "An array takes one index");
				auto index = constantInt(value->expr());
				if (!index)
					return &graph().addErrorNode();
				if (*index < 0 || *index >= static_cast<int>(array->elements.size()))
					return error(
							value->start,
							std::format("Index {} is out of bounds (array of {} elements)", *index, array->elements.size())
					);
				return array->elements[*index];
			}

			// Tensors: integers select (and remove) an axis, `:` keeps it, `...` stands for the axes not listed.
			const auto* node = std::get_if<const Node*>(&base);
			if (node == nullptr || !std::holds_alternative<TensorType>(mdl::getDataType(**node)))
				return error(ctx->base->start, "Only arrays and tensors can be indexed");
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
				auto index = constantInt(value->expr());
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
	if (const char* env = std::getenv("MDL_STDLIB_PATH")) {
		std::string_view paths{env};
		while (!paths.empty()) {
			auto sep = paths.find(':');
			if (auto dir = paths.substr(0, sep); !dir.empty())
				dirs.emplace_back(dir);
			paths = (sep == std::string_view::npos) ? std::string_view{} : paths.substr(sep + 1);
		}
	}
#ifdef MDL_DEFAULT_STDLIB_DIR
	dirs.emplace_back(MDL_DEFAULT_STDLIB_DIR);
#endif
	return dirs;
}

namespace {
	std::optional<mdl::CompilationUnit> finish(std::shared_ptr<mdl::parser::FrontendState> state) {
		auto& visitor = *state->visitor;
		if (visitor.getNumErrors() > 0)
			return std::nullopt;
		mdl::CompilationUnit unit;
		unit.definitions = std::move(visitor.getDefinitions());
		unit.sources = std::move(visitor.getSources());
		unit.state = std::move(state);
		return unit;
	}
} // namespace

std::optional<mdl::CompilationUnit>
mdl::parser::parse(std::istream& stream, std::string_view sourcename, const ParseOptions& options) {
	auto state = std::make_shared<FrontendState>();
	state->visitor = std::make_unique<TranslatingVisitor>(*state, options);
	state->visitor->translate(stream, std::string{sourcename}, std::filesystem::current_path(), /*main=*/true);
	return finish(std::move(state));
}

std::optional<mdl::CompilationUnit> mdl::parser::parse(const std::filesystem::path& file, const ParseOptions& options) {
	auto state = std::make_shared<FrontendState>();
	state->visitor = std::make_unique<TranslatingVisitor>(*state, options);
	state->visitor->translateFile(file, /*main=*/true);
	return finish(std::move(state));
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
