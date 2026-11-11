#include <mdl/parser.hpp>

#include "./parser/errorlistener.hpp"

#include <mdl/computegraphv2/function.hpp>
#include <mdl/computegraphv2/object.hpp>
#include <mdl/namespace.hpp>
#include <mdl/utils/einops.hpp>

#include <mdlBaseVisitor.h>
#include <mdlLexer.h>
#include <mdlParser.h>

#include <cassert>
#include <format>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string_view>
#include <functional>
#include <utility>

namespace mdl {
	class TranslatingVisitor : public antlr4::mdlBaseVisitor {
	private:
		CompilationUnit& unit;
		const parser::ParseOptions& options;
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
		TranslatingVisitor(CompilationUnit& unit, const parser::ParseOptions& options) : unit(unit), options(options) {
			pushGraph(&unit.graph);
			nsstack.push(mdl::Namespace::baseNamespace());
		}

		size_t getNumErrors() const { return numErrors; }

		/**
		 * @brief Translates the program read from @p stream into the compilation unit.
		 * @param main whether the definitions are added to CompilationUnit::definitions (false for imported files).
		 */
		void translate(std::istream& stream, const std::string& sourcename, const std::filesystem::path& dir, bool main) {
			antlr4::ANTLRInputStream input{stream};
			antlr4::mdlLexer lexer(&input);
			antlr4::CommonTokenStream tokens(&lexer);
			antlr4::mdlParser parser(&tokens);
			parser.removeErrorListeners();
			PrettyErrorListener fileErrors(sourcename);
			parser.addErrorListener(&fileErrors);

			auto* outerErrors = std::exchange(errorlistener, &fileErrors);
			auto outerDir = std::exchange(currentDir, dir);
			bool outerRecord = std::exchange(recordDefinitions, main);
			parser.program()->accept(this);
			errorlistener = outerErrors;
			currentDir = std::move(outerDir);
			recordDefinitions = outerRecord;

			fileErrors.printSummary();
			numErrors += fileErrors.getNumErrors();
		}

		void translateFile(const std::filesystem::path& file, bool main) {
			std::error_code ec;
			auto canonical = std::filesystem::weakly_canonical(file, ec);
			if (!translatedFiles.insert(ec ? file : canonical).second)
				return; // Already imported (or currently being imported).
			unit.sources.push_back(ec ? file : canonical);
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
					unit.definitions.try_emplace(std::move(name), std::move(member));
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

		virtual std::any visitModule_def(mdlParser::Module_defContext* ctx) override {
			ModuleType m;
			auto name = ctx->IDENT()->getText();
			const std::string outerScope = std::exchange(scope, name);
			pushNamespace();
			for (auto& param : ctx->param()) {
				const auto [pname, dtype] = std::any_cast<ModuleParam>(param->accept(this));
				const std::string qualifiedName = name + "." + pname;
				const Node& node = graph().addHyperparameterNode(dtype, qualifiedName);
				m.typeargs.emplace_back(pname, std::cref(std::get<HyperparameterNode>(node)));
				if (auto* agg = std::get_if<AggregateType>(&dtype))
					currentNamespace().put(
							pname, makeAggregateObject(*agg, qualifiedName, [&](DataType t, std::string n) -> const Node& {
								return graph().addHyperparameterNode(std::move(t), std::move(n));
							})
					);
				else
					currentNamespace().put(pname, Object{&node});
			}
			for (auto& var : ctx->vardecl()) {
				const auto [vname, dtype] = std::any_cast<VarDecl>(var->accept(this));
				// Weights are registered in the module's scope so that all functions of the module share them.
				const Node& node = graph().getOrAddParameterNode(scope, vname, dtype);
				m.variables.emplace_back(vname, std::cref(std::get<ParameterNode>(node)));
				if (auto* agg = std::get_if<AggregateType>(&dtype))
					currentNamespace().put(
							vname, makeAggregateObject(*agg, vname, [&](DataType t, std::string path) -> const Node& {
								return graph().getOrAddParameterNode(scope, path, std::move(t));
							})
					);
				else
					currentNamespace().put(vname, Object{&node});
			}
			for (auto& func : ctx->funcdef()) {
				auto [name, decl] = std::any_cast<std::pair<std::string, FunctionDecl>>(func->accept(this));
				auto [_, inserted] = m.functions.insert_or_assign(std::move(name), std::move(decl));
				if (!inserted) {
					/** \todo output a helpful warning message that this definition shadows another member, which is undefined behavior **/
				}
			}
			popNamespace();
			scope = outerScope;
			if (!checkRedefinition(ctx->IDENT()))
				currentNamespace().putType(name, DataType{m});
			return std::make_pair(name, m);
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
			auto dtype = currentNamespace().getType(ctx->typename_->getText());
			if (!dtype.has_value()) {
				errorlistener->semanticError(
						PrettyErrorListener::ErrorLevel::Error, ctx->typename_, ctx->typename_->getLine(),
						ctx->typename_->getCharPositionInLine(),
						std::format("{} does not name a type", ctx->typename_->getText())
				);
				return mdl::DataType{mdl::PrimitiveType::error};
			} else {
				// Module types are always bound (also without arguments, e.g. `Inner[]`), so that arity errors are
				// reported and instances are callable.
				if (ctx->type_arg().size() != 0 || std::holds_alternative<ModuleType>(dtype.value().get())) {
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
			std::vector<std::reference_wrapper<const Node>> outputs;
			std::vector<std::pair<std::string, std::reference_wrapper<const HyperparameterNode>>> tparams;
			std::vector<std::pair<std::string, std::reference_wrapper<const InputNode>>> params;
			pushNamespace();
			if (ctx->templatespec() != nullptr) {
				for (auto decl : ctx->templatespec()->vardecl()) {
					auto [name, dtype] = std::any_cast<VarDecl>(decl->accept(this));
					auto& p = currentNamespace().put(name, Object{&graph().addHyperparameterNode(dtype, name)});
					tparams.emplace_back(name, std::get<mdl::HyperparameterNode>(*std::get<const mdl::Node*>(p)));
				}
			}
			for (auto& param : ctx->param()) {
				auto [name, dtype] = std::any_cast<VarDecl>(param->accept(this));
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
			if (!checkRedefinition(ctx->IDENT()))
				currentNamespace().put<mdl::Object>(ctx->IDENT()->getText(), Object{decl});
			return std::make_pair(ctx->IDENT()->getText(), std::move(decl));
		}

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

			// If the callee is a module instance node, use addModuleInvocationNode so that each instance (weightQuery,
			// weightKey, ...) gets its own weights (with fully qualified paths, e.g. `weightQuery.W`) instead of sharing
			// the module type's nodes.
			auto resultOrErr = [&]() {
				if (auto* calleeNodePtr = std::get_if<const Node*>(&calleeObj)) {
					if (!std::holds_alternative<ErrorNode>(**calleeNodePtr)) {
						if (auto* bmt = std::get_if<BoundModuleType>(&mdl::getDataType(**calleeNodePtr))) {
							std::string instanceName;
							if (const auto* pn = std::get_if<ParameterNode>(*calleeNodePtr))
								instanceName = pn->name;
							return graph().addModuleInvocationNode(
									*bmt, scope, instanceName, decl.value(), std::move(args)
							);
						}
					}
				}
				return graph().addInvocationNode(decl.value(), std::move(args));
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
			return &graph().addInvocationNode(decl.value(), {std::cref(*left), std::cref(*right)}).value().front().get();
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
					builtins::FunctionIDs::TENSOR_SUB, mdl::getDataType(*operand), {std::cref(zero), std::cref(*operand)}
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

std::optional<mdl::CompilationUnit>
mdl::parser::parse(std::istream& stream, std::string_view sourcename, const ParseOptions& options) {
	CompilationUnit unit;
	TranslatingVisitor visitor{unit, options};
	visitor.translate(stream, std::string{sourcename}, std::filesystem::current_path(), /*main=*/true);
	if (visitor.getNumErrors() > 0)
		return std::nullopt;
	return std::make_optional<CompilationUnit>(std::move(unit));
}

std::optional<mdl::CompilationUnit> mdl::parser::parse(const std::filesystem::path& file, const ParseOptions& options) {
	CompilationUnit unit;
	TranslatingVisitor visitor{unit, options};
	visitor.translateFile(file, /*main=*/true);
	if (visitor.getNumErrors() > 0)
		return std::nullopt;
	return std::make_optional<CompilationUnit>(std::move(unit));
}
