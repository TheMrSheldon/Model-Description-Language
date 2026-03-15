#include <mdl/parser.hpp>

#include "./parser/errorlistener.hpp"

#include <mdl/computegraphv2/function.hpp>
#include <mdl/namespace.hpp>

#include <mdlBaseVisitor.h>
#include <mdlLexer.h>
#include <mdlParser.h>

#include <cassert>
#include <format>
#include <fstream>

namespace mdl {
	class TranslatingVisitor : public antlr4::mdlBaseVisitor {
	private:
		PrettyErrorListener& errorlistener;

		std::stack<mdl::Computegraph*> graphstack;
		std::stack<mdl::Namespace> nsstack;

		using ModuleParam = std::pair<std::string, mdl::DataType>;
		using VarDecl = std::pair<std::string, mdl::DataType>;

		mdl::Computegraph& graph() { return *graphstack.top(); }
		void pushGraph(mdl::Computegraph* graph) { graphstack.push(graph); }
		void popGraph() { graphstack.pop(); }

		mdl::Namespace& currentNamespace() { return nsstack.top(); }
		void pushNamespace() { nsstack.push(nsstack.top().createSubnamespace()); }
		void popNamespace() { nsstack.pop(); }

	public:
		TranslatingVisitor(PrettyErrorListener& errorlistener) : errorlistener(errorlistener) {}

		virtual std::any visitProgram(mdlParser::ProgramContext* ctx) override {
			assert(nsstack.empty());
			CompilationUnit program;
			pushGraph(&program.graph);
			nsstack.push(mdl::Namespace::baseNamespace());
			// Parse Structs first (no matter where they are in the file)
			// Does this make sense? Only if Structs can not have instances of modules as members.
			for (auto&& def : ctx->struct_def()) {
				auto [name, s] = std::any_cast<std::pair<std::string, AggregateType>>(def->accept(this));
				program.definitions.try_emplace(name, std::move(s));
			}
			// Parse Modules second (no matter where they are in the file)
			for (auto&& def : ctx->module_def()) {
				auto [name, m] = std::any_cast<std::pair<std::string, ModuleType>>(def->accept(this));
				program.definitions.try_emplace(name, std::move(m));
			}
			nsstack.pop();
			popGraph();
			assert(nsstack.empty());
			return std::make_shared<CompilationUnit>(std::move(program));
		}

		void updateTypeWithVarDecl(
				AggregateType& type, const std::string& tname, std::vector<antlr4::mdlParser::VardeclContext*> vars
		) {
			for (auto& var : vars) {
				auto [name, dtype] = std::any_cast<VarDecl>(var->accept(this));
				auto [_, inserted] = type.members.insert_or_assign(name, dtype);
				if (!inserted) {
					auto offendingToken = var->IDENT()->getSymbol();
					errorlistener.semanticError(
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
			currentNamespace().putType(name, DataType{s});
			return std::make_pair(name, s);
		}

		virtual std::any visitModule_def(mdlParser::Module_defContext* ctx) override {
			ModuleType m;
			auto name = ctx->IDENT()->getText();
			pushNamespace();
			/** \todo add params and vars to namespace */
			for (auto& param : ctx->param()) {
				const auto [name, dtype] = std::any_cast<ModuleParam>(param->accept(this));
				auto& node = *currentNamespace().put(name, &graph().addHyperparameterNode(dtype));
				m.typeargs.emplace_back(name, std::cref(std::get<HyperparameterNode>(node)));
			}
			for (auto& var : ctx->vardecl()) {
				const auto [name, dtype] = std::any_cast<VarDecl>(var->accept(this));
				auto& node = *currentNamespace().put(name, &graph().addParameterNode(dtype));
				m.variables.emplace_back(name, std::cref(std::get<ParameterNode>(node)));
			}
			for (auto& func : ctx->funcdef()) {
				auto decl = std::any_cast<FunctionDecl>(func->accept(this));
				auto name = decl.name;
				auto [_, inserted] = m.functions.insert_or_assign(std::move(name), std::move(decl));
				if (!inserted) {
					/** \todo output a helpful warning message that this definition shadows another member, which is undefined behavior **/
				}
			}
			popNamespace();
			currentNamespace().putType(name, DataType{BoundModuleType{.definition = m}});
			return std::make_pair(name, m);
		}

		virtual std::any visitParam(mdlParser::ParamContext* ctx) override {
			if (ctx->default_ != nullptr) {
				auto firsttok = ctx->default_->start;
				auto lasttok = ctx->default_->stop;
				errorlistener.semanticError(
						PrettyErrorListener::ErrorLevel::Error,
						{{.linenr = firsttok->getLine(),
						  .line = errorlistener.getline(*firsttok->getInputStream(), 0, firsttok->getStartIndex()),
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
				errorlistener.semanticError(
						PrettyErrorListener::ErrorLevel::Error, ctx->typename_, ctx->typename_->getLine(),
						ctx->typename_->getCharPositionInLine(),
						std::format("{} does not name a type", ctx->typename_->getText())
				);
				return mdl::DataType{mdl::PrimitiveType::error};
			} else {
				if (ctx->type_arg().size() != 0) {
					std::vector<TypeArg> args;
					for (auto arg : ctx->type_arg()) {
						args.emplace_back(std::any_cast<TypeArg>(arg->accept(this)));
					}
					auto ta = setTypeArgs(dtype.value(), args);
					if (!ta.has_value()) {
						auto firsttok = ctx->type_arg().front()->start;
						auto lasttok = ctx->type_arg().back()->stop;
						errorlistener.semanticError(
								PrettyErrorListener::ErrorLevel::Error,
								{{.linenr = ctx->typename_->getLine(),
								  .line = errorlistener.getline(
										  *firsttok->getInputStream(), 0, firsttok->getStartIndex()
								  ),
								  .beginMark = firsttok->getCharPositionInLine(),
								  .markWidth = lasttok->getStopIndex() - firsttok->getStartIndex() + 1}},
								ctx->typename_->getLine(), firsttok->getCharPositionInLine(), ta.error()
						);
						return dtype.value(); /** Or return some "error" datatype here? */
					}
					return ta.value();
				}
				return dtype.value();
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
				auto tmp = ctx->expr()->accept(this);
				const auto& node = *std::any_cast<const Node*>(tmp);
				if (!mdl::isCompiletimeEvalable(node)) {
					auto firsttok = ctx->expr()->start;
					auto lasttok = ctx->expr()->stop;
					errorlistener.semanticError(
							PrettyErrorListener::ErrorLevel::Error,
							{{.linenr = firsttok->getLine(),
							  .line = errorlistener.getline(*firsttok->getInputStream(), 0, firsttok->getStartIndex()),
							  .beginMark = firsttok->getCharPositionInLine(),
							  .markWidth = lasttok->getStopIndex() - firsttok->getStartIndex() + 1}},
							firsttok->getLine(), firsttok->getCharPositionInLine(),
							"Parameters must be compile time constant"
					);
					return TypeArg{&node}; /** return this node anyway for now **/
				} else if (auto n = std::get_if<ConstNode>(&node)) {
					return TypeArg{n->value};
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
			assert(false); // Invalid case
		}

		virtual std::any visitVardecl(mdlParser::VardeclContext* ctx) override {
			return std::make_pair(ctx->IDENT()->getText(), std::any_cast<mdl::DataType>(ctx->type()->accept(this)));
		}

		virtual std::any visitFuncdef(mdlParser::FuncdefContext* ctx) override {
			const Node* output = nullptr;
			pushNamespace();
			if (ctx->templatespec() != nullptr) {
				for (auto decl : ctx->templatespec()->vardecl()) {
					auto [name, dtype] = std::any_cast<VarDecl>(decl->accept(this));
					currentNamespace().put(name, &graph().addHyperparameterNode(dtype));
					/** \todo register template in FunctionDecl */
				}
			}
			for (auto& param : ctx->param()) {
				auto [name, dtype] = std::any_cast<VarDecl>(param->accept(this));
				currentNamespace().put(name, &graph().addParameterNode(dtype));
				/** \todo register parameter in FunctionDecl */
			}
			if (ctx->expr() != nullptr) {
				output = std::any_cast<const Node*>(ctx->expr()->accept(this));
			} else {
				for (auto stmt : ctx->statement()) {
					auto result = stmt->accept(this);
					if (result.type() == typeid(const Node*)) {
						/** \todo create a descriptive error message; the function has two return statements. **/
						assert(output == nullptr);
						// The statement is a return statement; the others return nullopt
						output = std::any_cast<const Node*>(result);
					}
				}
				/** \todo track output statement **/
			}
			/** \todo create a descriptive error message; the function did not have a return statement. **/
			assert(output != nullptr);
			popNamespace();
			return FunctionDecl{.name = ctx->IDENT()->getText(), .output = *output};
		}

		virtual std::any visitVarAccessExpression(mdlParser::VarAccessExpressionContext* ctx) override {
			auto value = currentNamespace().get<const Node*>(ctx->IDENT()->getText());
			if (!value.has_value()) {
				auto offendingSymbol = ctx->IDENT()->getSymbol();
				errorlistener.semanticError(
						PrettyErrorListener::ErrorLevel::Error, offendingSymbol, offendingSymbol->getLine(),
						offendingSymbol->getCharPositionInLine(),
						std::format("Variable {} not defined", ctx->IDENT()->getText())
				);
				return &graph().addErrorNode();
			}
			return value.value().get();
		}
		virtual std::any visitConstantExpression(mdlParser::ConstantExpressionContext* ctx) override {
			return &graph().addConstNode(std::any_cast<mdl::Atomic>(ctx->constant()->accept(this)));
		}
		virtual std::any visitParenExpression(mdlParser::ParenExpressionContext* ctx) override {
			return ctx->expr()->accept(this);
		}

		virtual std::any visitAccessExpression(mdlParser::AccessExpressionContext* ctx) override {
			auto node = std::any_cast<const Node*>(ctx->expr()->accept(this));

			auto firsttok = ctx->start;
			auto lasttok = ctx->stop;
			errorlistener.semanticError(
					PrettyErrorListener::ErrorLevel::Error,
					{{.linenr = firsttok->getLine(),
					  .line = errorlistener.getline(*firsttok->getInputStream(), 0, firsttok->getStartIndex()),
					  .beginMark = firsttok->getCharPositionInLine(),
					  .markWidth = lasttok->getStopIndex() - firsttok->getStartIndex() + 1}},
					firsttok->getLine(), firsttok->getCharPositionInLine(),
					"Member access expressions are not yet implemented"
			);
			return &graph().addErrorNode();
		}

		virtual std::any visitEinopsExpression(mdlParser::EinopsExpressionContext* ctx) override {
			auto firsttok = ctx->start;
			auto lasttok = ctx->stop;
			errorlistener.semanticError(
					PrettyErrorListener::ErrorLevel::Error,
					{{.linenr = firsttok->getLine(),
					  .line = errorlistener.getline(*firsttok->getInputStream(), 0, firsttok->getStartIndex()),
					  .beginMark = firsttok->getCharPositionInLine(),
					  .markWidth = lasttok->getStopIndex() - firsttok->getStartIndex() + 1}},
					firsttok->getLine(), firsttok->getCharPositionInLine(), "Einops expressions are not yet implemented"
			);
			return &graph().addErrorNode();
		}

		virtual std::any visitMethodcallExpression(mdlParser::MethodcallExpressionContext* ctx) override {
			auto firsttok = ctx->start;
			auto lasttok = ctx->stop;
			errorlistener.semanticError(
					PrettyErrorListener::ErrorLevel::Error,
					{{.linenr = firsttok->getLine(),
					  .line = errorlistener.getline(*firsttok->getInputStream(), 0, firsttok->getStartIndex()),
					  .beginMark = firsttok->getCharPositionInLine(),
					  .markWidth = lasttok->getStopIndex() - firsttok->getStartIndex() + 1}},
					firsttok->getLine(), firsttok->getCharPositionInLine(), "Method calls are not yet implemented"
			);
			return &graph().addErrorNode();
		}

		virtual std::any visitArithmeticExpression(mdlParser::ArithmeticExpressionContext* ctx) override {
			const auto* left = std::any_cast<const Node*>(ctx->left->accept(this));
			const auto* right = std::any_cast<const Node*>(ctx->right->accept(this));
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
				errorlistener.semanticError(
						PrettyErrorListener::ErrorLevel::Error,
						{{.linenr = firsttok->getLine(),
						  .line = errorlistener.getline(*firsttok->getInputStream(), 0, firsttok->getStartIndex()),
						  .beginMark = firsttok->getCharPositionInLine(),
						  .markWidth = lasttok->getStopIndex() - firsttok->getStartIndex() + 1}},
						firsttok->getLine(), firsttok->getCharPositionInLine(),
						std::format("Operator {} not supported on {}", ctx->bop->getText(), "TODO")
				);
				return &graph().addErrorNode();
			}
			return &graph().addInvocationNode(decl.value(), {std::cref(*left), std::cref(*right)});
		}

		virtual std::any visitSwitchExpression(mdlParser::SwitchExpressionContext* ctx) override {
			auto firsttok = ctx->start;
			auto lasttok = ctx->stop;
			std::vector<PrettyErrorListener::ContextLine> context;
			for (size_t i = firsttok->getLine(); i <= lasttok->getLine(); ++i) {
				context.push_back(
						{.linenr = i,
						 .line = errorlistener.getline(
								 *firsttok->getInputStream(), i - firsttok->getLine(), firsttok->getStartIndex()
						 )}
				);
			}
			errorlistener.semanticError(
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
			const auto node = std::any_cast<const Node*>(ctx->expr()->accept(this));
			/** \todo check datatype with ctx->type() **/
			currentNamespace().put(ctx->IDENT()->getText(), node);
			return std::nullopt;
		}

		virtual std::any visitReturnStatement(mdlParser::ReturnStatementContext* ctx) override {
			return ctx->expr()->accept(this); // Datatype: const Node*
		}
	};
} // namespace mdl

std::optional<mdl::CompilationUnit> mdl::parser::parse(std::istream& stream, std::string_view sourcename) {
	antlr4::ANTLRInputStream input{stream};
	antlr4::mdlLexer lexer(&input);
	antlr4::CommonTokenStream tokens(&lexer);
	antlr4::mdlParser parser(&tokens);
	parser.removeErrorListeners();
	PrettyErrorListener errorlistener(std::string{sourcename});
	parser.addErrorListener(&errorlistener);
	TranslatingVisitor visitor{errorlistener};
	auto programptr = std::any_cast<std::shared_ptr<CompilationUnit>>(parser.program()->accept(&visitor));
	errorlistener.printSummary();
	if (errorlistener.getNumErrors() > 0)
		return std::nullopt;
	return std::make_optional<CompilationUnit>(std::move(*programptr));
}
std::optional<mdl::CompilationUnit> mdl::parser::parse(const std::filesystem::path& file) {
	std::ifstream stream{file.c_str()};
	return parse(stream, std::filesystem::absolute(file).c_str());
}