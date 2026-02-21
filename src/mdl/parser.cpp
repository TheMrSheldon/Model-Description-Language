#include <mdl/parser.hpp>

#include "./parser/errorlistener.hpp"

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
			nsstack.push(mdl::Namespace::baseNamespace());
			Program program;
			// Parse Structs first (no matter where they are in the file)
			// Does this make sense? Only if Structs can not have instances of modules as members.
			for (auto&& def : ctx->struct_def()) {
				auto s = std::any_cast<Struct>(def->accept(this));
				program.definitions.try_emplace(s.name, std::move(s));
			}
			// Parse Modules second (no matter where they are in the file)
			for (auto&& def : ctx->module_def()) {
				auto m = std::any_cast<std::shared_ptr<Module>>(def->accept(this));
				program.definitions.try_emplace(m->name, std::move(*m));
			}
			nsstack.pop();
			assert(nsstack.empty());
			return std::make_shared<Program>(std::move(program));
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
			Struct s{.name = ctx->IDENT()->getText()};
			updateTypeWithVarDecl(s.type, s.name, ctx->vardecl());
			currentNamespace().putType(s.name, DataType{s.type});
			return s;
		}

		virtual std::any visitModule_def(mdlParser::Module_defContext* ctx) override {
			Module m{.name = ctx->IDENT()->getText()};
			pushGraph(&m.graph);
			pushNamespace();
			/** \todo add params and vars to namespace */
			for (auto& param : ctx->param()) {
				const auto [name, dtype] = std::any_cast<ModuleParam>(param->accept(this));
				auto& node = currentNamespace().put(name, &graph().addArgumentNode(name, dtype));
				m.arguments.emplace_back(VarDesc{.name = name, .node = node});
			}
			for (auto& var : ctx->vardecl()) {
				const auto [name, dtype] = std::any_cast<VarDecl>(var->accept(this));
				auto& node = currentNamespace().put(name, &graph().addInputNode(name, dtype));
				m.variables.emplace_back(VarDesc{.name = name, .node = node});
			}
			for (auto& func : ctx->funcdef()) {
				m.functions.emplace_back(std::any_cast<FuncDesc>(func->accept(this)));
			}
			popNamespace();
			popGraph();
			currentNamespace().putType(m.name, DataType{std::cref(m)});
			return std::make_shared<Module>(std::move(m));
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
				return mdl::DataType{Primitive::integer}; /** Use integer for error types or define a new one? **/
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
				} else if (mdl::getOperation(node) == mdl::Computegraph::Operation::Const) {
					return TypeArg{mdl::getConst(node)};
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
				return mdl::Computegraph::Value{std::atoi(ctx->INTEGER()->getText().c_str())};
			if (ctx->FLOAT() != nullptr)
				return mdl::Computegraph::Value{std::atof(ctx->FLOAT()->getText().c_str())};
			assert(false); // Invalid case
		}

		virtual std::any visitVardecl(mdlParser::VardeclContext* ctx) override {
			return std::make_pair(ctx->IDENT()->getText(), std::any_cast<mdl::DataType>(ctx->type()->accept(this)));
		}

		virtual std::any visitFuncdef(mdlParser::FuncdefContext* ctx) override {
			FuncDesc f{.name = ctx->IDENT()->getText(), .graph = &graph().addSubgraph()};
			pushGraph(f.graph);
			pushNamespace();
			if (ctx->templatespec() != nullptr) {
				for (auto decl : ctx->templatespec()->vardecl()) {
					auto [name, dtype] = std::any_cast<VarDecl>(decl->accept(this));
					currentNamespace().put(name, &graph().addArgumentNode(name, dtype));
				}
			}
			for (auto& param : ctx->param()) {
				auto [name, dtype] = std::any_cast<VarDecl>(param->accept(this));
				currentNamespace().put(name, &graph().addInputNode(name, dtype));
			}
			if (ctx->expr() != nullptr) {
				auto node = std::any_cast<const Node*>(ctx->expr()->accept(this));
				graph().setOutputNode(node);
			} else {
				for (auto stmt : ctx->statement())
					stmt->accept(this);
			}
			popNamespace();
			popGraph();
			return f;
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
			return &graph().addConstNode(std::any_cast<mdl::Computegraph::Value>(ctx->constant()->accept(this)));
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
			static const std::map<std::string, mdl::Computegraph::Operation> toOp{
					{"+", mdl::Computegraph::Operation::ADD},	  {"-", mdl::Computegraph::Operation::SUB},
					{"*", mdl::Computegraph::Operation::MULT},	  {"/", mdl::Computegraph::Operation::DIV},
					{"@", mdl::Computegraph::Operation::MATMULT},
			};
			const auto op = toOp.at(ctx->bop->getText());
			const auto* left = std::any_cast<const Node*>(ctx->left->accept(this));
			const auto* right = std::any_cast<const Node*>(ctx->right->accept(this));
			return &graph().addOpNode(op, {left, right});
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
			auto node = std::any_cast<const Node*>(ctx->expr()->accept(this));
			graph().setOutputNode(node);
			return std::nullopt;
		}
	};
} // namespace mdl

std::optional<mdl::Program> mdl::parser::parse(std::istream& stream, std::string_view sourcename) {
	antlr4::ANTLRInputStream input{stream};
	antlr4::mdlLexer lexer(&input);
	antlr4::CommonTokenStream tokens(&lexer);
	antlr4::mdlParser parser(&tokens);
	parser.removeErrorListeners();
	PrettyErrorListener errorlistener(std::string{sourcename});
	parser.addErrorListener(&errorlistener);
	TranslatingVisitor visitor{errorlistener};
	auto programptr = std::any_cast<std::shared_ptr<Program>>(parser.program()->accept(&visitor));
	errorlistener.printSummary();
	if (errorlistener.getNumErrors() > 0)
		return std::nullopt;
	return std::make_optional<Program>(std::move(*programptr));
}
std::optional<mdl::Program> mdl::parser::parse(const std::filesystem::path& file) {
	std::ifstream stream{file.c_str()};
	return parse(stream, std::filesystem::absolute(file).c_str());
}