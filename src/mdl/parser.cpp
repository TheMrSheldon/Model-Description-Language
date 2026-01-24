#include <mdl/parser.hpp>

#include "./parser/errorlistener.hpp"

#include <mdlBaseVisitor.h>
#include <mdlLexer.h>
#include <mdlParser.h>

#include <cassert>
#include <fstream>

class TranslatingVisitor : public antlr4::mdlBaseVisitor {
private:
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

	virtual std::any visitStruct_def(mdlParser::Struct_defContext* ctx) override {
		Struct s{.name = ctx->IDENT()->getText()};
		for (auto& var : ctx->vardecl()) {
			auto [name, dtype] = std::any_cast<VarDecl>(var->accept(this));
			auto [_, inserted] = s.type.members.insert_or_assign(name, dtype);
			if (!inserted) {
				/** \todo Proper warning that a member was redeclared **/
				std::cout << "Member " << name << " was declared multiple times." << std::endl;
			}
			s.type.members[name] = dtype;
		}
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
		return std::make_shared<Module>(std::move(m));
	}

	virtual std::any visitParam(mdlParser::ParamContext* ctx) override {
		if (ctx->default_ != nullptr) {
			throw std::runtime_error("Default Values have not yet been implemented");
		}
		return std::make_pair(ctx->IDENT()->getText(), std::any_cast<mdl::DataType>(ctx->type()->accept(this)));
	}

	virtual std::any visitTypeSpec(mdlParser::TypeSpecContext* ctx) override {
		auto dtype = currentNamespace().getType(ctx->typename_->getText());
		if (ctx->type_arg().size() != 0) {
			std::vector<TypeArg> args;
			for (auto arg : ctx->type_arg()) {
				args.emplace_back(std::any_cast<TypeArg>(arg->accept(this)));
			}
			return setTypeArgs(dtype, args).value();
		}
		return dtype;
	}

	virtual std::any visitAnonType(mdlParser::AnonTypeContext* ctx) override {
		throw std::runtime_error("Not implemented");
	}

	virtual std::any visitType_arg(mdlParser::Type_argContext* ctx) override {
		if (ctx->elipsis != nullptr) {
			return TypeArg{mdl::Elipsis{}};
		} else if (ctx->expr() != nullptr) {
			auto tmp = ctx->expr()->accept(this);
			const auto& node = *std::any_cast<const Node*>(tmp);
			switch (mdl::getOperation(node)) {
			case Operation::Const:
				return TypeArg{mdl::getConst(node)};
			case Operation::Argument:
				return TypeArg{&node};
			default:
				/** \todo write descriptive error message **/
				assert(false);
				break;
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
		return currentNamespace().get<const Node*>(ctx->IDENT()->getText());
	}
	virtual std::any visitConstantExpression(mdlParser::ConstantExpressionContext* ctx) override {
		return &graph().addConstNode(std::any_cast<mdl::Computegraph::Value>(ctx->constant()->accept(this)));
	}
	virtual std::any visitParenExpression(mdlParser::ParenExpressionContext* ctx) override {
		return ctx->expr()->accept(this);
	}

	virtual std::any visitAccessExpression(mdlParser::AccessExpressionContext* ctx) override {
		throw std::runtime_error("Not implemented");
	}

	virtual std::any visitEinopsExpression(mdlParser::EinopsExpressionContext* ctx) override {
		throw std::runtime_error("Not implemented");
	}

	virtual std::any visitMethodcallExpression(mdlParser::MethodcallExpressionContext* ctx) override {
		throw std::runtime_error("Not implemented");
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
		throw std::runtime_error("Not implemented");
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

Program mdl::parser::parse(std::istream& stream, std::string_view sourcename) {
	antlr4::ANTLRInputStream input{stream};
	antlr4::mdlLexer lexer(&input);
	antlr4::CommonTokenStream tokens(&lexer);
	antlr4::mdlParser parser(&tokens);
	TranslatingVisitor visitor;
	parser.removeErrorListeners();
	PrettyErrorListener errorlistener(std::string{sourcename});
	parser.addErrorListener(&errorlistener);
	return std::move(*std::any_cast<std::shared_ptr<Program>>(parser.program()->accept(&visitor)));
}
Program mdl::parser::parse(const std::filesystem::path& file) {
	std::ifstream stream{file.c_str()};
	return parse(stream, std::filesystem::absolute(file).c_str());
}