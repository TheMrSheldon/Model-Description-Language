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

		// Convert any expression result (const Node* or Object) to Object.
		Object asObject(std::any value) {
			if (auto* node = std::any_cast<const Node*>(&value))
				return Object{*node};
			if (auto* obj = std::any_cast<Object>(&value))
				return *obj;
			std::abort();
		}

		// Extract const Node* from an expression result, emitting an error if it is not a node.
		const Node* asNode(std::any value, antlr4::Token* tok) {
			auto obj = asObject(std::move(value));
			return std::visit(
					overloaded{
							[](const Node* node) { return node; },
							[&](const FunctionDecl&) -> const Node* {
								errorlistener.semanticError(
										PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(),
										tok->getCharPositionInLine(), "Expected a value, got a function"
								);
								return &graph().addErrorNode();
							},
							[&](const AggregateInstance&) -> const Node* {
								errorlistener.semanticError(
										PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(),
										tok->getCharPositionInLine(), "Expected a scalar/tensor value, got a struct"
								);
								return &graph().addErrorNode();
							},
							[&](const ModuleInstance&) -> const Node* {
								errorlistener.semanticError(
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
		template <typename NodeType>
		Object makeAggregateObject(
				const AggregateType& agg, const std::string& prefix, const Node& (Computegraph::*addNode)(DataType)
		) {
			std::map<std::string, Object> values;
			for (auto& [memberName, memberType] : agg.members) {
				if (auto* subAgg = std::get_if<AggregateType>(&memberType))
					values[memberName] = makeAggregateObject<NodeType>(*subAgg, prefix + "." + memberName, addNode);
				else
					values[memberName] = Object{&(graph().*addNode)(memberType)};
			}
			return Object{AggregateInstance::create(agg, std::move(values)).value()};
		}

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
			// Parse func definitions
			for (auto&& def : ctx->funcdef()) {
				auto [name, f] = std::any_cast<std::pair<std::string, FunctionDecl>>(def->accept(this));
				program.definitions.try_emplace(name, std::move(f));
			}
			// Parse Modules third (no matter where they are in the file)
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
			for (auto& param : ctx->param()) {
				const auto [pname, dtype] = std::any_cast<ModuleParam>(param->accept(this));
				const Node& node = graph().addHyperparameterNode(dtype);
				m.typeargs.emplace_back(pname, std::cref(std::get<HyperparameterNode>(node)));
				if (auto* agg = std::get_if<AggregateType>(&dtype))
					currentNamespace().put(
							pname,
							makeAggregateObject<HyperparameterNode>(*agg, pname, &Computegraph::addHyperparameterNode)
					);
				else
					currentNamespace().put(pname, Object{&node});
			}
			for (auto& var : ctx->vardecl()) {
				const auto [vname, dtype] = std::any_cast<VarDecl>(var->accept(this));
				const Node& node = graph().addParameterNode(dtype);
				m.variables.emplace_back(vname, std::cref(std::get<ParameterNode>(node)));
				if (auto* agg = std::get_if<AggregateType>(&dtype))
					currentNamespace().put(
							vname, makeAggregateObject<ParameterNode>(*agg, vname, &Computegraph::addParameterNode)
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
			currentNamespace().putType(name, DataType{m});
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
				const auto& node = *asNode(ctx->expr()->accept(this), ctx->expr()->start);
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
			const Node* output = nullptr;
			std::vector<std::pair<std::string, std::reference_wrapper<const HyperparameterNode>>> tparams;
			std::vector<std::pair<std::string, std::reference_wrapper<const ParameterNode>>> params;
			pushNamespace();
			if (ctx->templatespec() != nullptr) {
				for (auto decl : ctx->templatespec()->vardecl()) {
					auto [name, dtype] = std::any_cast<VarDecl>(decl->accept(this));
					auto& p = currentNamespace().put(name, Object{&graph().addHyperparameterNode(dtype)});
					tparams.emplace_back(name, std::get<mdl::HyperparameterNode>(*std::get<const mdl::Node*>(p)));
				}
			}
			for (auto& param : ctx->param()) {
				auto [name, dtype] = std::any_cast<VarDecl>(param->accept(this));
				auto& p = currentNamespace().put(name, Object{&graph().addParameterNode(dtype)});
				params.emplace_back(name, std::get<mdl::ParameterNode>(*std::get<const mdl::Node*>(p)));
			}
			if (ctx->expr() != nullptr) {
				output = asNode(ctx->expr()->accept(this), ctx->expr()->start);
			} else {
				for (auto stmt : ctx->statement()) {
					auto result = stmt->accept(this);
					if (result.type() == typeid(const Node*) || result.type() == typeid(Object)) {
						/** \todo create a descriptive error message; the function has two return statements. **/
						assert(output == nullptr);
						// The statement is a return statement; the others return nullopt
						output = asNode(std::move(result), stmt->start);
					}
				}
				assert(output != nullptr);
			}
			/** \todo create a descriptive error message; the function did not have a return statement. **/
			assert(output != nullptr);
			popNamespace();
			auto& obj = currentNamespace().put<mdl::Object>(
					ctx->IDENT()->getText(), FunctionDecl{
													 .name = ctx->IDENT()->getText(),
													 .templateParameters = std::move(tparams),
													 .parameters = std::move(params),
													 .output = *output
											 }
			);
			return std::make_pair(ctx->IDENT()->getText(), std::get<FunctionDecl>(obj));
		}

		virtual std::any visitVarAccessExpression(mdlParser::VarAccessExpressionContext* ctx) override {
			auto value = currentNamespace().get<Object>(ctx->IDENT()->getText());
			if (!value.has_value()) {
				auto* sym = ctx->IDENT()->getSymbol();
				errorlistener.semanticError(
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
					errorlistener.semanticError(
							PrettyErrorListener::ErrorLevel::Error, memberTok, memberTok->getLine(),
							memberTok->getCharPositionInLine(), std::format("No member '{}' in struct", memberName)
					);
					return &graph().addErrorNode();
				}
				return mem.value().get();
			} else if (auto* mod = std::get_if<ModuleInstance>(&baseObj)) {
				auto mem = mod->getMember(memberName);
				if (!mem.has_value()) {
					errorlistener.semanticError(
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
					auto& funcs = bmt->definition.get().functions;
					auto it = funcs.find(memberName);
					if (it == funcs.end()) {
						errorlistener.semanticError(
								PrettyErrorListener::ErrorLevel::Error, memberTok, memberTok->getLine(),
								memberTok->getCharPositionInLine(),
								std::format("No function '{}' in module", memberName)
						);
						return &graph().addErrorNode();
					}
					return Object{it->second};
				} else if (auto* tensor = std::get_if<TensorType>(&mdl::getDataType(**nodePtr))) {
					/** \todo this "T" has "property semantic"; i.e. it is a function but automatically invoked; this should be implemented more cleanly */
					auto member = mdl::getMember(mdl::Object{*nodePtr}, memberName, {mdl::getDataType(**nodePtr)});
					if (!member.has_value()) {
						errorlistener.semanticError(
								PrettyErrorListener::ErrorLevel::Error, memberTok, memberTok->getLine(),
								memberTok->getCharPositionInLine(), std::format("No member '{}' for tensor", memberName)
						);
						return &graph().addErrorNode();
					}
					std::vector<std::reference_wrapper<const Node>> args{std::cref(**nodePtr)};
					return &graph().addInvocationNode(member.value().get(), std::move(args));
				}
			}

			auto* firsttok = ctx->start;
			auto* lasttok = ctx->stop;
			errorlistener.semanticError(
					PrettyErrorListener::ErrorLevel::Error,
					{{.linenr = firsttok->getLine(),
					  .line = errorlistener.getline(*firsttok->getInputStream(), 0, firsttok->getStartIndex()),
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
				errorlistener.semanticError(
						PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
						std::format("einops error: {}", result.error())
				);
				return &graph().addErrorNode();
			}
			return static_cast<const Node*>(&result->get());
		}

		virtual std::any visitMethodcallExpression(mdlParser::MethodcallExpressionContext* ctx) override {
			auto calleeObj = asObject(ctx->func->accept(this));
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
				errorlistener.semanticError(
						PrettyErrorListener::ErrorLevel::Error, tok, tok->getLine(), tok->getCharPositionInLine(),
						"Object is not callable with these argument types"
				);
				return &graph().addErrorNode();
			}
			/** \todo descriptive error message when arguments / argument count does not match **/
			/** \todo For member function calls (obj.func(...)), obj should be passed as the first ("this") argument.
			 *  This is not yet implemented. */
			return &graph().addInvocationNode(decl.value(), std::move(args));
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
			const auto* node = asNode(ctx->expr()->accept(this), ctx->expr()->start);
			/** \todo check datatype with ctx->type() **/
			currentNamespace().put(ctx->IDENT()->getText(), Object{node});
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