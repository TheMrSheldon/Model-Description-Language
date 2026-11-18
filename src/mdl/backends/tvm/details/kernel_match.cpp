// Matching kernel packages at the matching boundary (see kernel_match.hpp and docs/dev/spec/kernel-packages.md).
#include "./kernel_match.hpp"
#include "./extern_call.hpp"
#include "./legalize.hpp"
#include "./tvm_compat.hpp"

#include <mdl/kernels/expr.hpp>

#include <tvm/ffi/reflection/accessor.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/utils.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace mdl::backend::tvm {

	using ::tvm::Expr;
	using ::tvm::PrimExpr;
	using ::tvm::Var;
	using ::tvm::ffi::Array;
	using ::tvm::ffi::Map;
	namespace relax = ::tvm::relax;
	namespace expr = mdl::kernels::expr;

	namespace {
		/** @brief An attribute value of a manifest (as written, e.g. `-1`, `[-1]`, `true`, `1e-5`) for an AttrPattern. */
		::tvm::ffi::Any attrValue(const std::string& text) {
			if (text == "true" || text == "false")
				return text == "true";
			if (text.starts_with("[")) {
				Array<int64_t> values;
				std::stringstream list{text.substr(1, text.size() - 2)};
				std::string item;
				while (std::getline(list, item, ','))
					if (!item.empty())
						values.push_back(std::stoll(item));
				return values;
			}
			if (text.find_first_of(".eE") != std::string::npos)
				return std::stod(text);
			return static_cast<int64_t>(std::stoll(text));
		}

		/** @brief A compiled pattern: the DFPattern and the nodes that need checks after matching. */
		struct CompiledPattern {
			relax::DFPattern root;
			std::vector<std::pair<std::string, relax::DFPattern>> captures; ///< by name (one pattern per name)
			std::vector<std::pair<const kernels::PatternNode*, relax::DFPattern>> constants;
			std::vector<relax::DFPattern> ops; ///< the operator nodes (their matches are the replaced bindings)
			std::vector<std::pair<std::string, std::string>> notOps; ///< (capture, operator that must not produce it)
		};

		relax::DFPattern compileNode(const kernels::PatternNode& node, CompiledPattern& out) {
			switch (node.kind) {
			case kernels::PatternNode::Kind::capture: {
				auto it =
						std::ranges::find(out.captures, node.capture, &std::pair<std::string, relax::DFPattern>::first);
				if (it != out.captures.end())
					return it->second; // the same name: the same value
				auto w = relax::WildcardPattern();
				out.captures.emplace_back(node.capture, w);
				for (const auto& op : node.notOps) {
					(void)::tvm::Op::Get(op); // (throws for an unknown operator)
					out.notOps.emplace_back(node.capture, op);
				}
				return w;
			}
			case kernels::PatternNode::Kind::constant: {
				auto w = relax::WildcardPattern();
				out.constants.emplace_back(&node, w);
				return w;
			}
			case kernels::PatternNode::Kind::op: {
				Array<relax::DFPattern> args;
				for (const auto& arg : node.args)
					args.push_back(compileNode(arg, out));
				relax::DFPattern p = relax::IsOp(node.op)(args); // throws for an unknown operator
				if (!node.attrs.empty()) {
					Map<::tvm::ffi::String, ::tvm::ffi::Any> attrs;
					for (const auto& [name, value] : node.attrs)
						attrs.Set(name, attrValue(value));
					p = p.HasAttr(attrs);
				}
				out.ops.push_back(p);
				if (!node.capture.empty())
					out.captures.emplace_back(node.capture, p);
				return p;
			}
			}
			return relax::WildcardPattern();
		}

		/** @brief A value of the expression language during evaluation. */
		struct Value {
			expr::Type type;
			PrimExpr scalar; ///< integer, boolean, or real
			Array<PrimExpr> shape;
			::tvm::PrimType dtype = ::tvm::PrimType::Void();
			std::vector<::tvm::PrimType> dtypes; ///< a list of element types (for `in`)
		};

		struct EvalError {
			std::string message;
		};

		PrimExpr i64(int64_t v) { return ::tvm::IntImm(::tvm::PrimType::Int(64), v); }
		PrimExpr asI64(const PrimExpr& e) { return ::tvm::prim::cast(::tvm::PrimType::Int(64), e); }
		PrimExpr boolean(bool b) { return ::tvm::IntImm::Bool(b); }

		::tvm::PrimType dtypeNamed(const std::string& name) {
			if (name == "bfloat16")
				return ::tvm::PrimType::BFloat(16);
			if (name == "bool")
				return ::tvm::PrimType::Bool();
			const bool isUInt = name.starts_with("uint"), isInt = !isUInt && name.starts_with("int");
			const int bits = std::stoi(name.substr(isUInt ? 4 : isInt ? 3 : 5));
			return isUInt  ? ::tvm::PrimType::UInt(bits)
				   : isInt ? ::tvm::PrimType::Int(bits)
						   : ::tvm::PrimType::Float(bits);
		}

		/** @brief Evaluates expressions over the captures of a match (symbolic dimensions stay symbolic). */
		class Evaluator {
		public:
			/** @param calls the operator calls whose attributes are available (`out` and the captured operators) */
			explicit Evaluator(
					const std::map<std::string, Expr>& captures, const std::map<std::string, Expr>& calls = {}
			)
					: captures(captures), calls(calls) {}

			Value eval(const expr::Node& n) {
				using K = expr::Node::Kind;
				switch (n.kind) {
				case K::integer:
					return {.type = expr::Type::integer, .scalar = i64(n.value)};
				case K::dtypeName:
					return {.type = expr::Type::dtype, .dtype = dtypeNamed(n.name)};
				case K::list: {
					Value v{.type = n.type};
					for (const auto& e : n.args) {
						auto element = eval(*e);
						if (element.type == expr::Type::dtype)
							v.dtypes.push_back(element.dtype);
						else
							v.shape.push_back(element.scalar);
					}
					return v;
				}
				case K::access:
					return access(n);
				case K::unary: {
					auto x = eval(*n.args[0]);
					if (n.op == "not")
						return {.type = expr::Type::boolean, .scalar = !x.scalar};
					return {.type = expr::Type::integer, .scalar = -x.scalar};
				}
				case K::binary:
					return binary(n);
				case K::in: {
					auto x = eval(*n.args[0]);
					auto list = eval(*n.args[1]);
					if (x.type == expr::Type::dtype)
						return {.type = expr::Type::boolean,
								.scalar = boolean(std::ranges::find(list.dtypes, x.dtype) != list.dtypes.end())};
					PrimExpr any = boolean(false);
					for (const auto& e : list.shape)
						any = any || (x.scalar == e);
					return {.type = expr::Type::boolean, .scalar = any};
				}
				case K::index: {
					auto shape = eval(*n.args[0]).shape;
					auto index = ::tvm::sym::Analyzer()->Simplify(eval(*n.args[1]).scalar);
					const auto* imm = index.as<::tvm::IntImmNode>();
					if (imm == nullptr)
						throw EvalError{"an index must be a constant"};
					int64_t i = intValue(imm);
					const auto size = static_cast<int64_t>(shape.size());
					if (i < 0)
						i += size;
					if (i < 0 || i >= size)
						throw EvalError{std::format("index {} is out of range for {} dimension(s)", intValue(imm), size)
						};
					return {.type = expr::Type::integer, .scalar = asI64(shape[i])};
				}
				case K::slice: {
					auto shape = eval(*n.args[0]).shape;
					const auto size = static_cast<int64_t>(shape.size());
					auto clamp = [&](int64_t i) { return std::clamp(i < 0 ? i + size : i, int64_t{0}, size); };
					const int64_t from = n.hasFrom ? clamp(n.from) : 0, to = n.hasTo ? clamp(n.to) : size;
					Value v{.type = expr::Type::shape};
					for (int64_t i = from; i < to; ++i)
						v.shape.push_back(shape[i]);
					return v;
				}
				case K::call: {
					auto shape = eval(*n.args[0]).shape;
					if (n.name == "len")
						return {.type = expr::Type::integer, .scalar = i64(static_cast<int64_t>(shape.size()))};
					PrimExpr p = i64(1);
					for (const auto& d : shape)
						p = p * asI64(d);
					return {.type = expr::Type::integer, .scalar = p};
				}
				}
				throw EvalError{"unsupported expression"};
			}

		private:
			const std::map<std::string, Expr>& captures;
			const std::map<std::string, Expr>& calls;

			/** @brief A numeric attribute of a captured operator call (by reflection, as TVM's matcher reads them). */
			Value attribute(const std::string& capture, const std::string& name) {
				auto it = calls.find(capture);
				const auto* call = it == calls.end() ? nullptr : it->second.as<::tvm::CallNode>();
				if (call == nullptr || !call->attrs.defined())
					throw EvalError{std::format("${} is not an operator with attributes", capture)};
				std::optional<double> value;
				const auto* info = TVMFFIGetTypeInfo(call->attrs->type_index());
				::tvm::ffi::reflection::ForEachFieldInfoWithEarlyStop(info, [&](const TVMFFIFieldInfo* field) {
					if (std::string_view(field->name.data, field->name.size) != name)
						return false;
					::tvm::ffi::Any v = ::tvm::ffi::reflection::FieldGetter(field)(call->attrs.get());
					if (auto d = v.try_cast<double>())
						value = *d;
					else if (auto i = v.try_cast<int64_t>())
						value = static_cast<double>(*i);
					return true;
				});
				if (!value)
					throw EvalError{std::format("${} has no numeric attribute '{}'", capture, name)};
				return {.type = expr::Type::real, .scalar = ::tvm::FloatImm(::tvm::PrimType::Float(64), *value)};
			}

			Value access(const expr::Node& n) {
				if (n.field.starts_with("attr."))
					return attribute(n.name, n.field.substr(5));
				auto it = captures.find(n.name);
				if (it == captures.end())
					throw EvalError{std::format("${} is not available", n.name)};
				const auto* info = tensorInfo(it->second);
				if (n.field == "data")
					return {.type = expr::Type::pointer};
				if (info == nullptr)
					throw EvalError{std::format("${} is not a tensor", n.name)};
				if (n.field == "dtype") {
					if (info->IsUnknownDtype())
						throw EvalError{std::format("the element type of ${} is unknown", n.name)};
					return {.type = expr::Type::dtype, .dtype = info->dtype.value()};
				}
				auto shape = knownShape(it->second);
				if (!shape)
					throw EvalError{std::format("the shape of ${} is unknown", n.name)};
				if (n.field == "shape")
					return {.type = expr::Type::shape, .shape = *shape};
				if (n.field == "ndim")
					return {.type = expr::Type::integer, .scalar = i64(static_cast<int64_t>(shape->size()))};
				throw EvalError{std::format("'{}' is not supported yet", n.field)};
			}

			Value binary(const expr::Node& n) {
				auto a = eval(*n.args[0]), b = eval(*n.args[1]);
				const auto& op = n.op;
				auto result = [](PrimExpr e, expr::Type t) { return Value{.type = t, .scalar = std::move(e)}; };
				if (op == "and")
					return result(a.scalar && b.scalar, expr::Type::boolean);
				if (op == "or")
					return result(a.scalar || b.scalar, expr::Type::boolean);
				if (a.type == expr::Type::dtype)
					return result(boolean((a.dtype == b.dtype) == (op == "==")), expr::Type::boolean);
				if (a.type == expr::Type::shape) {
					PrimExpr equal = boolean(a.shape.size() == b.shape.size());
					for (size_t i = 0; i < a.shape.size() && i < b.shape.size(); ++i)
						equal = equal && (asI64(a.shape[i]) == asI64(b.shape[i]));
					return result(op == "==" ? equal : !equal, expr::Type::boolean);
				}
				const PrimExpr x = a.scalar, y = b.scalar;
				if (op == "+")
					return result(x + y, expr::Type::integer);
				if (op == "-")
					return result(x - y, expr::Type::integer);
				if (op == "*")
					return result(x * y, expr::Type::integer);
				if (op == "//" || op == "%") {
					if (::tvm::prim::is_zero(::tvm::sym::Analyzer()->Simplify(y)))
						throw EvalError{"division by zero"};
					return result(op == "//" ? ::tvm::floordiv(x, y) : ::tvm::floormod(x, y), expr::Type::integer);
				}
				if (op == "==")
					return result(x == y, expr::Type::boolean);
				if (op == "!=")
					return result(x != y, expr::Type::boolean);
				if (op == "<")
					return result(x < y, expr::Type::boolean);
				if (op == "<=")
					return result(x <= y, expr::Type::boolean);
				if (op == ">")
					return result(x > y, expr::Type::boolean);
				return result(x >= y, expr::Type::boolean);
			}
		};

		std::string text(const PrimExpr& e) {
			std::ostringstream out;
			out << e;
			// Integer constants without TVM's type wrapper: `T.int64(16)` -> `16`.
			static const std::regex constant{R"(T\.u?int(8|16|32|64)\((-?[0-9]+)\))"};
			return std::regex_replace(out.str(), constant, "$2");
		}

		/** @brief The verdict of a condition: proven, refuted, or unknown (with its simplified form). */
		Verdict condition(const std::string& what, const PrimExpr& cond) {
			auto simplified = ::tvm::sym::Analyzer()->Simplify(cond);
			if (const auto* imm = simplified.as<::tvm::IntImmNode>())
				return {what, intValue(imm) != 0 ? Verdict::Result::proven : Verdict::Result::refuted, "", {}};
			const bool symbolic = !::tvm::tirx::UndefinedVars(simplified).empty();
			if (!symbolic)
				return {what, Verdict::Result::unknown, "undecided at compile time: " + text(simplified), {}};
			return {what, Verdict::Result::unknown, "checked at runtime: " + text(simplified),
					Verdict::Class::shapeArithmetic, simplified};
		}

		std::string dtypeName(const ::tvm::PrimType& t) {
			std::ostringstream out;
			out << t;
			std::string s = out.str();
			if (s.starts_with("T."))
				s = s.substr(2);
			return s;
		}

		/** @brief The uses of every variable (bindings and the function's result). */
		std::unordered_map<const ::tvm::VarNode*, std::vector<const ::tvm::VarNode*>>
		usersOf(const relax::SeqExprNode& seq, const ::tvm::VarNode* result) {
			std::unordered_map<const ::tvm::VarNode*, std::vector<const ::tvm::VarNode*>> users;
			for (const auto& block : seq.blocks)
				for (const auto& binding : block->bindings)
					if (const auto* vb = binding.as<relax::VarBindingNode>())
						relax::PostOrderVisit(vb->value, [&](const Expr& e) {
							if (const auto* v = e.as<::tvm::VarNode>())
								users[v].push_back(vb->var.get());
						});
			relax::PostOrderVisit(seq.body, [&](const Expr& e) {
				if (const auto* v = e.as<::tvm::VarNode>())
					users[v].push_back(result);
			});
			return users;
		}

		/** @brief The first artifact of @p impl that can be used for @p target, and why the others cannot. */
		std::pair<std::string, std::string> chooseArtifact(
				const kernels::Package& p, const kernels::Implementation& impl, const kernels::Platform& target
		) {
			std::string why;
			for (const auto& id : impl.artifacts) {
				const auto* a = p.artifact(id);
				std::error_code ec;
				std::string reason = !kernels::forDevice(*a, target)
											 ? (a->deviceArch.empty() ? std::string("not for the device ") + target.deviceArch
																	  : "not for " + (target.deviceArch.empty() ? std::string("the CPU") : target.deviceArch))
									 : !a->supported() ? "cubin and PTX artifacts are not supported yet"
									 : !kernels::compatible(a->platform, target)	  ? "not for " + target.str()
									 : !std::filesystem::is_regular_file(a->path, ec) ? "missing " + a->path.string()
																					  : "";
				if (reason.empty())
					return {id, ""};
				why += (why.empty() ? "" : "; ") + id + ": " + reason;
			}
			return {"", why};
		}
	} // namespace

	namespace {
		/** @brief The plan of a named adapter (see `adapter:` in docs/dev/spec/kernel-packages.md) for the root call of a region. */
		std::expected<ExternPlan, std::string> adapterPlan(const std::string& adapter, const Expr& root) {
			const auto* call = root.as<::tvm::CallNode>();
			if (call == nullptr)
				return std::unexpected("the region's result is not an operator call");
			const auto c = ::tvm::ffi::GetRef<Call>(call);
			if (adapter == "matmul-strided" || adapter == "matmul-quantized")
				return matmulAdapter(c, adapter == "matmul-quantized");
			return std::unexpected("unknown adapter " + adapter);
		}
	} // namespace

	MatchResult matchKernels(
			const ::tvm::IRModule& mod, const std::vector<kernels::Package>& packages, const kernels::Platform& target
	) {
		MatchResult result;
		// Compile the patterns of all implementations (an unknown operator is a problem of the package).
		struct Entry {
			const kernels::Package* package;
			const kernels::Implementation* impl;
			const kernels::Pattern* pattern;
			CompiledPattern compiled;
		};
		std::vector<Entry> entries;
		for (const auto& p : packages)
			for (const auto& impl : p.implementations) {
				const auto* pattern = p.pattern(impl.pattern);
				Entry e{&p, &impl, pattern, {}};
				try {
					e.compiled.root = compileNode(pattern->root, e.compiled);
				} catch (const std::exception& ex) {
					result.problems.push_back({&p, pattern->id, ex.what()});
					continue;
				}
				entries.push_back(std::move(e));
			}

		for (const auto& [gv, base] : mod->functions) {
			const auto* fn = base.as<relax::FunctionNode>();
			const auto* seq = fn ? fn->body.as<relax::SeqExprNode>() : nullptr;
			if (seq == nullptr)
				continue;
			const auto func = ::tvm::ffi::GetRef<relax::Function>(fn);
			const auto bindings = relax::AnalyzeVar2Value(func);
			std::unordered_map<const ::tvm::ExprNode*, Var> varOf; // the variable each call is bound to
			for (const auto& [var, value] : bindings)
				varOf.emplace(value.get(), var);
			const auto users = usersOf(*seq, nullptr);

			for (const auto& block : seq->blocks)
				for (const auto& binding : block->bindings) {
					const auto* vb = binding.as<relax::VarBindingNode>();
					if (vb == nullptr || vb->value.as<::tvm::CallNode>() == nullptr)
						continue;
					for (const auto& e : entries) {
						auto matched = relax::ExtractMatchedExpr(e.compiled.root, vb->value, bindings);
						if (!matched)
							continue;
						const auto& m = matched.value();

						Candidate c{.package = e.package, .implementation = e.impl, .function = gv->name_hint};
						// The region: the bindings of the matched operator calls (the root last).
						for (const auto& op : e.compiled.ops) {
							auto it = varOf.find(m.at(op).get());
							const Var var = it != varOf.end() ? it->second : vb->var;
							if (std::ranges::none_of(c.replaced, [&](const Var& v) { return v.same_as(var); }))
								c.replaced.push_back(var);
						}
						std::ranges::stable_partition(c.replaced, [&](const Var& v) { return !v.same_as(vb->var); });
						std::map<std::string, Expr> captures, calls{{"out", vb->value}};
						c.calls.emplace_back("out", vb->value);
						for (const auto& [name, pattern] : e.compiled.captures) {
							// A value bound to a variable is captured as the variable (the matcher looks through it).
							Expr value = m.at(pattern);
							if (value.as<::tvm::CallNode>() != nullptr) { // a captured operator (its attributes)
								calls[name] = value;
								c.calls.emplace_back(name, value);
							}
							if (auto it = varOf.find(value.get()); it != varOf.end())
								value = it->second;
							captures[name] = value;
							c.captures.emplace_back(name, value);
						}
						captures["out"] = vb->var;

						// Captures that must not be produced by certain operators.
						for (const auto& [name, op] : e.compiled.notOps) {
							Expr value = captures.at(name);
							if (const auto* var = value.as<::tvm::VarNode>())
								if (auto it = bindings.find(::tvm::ffi::GetRef<Var>(var)); it != bindings.end())
									value = (*it).second;
							const auto* producer = value.as<::tvm::CallNode>();
							const bool produced = producer != nullptr && producer->op.same_as(::tvm::Op::Get(op));
							c.verdicts.push_back(
									{std::format("${} is not produced by {}", name, op),
									 produced ? Verdict::Result::refuted : Verdict::Result::proven,
									 "",
									 {}}
							);
						}

						// Region rules: an intermediate result must not be used outside the region.
						std::unordered_set<const ::tvm::VarNode*> region;
						for (const auto& v : c.replaced)
							region.insert(v.get());
						for (const auto& v : c.replaced) {
							if (v.same_as(vb->var))
								continue;
							auto it = users.find(v.get());
							const bool escapes = it != users.end() && std::ranges::any_of(it->second, [&](auto* user) {
													 return user == nullptr || !region.contains(user);
												 });
							c.verdicts.push_back(
									{std::format("{} is used only inside the region", std::string(v->name)),
									 escapes ? Verdict::Result::refuted : Verdict::Result::proven,
									 "",
									 {}}
							);
						}

						// Constants: exact unless a tolerance is given; the value may be an expression over the captures.
						for (const auto& [node, pattern] : e.compiled.constants) {
							const auto actual = scalarConstant(m.at(pattern));
							std::optional<double> expected = node->value;
							std::string spec = std::format("{}", node->value);
							if (!node->valueExpr.empty()) {
								spec = node->valueExpr;
								expected.reset();
								try {
									auto parsed = expr::parse(node->valueExpr, [&] {
										std::vector<std::string> names;
										for (const auto& [n, _] : captures)
											names.push_back(n);
										return names;
									}());
									auto value =
											::tvm::sym::Analyzer()->Simplify(Evaluator{captures}.eval(**parsed).scalar);
									if (const auto* imm = value.as<::tvm::IntImmNode>())
										expected = static_cast<double>(intValue(imm));
								} catch (const EvalError&) {
								}
							}
							const std::string what = std::format("the constant is {}", spec);
							if (!actual)
								c.verdicts.push_back({what, Verdict::Result::refuted, "it is not a scalar constant", {}}
								);
							else if (!expected)
								c.verdicts.push_back(
										{what, Verdict::Result::unknown, "the expected value is not a constant", {}}
								);
							else {
								const bool ok = std::abs(*actual - *expected) <= node->tolerance.value_or(0.0);
								c.verdicts.push_back(
										{what,
										 ok ? Verdict::Result::proven : Verdict::Result::refuted,
										 ok ? "" : std::format("it is {:.17g}", *actual),
										 {}}
								);
							}
						}

						// The artifact.
						auto [artifact, why] = chooseArtifact(*e.package, *e.impl, target);
						c.artifact = artifact;
						c.verdicts.push_back(
								{"a usable artifact",
								 artifact.empty() ? Verdict::Result::refuted : Verdict::Result::proven,
								 artifact.empty() ? why : artifact,
								 {}}
						);

						// Applicability and the declared outputs.
						Evaluator eval{captures, calls};
						for (const auto& [name, allowed] : e.impl->dtypes) {
							const auto* info = tensorInfo(captures.at(name));
							const std::string actual =
									info && !info->IsUnknownDtype() ? dtypeName(info->dtype.value()) : "unknown";
							const bool ok = std::ranges::find(allowed, actual) != allowed.end();
							c.verdicts.push_back(
									{std::format("${}.dtype is {}", name, allowed.size() == 1 ? allowed[0] : "allowed"),
									 ok ? Verdict::Result::proven : Verdict::Result::refuted,
									 ok ? "" : "it is " + actual,
									 {}}
							);
						}
						for (const auto& [name, layout] : e.impl->layouts)
							c.verdicts.push_back(
									{std::format("${} is {}", name, layout),
									 Verdict::Result::proven,
									 "tensors are dense and row-major at the matching boundary",
									 {}}
							);
						auto evaluate = [&](const std::string& text, const std::string& what) {
							try {
								auto parsed = expr::parse(text, [&] {
									std::vector<std::string> names;
									for (const auto& [n, _] : captures)
										names.push_back(n);
									return names;
								}());
								if (!parsed)
									throw EvalError{parsed.error()};
								c.verdicts.push_back(condition(what, eval.eval(**parsed).scalar));
							} catch (const EvalError& err) {
								c.verdicts.push_back({what, Verdict::Result::unknown, err.message, {}});
							}
						};
						for (const auto& w : e.impl->where)
							evaluate(w, "where " + w);
						// The arguments: integers must be computable, and fit their type (a checked conversion).
						if (!e.impl->adapter.empty()) {
							auto plan = adapterPlan(e.impl->adapter, vb->value);
							c.verdicts.push_back(
									{"the argument adapter " + e.impl->adapter + " applies",
									 plan ? Verdict::Result::proven : Verdict::Result::refuted,
									 plan ? "" : plan.error(),
									 {}}
							);
						}
						for (size_t i = 0; i < e.impl->arguments.size(); ++i) {
							const auto& arg = e.impl->arguments[i];
							if (arg.type == "pointer")
								continue;
							if (arg.type == "float32" || arg.type == "float64") {
								const std::string what =
										std::format("argument {} ({}) can be computed", i + 1, arg.value);
								try {
									auto parsed = expr::parse(arg.value, [&] {
										std::vector<std::string> names;
										for (const auto& [n, _] : captures)
											names.push_back(n);
										return names;
									}());
									if (!parsed)
										throw EvalError{parsed.error()};
									auto v = ::tvm::sym::Analyzer()->Simplify(eval.eval(**parsed).scalar);
									const bool constant = v.as<::tvm::FloatImmNode>() || v.as<::tvm::IntImmNode>();
									c.verdicts.push_back(
											{what,
											 constant ? Verdict::Result::proven : Verdict::Result::unknown,
											 constant ? "" : "not a constant",
											 {}}
									);
								} catch (const EvalError& err) {
									c.verdicts.push_back({what, Verdict::Result::unknown, err.message, {}});
								}
								continue;
							}
							if (arg.type == "int32")
								evaluate(
										std::format("{} >= -2147483648 and {} <= 2147483647", arg.value, arg.value),
										std::format("argument {} ({}) fits int32", i + 1, arg.value)
								);
							else
								evaluate(
										std::format("{} == {}", arg.value, arg.value),
										std::format("argument {} ({}) can be computed", i + 1, arg.value)
								);
						}
						if (!e.impl->outputs.empty()) {
							const auto& [shape, dtype] = e.impl->outputs.front();
							evaluate(std::format("{} == $out.shape", shape), "the result's shape is " + shape);
							evaluate(std::format("{} == $out.dtype", dtype), "the result's element type is " + dtype);
						}

						// Unknown shape arithmetic becomes a runtime guard (the original region is the fallback).
						c.eligible = std::ranges::all_of(c.verdicts, [](const Verdict& v) {
							return v.result == Verdict::Result::proven || v.guard.defined();
						});
						c.guarded = c.eligible && std::ranges::any_of(c.verdicts, [](const Verdict& v) {
										return v.result != Verdict::Result::proven;
									});
						if (!c.eligible) {
							const auto refuted = std::ranges::count_if(c.verdicts, [](const Verdict& v) {
								return v.result == Verdict::Result::refuted;
							});
							const auto unknown = std::ranges::count_if(c.verdicts, [](const Verdict& v) {
								return v.result == Verdict::Result::unknown;
							});
							c.reason = refuted > 0 ? std::format("{} constraint(s) not met", refuted)
												   : std::format("{} constraint(s) undecided at compile time", unknown);
						}
						result.candidates.push_back(std::move(c));
					}
				}
		}

		// Selection: deterministic order, non-overlapping replaced bindings.
		std::vector<Candidate*> order;
		for (auto& c : result.candidates)
			if (c.eligible)
				order.push_back(&c);
		std::ranges::stable_sort(order, [](const Candidate* a, const Candidate* b) {
			return std::tuple{
						   -a->implementation->priority, -static_cast<int64_t>(a->replaced.size()), a->package->name,
						   a->implementation->id
				   } < std::tuple{
							   -b->implementation->priority, -static_cast<int64_t>(b->replaced.size()),
							   b->package->name, b->implementation->id
					   };
		});
		std::unordered_map<const ::tvm::VarNode*, const Candidate*> taken;
		for (auto* c : order) {
			const Candidate* conflict = nullptr;
			for (const auto& v : c->replaced)
				if (auto it = taken.find(v.get()); it != taken.end())
					conflict = it->second;
			if (conflict != nullptr) {
				c->outcome = Candidate::Outcome::overlapped;
				c->reason = std::format(
						"overlaps {}/{}, which takes precedence", conflict->package->name, conflict->implementation->id
				);
				continue;
			}
			c->outcome = Candidate::Outcome::selected;
			for (const auto& v : c->replaced)
				taken.emplace(v.get(), c);
		}
		return result;
	}

	namespace {
		/** @brief Rewrites the root bindings of the selected regions into calls of their implementations. */
		class Emitter : public relax::ExprMutator {
		public:
			Emitter(const ::tvm::IRModule& mod, const MatchResult& result) : relax::ExprMutator(mod) {
				for (const auto& c : result.candidates)
					if (c.outcome == Candidate::Outcome::selected)
						roots.emplace(c.replaced.back().get(), &c);
			}

			::tvm::IRModule run(const ::tvm::IRModule& mod) {
				for (const auto& [gv, base] : mod->functions)
					if (const auto* fn = base.as<relax::FunctionNode>()) {
						const auto func = ::tvm::ffi::GetRef<relax::Function>(fn);
						valueOf.clear();
						position.clear();
						for (const auto& [var, value] : relax::AnalyzeVar2Value(func))
							valueOf.emplace(var.get(), value);
						relax::PostOrderVisit(func, [&](const Expr& e) {
							if (const auto* vb = e.as<relax::SeqExprNode>())
								for (const auto& block : vb->blocks)
									for (const auto& binding : block->bindings)
										position.emplace(binding->var.get(), position.size());
						});
						auto updated = VisitExpr(func);
						builder_->UpdateFunction(gv, updated.as_or_throw<::tvm::BaseFunc>());
					}
				return builder_->GetContextIRModule();
			}

			using relax::ExprMutator::VisitBinding_;

			void VisitBinding_(const relax::VarBindingNode* binding) override {
				auto it = roots.find(binding->var.get());
				if (it == roots.end())
					return relax::ExprMutator::VisitBinding_(binding);
				const Candidate& c = *it->second;
				Expr kernel = call(c, binding->var);
				if (c.guarded)
					kernel = guarded(c, binding, kernel);
				ReEmitBinding(binding, builder_->Normalize(kernel));
			}

		private:
			std::unordered_map<const ::tvm::VarNode*, const Candidate*> roots;
			std::unordered_map<const ::tvm::VarNode*, Expr> valueOf; ///< the bindings of the current function
			std::unordered_map<const ::tvm::VarNode*, size_t> position;
			int64_t fallbacks = 0;

			/**
			 * @brief `if guard: kernel else: fallback(...)`: the region becomes a private function, which TVM fuses like
			 * any other and which is inlined after fusion (inlineGuardFallbacks; FuseOps does not look into `If`
			 * branches); its original bindings become dead code.
			 */
			Expr guarded(const Candidate& c, const relax::VarBindingNode* root, const Expr& kernel) {
				PrimExpr cond;
				for (const auto& v : c.verdicts)
					if (v.result != Verdict::Result::proven)
						cond = cond.defined() ? ::tvm::logical_and(cond, v.guard) : v.guard;
				std::vector<Var> order = c.replaced;
				std::ranges::sort(order, {}, [&](const Var& v) { return position.at(v.get()); });
				auto valueOfVar = [&](const Var& v) {
					return v.same_as(root->var) ? root->value : valueOf.at(v.get());
				};

				// The parameters: the values the region uses from outside.
				std::unordered_set<const ::tvm::VarNode*> region;
				for (const auto& v : order)
					region.insert(v.get());
				std::vector<Var> inputs;
				for (const auto& v : order)
					for (const auto& free : relax::FreeVars(valueOfVar(v)))
						if (!region.contains(free.get()) &&
							std::ranges::none_of(inputs, [&](const Var& u) { return u.same_as(free); }))
							inputs.push_back(free);
				::tvm::ffi::Map<Var, Expr> fresh;
				Array<Var> params;
				for (const auto& v : inputs) {
					Var p(std::string(v->name), relax::GetType(v));
					fresh.Set(v, p);
					params.push_back(p);
				}
				Array<relax::Binding> bindings;
				for (const auto& v : order) {
					Var copy(std::string(v->name), relax::GetType(v));
					bindings.push_back(relax::VarBinding(copy, relax::Bind(valueOfVar(v), fresh)));
					fresh.Set(v, copy);
				}
				auto body = relax::SeqExpr({relax::BindingBlock(bindings)}, fresh.at(root->var));
				// (The function uses the caller's symbolic variables: InlinePrivateFunctions does not substitute them.)
				auto func = relax::Function(params, body, relax::GetType(root->var));
				Expr fallback;
				if (relax::FreeSymbolicVars(func).empty()) {
					Array<Expr> args;
					for (const auto& v : inputs)
						args.push_back(VisitExpr(v));
					// (A distinct value per fallback: the builder would merge equal functions, but each must have a
					// single caller, as the inliner does not copy them.)
					func = ::tvm::WithAttr(std::move(func), kGuardFallbackAttr, ++fallbacks);
					auto gv = builder_->AddFunction(func, c.implementation->symbol + "_fallback");
					fallback = Call(::tvm::Type::Missing(), gv, args);
				} else {
					// A symbolic variable that the parameters do not define: the region stays in the branch (unfused).
					::tvm::ffi::Map<Var, Expr> outer;
					for (size_t i = 0; i < inputs.size(); ++i)
						outer.Set(params[i], VisitExpr(inputs[i]));
					fallback = relax::Bind(body, outer);
				}
				return relax::If(cond, relax::SeqExpr({}, kernel), relax::SeqExpr({}, fallback));
			}

			Expr call(const Candidate& c, const Var& root) {
				const auto& impl = *c.implementation;
				std::map<std::string, Expr> captures;
				for (const auto& [name, value] : c.captures)
					captures[name] = value;
				captures["out"] = root;
				std::vector<std::string> names;
				for (const auto& [name, _] : captures)
					names.push_back(name);
				std::map<std::string, Expr> calls(c.calls.begin(), c.calls.end());
				Evaluator eval{captures, calls};

				ExternCall ext{builder_, impl.symbol};
				if (!impl.adapter.empty()) {
					auto plan = adapterPlan(impl.adapter, valueOf.at(root.get())).value(); // (checked when matching)
					std::vector<Expr> pointers;
					for (const auto& input : plan.inputs)
						pointers.push_back(ext.input(VisitExpr(input), "in" + std::to_string(pointers.size())));
					auto out = ext.output(plan.outShape, plan.outDtype);
					std::vector<Expr> args;
					std::vector<PrimExpr> scalars;
					for (const auto& arg : plan.args)
						switch (arg.kind) {
						case ExternPlan::Arg::Kind::input:
							args.push_back(pointers[arg.input]);
							break;
						case ExternPlan::Arg::Kind::output:
							args.push_back(out);
							break;
						case ExternPlan::Arg::Kind::scalar:
							scalars.push_back(arg.value);
							args.push_back(arg.value);
							break;
						}
					return ext.finish(std::move(args), scalars);
				}
				const auto outShape = knownShape(root).value();
				const auto outDtype = tensorInfo(root)->dtype.value();
				auto out = ext.output(outShape, outDtype);
				std::map<std::string, Expr> pointers; // one buffer per captured tensor
				std::vector<Expr> args;
				std::vector<PrimExpr> scalars;
				for (const auto& arg : impl.arguments) {
					auto parsed = expr::parse(arg.value, names).value(); // checked when the package was loaded
					if (arg.type == "pointer") {
						const auto& name = parsed->name;
						if (name == "out") {
							args.push_back(out);
							continue;
						}
						auto [p, inserted] = pointers.try_emplace(name);
						if (inserted)
							p->second = ext.input(VisitExpr(captures.at(name)), name);
						args.push_back(p->second);
						continue;
					}
					PrimExpr value = eval.eval(*parsed).scalar;
					const auto type = arg.type == "int32"	  ? ::tvm::PrimType::Int(32)
									  : arg.type == "float32" ? ::tvm::PrimType::Float(32)
									  : arg.type == "float64" ? ::tvm::PrimType::Float(64)
															  : ::tvm::PrimType::Int(64);
					value = ::tvm::prim::cast(type, ::tvm::sym::Analyzer()->Simplify(value));
					scalars.push_back(value);
					args.push_back(value);
				}
				return ext.finish(std::move(args), scalars);
			}
		};
	} // namespace

	::tvm::IRModule applyKernels(::tvm::IRModule mod, const MatchResult& result) {
		if (std::ranges::none_of(result.candidates, [](const Candidate& c) {
				return c.outcome == Candidate::Outcome::selected;
			}))
			return mod;
		mod = Emitter{mod, result}.run(mod);
		return relax::transform::DeadCodeElimination({})(mod);
	}

	namespace {
		/** @brief Replaces the calls of guard fallbacks by their bodies (each fallback has a single caller). */
		class FallbackInliner : public relax::ExprMutator {
		public:
			explicit FallbackInliner(const ::tvm::IRModule& mod) : relax::ExprMutator(mod), mod(mod) {}

			using relax::ExprMutator::VisitExpr_;

			Expr VisitExpr_(const ::tvm::CallNode* op) override {
				auto call = relax::ExprMutator::VisitExpr_(op).as_or_throw<Call>();
				const auto* gv = call->op.as<::tvm::GlobalVarNode>();
				if (gv == nullptr)
					return call;
				const auto* callee = mod->Lookup(::tvm::ffi::GetRef<::tvm::GlobalVar>(gv)).as<relax::FunctionNode>();
				if (callee == nullptr || !callee->HasNonzeroAttr(kGuardFallbackAttr))
					return call;
				// The parameters' shapes use the caller's symbolic variables (see guarded): only the parameters
				// themselves are substituted.
				::tvm::ffi::Map<Var, Expr> args;
				for (size_t i = 0; i < callee->params.size(); ++i)
					args.Set(callee->params[i], call->args[i]);
				return relax::Bind(callee->body, args);
			}

		private:
			::tvm::IRModule mod;
		};
	} // namespace

	::tvm::transform::Pass inlineGuardFallbacks() {
		return ::tvm::transform::CreateModulePass(
				[](::tvm::IRModule mod, ::tvm::transform::PassContext) {
					std::vector<::tvm::GlobalVar> fallbacks;
					for (const auto& [gv, f] : mod->functions)
						if (f->HasNonzeroAttr(kGuardFallbackAttr))
							fallbacks.push_back(gv);
					for (const auto& [gv, f] : mod->functions)
						if (fallbacks.empty())
							return mod;
					std::vector<std::pair<::tvm::GlobalVar, relax::Function>> callers;
					for (const auto& [gv, f] : mod->functions)
						if (const auto* fn = f.as<relax::FunctionNode>(); fn && !fn->HasNonzeroAttr(kGuardFallbackAttr))
							callers.emplace_back(gv, ::tvm::ffi::GetRef<relax::Function>(fn));
					FallbackInliner inliner{mod};
					for (const auto& [gv, fn] : callers)
						mod->Update(gv, inliner.VisitExpr(fn).as_or_throw<::tvm::BaseFunc>());
					for (const auto& gv : fallbacks)
						mod->Remove(gv);
					return mod;
				},
				0, "mdl.InlineGuardFallbacks", {}
		);
	}

	bool needsNonDataflow(const MatchResult& result) {
		return std::ranges::any_of(result.candidates, [](const Candidate& c) {
			return c.outcome == Candidate::Outcome::selected && c.guarded;
		});
	}

	void explain(std::ostream& out, const MatchResult& result, const kernels::Discovery& discovery, bool verbose) {
		if (!verbose) {
			for (const auto& p : discovery.problems)
				out << "problem: " << p.str() << "\n";
			for (const auto& p : result.problems)
				out << std::format("problem: {}: pattern '{}': {}\n", p.package->name, p.pattern, p.message);
		}
		if (verbose) {
			out << "Kernel directories:\n";
			if (discovery.dirs.empty())
				out << "  (none)\n";
			for (const auto& d : discovery.dirs)
				out << std::format("  {:<14} {}\n", kernels::str(d.origin), d.path.string());
			out << std::format("Packages: {}\n", discovery.packages.size());
			for (const auto& p : discovery.problems)
				out << "  problem: " << p.str() << "\n";
			for (const auto& p : result.problems)
				out << std::format("  problem: {}: pattern '{}': {}\n", p.package->name, p.pattern, p.message);
		}

		auto mark = [](const Verdict& v) {
			return v.result == Verdict::Result::proven	  ? "proven"
				   : v.result == Verdict::Result::refuted ? "not met"
				   : v.guard.defined()					  ? "guard"
														  : "unknown";
		};
		std::string function;
		size_t selected = 0;
		for (const auto& c : result.candidates) {
			if (!verbose) {
				if (c.outcome == Candidate::Outcome::selected) {
					++selected;
					out << std::format(
							"{}: {}/{} at {}{}\n", c.function, c.package->name, c.implementation->id,
							std::string(c.replaced.back()->name), c.guarded ? " (with a runtime guard)" : ""
					);
				}
				continue;
			}
			if (c.function != function) {
				function = c.function;
				out << "\nfunction " << function << "\n";
			}
			out << std::format(
					"  {}/{} (priority {}) at {}\n", c.package->name, c.implementation->id, c.implementation->priority,
					std::string(c.replaced.back()->name)
			);
			std::string replaced, captures;
			for (const auto& v : c.replaced)
				replaced += (replaced.empty() ? "" : ", ") + std::string(v->name);
			for (const auto& [name, e] : c.captures) {
				std::ostringstream value;
				if (const auto* var = e.as<::tvm::VarNode>())
					value << std::string(var->name);
				else
					value << e;
				captures += (captures.empty() ? "" : ", ") + std::format("${} = {}", name, value.str());
			}
			out << std::format("    replaces {}; captures {}\n", replaced, captures.empty() ? "none" : captures);
			for (const auto& v : c.verdicts) {
				out << std::format("    {:<8} {}", mark(v), v.what);
				if (!v.detail.empty())
					out << " (" << v.detail << ")";
				out << "\n";
			}
			switch (c.outcome) {
			case Candidate::Outcome::selected:
				++selected;
				out << (c.guarded ? "    => selected with a runtime guard (otherwise TVM's code)\n"
								  : "    => selected\n");
				break;
			case Candidate::Outcome::overlapped:
				out << "    => not selected: " << c.reason << "\n";
				break;
			case Candidate::Outcome::ineligible:
				out << "    => ineligible: " << c.reason << "\n";
				break;
			}
		}
		out << std::format("\n{} candidate(s), {} selected\n", result.candidates.size(), selected);
	}

} // namespace mdl::backend::tvm
