#include "./tvm.hpp"

#include "../../argvals.hpp"
#include "../../logging.hpp"
#include "./details/aot.hpp"
#include "./details/pipeline.hpp"
#include "./details/relax_builtins.hpp"

#include <mdl/compiler.hpp>
#include <mdl/exports.hpp>
#include <mdl/computegraphv2/datatype.hpp>
#include <mdl/computegraphv2/function.hpp>
#include <mdl/computegraphv2/graph.hpp>
#include <mdl/computegraphv2/node.hpp>
#include <mdl/parserv2/compilation_unit.hpp>
#include <mdl/utils/utils.hpp>

#include <tvm/ir/cast.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tirx/var.h>

#include <algorithm>
#include <any>
#include <concepts>
#include <utility>
#include <cassert>
#include <cctype>
#include <cmath>
#include <optional>
#include <cstdlib>
#include <format>
#include <filesystem>
#include <fstream>
#include <ranges>
#include <sstream>
#include <unordered_map>

// ── type helpers ──────────────────────────────────────────────────────────────

static ::tvm::runtime::DataType toTVMType(mdl::PrimitiveType p) {
	switch (p) {
	case mdl::PrimitiveType::boolean:
		return ::tvm::runtime::DataType::Bool();
	case mdl::PrimitiveType::floating:
		return ::tvm::runtime::DataType::Float(32);
	case mdl::PrimitiveType::uint:
		return ::tvm::runtime::DataType::UInt(32);
	case mdl::PrimitiveType::integer:
		return ::tvm::runtime::DataType::Int(32);
	}
	assert(false);
}

static ::tvm::runtime::DataType toUnderlyingTVMType(mdl::DataType dtype) {
	return std::visit(
			mdl::overloaded{
					[](const mdl::PrimitiveType& p) { return toTVMType(p); },
					[](const mdl::TensorType& t) { return toTVMType(t.type); },
					[](const mdl::AggregateType&) {
						assert(false);
						return ::tvm::runtime::DataType{};
					},
					[](const mdl::ModuleType&) {
						std::abort();
						return ::tvm::runtime::DataType{};
					},
					[](const mdl::BoundModuleType&) {
						std::abort();
						return ::tvm::runtime::DataType{};
					},
			},
			dtype
	);
}

/**
 * Recursively evaluate a compile-time-evaluable Node to a TVM PrimExpr.
 * HyperparameterNodes are looked up in argVals; arithmetic Invocations are folded.
 */
static ::tvm::PrimExpr nodeToPrimExpr(const mdl::Node& node, const mdl::ArgVals& argVals) {
	using FIDs = mdl::builtins::FunctionIDs;
	return std::visit(
			mdl::overloaded{
					[](const mdl::ConstNode& cn) -> ::tvm::PrimExpr {
						return std::visit(
								mdl::overloaded{
										[](int v) -> ::tvm::PrimExpr {
											return ::tvm::IntImm(::tvm::DataType::Int(32), v);
										},
										[](double v) -> ::tvm::PrimExpr {
											return ::tvm::FloatImm(::tvm::DataType::Float(64), v);
										},
										[](const std::vector<int>&) -> ::tvm::PrimExpr {
											return ::tvm::IntImm(::tvm::DataType::Int(32), 0);
										},
								},
								cn.value
						);
					},
					[&argVals](const mdl::HyperparameterNode& hn) -> ::tvm::PrimExpr {
						if (auto val = argVals.lookup(hn.name))
							return ::tvm::IntImm(::tvm::DataType::Int(32), *val);
						return ::tvm::tirx::Var(hn.name.empty() ? "dim" : hn.name, ::tvm::DataType::Int(32));
					},
					[&argVals](const mdl::InvocationNode& in) -> ::tvm::PrimExpr {
						if (in.arguments.size() == 2) {
							auto lhs = nodeToPrimExpr(in.arguments[0].get(), argVals);
							auto rhs = nodeToPrimExpr(in.arguments[1].get(), argVals);
							switch (in.id) {
							case FIDs::TENSOR_ADD:
								return lhs + rhs;
							case FIDs::TENSOR_SUB:
								return lhs - rhs;
							case FIDs::TENSOR_PROD:
								return lhs * rhs;
							case FIDs::TENSOR_DIV:
								return ::tvm::truncdiv(lhs, rhs);
							default:
								break;
							}
						}
						return ::tvm::tirx::Var("dim", ::tvm::DataType::Int(32));
					},
					[](const auto&) -> ::tvm::PrimExpr { return ::tvm::tirx::Var("dim", ::tvm::DataType::Int(32)); },
			},
			node
	);
}

/**
 * @brief Context of the translation of one function.
 * @details All ellipses (`...`) in the signature of an exported function denote the same runtime dimension `B`
 * (\todo support multi-dimensional ellipses).
 */
struct TranslationContext {
	const mdl::ArgVals& argVals;
	std::vector<::tvm::tirx::Var> ellipsis;

	TranslationContext(const mdl::ArgVals& argVals, unsigned ellipsisRank) : argVals(argVals) {
		for (unsigned i = 0; i < ellipsisRank; ++i)
			ellipsis.emplace_back(ellipsisRank == 1 ? std::string{"B"} : "B" + std::to_string(i), ::tvm::DataType::Int(64));
	}
};

static ::tvm::ffi::Array<::tvm::PrimExpr> getShape(mdl::DataType dtype, const TranslationContext& ctx) {
	const auto& argVals = ctx.argVals;
	return std::visit(
			mdl::overloaded{
					[](const mdl::PrimitiveType&) { return ::tvm::ffi::Array<::tvm::PrimExpr>{1}; },
					[&](const mdl::TensorType& ttype) {
						::tvm::ffi::Array<::tvm::PrimExpr> shape;
						for (const auto i : ttype.shape) {
							if (std::holds_alternative<mdl::Elipsis>(i)) {
								for (const auto& var : ctx.ellipsis)
									shape.push_back(var);
								continue;
							}
							auto expr = std::visit(
									mdl::overloaded{
											[&](mdl::Elipsis) -> ::tvm::PrimExpr { return ctx.ellipsis.front(); }, // expanded above
											[](unsigned u) -> ::tvm::PrimExpr {
												return ::tvm::IntImm(::tvm::DataType::Int(64), u);
											},
											[&argVals](const mdl::Node* node) {
												return ::tvm::cast(
														::tvm::DataType::Int(64), nodeToPrimExpr(*node, argVals)
												);
											},
									},
									i
							);
							shape.emplace_back(expr);
						}
						return shape;
					},
					[](const auto&) -> ::tvm::ffi::Array<::tvm::PrimExpr> {
						throw std::runtime_error("TVM backend: only tensors and scalars can be function arguments");
					},
			},
			dtype
	);
}

// ── Relax translation ─────────────────────────────────────────────────────────

// Cache: MDL Node → its Relax Expr (Var or PrimValue or ShapeExpr)
using RelaxCache = std::unordered_map<const mdl::Node*, ::tvm::relax::Expr>;

static ::tvm::relax::Expr translateRelax(
		::tvm::relax::BlockBuilder& bb, RelaxCache& cache, std::vector<::tvm::relax::Var>& params,
		const mdl::Node& node, const mdl::ArgVals& argVals
);

static ::tvm::relax::Expr translateRelax(
		::tvm::relax::BlockBuilder& bb, RelaxCache&, std::vector<::tvm::relax::Var>& params, const mdl::ConstNode& node,
		const mdl::ArgVals&
) {
	return std::visit(
			mdl::overloaded{
					[](int v) -> ::tvm::relax::Expr {
						// Scalar int: used as shape dim (RESHAPE/PERMUTE/EXPAND sentinels)
						return ::tvm::relax::PrimValue(::tvm::IntImm(::tvm::DataType::Int(32), v));
					},
					[](double v) -> ::tvm::relax::Expr {
						// Scalar float: used in arithmetic (e.g. 1e-12 epsilon, 0.5 scale)
						float fval = static_cast<float>(v);
						DLDevice dev = {kDLCPU, 0};
						auto data = ::tvm::runtime::Tensor::Empty(::tvm::ffi::Shape{}, ::tvm::DataType::Float(32), dev);
						data.CopyFromBytes(&fval, sizeof(float));
						return ::tvm::relax::Constant(data);
					},
					[](const std::vector<int>& v) -> ::tvm::relax::Expr {
						// Int vector: used as permutation axes in PERMUTE
						::tvm::ffi::Array<::tvm::PrimExpr> vals;
						for (int x : v)
							vals.push_back(::tvm::IntImm(::tvm::DataType::Int(32), x));
						return ::tvm::relax::ShapeExpr(vals);
					},
			},
			node.value
	);
}

static ::tvm::relax::Expr translateRelax(
		::tvm::relax::BlockBuilder& bb, RelaxCache&, std::vector<::tvm::relax::Var>& /*params*/,
		const mdl::ErrorNode& /*node*/, const mdl::ArgVals&
) {
	std::abort(); // compile-time error: should not reach here
}

template <typename LeafNode>
	requires std::same_as<LeafNode, mdl::ParameterNode> || std::same_as<LeafNode, mdl::InputNode>
static ::tvm::relax::Expr translateRelax(
		::tvm::relax::BlockBuilder&, RelaxCache&, std::vector<::tvm::relax::Var>&, const LeafNode& node,
		const mdl::ArgVals&
) {
	// All arguments and weights of an exported function are Relax parameters (pre-seeded in the cache).
	throw std::runtime_error("TVM backend: " + mdl::getName(node) + " is not an argument of the compiled function");
}

static ::tvm::relax::Expr translateRelax(
		::tvm::relax::BlockBuilder& bb, RelaxCache&, std::vector<::tvm::relax::Var>& /*params*/,
		const mdl::HyperparameterNode& node, const mdl::ArgVals& argVals
) {
	if (auto val = argVals.lookup(node.name))
		return ::tvm::relax::PrimValue(::tvm::IntImm(::tvm::DataType::Int(32), *val));
	// Symbolic hyperparameter
	return ::tvm::relax::PrimValue(::tvm::tirx::Var(node.name.empty() ? "dim" : node.name, ::tvm::DataType::Int(32)));
}

/**
 * @brief Evaluates a compile-time scalar expression (literals, hyperparameters, and elementwise builtins over them).
 * @return the value, or std::nullopt if @p node is not a compile-time scalar.
 */
static std::optional<double> evalScalar(const mdl::Node& node, const mdl::ArgVals& argVals) {
	using FIDs = mdl::builtins::FunctionIDs;
	return std::visit(
			mdl::overloaded{
					[](const mdl::ConstNode& cn) -> std::optional<double> {
						if (const auto* i = std::get_if<int>(&cn.value))
							return *i;
						if (const auto* d = std::get_if<double>(&cn.value))
							return *d;
						return std::nullopt;
					},
					[&](const mdl::HyperparameterNode& hn) -> std::optional<double> {
						if (auto v = argVals.lookup(hn.name))
							return static_cast<double>(*v);
						return std::nullopt;
					},
					[&](const mdl::InvocationNode& in) -> std::optional<double> {
						std::vector<double> args;
						for (const auto& arg : in.arguments) {
							auto v = evalScalar(arg.get(), argVals);
							if (!v)
								return std::nullopt;
							args.push_back(*v);
						}
						auto unary = [&](double (*f)(double)) -> std::optional<double> {
							return args.size() == 1 ? std::optional<double>{f(args[0])} : std::nullopt;
						};
						if (args.size() == 2) {
							switch (in.id) {
							case FIDs::TENSOR_ADD:
								return args[0] + args[1];
							case FIDs::TENSOR_SUB:
								return args[0] - args[1];
							case FIDs::TENSOR_PROD:
								return args[0] * args[1];
							case FIDs::TENSOR_DIV:
								return args[0] / args[1];
							default:
								return std::nullopt;
							}
						}
						switch (in.id) {
						case FIDs::SQRT:
							return unary([](double x) { return std::sqrt(x); });
						case FIDs::RSQRT:
							return unary([](double x) { return 1.0 / std::sqrt(x); });
						case FIDs::EXP:
							return unary([](double x) { return std::exp(x); });
						case FIDs::LOG:
							return unary([](double x) { return std::log(x); });
						case FIDs::ERF:
							return unary([](double x) { return std::erf(x); });
						case FIDs::TANH:
							return unary([](double x) { return std::tanh(x); });
						case FIDs::COS:
							return unary([](double x) { return std::cos(x); });
						case FIDs::SIN:
							return unary([](double x) { return std::sin(x); });
						case FIDs::TAN:
							return unary([](double x) { return std::tan(x); });
						default:
							return std::nullopt;
						}
					},
					[](const auto&) -> std::optional<double> { return std::nullopt; },
			},
			node
	);
}

// Promote a PrimValue(int) to a float32 0-D Constant for use in arithmetic ops.
static ::tvm::relax::Expr promoteToFloatTensor(const ::tvm::relax::Expr& e) {
	const auto* pv = e.as<::tvm::relax::PrimValueNode>();
	if (!pv)
		return e;
	float fval = 0.0f;
	if (const auto* imm = pv->value.as<::tvm::IntImmNode>())
		fval = static_cast<float>(imm->value);
	else if (const auto* fimm = pv->value.as<::tvm::FloatImmNode>())
		fval = static_cast<float>(fimm->value);
	DLDevice dev = {kDLCPU, 0};
	auto data = ::tvm::runtime::Tensor::Empty(::tvm::ffi::Shape{}, ::tvm::DataType::Float(32), dev);
	data.CopyFromBytes(&fval, sizeof(float));
	return ::tvm::relax::Constant(data);
}

static ::tvm::relax::Expr translateRelax(
		::tvm::relax::BlockBuilder& bb, RelaxCache& cache, std::vector<::tvm::relax::Var>& params,
		const mdl::InvocationNode& node, const mdl::ArgVals& argVals
) {
	// RESHAPE/PERMUTE/EXPAND pass raw PrimValues as shape sentinels.
	// All other ops operate on tensors: compile-time scalar arguments (e.g. `sqrt(2)` or `sqrt(conf.dims.key)`) are
	// evaluated here and passed as float32 constants. (Relax's FoldConstant cannot evaluate operators without TVM's
	// Python legalizations, and backend patterns rely on seeing the constant values.)
	using FIDs = mdl::builtins::FunctionIDs;
	bool isShapeOp = (node.id == FIDs::RESHAPE || node.id == FIDs::PERMUTE || node.id == FIDs::EXPAND);
	std::vector<::tvm::relax::Expr> translatedArgs;
	translatedArgs.reserve(node.arguments.size());
	for (const auto& argRef : node.arguments) {
		if (!isShapeOp) {
			if (auto value = evalScalar(argRef.get(), argVals)) {
				translatedArgs.push_back(mdl::backend::tvm::makeFloatConst(static_cast<float>(*value)));
				continue;
			}
		}
		auto arg = translateRelax(bb, cache, params, argRef.get(), argVals);
		translatedArgs.push_back(isShapeOp ? arg : promoteToFloatTensor(arg));
	}

	auto& fn = mdl::backend::tvm::relaxFunctions[std::to_underlying(node.id)];
	assert(fn != nullptr);
	auto result = fn(bb, translatedArgs);

	// Emit non-leaf expressions into the dataflow block
	if (!result.as<::tvm::relax::VarNode>() && !result.as<::tvm::relax::PrimValueNode>() &&
		!result.as<::tvm::relax::ShapeExprNode>() && !result.as<::tvm::relax::ConstantNode>()) {
		return bb->Emit(result, "v");
	}
	return result;
}

static ::tvm::relax::Expr translateRelax(
		::tvm::relax::BlockBuilder& bb, RelaxCache& cache, std::vector<::tvm::relax::Var>& params,
		const mdl::Node& node, const mdl::ArgVals& argVals
) {
	if (auto it = cache.find(&node); it != cache.end())
		return it->second;
	auto result = std::visit([&](const auto& n) { return translateRelax(bb, cache, params, n, argVals); }, node);
	cache[&node] = result;
	return result;
}

// ── per-function Relax IR builder ─────────────────────────────────────────────

static ::tvm::relax::Var makeRelaxParam(const std::string& name, const mdl::DataType& dtype, const TranslationContext& ctx) {
	auto sinfo = ::tvm::relax::TensorStructInfo(::tvm::relax::ShapeExpr(getShape(dtype, ctx)), toUnderlyingTVMType(dtype));
	return ::tvm::relax::Var(name, sinfo);
}

/**
 * @brief Checks that a compiled output has the declared shape (which the generated bindings rely on).
 */
static void checkOutputShape(const mdl::ExportedFunction& fn, size_t i, const ::tvm::relax::Expr& result) {
	const auto& declared = fn.outputs[i];
	const auto* sinfo = ::tvm::relax::GetStructInfoAs<::tvm::relax::TensorStructInfoNode>(result);
	auto error = [&](const std::string& what) {
		throw std::runtime_error(
				std::format("{}: output {} {} (declared {} dimension(s))", fn.symbol, i, what, declared.shape.size())
		);
	};
	if (sinfo == nullptr || !sinfo->GetShape().defined())
		error("has no known shape");
	auto shape = sinfo->GetShape().value();
	if (shape.size() != declared.shape.size())
		error(std::format("has {} dimension(s)", shape.size()));
	for (size_t d = 0; d < shape.size(); ++d) {
		const auto* imm = shape[d].as<::tvm::IntImmNode>();
		if (declared.shape[d] >= 0 && (imm == nullptr || imm->value != declared.shape[d]))
			error(std::format("has a different size in dimension {}", d));
	}
}

/**
 * @brief Translates an exported function into a Relax function.
 * @details The Relax parameters follow the exported calling convention: the inputs (in declaration order) followed
 * by the weights (in canonical order); see mdl::ExportedFunction.
 */
static void translateFuncRelax(
		::tvm::relax::BlockBuilder& bb, const mdl::ExportedFunction& fn, const mdl::ArgVals& argVals,
		unsigned ellipsisRank
) {
	auto logger = mdl::log::getLogger("compiler");
	logger->info("Translating {} (Relax)", fn.symbol);

	TranslationContext ctx{argVals, ellipsisRank};
	RelaxCache cache;
	::tvm::ffi::Array<::tvm::relax::Var> bbParams;
	auto addParams = [&](const std::vector<mdl::TensorSignature>& tensors, auto&& typeOf) {
		for (const auto& t : tensors) {
			auto var = makeRelaxParam(t.name, typeOf(t), ctx);
			bbParams.push_back(var);
			if (t.node != nullptr)
				cache[t.node] = var;
		}
	};
	for (size_t i = 0; i < fn.inputs.size(); ++i)
		addParams({fn.inputs[i]}, [&](const auto&) { return fn.decl->parameters[i].second.get().type; });
	addParams(fn.weights, [](const auto& t) { return mdl::getDataType(*t.node); });

	bb->BeginScope(bbParams);
	bb->BeginDataflowBlock();

	std::vector<::tvm::relax::Var> unusedParams; // params are already in the cache
	::tvm::ffi::Array<::tvm::relax::Expr> results;
	for (size_t i = 0; i < fn.decl->outputs.size(); ++i) {
		auto result = translateRelax(bb, cache, unusedParams, fn.decl->outputs[i].get(), ctx.argVals);
		if (!result.as<::tvm::relax::VarNode>())
			result = bb->Emit(result, "out");
		checkOutputShape(fn, i, result);
		results.push_back(result);
	}
	::tvm::relax::Var outputVar = results.size() == 1 ? bb->EmitOutput(results[0], "output")
													  : bb->EmitOutput(::tvm::relax::Tuple(results), "output");

	auto block = bb->EndBlock();
	bb->EndScope();

	auto body = ::tvm::relax::SeqExpr(::tvm::ffi::Array<::tvm::relax::BindingBlock>{block}, outputVar);
	auto retSI = ::tvm::Downcast<::tvm::relax::StructInfo>(outputVar->struct_info_.value());
	::tvm::relax::Function func(bbParams, body, retSI, /*is_pure=*/true);
	func = ::tvm::WithAttr(func, ::tvm::attr::kGlobalSymbol, ::tvm::ffi::String{fn.symbol});
	bb->AddFunction(func, fn.symbol);
}

// ── main entry point ──────────────────────────────────────────────────────────

/** @brief A C identifier derived from the output file name (prefix for the symbols of the compiled kernels). */
static std::string symbolPrefix(const std::filesystem::path& output) {
	std::string prefix = "mdl_";
	for (char c : output.stem().string())
		prefix += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
	return prefix + "_";
}

void mdl::backends::tvm::compileToFile(
		const std::vector<mdl::ExportedFunction>& exports, const std::filesystem::path& output,
		const mdl::ArgVals& argvals, const mdl::CompileOptions& opts
) {
	auto logger = mdl::log::getLogger("compiler");
	namespace backend = mdl::backend::tvm;

	auto dump = [&](const ::tvm::IRModule& mod, const std::string& stage) {
		if (opts.dumpDir.empty())
			return;
		std::filesystem::create_directories(opts.dumpDir);
		std::ofstream out(opts.dumpDir / (output.stem().string() + "." + stage + ".txt"));
		out << mod;
	};

	// 1. Relax IRModule with one function per exported function.
	auto bb = ::tvm::relax::BlockBuilder::Create(::tvm::ffi::Optional<::tvm::IRModule>{});
	for (const auto& fn : exports)
		translateFuncRelax(bb, fn, argvals, opts.ellipsisRank);
	auto mod = bb->Finalize();
	dump(mod, "relax");

	// 2. Relax passes, AOT lowering, TIR passes, and code generation.
	auto target = backend::makeTarget(opts);
	mod = backend::runRelaxPipeline(mod, target, opts);
	dump(mod, "relax-lowered");
	const auto prefix = symbolPrefix(output);
	mod = backend::lowerToAOT(mod, prefix + "k_");
	dump(mod, "tir");
	backend::buildObject(mod, target, prefix, output);
}
