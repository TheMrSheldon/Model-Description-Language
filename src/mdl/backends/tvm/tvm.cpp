#include "./tvm.hpp"

#include "../../argvals.hpp"
#include "../../logging.hpp"
#include "./details/aot.hpp"
#include "./details/kernel_match.hpp"
#include "./details/pipeline.hpp"
#include "./details/relax_builtins.hpp"
#include "./details/tvm_compat.hpp"

#include <mdl/compiler.hpp>
#include <mdl/computegraphv2/datatype.hpp>
#include <mdl/computegraphv2/function.hpp>
#include <mdl/computegraphv2/graph.hpp>
#include <mdl/computegraphv2/node.hpp>
#include <mdl/exports.hpp>
#include <mdl/parserv2/compilation_unit.hpp>
#include <mdl/passes/shapes.hpp>
#include <mdl/utils/utils.hpp>

#include <tvm/ir/prim/op.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/var.h>

#include <algorithm>
#include <any>
#include <cassert>
#include <cctype>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <optional>
#include <ranges>
#include <sstream>
#include <unordered_map>
#include <utility>

// ── type helpers ──────────────────────────────────────────────────────────────

static ::tvm::PrimType toTVMType(mdl::PrimitiveType p) {
	switch (p) {
	case mdl::PrimitiveType::boolean:
		return ::tvm::PrimType::Bool();
	case mdl::PrimitiveType::float32:
		return ::tvm::PrimType::Float(32);
	case mdl::PrimitiveType::float16:
		return ::tvm::PrimType::Float(16);
	case mdl::PrimitiveType::bfloat16:
		return ::tvm::PrimType::BFloat(16);
	case mdl::PrimitiveType::int8:
		return ::tvm::PrimType::Int(8);
	case mdl::PrimitiveType::uint:
		return ::tvm::PrimType::UInt(32);
	case mdl::PrimitiveType::integer:
		return ::tvm::PrimType::Int(32);
	}
	assert(false);
}

static ::tvm::PrimType toUnderlyingTVMType(mdl::DataType dtype) {
	return std::visit(
			mdl::overloaded{
					[](const mdl::PrimitiveType& p) { return toTVMType(p); },
					[](const mdl::TensorType& t) { return toTVMType(t.type); },
					[](const mdl::AggregateType&) {
						assert(false);
						return ::tvm::PrimType::Void();
					},
					[](const mdl::ModuleType&) {
						std::abort();
						return ::tvm::PrimType::Void();
					},
					[](const mdl::BoundModuleType&) {
						std::abort();
						return ::tvm::PrimType::Void();
					},
					[](const mdl::ArrayType&) {
						std::abort(); // arrays are compile-time values
						return ::tvm::PrimType::Void();
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
											return ::tvm::IntImm(::tvm::PrimType::Int(32), v);
										},
										[](double v) -> ::tvm::PrimExpr {
											return ::tvm::FloatImm(::tvm::PrimType::Float(64), v);
										},
										[](const std::vector<int>&) -> ::tvm::PrimExpr {
											return ::tvm::IntImm(::tvm::PrimType::Int(32), 0);
										},
								},
								cn.value
						);
					},
					[&argVals](const mdl::HyperparameterNode& hn) -> ::tvm::PrimExpr {
						if (auto val = argVals.lookup(hn.name))
							return ::tvm::IntImm(::tvm::PrimType::Int(32), *val);
						return ::tvm::PrimVar(hn.name.empty() ? "dim" : hn.name, ::tvm::PrimType::Int(32));
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
						return ::tvm::PrimVar("dim", ::tvm::PrimType::Int(32));
					},
					[](const auto&) -> ::tvm::PrimExpr { return ::tvm::PrimVar("dim", ::tvm::PrimType::Int(32)); },
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
	std::vector<::tvm::PrimVar> ellipsis;

	TranslationContext(const mdl::ArgVals& argVals, unsigned ellipsisRank) : argVals(argVals) {
		for (unsigned i = 0; i < ellipsisRank; ++i)
			ellipsis.emplace_back(
					ellipsisRank == 1 ? std::string{"B"} : "B" + std::to_string(i), ::tvm::PrimType::Int(64)
			);
	}
};

static ::tvm::ffi::Array<::tvm::PrimExpr> getShape(mdl::DataType dtype, const TranslationContext& ctx, int rank = -1) {
	const auto& argVals = ctx.argVals;
	return std::visit(
			mdl::overloaded{
					[](const mdl::PrimitiveType&) { return ::tvm::ffi::Array<::tvm::PrimExpr>{1}; },
					[&](const mdl::TensorType& ttype) {
						::tvm::ffi::Array<::tvm::PrimExpr> shape;
						for (const auto i : ttype.shape) {
							if (std::holds_alternative<mdl::Elipsis>(i)) {
								// The leading `rank` dimensions for an input with a smaller rank (see ExportedFunction).
								const size_t n = rank < 0 ? ctx.ellipsis.size() : static_cast<size_t>(rank);
								for (size_t k = 0; k < n && k < ctx.ellipsis.size(); ++k)
									shape.push_back(ctx.ellipsis[k]);
								continue;
							}
							auto expr = std::visit(
									mdl::overloaded{
											[&](mdl::Elipsis) -> ::tvm::PrimExpr {
												return ctx.ellipsis.front();
											}, // expanded above
											[](unsigned u) -> ::tvm::PrimExpr {
												return ::tvm::IntImm(::tvm::PrimType::Int(64), u);
											},
											[&argVals](const mdl::Node* node) {
												return ::tvm::prim::cast(
														::tvm::PrimType::Int(64), nodeToPrimExpr(*node, argVals)
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
		::tvm::relax::BlockBuilder& bb, RelaxCache& cache, std::vector<::tvm::Var>& params, const mdl::Node& node,
		const mdl::ArgVals& argVals
);

static ::tvm::relax::Expr
translateRelax(::tvm::relax::BlockBuilder& bb, RelaxCache&, std::vector<::tvm::Var>& params, const mdl::ConstNode& node, const mdl::ArgVals&) {
	return std::visit(
			mdl::overloaded{
					[](int v) -> ::tvm::relax::Expr {
						// Scalar int: used as shape dim (RESHAPE/PERMUTE/EXPAND sentinels)
						return ::tvm::IntImm(::tvm::PrimType::Int(32), v);
					},
					[](double v) -> ::tvm::relax::Expr {
						// Scalar float: used in arithmetic (e.g. 1e-12 epsilon, 0.5 scale)
						float fval = static_cast<float>(v);
						DLDevice dev = {kDLCPU, 0};
						auto data = ::tvm::runtime::Tensor::Empty(
								::tvm::ffi::Shape{}, ::tvm::PrimType::Float(32)->dtype, dev
						);
						data.CopyFromBytes(&fval, sizeof(float));
						return ::tvm::relax::MakeTensorConst(data);
					},
					[](const std::vector<int>& v) -> ::tvm::relax::Expr {
						// Int vector: used as permutation axes in PERMUTE
						::tvm::ffi::Array<::tvm::PrimExpr> vals;
						for (int x : v)
							vals.push_back(::tvm::IntImm(::tvm::PrimType::Int(32), x));
						return ::tvm::relax::ShapeExpr(vals);
					},
			},
			node.value
	);
}

static ::tvm::relax::Expr
translateRelax(::tvm::relax::BlockBuilder& bb, RelaxCache&, std::vector<::tvm::Var>& /*params*/, const mdl::ErrorNode& /*node*/, const mdl::ArgVals&) {
	std::abort(); // compile-time error: should not reach here
}

template <typename LeafNode>
	requires std::same_as<LeafNode, mdl::ParameterNode> || std::same_as<LeafNode, mdl::InputNode>
static ::tvm::relax::Expr
translateRelax(::tvm::relax::BlockBuilder&, RelaxCache&, std::vector<::tvm::Var>&, const LeafNode& node, const mdl::ArgVals&) {
	// All arguments and weights of an exported function are Relax parameters (pre-seeded in the cache).
	throw std::runtime_error("TVM backend: " + mdl::getName(node) + " is not an argument of the compiled function");
}

static ::tvm::relax::Expr translateRelax(
		::tvm::relax::BlockBuilder& bb, RelaxCache&, std::vector<::tvm::Var>& /*params*/,
		const mdl::HyperparameterNode& node, const mdl::ArgVals& argVals
) {
	if (auto val = argVals.lookup(node.name))
		return ::tvm::IntImm(::tvm::PrimType::Int(32), *val);
	// Symbolic hyperparameter
	return ::tvm::PrimVar(node.name.empty() ? "dim" : node.name, ::tvm::PrimType::Int(32));
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

// Promote a PrimValue(int) to a 0-D float Constant (of @p dtype) for use in arithmetic ops.
static ::tvm::relax::Expr promoteToFloatTensor(const ::tvm::relax::Expr& e, ::tvm::PrimType dtype) {
	if (!mdl::backend::tvm::scalarValue(e))
		return e;
	float fval = 0.0f;
	if (const auto* imm = e.as<::tvm::IntImmNode>())
		fval = static_cast<float>(mdl::backend::tvm::intValue(imm));
	else if (const auto* fimm = e.as<::tvm::FloatImmNode>())
		fval = static_cast<float>(fimm->value);
	return mdl::backend::tvm::makeFloatConst(fval, dtype);
}

static ::tvm::relax::Expr translateRelax(
		::tvm::relax::BlockBuilder& bb, RelaxCache& cache, std::vector<::tvm::Var>& params,
		const mdl::InvocationNode& node, const mdl::ArgVals& argVals
) {
	// RESHAPE/PERMUTE/EXPAND pass raw PrimValues as shape sentinels.
	// All other ops operate on tensors: compile-time scalar arguments (e.g. `sqrt(2)` or `sqrt(conf.dims.key)`) are
	// evaluated here and passed as float32 constants. (Relax's FoldConstant cannot evaluate operators without TVM's
	// Python legalizations, and backend patterns rely on seeing the constant values.)
	using FIDs = mdl::builtins::FunctionIDs;
	bool isShapeOp =
			(node.id == FIDs::RESHAPE || node.id == FIDs::PERMUTE || node.id == FIDs::EXPAND ||
			 node.id == FIDs::SELECT || node.id == FIDs::UNSELECT_LIKE ||
			 node.id == FIDs::DEQUANT); // integer arguments stay integers
	// The element type of the result: float constants are created in it (the operands have it already: conversions
	// between float types are explicit CASTs).
	const auto* tensorType = std::get_if<mdl::TensorType>(&node.type);
	const auto* primType = std::get_if<mdl::PrimitiveType>(&node.type);
	const mdl::PrimitiveType element = tensorType ? tensorType->type : primType ? *primType : mdl::PrimitiveType::error;
	const auto floatType = mdl::isFloat(element) ? toTVMType(element) : ::tvm::PrimType::Float(32);
	auto isFloatTensor = [](const ::tvm::relax::Expr& e) {
		const auto* sinfo = e->ty.as<::tvm::relax::TensorTypeNode>();
		return sinfo != nullptr && !sinfo->IsUnknownDtype() &&
			   (mdl::backend::tvm::isFloat(sinfo->dtype.value()) || mdl::backend::tvm::isBFloat16(sinfo->dtype.value())
			   );
	};
	std::vector<::tvm::relax::Expr> translatedArgs;
	translatedArgs.reserve(node.arguments.size());
	for (size_t i = 0; i < node.arguments.size(); ++i) {
		const auto& argRef = node.arguments[i];
		if (!isShapeOp && node.id != FIDs::CAST) {
			if (auto value = evalScalar(argRef.get(), argVals)) {
				translatedArgs.push_back(mdl::backend::tvm::makeFloatConst(static_cast<float>(*value), floatType));
				continue;
			}
		}
		auto arg = translateRelax(bb, cache, params, argRef.get(), argVals);
		if (!isShapeOp)
			arg = promoteToFloatTensor(arg, floatType);
		if (node.id == FIDs::CAST && i == 0) // the explicit conversion
			arg = bb->Emit(mdl::backend::tvm::callRelaxOp("relax.op.astype", arg, floatType->dtype), "cast");
		translatedArgs.push_back(arg);
	}

	auto& fn = mdl::backend::tvm::relaxFunctions[std::to_underlying(node.id)];
	assert(fn != nullptr);
	auto result = fn(bb, translatedArgs);

	// Emit non-leaf expressions into the dataflow block
	if (!result.as<::tvm::VarNode>() && !mdl::backend::tvm::scalarValue(result) &&
		!result.as<::tvm::relax::ShapeExprNode>() && !mdl::backend::tvm::tensorConstant(result))
		result = bb->Emit(result, "v");
	// Results that a translation computes in float32 (e.g. comparisons, one-hot vectors, kernels that accumulate in
	// float32) are converted back to the node's type.
	if (mdl::isFloat(element) && isFloatTensor(result) && mdl::backend::tvm::dtypeOf(result) != floatType)
		result = bb->Emit(mdl::backend::tvm::callRelaxOp("relax.op.astype", result, floatType->dtype), "v");
	return result;
}

static ::tvm::relax::Expr translateRelax(
		::tvm::relax::BlockBuilder& bb, RelaxCache& cache, std::vector<::tvm::Var>& params, const mdl::Node& node,
		const mdl::ArgVals& argVals
) {
	if (auto it = cache.find(&node); it != cache.end())
		return it->second;
	auto result = std::visit([&](const auto& n) { return translateRelax(bb, cache, params, n, argVals); }, node);
	cache[&node] = result;
	return result;
}

// ── per-function Relax IR builder ─────────────────────────────────────────────

static ::tvm::Var
makeRelaxParam(const std::string& name, const mdl::DataType& dtype, const TranslationContext& ctx, int rank = -1) {
	auto sinfo =
			::tvm::relax::TensorType(::tvm::relax::ShapeExpr(getShape(dtype, ctx, rank)), toUnderlyingTVMType(dtype));
	return ::tvm::Var(name, sinfo);
}

/**
 * @brief Checks that a compiled output has the declared shape (which the generated bindings rely on).
 */
static void checkOutputShape(const mdl::ExportedFunction& fn, size_t i, const ::tvm::relax::Expr& result) {
	const auto& declared = fn.outputs[i];
	const auto* sinfo = ::tvm::relax::GetTypeAs<::tvm::relax::TensorTypeNode>(result);
	auto error = [&](const std::string& what) {
		throw std::runtime_error(
				std::format("{}: output {} {} (declared {} dimension(s))", fn.symbol, i, what, declared.shape.size())
		);
	};
	if (sinfo == nullptr || !sinfo->GetShape().defined())
		error("has no known shape");
	auto shape = sinfo->GetShape().value();
	if (shape.size() != declared.shape.size()) {
		std::ostringstream dims;
		dims << shape;
		error(std::format("has {} dimension(s) {}", shape.size(), dims.str()));
	}
	for (size_t d = 0; d < shape.size(); ++d) {
		const auto* imm = shape[d].as<::tvm::IntImmNode>();
		if (declared.shape[d] >= 0 && (imm == nullptr || mdl::backend::tvm::intValue(imm) != declared.shape[d]))
			error(std::format("has a different size in dimension {}: {}", d, [&] {
				std::ostringstream o;
				o << shape;
				return o.str();
			}()));
	}
}

/** @brief A Relax dimension as a shape-inference dimension (constants, symbols, and their products). */
static mdl::shapes::Dim toDim(const ::tvm::PrimExpr& e) {
	if (const auto* imm = e.as<::tvm::IntImmNode>())
		return mdl::shapes::Dim::constant(mdl::backend::tvm::intValue(imm));
	if (const auto* var = e.as<::tvm::VarNode>())
		return mdl::shapes::Dim::symbol(std::string(var->name));
	if (const auto* mul = e.as<::tvm::prim::MulNode>())
		return toDim(mul->a) * toDim(mul->b);
	if (const auto* cast = e.as<::tvm::prim::CastNode>())
		return toDim(cast->value);
	return mdl::shapes::Dim::unknown();
}

/** @brief A short description of a node for diagnostics. */
static std::string describe(const mdl::Node& node) {
	return std::visit(
			mdl::overloaded{
					[](const mdl::InvocationNode& inv) {
						return std::format("builtin #{}", std::to_underlying(inv.id));
					},
					[](const mdl::InputNode& in) { return "input " + in.name; },
					[](const mdl::ParameterNode& p) { return "weight " + p.name; },
					[](const auto&) { return std::string{"constant"}; },
			},
			node
	);
}

/**
 * @brief Checks the shapes inferred on the computegraph (mdl::shapes::infer) against the shapes Relax computed for the
 * translated nodes (enabled with the environment variable MDL_VERIFY_SHAPES=1; a development aid for the inference).
 */
static void verifyShapes(
		const mdl::ExportedFunction& fn, const RelaxCache& cache, const mdl::ArgVals& argVals, unsigned ellipsisRank
) {
	// The packed inputs (residuals) have the shapes their producer computed; their declared types are imprecise.
	std::unordered_map<const mdl::Node*, mdl::shapes::Shape> known;
	for (const auto& t : fn.inputs)
		if ((t.packed || t.ellipsisRank >= 0) && t.node != nullptr)
			if (auto it = cache.find(t.node); it != cache.end())
				if (const auto* si = ::tvm::relax::GetTypeAs<::tvm::relax::TensorTypeNode>(it->second)) {
					mdl::shapes::Shape shape;
					for (const auto& d : si->GetShape().value())
						shape.push_back(toDim(d));
					known.emplace(t.node, std::move(shape));
				}
	auto shapes = mdl::shapes::infer(
			*fn.decl, ellipsisRank, [&](const mdl::Node& n) { return mdl::evalConstant(n, argVals); }, known
	);
	if (!shapes)
		throw std::runtime_error("shape inference failed: " + shapes.error());
	std::string mismatches;
	size_t checked = 0;
	for (const auto& [node, expr] : cache) {
		const auto* sinfo = ::tvm::relax::GetTypeAs<::tvm::relax::TensorTypeNode>(expr);
		const auto* inferred = shapes->find(*node);
		if (sinfo == nullptr || !sinfo->GetShape().defined() || inferred == nullptr)
			continue;
		mdl::shapes::Shape actual;
		for (const auto& d : sinfo->GetShape().value())
			actual.push_back(toDim(d));
		bool same = actual.size() == inferred->size();
		for (size_t i = 0; same && i < actual.size(); ++i)
			same = !actual[i].known || !(*inferred)[i].known || actual[i] == (*inferred)[i];
		++checked;
		if (!same)
			mismatches += std::format(
					"\n  {}: inferred {}, Relax {}", describe(*node), mdl::shapes::str(*inferred),
					mdl::shapes::str(actual)
			);
	}
	if (!mismatches.empty())
		throw std::runtime_error(fn.symbol + ": inferred shapes differ from Relax:" + mismatches);
	mdl::log::getLogger("compiler")->info("{}: {} inferred shapes match Relax", fn.symbol, checked);
}

/**
 * @brief The packed tensors of a context buffer, as produced by its producer (see mdl::ContextBuffer).
 * @details The consumer declares its packed inputs with exactly these shapes (in its own dimension variables), so that
 * both compute the same layout.
 */
struct ContextLayout {
	std::vector<::tvm::relax::TensorType> tensors;
	std::vector<::tvm::PrimVar> ellipsis; ///< the producer's dimension variables of `...`
};

/**
 * @brief Translates an exported function into a Relax function.
 * @details The Relax parameters follow the exported calling convention: the inputs (in declaration order) followed
 * by the weights (in canonical order); see mdl::ExportedFunction. Packed inputs are Relax parameters as well (the AOT
 * lowering reads them from the context buffer).
 *
 * @param contexts the layouts of the context buffers translated so far (a producer adds its layout).
 * @return which parameters and outputs are packed.
 */
static mdl::backend::tvm::EntrySignature translateFuncRelax(
		::tvm::relax::BlockBuilder& bb, const mdl::ExportedFunction& fn, const mdl::ArgVals& argVals,
		unsigned ellipsisRank, std::map<std::string, ContextLayout>& contexts
) {
	auto logger = mdl::log::getLogger("compiler");
	logger->info("Translating {} (Relax)", fn.symbol);

	TranslationContext ctx{argVals, ellipsisRank};
	RelaxCache cache;
	mdl::backend::tvm::EntrySignature packing;
	packing.context = fn.context.has_value();
	::tvm::ffi::Array<::tvm::Var> bbParams;
	auto bind = [&](const mdl::TensorSignature& t, ::tvm::Var var, bool packed) {
		bbParams.push_back(var);
		packing.params.push_back(packed);
		if (t.node != nullptr)
			cache[t.node] = var;
	};
	size_t nextPacked = 0;
	for (size_t i = 0; i < fn.inputs.size(); ++i) {
		const auto& t = fn.inputs[i];
		if (!t.packed) {
			bind(t, makeRelaxParam(t.name, fn.decl->parameters[i].second.get().type, ctx, t.ellipsisRank), false);
			continue;
		}
		// A packed input has the shape its producer computed, in this function's dimension variables.
		auto layout = fn.context ? contexts.find(fn.context->name) : contexts.end();
		if (layout == contexts.end() || nextPacked >= layout->second.tensors.size())
			throw std::runtime_error(fn.symbol + ": the producer of the context buffer must be compiled first");
		::tvm::ffi::Map<::tvm::Var, ::tvm::PrimExpr> rename;
		for (size_t d = 0; d < ctx.ellipsis.size() && d < layout->second.ellipsis.size(); ++d)
			rename.Set(layout->second.ellipsis[d], ctx.ellipsis[d]);
		const auto& sinfo = layout->second.tensors[nextPacked++];
		auto shape = mdl::backend::tvm::substitute(sinfo->GetShape().value(), rename);
		bind(t, ::tvm::Var(t.name, ::tvm::relax::TensorType(::tvm::relax::ShapeExpr(shape), sinfo->dtype)), true);
	}
	for (const auto& w : fn.weights)
		bind(w, makeRelaxParam(w.name, mdl::getDataType(*w.node), ctx), false);

	bb->BeginScope(bbParams);
	bb->BeginDataflowBlock();

	std::vector<::tvm::Var> unusedParams; // params are already in the cache
	::tvm::ffi::Array<::tvm::relax::Expr> results;
	ContextLayout produced{.tensors = {}, .ellipsis = ctx.ellipsis};
	for (size_t i = 0; i < fn.decl->outputs.size(); ++i) {
		auto result = translateRelax(bb, cache, unusedParams, fn.decl->outputs[i].get(), ctx.argVals);
		if (!result.as<::tvm::VarNode>())
			result = bb->Emit(result, "out");
		const bool packed = fn.outputs[i].packed;
		packing.outputs.push_back(packed);
		packing.accumulate.push_back(fn.outputs[i].accumulate);
		if (packed) {
			// Laid out by the backend: the (possibly imprecise) declared type is not part of the ABI.
			const auto* sinfo = ::tvm::relax::GetTypeAs<::tvm::relax::TensorTypeNode>(result);
			if (sinfo == nullptr || !sinfo->GetShape().defined())
				throw std::runtime_error(std::format("{}: packed output {} has no known shape", fn.symbol, i));
			produced.tensors.push_back(::tvm::ffi::GetRef<::tvm::relax::TensorType>(sinfo));
		} else {
			checkOutputShape(fn, i, result);
		}
		results.push_back(result);
	}
	if (std::getenv("MDL_VERIFY_SHAPES"))
		verifyShapes(fn, cache, argVals, ellipsisRank);
	if (fn.context) {
		for (const auto& var : ctx.ellipsis)
			packing.dims.push_back(var);
		if (fn.context->producer) {
			contexts[fn.context->name] = produced;
			packing.sizeSymbol = fn.context->sizeSymbol;
		}
	}
	::tvm::Var outputVar = results.size() == 1 ? bb->EmitOutput(results[0], "output")
											   : bb->EmitOutput(::tvm::relax::Tuple(results), "output");

	auto block = bb->EndBlock();
	bb->EndScope();

	auto body = ::tvm::relax::SeqExpr(::tvm::ffi::Array<::tvm::relax::BindingBlock>{block}, outputVar);
	auto retSI = outputVar->ty;
	::tvm::relax::Function func(bbParams, body, retSI, /*is_pure=*/true);
	func = ::tvm::WithAttr(func, ::tvm::attr::kGlobalSymbol, ::tvm::ffi::String{fn.symbol});
	bb->AddFunction(func, fn.symbol);
	return packing;
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
	std::map<std::string, ContextLayout> contexts;
	std::unordered_map<std::string, backend::EntrySignature> packing;
	for (const auto& fn : exports) {
		packing.emplace(fn.symbol, translateFuncRelax(bb, fn, argvals, opts.ellipsisRank, contexts));
	}
	auto mod = bb->Finalize();
	dump(mod, "relax");

	// 2. Relax passes, AOT lowering, TIR passes, and code generation.
	auto target = backend::makeTarget(opts);
	mod = backend::canonicalize(mod, target);
	dump(mod, "relax-match"); // the matching boundary of kernel packages (see docs/kernels.md)
	using Inspection = mdl::CompileOptions::KernelInspection;
	if (opts.inspectKernels == Inspection::dump) {
		*opts.inspectOut << mod;
		return;
	}
	// Kernel packages: match at the boundary, then call the selected implementations (see docs/kernels.md).
	auto discovery = mdl::kernels::discover(mdl::kernels::kernelDirs(opts.kernelDirs, opts.source));
	for (const auto& problem : discovery.problems)
		logger->warn("kernel package skipped: {}", problem.str());
	std::erase_if(discovery.packages, [&](const mdl::kernels::Package& p) {
		const bool allowed = opts.kernelPackages.empty() || std::ranges::contains(opts.kernelPackages, p.name);
		return !opts.useKernelPackages || opts.kernels == mdl::Kernels::None || !allowed ||
			   std::ranges::contains(opts.noKernelPackages, p.name);
	});
	auto matches = backend::matchKernels(mod, discovery.packages, mdl::kernels::hostPlatform());
	if (opts.inspectKernels != Inspection::none) {
		backend::explain(*opts.inspectOut, matches, discovery, opts.inspectKernels == Inspection::explain);
		return;
	}
	if (backend::needsNonDataflow(matches)) {
		// Runtime guards are `If`s, which cannot be inside dataflow blocks: match again without them (the same
		// selection, of the new variables), then restore the blocks and their DataflowVars (only these can be fused).
		mod = ::tvm::relax::transform::ToNonDataflow()(mod);
		matches = backend::matchKernels(mod, discovery.packages, mdl::kernels::hostPlatform());
		mod = backend::applyKernels(mod, matches);
		mod = ::tvm::relax::transform::ConvertToDataflow()(mod);
		mod = ::tvm::relax::transform::CanonicalizeBindings()(mod);
	} else
		mod = backend::applyKernels(mod, matches);
	for (const auto& c : matches.candidates) {
		if (c.outcome != backend::Candidate::Outcome::selected)
			continue;
		logger->info(
				"{}: {}/{} at {}", c.function, c.package->name, c.implementation->id,
				std::string(c.replaced.back()->name)
		);
		if (opts.selectedKernels == nullptr)
			continue;
		mdl::SelectedKernel k{
				.function = c.function,
				.package = c.package->name,
				.version = c.package->version,
				.implementation = c.implementation->id,
				.symbol = c.implementation->symbol,
				.manifest = c.package->manifest,
				.artifacts = {}
		};
		// The artifact and (transitively) the artifacts it depends on.
		std::vector<std::string> pending{c.artifact};
		while (!pending.empty()) {
			const auto* artifact = c.package->artifact(pending.front());
			pending.erase(pending.begin());
			if (artifact == nullptr || std::ranges::contains(k.artifacts, artifact->id, &mdl::LinkedArtifact::id))
				continue;
			mdl::LinkedArtifact linked{
					.package = c.package->name,
					.id = artifact->id,
					.path = artifact->path,
					.kind = artifact->kind == mdl::kernels::Artifact::Kind::sharedLibrary ? "shared-library"
																						  : "static-library",
					.dependencies = {}
			};
			for (const auto& d : artifact->dependencies) {
				using Kind = mdl::kernels::Dependency::Kind;
				linked.dependencies.push_back(
						std::string(
								d.kind == Kind::runtime	   ? "runtime:"
								: d.kind == Kind::artifact ? "artifact:"
														   : "system-library:"
						) +
						d.name
				);
				if (d.kind == Kind::artifact)
					pending.push_back(d.name);
			}
			k.artifacts.push_back(std::move(linked));
		}
		opts.selectedKernels->push_back(std::move(k));
	}
	mod = backend::lowerRelax(mod, target, opts);
	dump(mod, "relax-lowered");
	const auto prefix = symbolPrefix(output);
	mod = backend::lowerToAOT(mod, prefix + "k_", packing, opts.profile);
	dump(mod, "tir");
	backend::buildObject(mod, target, prefix, output);
}
