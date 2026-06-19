#include "./tvm.hpp"

#include "../../argvals.hpp"
#include "../../logging.hpp"
#include "./details/pattern_registry.hpp"
#include "./details/pipeline.hpp"
#include "./details/relax_builtins.hpp"

#include <mdl/compiler.hpp>
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

#include <any>
#include <cassert>
#include <cstdlib>
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

static int ellipsisDimCounter = 0;

static ::tvm::ffi::Array<::tvm::PrimExpr> getShape(mdl::DataType dtype, const mdl::ArgVals& argVals) {
	return std::visit(
			mdl::overloaded{
					[](const mdl::PrimitiveType&) { return ::tvm::ffi::Array<::tvm::PrimExpr>{1}; },
					[&argVals](const mdl::TensorType& ttype) {
						::tvm::ffi::Array<::tvm::PrimExpr> shape;
						for (const auto i : ttype.shape) {
							auto expr = std::visit(
									mdl::overloaded{
											[](mdl::Elipsis) -> ::tvm::PrimExpr {
												return ::tvm::tirx::Var(
														"B" + std::to_string(ellipsisDimCounter++),
														::tvm::DataType::Int(64)
												);
											},
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
					[](const mdl::AggregateType&) {
						assert(false);
						return ::tvm::ffi::Array<::tvm::PrimExpr>{};
					},
					[](const mdl::ModuleType&) {
						std::abort();
						return ::tvm::ffi::Array<::tvm::PrimExpr>{};
					},
					[](const mdl::BoundModuleType&) {
						std::abort();
						return ::tvm::ffi::Array<::tvm::PrimExpr>{};
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

static ::tvm::relax::Expr translateRelax(
		::tvm::relax::BlockBuilder& bb, RelaxCache&, std::vector<::tvm::relax::Var>& params,
		const mdl::ParameterNode& node, const mdl::ArgVals& argVals
) {
	// Create a Relax Var with full tensor StructInfo
	auto shape = getShape(mdl::getDataType(node), argVals);
	auto dtype = toUnderlyingTVMType(mdl::getDataType(node));
	auto sinfo = ::tvm::relax::TensorStructInfo(::tvm::relax::ShapeExpr(shape), dtype);
	auto var = ::tvm::relax::Var(mdl::getName(node), sinfo);
	params.push_back(var);
	return var;
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
	// Recursively translate all arguments
	std::vector<::tvm::relax::Expr> translatedArgs;
	translatedArgs.reserve(node.arguments.size());
	for (const auto& argRef : node.arguments)
		translatedArgs.push_back(translateRelax(bb, cache, params, argRef.get(), argVals));

	// RESHAPE/PERMUTE/EXPAND pass raw PrimValues as shape sentinels.
	// All other ops operate on tensors — promote any scalar PrimValues to float32 constants.
	using FIDs = mdl::builtins::FunctionIDs;
	bool isShapeOp = (node.id == FIDs::RESHAPE || node.id == FIDs::PERMUTE || node.id == FIDs::EXPAND);
	if (!isShapeOp) {
		for (auto& arg : translatedArgs)
			arg = promoteToFloatTensor(arg);
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

static void
translateFuncRelax(::tvm::relax::BlockBuilder& bb, const mdl::FunctionDecl& func, const mdl::ArgVals& argVals) {
	auto logger = mdl::log::getLogger("compiler");
	logger->info("Translating {} (Relax)", func.name);

	RelaxCache cache;
	std::vector<::tvm::relax::Var> paramVars;

	// Set up BeginScope with an empty params list first; params get populated during
	// translation (ParameterNodes push themselves via translateRelax → ParameterNode).
	// We actually need params before BeginScope, so we do a pre-DFS to discover them.

	// Pre-DFS: collect ParameterNodes in visitation order
	std::vector<const mdl::Node*> toVisit;
	std::unordered_map<const mdl::Node*, bool> visited;
	std::function<void(const mdl::Node&)> preTraverse = [&](const mdl::Node& node) {
		if (visited[&node])
			return;
		visited[&node] = true;
		if (std::holds_alternative<mdl::InvocationNode>(node)) {
			for (const auto& argRef : std::get<mdl::InvocationNode>(node).arguments)
				preTraverse(argRef.get());
		} else if (std::holds_alternative<mdl::ParameterNode>(node)) {
			toVisit.push_back(&node);
		}
	};
	preTraverse(func.output);

	// Create Relax Var for each parameter in DFS order
	::tvm::ffi::Array<::tvm::relax::Var> bbParams;
	for (const auto* nodePtr : toVisit) {
		const auto& pn = std::get<mdl::ParameterNode>(*nodePtr);
		auto shape = getShape(mdl::getDataType(pn), argVals);
		auto dtype = toUnderlyingTVMType(mdl::getDataType(pn));
		auto sinfo = ::tvm::relax::TensorStructInfo(::tvm::relax::ShapeExpr(shape), dtype);
		auto var = ::tvm::relax::Var(mdl::getName(pn), sinfo);
		bbParams.push_back(var);
		cache[nodePtr] = var;
		logger->info("  param: {} {}", mdl::getName(pn), std::to_string(mdl::getDataType(pn)));
	}

	// Log param shapes
	for (size_t i = 0; i < bbParams.size(); ++i) {
		const auto* sinfo = bbParams[i]->struct_info_.as<::tvm::relax::TensorStructInfoNode>();
		std::string shape_str = "[";
		if (sinfo) {
			auto maybe = sinfo->GetShape();
			if (maybe.defined()) {
				for (size_t j = 0; j < maybe.value().size(); ++j) {
					if (j)
						shape_str += ", ";
					std::ostringstream ss;
					ss << maybe.value()[j];
					shape_str += ss.str();
				}
			}
		}
		shape_str += "]";
		logger->info("  arg[{}] = {} {}", i, std::string(bbParams[i]->name_hint()), shape_str);
	}

	bb->BeginScope(bbParams);
	bb->BeginDataflowBlock();

	// Translate the output DAG
	// The paramVars vector is unused here; cache already has the params
	std::vector<::tvm::relax::Var> unusedParams; // not used — params already in cache
	auto result = translateRelax(bb, cache, unusedParams, func.output, argVals);

	// Emit as output binding
	// result might already be a Var or might need emitting
	::tvm::relax::Var outputVar;
	if (auto* v = result.as<::tvm::relax::VarNode>()) {
		outputVar = bb->EmitOutput(result, "output");
	} else {
		auto emitted = bb->Emit(result, "out");
		outputVar = bb->EmitOutput(emitted, "output");
	}

	auto block = bb->EndBlock();
	bb->EndScope();

	// Build the Relax Function
	auto body = ::tvm::relax::SeqExpr(::tvm::ffi::Array<::tvm::relax::BindingBlock>{block}, outputVar);

	// Downcast Optional<ObjectRef> → StructInfo for the return type
	auto retSI = ::tvm::Downcast<::tvm::relax::StructInfo>(outputVar->struct_info_.value());
	::tvm::relax::Function fn(bbParams, body, retSI, /*is_pure=*/true);

	// Attach global symbol attribute
	fn = ::tvm::WithAttr(fn, ::tvm::attr::kGlobalSymbol, ::tvm::ffi::String{func.name});

	bb->AddFunction(fn, func.name);
	logger->info("  → added Relax function '{}'", func.name);
}

// ── main entry point ──────────────────────────────────────────────────────────

void mdl::backends::tvm::compileToFile(
		const mdl::CompilationUnit& unit, const std::filesystem::path& output, const mdl::ArgVals& argvals,
		const mdl::CompileOptions& opts
) {
	auto logger = mdl::log::getLogger("compiler");

	// ── 1. Build Relax IRModule ───────────────────────────────────────────────
	auto bb = ::tvm::relax::BlockBuilder::Create(::tvm::ffi::Optional<::tvm::IRModule>{});

	for (const auto& [name, member] : unit.definitions) {
		if (const auto* func = std::get_if<mdl::FunctionDecl>(&member)) {
			// Skip standalone functions — they have dynamic shapes (ellipsis dims)
			// and are inlined into module compute graphs anyway.
			(void)func;
			continue;
		} else if (const auto* module = std::get_if<mdl::ModuleType>(&member)) {
			// Skip template modules whose scalar type args aren't in argVals.
			// Struct-typed typeargs (InvalidDataType) are OK — their leaf fields are in argVals.
			bool hasUnresolvable = false;
			for (const auto& [tname, hn] : module->typeargs) {
				auto res = argvals.lookup(hn.get().name);
				if (!res.has_value() && res.error() == mdl::ArgVals::ErrorCode::NotFound) {
					hasUnresolvable = true;
					break;
				}
			}
			if (hasUnresolvable) {
				logger->info("Skipping module '{}' (unresolvable type args)", name);
				continue;
			}
			for (const auto& [funcName, funcDecl] : module->functions) {
				if (!funcDecl.templateParameters.empty())
					continue;
				mdl::FunctionDecl mangledDecl = funcDecl;
				mangledDecl.name = mdl::mangledName(name, funcName);
				translateFuncRelax(bb, mangledDecl, argvals);
			}
		}
	}

	auto irMod = bb->Finalize();
	logger->info("Relax IRModule built ({} functions)", irMod->functions.size());

	// ── 1b. Apply C++ pattern rewrites ────────────────────────────────────────
	//   Replaces decomposed softmax/gelu/layer_norm with high-level Relax ops
	//   before handing the module to Python for legalization and compilation.
	auto patterns = mdl::backend::tvm::defaultPatterns();
	irMod = mdl::backend::tvm::applyPatterns(irMod, patterns);

	// ── 2. Determine CPU target string ────────────────────────────────────────
	std::string cpuName = "generic";
	if (auto fn = ::tvm::ffi::Function::GetGlobal("tvm.codegen.llvm.GetHostCPUName"))
		cpuName = (*fn)().cast<std::string>();
	std::string targetStr = "llvm -mcpu=" + cpuName;

	// ── 3. Serialize pre-rewrite IRModule to JSON ─────────────────────────────
	auto save_json_fn = *::tvm::ffi::Function::GetGlobal("node.SaveJSON");
	auto load_json_fn = *::tvm::ffi::Function::GetGlobal("node.LoadJSON");

	auto saveIRMod = [&](const ::tvm::IRModule& mod, const std::string& path) {
		std::string json = save_json_fn(mod).cast<std::string>();
		std::ofstream f(path);
		if (!f) throw std::runtime_error("Cannot write: " + path);
		f << json;
	};
	auto loadIRMod = [&](const std::string& path) -> ::tvm::IRModule {
		std::ifstream f(path);
		if (!f) throw std::runtime_error("Cannot read: " + path);
		std::string json((std::istreambuf_iterator<char>(f)), {});
		return load_json_fn(json).cast<::tvm::IRModule>();
	};

	std::string jsonPath      = output.string() + ".pre_rewrite.json";
	std::string rewrittenPath = output.string() + ".rewritten.json";
	std::string loweredPath   = output.string() + ".lowered.json";
	std::string helperPath    = output.string() + ".build_helper.py";

	saveIRMod(irMod, jsonPath);
	logger->info("IRModule serialized to {}", jsonPath);

	// ── 4. Python rewrite pass: softmax DAG + attention (Python dpl only) ─────
	//    These patterns require Python's dpl.rewrite_call for DAG-aware matching.
	//    Output: rewritten.json (consumed by C++ pipeline below).
	{
		std::ofstream f(helperPath);
		f << R"PYHELPER(
import sys, os, tvm, tvm.relax, tvm.relax.dpl as dpl

json_in, json_out = sys.argv[1], sys.argv[2]

with open(json_in) as f:
    mod = tvm.ir.load_json(f.read())

def _rewrite_softmax(mod):
    x1 = dpl.wildcard()
    exp_num = dpl.is_op("relax.exp")(dpl.is_op("relax.subtract")(x1, dpl.is_op("relax.max")(x1)))
    x2 = dpl.wildcard()
    exp_den = dpl.is_op("relax.exp")(dpl.is_op("relax.subtract")(x2, dpl.is_op("relax.max")(x2)))
    pat = dpl.is_op("relax.divide")(exp_num, dpl.is_op("relax.sum")(exp_den))
    def rewriter(expr, bindings):
        inp1, inp2 = bindings[x1], bindings[x2]
        if not tvm.ir.structural_equal(inp1, inp2): return expr
        return tvm.relax.op.nn.softmax(inp1, axis=-1)
    for gv in mod.get_global_vars():
        fn = mod[gv]
        if isinstance(fn, tvm.relax.Function):
            new_fn = dpl.rewrite_call(pat, rewriter, fn)
            if not new_fn.same_as(fn): mod.update_func(gv, new_fn)
    return mod

def _rewrite_attention(mod):
    q_proj, k_proj, v_proj = dpl.wildcard(), dpl.wildcard(), dpl.wildcard()
    q_heads = dpl.is_op("relax.permute_dims")(dpl.is_op("relax.reshape")(q_proj, dpl.wildcard()))
    k_heads = dpl.is_op("relax.permute_dims")(dpl.is_op("relax.reshape")(k_proj, dpl.wildcard()))
    v_heads = dpl.is_op("relax.permute_dims")(dpl.is_op("relax.reshape")(v_proj, dpl.wildcard()))
    k_T     = dpl.is_op("relax.permute_dims")(k_heads)
    scores  = dpl.is_op("relax.matmul")(q_heads, k_T)
    scaled  = dpl.is_op("relax.divide")(scores, dpl.wildcard())
    attn    = dpl.is_op("relax.nn.softmax")(scaled)
    ctx     = dpl.is_op("relax.matmul")(attn, v_heads)
    ctx_p   = dpl.is_op("relax.permute_dims")(ctx)
    pat     = dpl.is_op("relax.reshape")(ctx_p, dpl.wildcard())
    def rewriter(expr, bindings):
        q, k, v = bindings[q_proj], bindings[k_proj], bindings[v_proj]
        kh = bindings[k_heads]; vh = bindings[v_heads]
        for t in [q, k, v, kh, vh]:
            if t.struct_info is None or t.struct_info.shape is None: return expr
        qs = q.struct_info.shape.values; ks = k.struct_info.shape.values; vs = v.struct_info.shape.values
        khs = kh.struct_info.shape.values; vhs = vh.struct_info.shape.values
        if len(khs) != 3 or len(vhs) != 3: return expr
        S_q, H, D_k, D_v = qs[0], khs[0], khs[2], vhs[2]
        S_k, S_v = ks[0], vs[0]
        q_bsnh = tvm.relax.op.reshape(q, [1, S_q, H, D_k])
        k_bsnh = tvm.relax.op.reshape(k, [1, S_k, H, D_k])
        v_bsnh = tvm.relax.op.reshape(v, [1, S_v, H, D_v])
        attn_out = tvm.relax.op.nn.attention(q_bsnh, k_bsnh, v_bsnh)
        return tvm.relax.op.reshape(attn_out, [S_q, H * D_v])
    n = 0
    for gv in mod.get_global_vars():
        fn = mod[gv]
        if isinstance(fn, tvm.relax.Function):
            new_fn = dpl.rewrite_call(pat, rewriter, fn)
            if not new_fn.same_as(fn): mod.update_func(gv, new_fn); n += 1
    if n: print(f"[mdl] attention pattern: {n} function(s) rewritten", flush=True)
    return mod

mod = _rewrite_softmax(mod)
mod = _rewrite_attention(mod)
mod = tvm.relax.transform.Normalize()(mod)
with open(json_out, "w") as f:
    f.write(tvm.ir.save_json(mod))
print("[mdl] Python rewrites done", flush=True)
)PYHELPER";
	}
	{
		std::string cmd = "python3 " + helperPath + " " + jsonPath + " " + rewrittenPath;
		logger->info("Running Python rewrite pass: {}", cmd);
		int rc = std::system(cmd.c_str());
		if (rc != 0)
			throw std::runtime_error("Python rewrite pass failed (exit " + std::to_string(rc) + ")");
	}

	// ── 5. Python full pipeline + export ─────────────────────────────────────
	//    LegalizeOps (CBLAS + oneDNN), scheduling, VM lowering, and export_library
	//    are all handled by the Python helper (default_pipeline + tvm.relax.build).
	std::string exportHelperPath = output.string() + ".export_helper.py";
	{
		std::ofstream f(exportHelperPath);
		f << R"PYHELPER(
import sys, os, json, subprocess, shutil, tempfile, tvm, tvm.relax

json_in    = sys.argv[1]   # lowered IRModule JSON
output_path = sys.argv[2]  # output .so path
target_str  = sys.argv[3]  # "llvm -mcpu=xxx"
dnnl_src_path = sys.argv[4]  # path to mdl_dnnl_ops.cpp

# Compile oneDNN wrapper into an object file
obj_path = dnnl_src_path.replace(".cpp", ".o")
subprocess.check_call([
    "g++", "-O3", "-fPIC", "-std=c++17",
    "-I/usr/include", "-I/usr/local/include",
    "-I/opt/intel/oneapi/mkl/latest/include",
    "-c", dnnl_src_path, "-o", obj_path
])
print("[mdl] oneDNN wrappers compiled", flush=True)

def fcompile(output, objects, options=None):
    cc_cmd = shutil.which("g++") or "g++"
    cmd = ([cc_cmd, "-shared", "-fPIC", "-o", output]
           + list(options or [])
           + list(objects) + [obj_path]
           + ["-ldnnl"])
    subprocess.check_call(cmd)

parts = target_str.split()
d = {"kind": parts[0]}
for p in parts[1:]:
    if "=" in p:
        k, v = p.lstrip("-").split("=", 1)
        d[k] = v
if d.get("kind") == "llvm":
    d["fast-math"] = True
target = tvm.target.Target(json.dumps(d))

with open(json_in) as f:
    mod = tvm.ir.load_json(f.read())

ex = tvm.relax.build(mod, target=target, relax_pipeline=None)
ex.export_library(output_path, fcompile=fcompile)
print(f"[mdl] Relax compiled -> {output_path}", flush=True)
)PYHELPER";
	}

	{
		std::ofstream f(helperPath);
		f << R"PYHELPER(
import sys, os

json_path   = sys.argv[1]
output_path = sys.argv[2]
target_str  = sys.argv[3] if len(sys.argv) > 3 else "llvm"
opt_level   = sys.argv[4] if len(sys.argv) > 4 else "default"

_here = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    os.path.join(_here, "../out/build/conf-dev/_deps/tvm-src/python"),
    "/workspaces/model-description-language/out/build/conf-dev/_deps/tvm-src/python",
    os.environ.get("TVM_PYTHON_PATH", ""),
]
_lib_candidates = [
    "/workspaces/model-description-language/out/build/conf-dev/lib",
    "/workspaces/model-description-language/out/build/conf-dev/_deps/tvm-build",
]
for c in _candidates:
    if c and os.path.isdir(c):
        sys.path.insert(0, c)
        os.environ["PYTHONPATH"] = c + ":" + os.environ.get("PYTHONPATH", "")
        break
for c in _lib_candidates:
    if c and os.path.isdir(c):
        os.environ["LD_LIBRARY_PATH"] = c + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        break

import tvm
import tvm.relax
import tvm.relax.transform as RT

with open(json_path) as f:
    mod = tvm.ir.load_json(f.read())

# TVM 0.24.0 dropped the "llvm -mcpu=xxx" CLI string form; build a Target from a dict.
def _parse_target(s):
    import json
    parts = s.split()
    d = {"kind": parts[0]}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.lstrip("-").split("=", 1)
            d[k] = v
    # fast-math enables vectorized exp/erf via LLVM's approximation (significant speedup
    # for softmax and GELU kernels which would otherwise fall back to scalar libm calls).
    if d.get("kind") == "llvm":
        d["fast-math"] = True
    return tvm.target.Target(json.dumps(d))

target = _parse_target(target_str)

# Dynamic shapes: symbolic dims left unbound so any seq length works at runtime.

# ── CBLAS matmul legalization ─────────────────────────────────────────────────
# cblas_sgemm is linked into libtvm.so via OpenBLAS.  tir.call_extern emits a
# direct LLVM call that survives the full lowering pipeline.
#
# Two variants:
#   _cblas_matmul_te       — C = A @ B      (ColMajor swap trick)
#   _cblas_matmul_transb_te — C = A @ B.T   (CblasTrans; eliminates explicit transpose copy)
#
# LegalizeOps lowers permute_dims before matmul in topological order, so when
# _legalize_matmul_cblas runs, a weight transpose W.T() has already become a
# call_tir(T_transpose, [W]).  We detect this via a shape-swap check and fuse
# it directly into the BLAS call.

_CblasColMajor, _CblasRowMajor = 102, 101
_CblasNoTrans,  _CblasTrans    = 111, 112
_call_tir_op = tvm.ir.Op.get("relax.call_tir")

def _cblas_matmul_te(A, B):
    """C[M,N] = A[M,K] @ B[K,N]  (2-D, ColMajor swap trick)."""
    M, K, N = A.shape[0], A.shape[1], B.shape[1]
    return tvm.te.extern(
        (M, N), [A, B],
        lambda ins, outs: tvm.tirx.Evaluate(tvm.tirx.call_extern(
            "int32", "cblas_sgemm",
            _CblasColMajor, _CblasNoTrans, _CblasNoTrans,
            N, M, K,
            tvm.tirx.const(1.0, "float32"),
            ins[1].access_ptr("r"), N,
            ins[0].access_ptr("r"), K,
            tvm.tirx.const(0.0, "float32"),
            outs[0].access_ptr("w"), N,
        )),
        name="C", dtype="float32",
    )

def _cblas_matmul_transb_te(A, B):
    """C[M,N] = A[M,K] @ B[N,K].T  (2-D, CblasTrans — no explicit copy)."""
    M, K, N = A.shape[0], A.shape[1], B.shape[0]
    return tvm.te.extern(
        (M, N), [A, B],
        lambda ins, outs: tvm.tirx.Evaluate(tvm.tirx.call_extern(
            "int32", "cblas_sgemm",
            _CblasRowMajor, _CblasNoTrans, _CblasTrans,
            M, N, K,
            tvm.tirx.const(1.0, "float32"),
            ins[0].access_ptr("r"), K,
            ins[1].access_ptr("r"), K,
            tvm.tirx.const(0.0, "float32"),
            outs[0].access_ptr("w"), N,
        )),
        name="C", dtype="float32",
    )

def _cblas_batmul_te(A, B):
    """C[b,M,N] = A[b,M,K] @ B[b,K,N].

    Uses cblas_sgemm_batch_strided (MKL) when available — one library call for all
    batch heads, avoids per-head function-call overhead and allows MKL to fuse the
    batch into one parallelised kernel.  Falls back to a serial loop otherwise.
    The compiled .so will link cblas_sgemm_batch_strided as an unresolved symbol;
    it must be satisfied at runtime (e.g. via LD_PRELOAD or linking against MKL).
    """
    batch, M, K = A.shape[0], A.shape[1], A.shape[2]
    N = B.shape[2]
    stride_a = M * K
    stride_b = K * N
    stride_c = M * N
    def body(ins, outs):
        # ColMajor swap trick: C[M,N] = A[M,K]@B[K,N] in row-major is equivalent
        # to C^T[N,M] = B^T[N,K]@A^T[K,M] in col-major (no data copies needed).
        # cblas_sgemm_batch_strided with ColMajor and NoTrans on B,A (swapped):
        #   lda=N (leading dim of B^T in col-major = N cols of B)  stride_a=K*N
        #   ldb=K (leading dim of A^T in col-major = K cols of A)  stride_b=M*K
        #   ldc=N (leading dim of C^T in col-major)               stride_c=M*N
        return tvm.tirx.Evaluate(tvm.tirx.call_extern(
            "int32", "cblas_sgemm_batch_strided",
            _CblasColMajor, _CblasNoTrans, _CblasNoTrans,
            N, M, K,
            tvm.tirx.const(1.0, "float32"),
            ins[1].access_ptr("r"), N, stride_b,   # B^T col-major
            ins[0].access_ptr("r"), K, stride_a,   # A^T col-major
            tvm.tirx.const(0.0, "float32"),
            outs[0].access_ptr("w"), N, stride_c,  # C^T col-major
            batch,
        ))
    return tvm.te.extern(
        (batch, M, N), [A, B], body, name="C_batch", dtype="float32",
    )

def _cblas_batmul_scaled_te(A, B, alpha_val=1.0):
    """C[b,M,N] = alpha_val * A[b,M,K] @ B[b,K,N]  (same ColMajor swap as _cblas_batmul_te)."""
    batch, M, K = A.shape[0], A.shape[1], A.shape[2]
    N = B.shape[2]
    stride_a = M * K
    stride_b = K * N
    stride_c = M * N
    alpha = float(alpha_val)
    def body(ins, outs):
        return tvm.tirx.Evaluate(tvm.tirx.call_extern(
            "int32", "cblas_sgemm_batch_strided",
            _CblasColMajor, _CblasNoTrans, _CblasNoTrans,
            N, M, K,
            tvm.tirx.const(alpha, "float32"),
            ins[1].access_ptr("r"), N, stride_b,
            ins[0].access_ptr("r"), K, stride_a,
            tvm.tirx.const(0.0, "float32"),
            outs[0].access_ptr("w"), N, stride_c,
            batch,
        ))
    return tvm.te.extern(
        (batch, M, N), [A, B], body, name="C_batch_scaled", dtype="float32",
    )

def _cblas_matmul_biased_te(A, B, bias, trans_b=False):
    """C[M,N] = A[M,K] @ B + bias[N] via oneDNN fused matmul+bias.
    B shape is [N,K] if trans_b else [K,N].
    Uses mdl_dnnl_matmul_bias_f32: bias is fused into the matmul kernel,
    avoiding the separate memcpy-bias-then-CBLAS-beta=1 approach.
    """
    M, K = A.shape[0], A.shape[1]
    N = B.shape[0] if trans_b else B.shape[1]
    trans_b_int = 1 if trans_b else 0
    return tvm.te.extern(
        (M, N), [A, B, bias],
        lambda ins, outs: tvm.tirx.Evaluate(tvm.tirx.call_extern(
            "int32", "mdl_dnnl_matmul_bias_f32",
            outs[0].access_ptr("w"),
            ins[0].access_ptr("r"), ins[1].access_ptr("r"), ins[2].access_ptr("r"),
            M, K, N, tvm.tirx.const(trans_b_int, "int32"),
        )),
        name="_cblas_matmul_biased", dtype="float32",
    )

def _is_2d_transpose(b_defn, B):
    """Return the pre-transpose input if b_defn is a call_tir 2-D transpose, else None."""
    if not (isinstance(b_defn, tvm.relax.Call) and
            isinstance(b_defn.op, tvm.ir.Op) and
            b_defn.op == _call_tir_op):
        return None
    args_tuple = b_defn.args[1]
    if not (isinstance(args_tuple, tvm.relax.Tuple) and len(args_tuple.fields) == 1):
        return None
    B_input = args_tuple.fields[0]
    try:
        in_shape  = B_input.struct_info.shape.values
        out_shape = B.struct_info.shape.values
    except Exception:
        return None
    if (len(in_shape) == 2 and len(out_shape) == 2 and
            int(in_shape[0]) == int(out_shape[1]) and
            int(in_shape[1]) == int(out_shape[0])):
        return B_input
    return None

def _legalize_matmul_cblas(bb, call):
    A, B = call.args
    ndim_a = len(A.struct_info.shape.values) if A.struct_info.shape else 0
    # 3-D batched matmul: A[batch,M,K] @ B[batch,K,N]
    if ndim_a == 3:
        return bb.call_te(_cblas_batmul_te, A, B)
    # 2-D: check for fused transpose on B
    if isinstance(B, tvm.relax.Var):
        B_orig = _is_2d_transpose(bb.lookup_binding(B), B)
        if B_orig is not None:
            return bb.call_te(_cblas_matmul_transb_te, A, B_orig)
    return bb.call_te(_cblas_matmul_te, A, B)

def _legalize_divide_fold_batmul_scale(bb, call):
    """Fold relax.divide(3D_batmul_call_tir, const_scalar) into CBLAS alpha.

    After LegalizeOps processes relax.matmul → call_tir(C_batch...), the downstream
    divide by the attention scale 1/sqrt(dk) can be folded into the alpha parameter
    of cblas_sgemm_batch_strided, eliminating a separate kernel launch.
    Non-matching cases fall back to the standard topi.divide legalization.
    """
    import tvm.topi as topi
    dividend, divisor = call.args
    if not isinstance(dividend, tvm.relax.Var):
        return bb.call_te(topi.divide, dividend, divisor)
    if not isinstance(divisor, tvm.relax.Constant):
        return bb.call_te(topi.divide, dividend, divisor)
    divisor_np = divisor.data.numpy()
    if divisor_np.size != 1:
        return bb.call_te(topi.divide, dividend, divisor)
    scale_val = float(divisor_np.flat[0])
    if scale_val == 0.0:
        return bb.call_te(topi.divide, dividend, divisor)
    binding = bb.lookup_binding(dividend)
    if binding is None:
        return bb.call_te(topi.divide, dividend, divisor)
    if not (isinstance(binding, tvm.relax.Call) and
            isinstance(binding.op, tvm.ir.Op) and
            binding.op == _call_tir_op):
        return bb.call_te(topi.divide, dividend, divisor)
    gv = binding.args[0]
    # TVM names TIR functions after the Python function passed to call_te
    if not (hasattr(gv, 'name_hint') and gv.name_hint.startswith("_cblas_batmul_te")):
        return bb.call_te(topi.divide, dividend, divisor)
    args_tuple = binding.args[1]
    if not (isinstance(args_tuple, tvm.relax.Tuple) and len(args_tuple.fields) == 2):
        return bb.call_te(topi.divide, dividend, divisor)
    A, B = args_tuple.fields
    if not (A.struct_info.shape and len(A.struct_info.shape.values) == 3):
        return bb.call_te(topi.divide, dividend, divisor)
    alpha = 1.0 / scale_val
    print(f"[mdl] folded attention scale (1/{scale_val:.4f}) into CBLAS alpha={alpha:.6f}", flush=True)
    def _cblas_batmul_scaled_call(a, b, _alpha=alpha):
        return _cblas_batmul_scaled_te(a, b, _alpha)
    return bb.call_te(_cblas_batmul_scaled_call, A, B)

def _legalize_add_fold_matmul_bias(bb, call):
    """Fold relax.add(2D_matmul_call_tir, 1D_bias) into fused CBLAS bias call.

    Detects add(call_tir(C/C_biased..., [A, B]), bias) where bias is rank-1 and
    replaces it with a single mdl_sgemm_bias_f32 call that initialises output with
    the bias and calls cblas_sgemm with beta=1.
    Non-matching cases fall back to the standard topi.add legalization.
    """
    import tvm.topi as topi
    lhs, rhs = call.args
    # Identify which side is the 1-D bias
    def is_1d(v):
        return (hasattr(v, 'struct_info') and v.struct_info is not None and
                v.struct_info.shape is not None and
                len(v.struct_info.shape.values) == 1)
    if is_1d(rhs):
        matmul_var, bias = lhs, rhs
    elif is_1d(lhs):
        matmul_var, bias = rhs, lhs
    else:
        return bb.call_te(topi.add, lhs, rhs)
    if not isinstance(matmul_var, tvm.relax.Var):
        return bb.call_te(topi.add, lhs, rhs)
    binding = bb.lookup_binding(matmul_var)
    if not (isinstance(binding, tvm.relax.Call) and
            isinstance(binding.op, tvm.ir.Op) and
            binding.op == _call_tir_op):
        return bb.call_te(topi.add, lhs, rhs)
    gv = binding.args[0]
    # TVM names TIR functions after the Python function passed to call_te
    if not (hasattr(gv, 'name_hint') and gv.name_hint.startswith("_cblas_matmul")):
        return bb.call_te(topi.add, lhs, rhs)
    args_tuple = binding.args[1]
    if not (isinstance(args_tuple, tvm.relax.Tuple) and len(args_tuple.fields) == 2):
        return bb.call_te(topi.add, lhs, rhs)
    A, B = args_tuple.fields
    if not (A.struct_info.shape and len(A.struct_info.shape.values) == 2):
        return bb.call_te(topi.add, lhs, rhs)
    # Determine if B is transposed: B[N,K] (trans_b=True) or B[K,N] (trans_b=False)
    if not (B.struct_info.shape and len(B.struct_info.shape.values) == 2):
        return bb.call_te(topi.add, lhs, rhs)
    K = int(A.struct_info.shape.values[1])
    b0, b1 = int(B.struct_info.shape.values[0]), int(B.struct_info.shape.values[1])
    if b1 == K:
        trans_b = True
    elif b0 == K:
        trans_b = False
    else:
        return bb.call_te(topi.add, lhs, rhs)
    print(f"[mdl] folded bias add into CBLAS matmul (trans_b={trans_b})", flush=True)
    def _cblas_matmul_biased_call(a, b, c, _trans_b=trans_b):
        return _cblas_matmul_biased_te(a, b, c, _trans_b)
    return bb.call_te(_cblas_matmul_biased_call, A, B, bias)


# ── oneDNN wrapper source ─────────────────────────────────────────────────────
# This C++ source is written to a temp file and compiled into the model .so
# alongside TVM's generated TIR code.  It wraps oneDNN softmax, GELU(erf), and
# layer_norm with a lightweight shape-keyed primitive cache so creation happens
# once per unique shape, not every call.
_DNNL_OPS_SRC = r"""
#include <oneapi/dnnl/dnnl.hpp>
#include <mkl_cblas.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

using namespace dnnl;

// ── global engine / stream (CPU, device 0) ──────────────────────────────────
static engine& dnnl_eng() {
    static engine e(engine::kind::cpu, 0);
    return e;
}
static stream& dnnl_strm() {
    static stream s(dnnl_eng());
    return s;
}

// ── softmax ──────────────────────────────────────────────────────────────────
struct SoftmaxCache {
    memory::dims dims;
    softmax_forward::primitive_desc pd;
    softmax_forward prim;
};
static std::unordered_map<size_t, SoftmaxCache> g_softmax_cache;
static std::mutex g_softmax_mu;

static SoftmaxCache& get_softmax(int64_t outer, int64_t axis) {
    size_t key = (size_t)outer * 0x9e3779b97f4a7c15ULL ^ (size_t)axis;
    std::lock_guard<std::mutex> lk(g_softmax_mu);
    auto it = g_softmax_cache.find(key);
    if (it != g_softmax_cache.end()) return it->second;
    memory::desc md({outer, axis}, memory::data_type::f32, memory::format_tag::nc);
    auto pd = softmax_forward::primitive_desc(dnnl_eng(),
        prop_kind::forward_inference, algorithm::softmax_accurate, md, md, /*axis=*/1);
    auto prim = softmax_forward(pd);
    g_softmax_cache.emplace(key, SoftmaxCache{{outer, axis}, std::move(pd), std::move(prim)});
    return g_softmax_cache.at(key);
}

extern "C" int mdl_dnnl_softmax_f32(
    float* dst, const float* src, int64_t outer, int64_t axis)
{
    auto& c = get_softmax(outer, axis);
    memory::desc md({outer, axis}, memory::data_type::f32, memory::format_tag::nc);
    memory src_m(md, dnnl_eng(), (void*)src);
    memory dst_m(md, dnnl_eng(), dst);
    c.prim.execute(dnnl_strm(), {{DNNL_ARG_SRC, src_m}, {DNNL_ARG_DST, dst_m}});
    dnnl_strm().wait();
    return 0;
}

// ── GELU (erf variant) ────────────────────────────────────────────────────────
struct EltwiseCache {
    memory::dims dims;
    eltwise_forward::primitive_desc pd;
    eltwise_forward prim;
};
static std::unordered_map<size_t, EltwiseCache> g_gelu_cache;
static std::mutex g_gelu_mu;

static EltwiseCache& get_gelu(int64_t n) {
    std::lock_guard<std::mutex> lk(g_gelu_mu);
    auto it = g_gelu_cache.find((size_t)n);
    if (it != g_gelu_cache.end()) return it->second;
    memory::desc md({{n}, memory::data_type::f32, memory::format_tag::a});
    auto pd = eltwise_forward::primitive_desc(dnnl_eng(),
        prop_kind::forward_inference, algorithm::eltwise_gelu_erf, md, md, 0.f, 0.f);
    auto prim = eltwise_forward(pd);
    g_gelu_cache.emplace(n, EltwiseCache{{n}, std::move(pd), std::move(prim)});
    return g_gelu_cache.at(n);
}

extern "C" int mdl_dnnl_gelu_f32(float* dst, const float* src, int64_t n) {
    auto& c = get_gelu(n);
    memory::desc md({n}, memory::data_type::f32, memory::format_tag::a);
    memory src_m(md, dnnl_eng(), (void*)src);
    memory dst_m(md, dnnl_eng(), dst);
    c.prim.execute(dnnl_strm(), {{DNNL_ARG_SRC, src_m}, {DNNL_ARG_DST, dst_m}});
    dnnl_strm().wait();
    return 0;
}

// ── layer normalisation ───────────────────────────────────────────────────────
struct LNCache {
    int64_t batch, channels;
    layer_normalization_forward::primitive_desc pd;
    layer_normalization_forward prim;
};
static std::unordered_map<size_t, LNCache> g_ln_cache;
static std::mutex g_ln_mu;

static LNCache& get_ln(int64_t batch, int64_t channels, float eps) {
    size_t key = (size_t)batch * 0xdeadbeef ^ (size_t)channels;
    std::lock_guard<std::mutex> lk(g_ln_mu);
    auto it = g_ln_cache.find(key);
    if (it != g_ln_cache.end()) return it->second;
    memory::desc md({{batch, channels}, memory::data_type::f32, memory::format_tag::nc});
    memory::desc stat_md({{batch}, memory::data_type::f32, memory::format_tag::a});
    auto pd = layer_normalization_forward::primitive_desc(dnnl_eng(),
        prop_kind::forward_inference, md, md, stat_md, eps,
        normalization_flags::use_scale | normalization_flags::use_shift);
    auto prim = layer_normalization_forward(pd);
    g_ln_cache.emplace(key, LNCache{batch, channels, std::move(pd), std::move(prim)});
    return g_ln_cache.at(key);
}

extern "C" int mdl_dnnl_layer_norm_f32(
    float* dst, const float* src,
    const float* scale, const float* shift,
    int64_t batch, int64_t channels, float eps)
{
    auto& c = get_ln(batch, channels, eps);
    auto& e = dnnl_eng();
    memory::desc data_md({batch, channels}, memory::data_type::f32, memory::format_tag::nc);
    memory::desc vec_md({channels}, memory::data_type::f32, memory::format_tag::a);
    memory src_m(data_md, e, (void*)src);
    memory dst_m(data_md, e, dst);
    // In oneDNN 3.x, scale (gamma) and shift (beta) are separate [C] vectors.
    memory sc_m(vec_md, e, (void*)scale);
    memory sh_m(vec_md, e, (void*)shift);
    c.prim.execute(dnnl_strm(), {
        {DNNL_ARG_SRC, src_m}, {DNNL_ARG_DST, dst_m},
        {DNNL_ARG_SCALE, sc_m}, {DNNL_ARG_SHIFT, sh_m}
    });
    dnnl_strm().wait();
    return 0;
}

// ── matmul with fused bias (oneDNN) ──────────────────────────────────────────
// C[M,N] = A[M,K] @ B + bias[N], where B is [N,K] if trans_b=1 else [K,N].
//
// Uses oneDNN matmul with bias as a first-class argument.  Compared to the old
// approach (memcpy bias into C rows, then cblas_sgemm with beta=1), this reads
// and writes C only once — no separate bias broadcast pass needed.
//
// For trans_b=1 (B stored row-major as [N,K]): describe weights with strides
// {1, K} so that weights[k,n] = B_ptr[n*K+k] = B[n,k], i.e. a column-major
// view of B, giving A @ B.T.

struct MatmulBiasCache {
    dnnl::matmul::primitive_desc pd;
    dnnl::matmul prim;
};
static std::unordered_map<size_t, MatmulBiasCache> g_matbias_cache;
static std::mutex g_matbias_mu;

static MatmulBiasCache& get_matmul_bias(int64_t M, int64_t K, int64_t N, int trans_b) {
    size_t key = (size_t)M * 0xdeadbeefULL
               ^ (size_t)K * 0x9e3779b9ULL
               ^ (size_t)N * 0x517cc1b7ULL
               ^ (size_t)trans_b;
    std::lock_guard<std::mutex> lk(g_matbias_mu);
    auto it = g_matbias_cache.find(key);
    if (it != g_matbias_cache.end()) return it->second;

    auto f32 = memory::data_type::f32;
    memory::desc src_md ({M, K}, f32, {K, 1});
    memory::desc wgt_md = trans_b
        ? memory::desc({K, N}, f32, {1, K})  // col-major view: wgt[k,n]=B[n,k]
        : memory::desc({K, N}, f32, {N, 1});
    memory::desc bias_md({1, N}, f32, {N, 1});
    memory::desc dst_md ({M, N}, f32, {N, 1});

    auto pd   = dnnl::matmul::primitive_desc(dnnl_eng(), src_md, wgt_md, bias_md, dst_md);
    auto prim = dnnl::matmul(pd);
    g_matbias_cache.emplace(key, MatmulBiasCache{std::move(pd), std::move(prim)});
    return g_matbias_cache.at(key);
}

extern "C" int mdl_dnnl_matmul_bias_f32(
    float* dst, const float* A, const float* B, const float* bias,
    int64_t M, int64_t K, int64_t N, int trans_b)
{
    auto& c = get_matmul_bias(M, K, N, trans_b);
    auto& e = dnnl_eng();
    auto f32 = memory::data_type::f32;
    memory::desc src_md ({M, K}, f32, {K, 1});
    memory::desc wgt_md = trans_b
        ? memory::desc({K, N}, f32, {1, K})
        : memory::desc({K, N}, f32, {N, 1});
    memory::desc bias_md({1, N}, f32, {N, 1});
    memory::desc dst_md ({M, N}, f32, {N, 1});
    memory src_m (src_md,  e, (void*)A);
    memory wgt_m (wgt_md,  e, (void*)B);
    memory bias_m(bias_md, e, (void*)bias);
    memory dst_m (dst_md,  e, dst);
    c.prim.execute(dnnl_strm(), {
        {DNNL_ARG_SRC,     src_m},
        {DNNL_ARG_WEIGHTS, wgt_m},
        {DNNL_ARG_BIAS,    bias_m},
        {DNNL_ARG_DST,     dst_m},
    });
    dnnl_strm().wait();
    return 0;
}

// ── fused scaled dot-product attention (BSNH → HSN transpose + batch SGEMM) ──
// Inputs/output: [1, S, H, D] in BSNH row-major layout.
//
// Problem with direct BSNH access in BLAS: Q_h[s,d] = Q[s*H*D + h*D + d], so
// consecutive rows of Q_h are H*D floats (=3KB for BERT) apart.  At large S
// (>128), MKL wastes (H-1)/H of every cache line fetched — the SGEMM becomes
// memory-bandwidth-bound on wasted fetches rather than on useful data.
//
// Fix: copy Q, K, V from interleaved BSNH into contiguous HSN [H, S, D] buffers
// (lda=D=64 bytes per row), then run cblas_sgemm_batch with packed inputs.
// The copy cost (4 × H*S*D floats) is amortised quickly at large S.
extern "C" int mdl_sdpa_bsnh_f32(
    float* out,        // [1, S, H, D]  BSNH
    const float* Q,    // [1, S, H, D]  BSNH
    const float* K,    // [1, S, H, D]  BSNH
    const float* V,    // [1, S, H, D]  BSNH
    int64_t S, int64_t H, int64_t D)
{
    const float scale = 1.0f / std::sqrt((float)D);

    // Thread-local scratch buffers — allocated once per thread, reused each call.
    // Avoids repeated mmap (glibc falls back to mmap for allocations > 128 KB)
    // which causes page-fault storms at large S.
    static thread_local std::vector<float> tl_q_hsn, tl_k_hsn, tl_v_hsn,
                                            tl_out_hsn, tl_scores;
    static thread_local std::vector<const float*> tl_A;
    static thread_local std::vector<float*>       tl_C;

    const size_t hsn_sz = (size_t)(H * S * D);
    const size_t sc_sz  = (size_t)(H * S * S);
    tl_scores.resize(sc_sz);

    const CBLAS_TRANSPOSE tN = CblasNoTrans, tT = CblasTrans;
    const MKL_INT iS = (MKL_INT)S, iD = (MKL_INT)D, iH = (MKL_INT)H;
    const MKL_INT iHD = (MKL_INT)(H * D);
    const float   one = 1.0f, zero = 0.0f;

    // Pointer arrays — size H, tiny, no mmap risk.
    tl_A.resize(H); tl_C.resize(H);
    std::vector<const float*>& A_arr = tl_A;
    std::vector<float*>&       C_arr = tl_C;

    // Threshold: for large S, copying Q/K/V from interleaved BSNH to packed HSN
    // [H,S,D] (lda=D) pays off because the SGEMM cache-waste from lda=H*D grows
    // O(S²) while the copy cost is O(S).  At small S the copy overhead dominates.
    // Empirically the crossover is around S=700 for BERT (H=12, D=64).
    if (S >= 700) {
        tl_q_hsn.resize(hsn_sz); tl_k_hsn.resize(hsn_sz);
        tl_v_hsn.resize(hsn_sz); tl_out_hsn.resize(hsn_sz);
        float* q_hsn = tl_q_hsn.data(), *k_hsn = tl_k_hsn.data();
        float* v_hsn = tl_v_hsn.data(), *out_hsn = tl_out_hsn.data();

        // BSNH → HSN (inner d loop is contiguous for both src/dst → AVX-512).
        for (int64_t s = 0; s < S; ++s) {
            const float* qr = Q + s*H*D, *kr = K + s*H*D, *vr = V + s*H*D;
            for (int64_t h = 0; h < H; ++h) {
                const float* qs = qr + h*D, *ks = kr + h*D, *vs = vr + h*D;
                float* qd = q_hsn + h*S*D + s*D;
                float* kd = k_hsn + h*S*D + s*D;
                float* vd = v_hsn + h*S*D + s*D;
                for (int64_t d = 0; d < D; ++d) { qd[d]=qs[d]; kd[d]=ks[d]; vd[d]=vs[d]; }
            }
        }

        std::vector<const float*> B_arr(H);
        for (int64_t h = 0; h < H; ++h) {
            A_arr[h] = q_hsn + h*S*D;
            B_arr[h] = k_hsn + h*S*D;
            C_arr[h] = tl_scores.data() + h*S*S;
        }
        cblas_sgemm_batch(CblasRowMajor, &tN, &tT, &iS, &iS, &iD, &scale,
            A_arr.data(), &iD, B_arr.data(), &iD,  // lda=D (packed)
            &zero, C_arr.data(), &iS, 1, &iH);

        mdl_dnnl_softmax_f32(tl_scores.data(), tl_scores.data(), H * S, S);

        for (int64_t h = 0; h < H; ++h) {
            A_arr[h] = tl_scores.data() + h*S*S;
            B_arr[h] = v_hsn + h*S*D;
            C_arr[h] = out_hsn + h*S*D;
        }
        cblas_sgemm_batch(CblasRowMajor, &tN, &tN, &iS, &iD, &iS, &one,
            A_arr.data(), &iS, B_arr.data(), &iD,  // lda=D (packed)
            &zero, C_arr.data(), &iD, 1, &iH);

        // HSN → BSNH scatter.
        for (int64_t s = 0; s < S; ++s) {
            float* outr = out + s*H*D;
            for (int64_t h = 0; h < H; ++h) {
                const float* src = out_hsn + h*S*D + s*D; float* dst = outr + h*D;
                for (int64_t d = 0; d < D; ++d) dst[d] = src[d];
            }
        }
    } else {
        // Small S: read BSNH heads directly with lda=H*D (no copy).
        std::vector<const float*> B_arr(H);
        for (int64_t h = 0; h < H; ++h) {
            A_arr[h] = Q + h*D; B_arr[h] = K + h*D;
            C_arr[h] = tl_scores.data() + h*S*S;
        }
        cblas_sgemm_batch(CblasRowMajor, &tN, &tT, &iS, &iS, &iD, &scale,
            A_arr.data(), &iHD, B_arr.data(), &iHD,
            &zero, C_arr.data(), &iS, 1, &iH);

        mdl_dnnl_softmax_f32(tl_scores.data(), tl_scores.data(), H * S, S);

        for (int64_t h = 0; h < H; ++h) {
            A_arr[h] = tl_scores.data() + h*S*S;
            B_arr[h] = V + h*D; C_arr[h] = out + h*D;
        }
        cblas_sgemm_batch(CblasRowMajor, &tN, &tN, &iS, &iD, &iS, &one,
            A_arr.data(), &iS, B_arr.data(), &iHD,
            &zero, C_arr.data(), &iHD, 1, &iH);
    }
    return 0;
}
"""

def _make_dnnl_fcompile(workspace_dir):
    """Return a fcompile callback that compiles mdl_dnnl_ops.cpp into the model .so."""
    import subprocess, os
    from tvm.contrib import cc

    src_path = os.path.join(workspace_dir, "mdl_dnnl_ops.cpp")
    obj_path = os.path.join(workspace_dir, "mdl_dnnl_ops.o")
    with open(src_path, "w") as f:
        f.write(_DNNL_OPS_SRC)
    subprocess.check_call([
        "g++", "-O3", "-fPIC", "-std=c++17",
        "-I/usr/include", "-I/usr/local/include",
        "-I/opt/intel/oneapi/mkl/latest/include",
        "-c", src_path, "-o", obj_path
    ])
    print("[mdl] oneDNN wrappers compiled", flush=True)

    def fcompile(output, objects, options=None):
        # Libraries (-l flags) must cThe thome AFTER object files in GCC's link order.
        # TVM's create_shared puts `options` BEFORE objects, so -ldnnl would be
        # ignored by --as-needed if passed there.  Build the command manually.
        import subprocess, shutil
        cc_cmd = shutil.which("g++") or "g++"
        cmd = ([cc_cmd, "-shared", "-fPIC", "-o", output]
               + list(options or [])
               + list(objects) + [obj_path]
               + ["-ldnnl"])
        subprocess.check_call(cmd)

    return fcompile


def _dnnl_softmax_te(x):
    """Softmax over the last axis via oneDNN (single-pass fused kernel)."""
    import tvm.te as te
    shape = x.shape
    outer = 1
    for d in shape[:-1]:
        outer = outer * d
    axis_size = shape[-1]
    def body(ins, outs):
        return te.extern_outs(
            outs, ins,
            lambda ins, outs: tvm.tirx.Evaluate(tvm.tirx.call_extern(
                "int32", "mdl_dnnl_softmax_f32",
                outs[0].access_ptr("w"),
                ins[0].access_ptr("r"),
                outer, axis_size,
            )),
            name="dnnl_softmax",
        )
    return te.extern(shape, [x],
                     lambda ins, outs: tvm.tirx.Evaluate(tvm.tirx.call_extern(
                         "int32", "mdl_dnnl_softmax_f32",
                         outs[0].access_ptr("w"),
                         ins[0].access_ptr("r"),
                         outer, axis_size,
                     )),
                     dtype=x.dtype, name="dnnl_softmax")

def _legalize_softmax_dnnl(bb, call):
    return bb.call_te(_dnnl_softmax_te, call.args[0])

def _dnnl_gelu_te(x):
    """GELU (erf variant) via oneDNN eltwise kernel."""
    shape = x.shape
    n = 1
    for d in shape:
        n = n * d
    return tvm.te.extern(
        shape, [x],
        lambda ins, outs: tvm.tirx.Evaluate(tvm.tirx.call_extern(
            "int32", "mdl_dnnl_gelu_f32",
            outs[0].access_ptr("w"),
            ins[0].access_ptr("r"),
            n,
        )),
        dtype=x.dtype, name="dnnl_gelu",
    )

def _legalize_gelu_dnnl(bb, call):
    return bb.call_te(_dnnl_gelu_te, call.args[0])

def _dnnl_layer_norm_te(x, scale, shift, axes, eps):
    """Layer normalisation via oneDNN (single fused kernel)."""
    # axes is a list; we always normalize over the last dim.
    shape = x.shape
    batch = 1
    for d in shape[:-1]:
        batch = batch * d
    channels = shape[-1]
    def body(ins, outs):
        return tvm.tirx.Evaluate(tvm.tirx.call_extern(
            "int32", "mdl_dnnl_layer_norm_f32",
            outs[0].access_ptr("w"),
            ins[0].access_ptr("r"),  # src
            ins[1].access_ptr("r"),  # scale (gamma)
            ins[2].access_ptr("r"),  # shift (beta)
            batch, channels, tvm.tirx.const(float(eps), "float32"),
        ))
    return tvm.te.extern(
        shape, [x, scale, shift],
        body,
        dtype=x.dtype, name="dnnl_layer_norm",
    )

def _legalize_layer_norm_dnnl(bb, call):
    x     = call.args[0]
    scale = call.args[1]
    shift = call.args[2]
    axes  = list(call.attrs.axes)
    eps   = float(call.attrs.epsilon)
    return bb.call_te(_dnnl_layer_norm_te, x, scale, shift, axes, eps)


def _rewrite_attention_python(mod):
    """
    Python DFPattern pass: replace the manual SDPA decomposition with
    relax.nn.attention so that LegalizeOps can dispatch it to
    _legalize_attention_sdpa_bsnh (CBLAS + oneDNN, no explicit transposes).

    Matched pattern (after softmax rewrite has run):
      q_heads = permute_dims(reshape(q_proj, [S, H, D]), [1,0,2])  -> [H,S,D]
      k_heads = permute_dims(reshape(k_proj, [S, H, D]), [1,0,2])  -> [H,S,D]
      v_heads = permute_dims(reshape(v_proj, [S, H, D]), [1,0,2])  -> [H,S,D]
      k_T     = permute_dims(k_heads, [0,2,1])                      -> [H,D,S]
      scores  = matmul(q_heads, k_T)                                -> [H,S,S]
      scaled  = divide(scores, sqrt_scale)
      attn    = relax.nn.softmax(scaled, axis=-1)
      ctx     = matmul(attn, v_heads)                               -> [H,S,D]
      ctx_p   = permute_dims(ctx, [1,0,2])                          -> [S,H,D]
      out     = reshape(ctx_p, [S, H*D])   <- TOP-LEVEL MATCH

    Replacement:
      relax.nn.attention(
          reshape(q_proj, [1, S_q, H, D_k]),   # BSNH
          reshape(k_proj, [1, S_k, H, D_k]),
          reshape(v_proj, [1, S_v, H, D_v]),
      )  -> [1, S_q, H, D_v]
      then reshape to [S_q, H*D_v]
    """
    import tvm.relax.dpl as dpl

    q_proj = dpl.wildcard()
    k_proj = dpl.wildcard()
    v_proj = dpl.wildcard()

    q_heads = dpl.is_op("relax.permute_dims")(dpl.is_op("relax.reshape")(q_proj, dpl.wildcard()))
    k_heads = dpl.is_op("relax.permute_dims")(dpl.is_op("relax.reshape")(k_proj, dpl.wildcard()))
    v_heads = dpl.is_op("relax.permute_dims")(dpl.is_op("relax.reshape")(v_proj, dpl.wildcard()))

    k_T    = dpl.is_op("relax.permute_dims")(k_heads)
    scores = dpl.is_op("relax.matmul")(q_heads, k_T)
    scaled = dpl.is_op("relax.divide")(scores, dpl.wildcard())
    attn   = dpl.is_op("relax.nn.softmax")(scaled)
    ctx    = dpl.is_op("relax.matmul")(attn, v_heads)
    ctx_p  = dpl.is_op("relax.permute_dims")(ctx)
    pat    = dpl.is_op("relax.reshape")(ctx_p, dpl.wildcard())

    def rewriter(expr, bindings):
        # Extract matched tensors
        q = bindings[q_proj]
        k = bindings[k_proj]
        v = bindings[v_proj]
        # Extract shapes
        q_si = q.struct_info
        k_si = k.struct_info
        v_si = v.struct_info
        if q_si is None or k_si is None or v_si is None:
            return expr
        q_shape = q_si.shape.values if q_si.shape else None
        k_shape = k_si.shape.values if k_si.shape else None
        v_shape = v_si.shape.values if v_si.shape else None
        if q_shape is None or k_shape is None or v_shape is None:
            return expr
        if len(q_shape) < 2 or len(k_shape) < 2 or len(v_shape) < 2:
            return expr
        S_q = q_shape[0]
        S_k = k_shape[0]
        S_v = v_shape[0]
        H_D_q = q_shape[1]  # H * D_k

        # Get H from k_heads shape [H, S_k, D_k]
        kh = bindings[k_heads]
        kh_shape = kh.struct_info.shape.values if kh.struct_info and kh.struct_info.shape else None
        if kh_shape is None or len(kh_shape) != 3:
            return expr
        H   = kh_shape[0]
        D_k = kh_shape[2]
        # Derive D_v from v_heads shape [H, S_v, D_v]
        vh = bindings[v_heads]
        vh_shape = vh.struct_info.shape.values if vh.struct_info and vh.struct_info.shape else None
        if vh_shape is None or len(vh_shape) != 3:
            return expr
        D_v = vh_shape[2]

        # Reshape to BSNH [1, S, H, D] using each tensor's own symbolic seq-length
        q_bsnh = tvm.relax.op.reshape(q, [1, S_q, H, D_k])
        k_bsnh = tvm.relax.op.reshape(k, [1, S_k, H, D_k])
        v_bsnh = tvm.relax.op.reshape(v, [1, S_v, H, D_v])

        # relax.nn.attention (scale=None → TVM default 1/sqrt(D_k))
        attn_bsnh = tvm.relax.op.nn.attention(q_bsnh, k_bsnh, v_bsnh)  # [1, S_q, H, D_v]

        # Reshape output [1, S_q, H, D_v] → [S_q, H*D_v]
        return tvm.relax.op.reshape(attn_bsnh, [S_q, H * D_v])

    n_rewrites = 0
    for gv in mod.get_global_vars():
        fn = mod[gv]
        if not isinstance(fn, tvm.relax.Function):
            continue
        new_fn = dpl.rewrite_call(pat, rewriter, fn)
        if not new_fn.same_as(fn):
            mod.update_func(gv, new_fn)
            n_rewrites += 1
    if n_rewrites:
        print(f"[mdl] attention pattern: {n_rewrites} function(s) rewritten", flush=True)
    return mod


def _sdpa_bsnh_te(q_bsnh, k_bsnh, v_bsnh):
    """Fused SDPA in BSNH layout via mdl_sdpa_bsnh_f32.

    Inputs: [1, S, H, D] each.  Output: [1, S, H, D].
    Uses CBLAS sgemm_batch_strided with non-unit lda to read heads directly
    from the BSNH layout — no explicit head-transpose tensors needed.
    """
    S = q_bsnh.shape[1]
    H = q_bsnh.shape[2]
    D = q_bsnh.shape[3]
    return tvm.te.extern(
        q_bsnh.shape, [q_bsnh, k_bsnh, v_bsnh],
        lambda ins, outs: tvm.tirx.Evaluate(tvm.tirx.call_extern(
            "int32", "mdl_sdpa_bsnh_f32",
            outs[0].access_ptr("w"),
            ins[0].access_ptr("r"),
            ins[1].access_ptr("r"),
            ins[2].access_ptr("r"),
            S, H, D,
        )),
        dtype="float32", name="sdpa_bsnh",
    )

def _legalize_attention_sdpa_bsnh(bb, call):
    """Legalize relax.nn.attention → mdl_sdpa_bsnh_f32 (CBLAS + oneDNN softmax)."""
    q, k, v = call.args[0], call.args[1], call.args[2]
    return bb.call_te(_sdpa_bsnh_te, q, k, v)


def _fast_gelu_te(x):
    """GELU using fast_erf polynomial — fallback when oneDNN is not available."""
    import tvm.te as te
    import tvm.topi as topi
    sqrt2 = te.const(1.4142135623730951, x.dtype)
    half  = te.const(0.5, x.dtype)
    one   = te.const(1.0, x.dtype)
    return topi.multiply(topi.multiply(x, half),
                         topi.add(one, topi.fast_erf(topi.divide(x, sqrt2))))

def _legalize_gelu_fast(bb, call):
    return bb.call_te(_fast_gelu_te, call.args[0])


def _rewrite_softmax_python(mod):
    """
    Python DFPattern pass: replace numerically-stable softmax decomposition
    (exp(x-max(x)) / sum(exp(x-max(x)))) with relax.nn.softmax.

    This is done in Python because TVM's C++ RewriteCall does not correctly
    propagate DAG constraints for patterns where the same node (exp_val) must
    appear as both the numerator and the argument to sum().  Python dpl handles
    this correctly by tracking bindings across the full match.

    Two sub-patterns are handled:
      (a) CSE form: divide(exp_val, sum(exp_val))   [shared exp node]
      (b) Dup form:  divide(exp_num, sum(exp_den))  [two separate exp nodes
                     from the same input — emitted when CSE is not applied]
    """
    import tvm.relax.dpl as dpl

    # Tree pattern — use separate wildcards to avoid DAG issues in matcher.
    x1 = dpl.wildcard()
    exp_num = dpl.is_op("relax.exp")(
        dpl.is_op("relax.subtract")(x1, dpl.is_op("relax.max")(x1)))

    x2 = dpl.wildcard()
    exp_den = dpl.is_op("relax.exp")(
        dpl.is_op("relax.subtract")(x2, dpl.is_op("relax.max")(x2)))
    sum_pat = dpl.is_op("relax.sum")(exp_den)

    pat = dpl.is_op("relax.divide")(exp_num, sum_pat)

    def rewriter(expr, bindings):
        inp1 = bindings[x1]
        inp2 = bindings[x2]
        # Both branches must come from the same input tensor.
        if not tvm.ir.structural_equal(inp1, inp2):
            return expr
        return tvm.relax.op.nn.softmax(inp1, axis=-1)

    for gv in mod.get_global_vars():
        fn = mod[gv]
        if not isinstance(fn, tvm.relax.Function):
            continue
        new_fn = dpl.rewrite_call(pat, rewriter, fn)
        if not new_fn.same_as(fn):
            mod.update_func(gv, new_fn)  # mutates in place
    return mod


def default_pipeline(mod):
    # Pattern rewrites (softmax, gelu, layer_norm) are applied in C++ before
    # this point; the module arrives here already containing the high-level ops.
    #
    # LegalizeOps runs FIRST so oneDNN/CBLAS legalizers intercept softmax/gelu/
    # layer_norm before FastMathTransform can replace them with TIR fast_exp kernels.
    # FastMathTransform then handles any residual relax.exp/erf/tanh.
    #
    # Additionally, a Python DFPattern pass replaces the softmax decomposition
    # with relax.nn.softmax (the C++ pass may miss some forms; Python handles all).
    mod = _rewrite_softmax_python(mod)
    mod = _rewrite_attention_python(mod)
    # Normalize to ANF: the Python attention rewriter may return compound expressions
    # (reshape(attention(reshape(q),...), [S,H*D])) which violates Relax's single-
    # call-per-binding rule.  Normalize() flattens nested calls into separate
    # DataflowVar bindings so LegalizeOps can legalize relax.nn.attention properly.
    mod = tvm.relax.transform.Normalize()(mod)
    # Remove dead SDPA bindings (Q_heads, K_T, scores, attn, ctx, etc.) that
    # the attention pattern left behind.  Eliminating them BEFORE LegalizeOps
    # avoids legalizing-then-DCE-ing dead CBLAS / oneDNN kernels, which would
    # waste compile time and leave ghost kernels in the runtime profile.
    with target:
        mod = RT.DeadCodeElimination()(mod)
    # FoldConstant before LegalizeOps so sqrt(const(64)) → const(8), enabling
    # _legalize_divide_fold_batmul_scale to recognise the attention scale as a Constant.
    with target:
        mod = RT.FoldConstant()(mod)
    with target:
        mod = tvm.transform.Sequential([
            RT.LegalizeOps({
                "relax.matmul":         _legalize_matmul_cblas,
                "relax.divide":         _legalize_divide_fold_batmul_scale,
                "relax.add":            _legalize_add_fold_matmul_bias,
                "relax.nn.softmax":     _legalize_softmax_dnnl,
                "relax.nn.gelu":        _legalize_gelu_dnnl,
                "relax.nn.layer_norm":  _legalize_layer_norm_dnnl,
                "relax.nn.attention":   _legalize_attention_sdpa_bsnh,
            }),
            RT.DeadCodeElimination(),
        ])(mod)
    print("[mdl] LegalizeOps done (CBLAS + oneDNN softmax/gelu/layer_norm)", flush=True)
    with target:
        mod = RT.FastMathTransform()(mod)
    print("[mdl] FastMathTransform done", flush=True)

    def try_schedule_reduction_func(func):
        try:
            sch = tvm.s_tir.Schedule(func)
            for bname in ["matmul", "C", "T_matmul_NN", "T_batch_matmul_NT",
                          "einsum", "T_einsum", "compute"]:
                try:
                    block = sch.get_block(bname)
                except Exception:
                    continue
                loops = sch.get_loops(block)
                if len(loops) < 3:
                    continue
                stmt = sch.get(block)
                ivs = stmt.iter_vars
                spatial = [i for i, iv in enumerate(ivs) if iv.iter_type == 0]
                reduc   = [i for i, iv in enumerate(ivs) if iv.iter_type == 2]
                if len(spatial) < 2 or not reduc:
                    continue
                TILE_J = 16
                last_s_idx = spatial[-1]
                j_loop = loops[last_s_idx]
                j_ext = sch.get(j_loop).extent
                if isinstance(j_ext, tvm.tirx.IntImm) and int(j_ext.value) % TILE_J == 0:
                    jo, ji = sch.split(j_loop, factors=[None, TILE_J])
                    loops2 = sch.get_loops(block)
                    outer_s = [loops2[i] for i in spatial[:-1]]
                    new_red = [loops2[i + 1] for i in reduc]
                    sch.reorder(*outer_s + [jo] + new_red + [ji])
                    try:
                        sch.vectorize(sch.get_loops(block)[-1])
                    except Exception:
                        pass
                else:
                    new_order = ([loops[i] for i in spatial[:-1]] +
                                 [loops[i] for i in reduc] +
                                 [loops[spatial[-1]]])
                    sch.reorder(*new_order)
                    inner = sch.get_loops(block)[-1]
                    if isinstance(sch.get(inner).extent, tvm.tirx.IntImm):
                        try:
                            sch.vectorize(inner)
                        except Exception:
                            pass
                print(f"[mdl]   tiled/vectorized block '{bname}'", flush=True)
            return sch.mod["main"]
        except Exception:
            return func

    n_scheduled = 0
    for gv in list(mod.get_global_vars()):
        fn = mod[gv]
        if not isinstance(fn, tvm.tirx.PrimFunc):
            continue
        opt = try_schedule_reduction_func(fn)
        if opt is not fn:
            mod[gv] = opt
            n_scheduled += 1
    print(f"[mdl] Scheduled {n_scheduled} PrimFuncs", flush=True)

    with target:
        mod = tvm.transform.Sequential([
            RT.AnnotateTIROpPattern(),
            RT.FoldConstant(),
            RT.FuseOps(),
            RT.FuseTIR(),
        ])(mod)
    print("[mdl] FuseOps+FuseTIR done", flush=True)

    with target:
        mod = tvm.transform.Sequential([
            RT.RewriteDataflowReshape(),
            RT.ToNonDataflow(),
            RT.RemovePurityChecking(),
            RT.CallTIRRewrite(),
            RT.StaticPlanBlockMemory(),
            RT.LowerAllocTensor(),
            RT.KillAfterLastUse(),
            RT.LowerRuntimeBuiltin(),
            RT.ComputePrimValue(),
            RT.VMShapeLower(),
            RT.AttachGlobalSymbol(),
        ])(mod)
    return mod


def metaschedule_pipeline(mod):
    import tvm.meta_schedule as ms
    import multiprocessing
    work_dir = output_path + ".ms_workdir"
    os.makedirs(work_dir, exist_ok=True)

    import json
    num_cores = multiprocessing.cpu_count()
    parts = target_str.split()
    ms_dict = {"kind": parts[0], "num-cores": num_cores, "libs": ["cblas"]}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.lstrip("-").split("=", 1)
            ms_dict[k] = v
    ms_target_str = json.dumps(ms_dict)
    ms_target = tvm.target.Target(ms_target_str)

    print(f"[mdl] MetaSchedule tuning (work_dir={work_dir}, cores={num_cores})", flush=True)

    import logging
    logging.getLogger("tvm.meta_schedule").setLevel(logging.INFO)
    logging.basicConfig(format="[mdl/ms] %(message)s", level=logging.INFO)

    with ms_target:
        mod_legalized = tvm.transform.Sequential([
            RT.LegalizeOps({"relax.matmul": _legalize_matmul_cblas}),
            RT.DeadCodeElimination(),
        ])(mod)

    try:
        import xgboost  # noqa: F401
        cost_model = "xgb"
    except ImportError:
        cost_model = "random"
        print("[mdl] xgboost not found, using random cost model", flush=True)

    database = ms.relax_integration.tune_relax(
        mod=mod_legalized,
        params={},
        target=ms_target_str,
        work_dir=work_dir,
        max_trials_global=512,
        num_trials_per_iter=16,
        cost_model=cost_model,
    )
    print("[mdl] MetaSchedule tuning done, compiling...", flush=True)

    mod = ms.relax_integration.compile_relax(database, mod_legalized, ms_target_str, params={})
    print("[mdl] MetaSchedule compile done", flush=True)
    return mod, ms_target


def _bind_sym_vars(mod):
    sym_vars = {}
    for gv in mod.get_global_vars():
        fn = mod[gv]
        if not isinstance(fn, tvm.relax.Function):
            continue
        for p in fn.params:
            si = p.struct_info
            if hasattr(si, 'shape') and si.shape is not None:
                for dim in si.shape.values:
                    if isinstance(dim, tvm.tirx.Var) and dim.name not in sym_vars:
                        sym_vars[dim.name] = 8
    return RT.BindSymbolicVars(sym_vars)(mod) if sym_vars else mod


if opt_level in ("metaschedule", "*"):
    try:
        ex, ms_target = metaschedule_pipeline(mod)
        ex.export_library(output_path)
        print(f"[mdl] Relax compiled (MetaSchedule) -> {output_path}", flush=True)
    except Exception as e:
        import traceback
        print(f"[mdl] MetaSchedule failed ({e}), falling back to default", flush=True)
        traceback.print_exc()
        with open(json_path) as f:
            mod = tvm.ir.load_json(f.read())
        mod = _bind_sym_vars(mod)
        mod = default_pipeline(mod)
        ex = tvm.relax.build(mod, target=target, relax_pipeline=None)
        ex.export_library(output_path)
        print(f"[mdl] Relax compiled (fallback) -> {output_path}", flush=True)
else:
    import tempfile
    _ws = tempfile.mkdtemp(prefix="mdl_dnnl_")
    _fcompile = _make_dnnl_fcompile(_ws)
    mod = default_pipeline(mod)
    ex = tvm.relax.build(mod, target=target, relax_pipeline=None)
    ex.export_library(output_path, fcompile=_fcompile)
    print(f"[mdl] Relax compiled (oneDNN) -> {output_path}", flush=True)
)PYHELPER";
	}

	std::string cmd = "python3 " + helperPath + " " + rewrittenPath + " " + output.string() + " \"" + targetStr + "\"" +
					  " \"" + (opts.optimization.empty() ? "default" : opts.optimization) + "\"";
	logger->info("Running: {}", cmd);
	int rc = std::system(cmd.c_str());
	if (rc != 0)
		throw std::runtime_error("Python relax.build() failed (exit " + std::to_string(rc) + ")");

	// ── 5. Clean up temp files ─────────────────────────────────────────────────
	// std::filesystem::remove(jsonPath);   // keep for debug
	// std::filesystem::remove(helperPath); // keep for debug
	logger->info("Done → {}", output.string());
}
