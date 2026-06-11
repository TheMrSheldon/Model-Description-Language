#include "./tvm.hpp"

#include "../../argvals.hpp"
#include "../../logging.hpp"
#include "./details/builtins.hpp"
#include "./details/utils.hpp"

#include <mdl/compiler.hpp>
#include <mdl/computegraphv2/datatype.hpp>
#include <mdl/computegraphv2/function.hpp>
#include <mdl/computegraphv2/graph.hpp>
#include <mdl/computegraphv2/node.hpp>
#include <mdl/parserv2/compilation_unit.hpp>
#include <mdl/utils/utils.hpp>

#include <tvm/target/codegen.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/transform.h>

#include <any>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <ranges>
#include <sstream>
#include <unordered_map>

namespace tvm {
	namespace tir {
		PrimFunc CreatePrimFunc(
				const ffi::Array<te::Tensor>& arg_list, std::optional<DataType> index_dtype_override = std::nullopt
		);
	}
} // namespace tvm

using Arguments = tvm::ffi::Array<tvm::te::Tensor>;

static tvm::runtime::DataType toTVMType(mdl::PrimitiveType p) {
	switch (p) {
	case mdl::PrimitiveType::boolean:
		return tvm::runtime::DataType::Bool();
	case mdl::PrimitiveType::floating:
		return tvm::runtime::DataType::Float(32); /** \todo make default floating point type configurable **/
	case mdl::PrimitiveType::uint:
		return tvm::runtime::DataType::UInt(32); /** \todo make default int width configurable **/
	case mdl::PrimitiveType::integer:
		return tvm::runtime::DataType::Int(32); /** \todo make default int width configurable **/
	}
	assert(false); // Should never be reached
}

static tvm::runtime::DataType toUnderlyingTVMType(mdl::DataType dtype) {
	return std::visit(
			mdl::overloaded{
					[](const mdl::PrimitiveType& ptype) { return toTVMType(ptype); },
					[](const mdl::TensorType& ttype) { return toTVMType(ttype.type); },
					[](const mdl::AggregateType&) {
						assert(false); /** \todo implement **/
						return tvm::runtime::DataType{};
					},
					[](const mdl::ModuleType&) {
						std::abort();
						/** Should not happen; at this point the modules should be resolved and their member functions inlined into the compute graph. **/
						return tvm::runtime::DataType{};
					},
					[](const mdl::BoundModuleType&) {
						std::abort();
						/** Should not happen; at this point the modules should be resolved and their member functions inlined into the compute graph. **/
						return tvm::runtime::DataType{};
					}
			},
			dtype
	);
}

/**
 * @brief Recursively evaluates a compile-time-evaluable Node to a TVM PrimExpr.
 * HyperparameterNodes are looked up by their dotted name in @p argVals;
 * arithmetic InvocationNodes are folded into TVM arithmetic expressions.
 */
static tvm::PrimExpr nodeToPrimExpr(const mdl::Node& node, const mdl::ArgVals& argVals) {
	using FIDs = mdl::builtins::FunctionIDs;
	return std::visit(
			mdl::overloaded{
					[](const mdl::ConstNode& cn) -> tvm::PrimExpr {
						return std::visit(
								mdl::overloaded{
										[](int v) -> tvm::PrimExpr { return tvm::IntImm(tvm::DataType::Int(32), v); },
										[](double v) -> tvm::PrimExpr {
											return tvm::FloatImm(tvm::DataType::Float(64), v);
										},
										[](const std::vector<int>&) -> tvm::PrimExpr {
											return tvm::IntImm(tvm::DataType::Int(32), 0);
										},
								},
								cn.value
						);
					},
					[&argVals](const mdl::HyperparameterNode& hn) -> tvm::PrimExpr {
						if (auto val = argVals.lookup(hn.name))
							return tvm::IntImm(tvm::DataType::Int(32), *val);
						return tvm::tir::Var(hn.name.empty() ? "dim" : hn.name, tvm::DataType::Int(32));
					},
					[&argVals](const mdl::InvocationNode& in) -> tvm::PrimExpr {
						if (in.arguments.size() == 2) {
							auto lhs = nodeToPrimExpr(in.arguments[0].get(), argVals);
							auto rhs = nodeToPrimExpr(in.arguments[1].get(), argVals);
							switch (in.id) {
							case FIDs::TENSOR_ADD: return lhs + rhs;
							case FIDs::TENSOR_SUB: return lhs - rhs;
							case FIDs::TENSOR_PROD: return lhs * rhs;
							case FIDs::TENSOR_DIV: return tvm::truncdiv(lhs, rhs);
							default: break;
							}
						}
						return tvm::tir::Var("dim", tvm::DataType::Int(32));
					},
					[](const auto&) -> tvm::PrimExpr { return tvm::tir::Var("dim", tvm::DataType::Int(32)); },
			},
			node
	);
}

static int ellipsisDimCounter = 0;

static tvm::ffi::Array<tvm::PrimExpr> getShape(mdl::DataType dtype, const mdl::ArgVals& argVals) {
	return std::visit(
			mdl::overloaded{
					[](const mdl::PrimitiveType& ptype) { return tvm::ffi::Array<tvm::PrimExpr>{1}; },
					[&argVals](const mdl::TensorType& ttype) {
						tvm::ffi::Array<tvm::PrimExpr> shape;
						for (const auto i : ttype.shape) {
							auto expr = std::visit(
									mdl::overloaded{
											[](mdl::Elipsis) {
												// Each Elipsis becomes a distinct symbolic batch dimension.
												return tvm::PrimExpr{tvm::tir::Var(
														"B" + std::to_string(ellipsisDimCounter++),
														tvm::DataType::Int(32)
												)};
											},
											[](unsigned u) {
												return tvm::PrimExpr{
														static_cast<int32_t>(u)
												}; /* May overflow since primexpr uses int32_t */
											},
											[&argVals](const mdl::Node* node) {
												return nodeToPrimExpr(*node, argVals);
											}
									},
									i
							);
							shape.emplace_back(expr);
						}
						return shape;
					},
					[](const mdl::AggregateType&) {
						assert(false); /** \todo implement **/
						return tvm::ffi::Array<tvm::PrimExpr>{};
					},
					[](const mdl::ModuleType&) {
						std::abort();
						/** Should not happen; at this point the modules should be resolved and their member functions inlined into the compute graph. **/
						return tvm::ffi::Array<tvm::PrimExpr>{};
					},
					[](const mdl::BoundModuleType&) {
						std::abort();
						/** Should not happen; at this point the modules should be resolved and their member functions inlined into the compute graph. **/
						return tvm::ffi::Array<tvm::PrimExpr>{};
					}
			},
			dtype
	);
}

static tvm::te::Tensor toPlaceholder(const std::string& name, mdl::DataType dtype, const mdl::ArgVals& argVals) {
	return tvm::te::placeholder(getShape(dtype, argVals), toUnderlyingTVMType(dtype), name);
}

using Cache = std::unordered_map<const mdl::Node*, ::tvm::te::Tensor>;

static tvm::te::Tensor translate(Arguments& args, Cache& cache, const mdl::Node& node, const mdl::ArgVals& argVals);

static tvm::te::Tensor translate(Arguments& args, Cache&, const mdl::ConstNode& node, const mdl::ArgVals&) {
	return std::visit(
			mdl::overloaded{
					[&](int v) -> tvm::te::Tensor {
						auto tvmtype = toUnderlyingTVMType(node.datatype());
						tvm::PrimExpr e = tvm::cast(tvmtype, tvm::IntImm(tvm::DataType::Int(32), v));
						return tvm::te::compute(
							{1}, [e](tvm::ffi::Array<tvm::tir::Var>) { return e; },
							mdl::backend::tvm::elemName("c")
						);
					},
					[&](double v) -> tvm::te::Tensor {
						auto tvmtype = toUnderlyingTVMType(node.datatype());
						tvm::PrimExpr e = tvm::cast(tvmtype, tvm::FloatImm(tvm::DataType::Float(64), v));
						return tvm::te::compute(
							{1}, [e](tvm::ffi::Array<tvm::tir::Var>) { return e; },
							mdl::backend::tvm::elemName("c")
						);
					},
					[&](const std::vector<int>& v) -> tvm::te::Tensor {
						int n = static_cast<int>(v.size());
						auto vals = v; // capture by value
						return ::tvm::te::compute(
								{n},
								[vals, n](::tvm::ffi::Array<::tvm::tir::Var> idx) -> ::tvm::PrimExpr {
									::tvm::PrimExpr result =
											::tvm::IntImm(::tvm::DataType::Int(32), vals.back());
									for (int i = n - 2; i >= 0; --i)
										result = ::tvm::tir::Select(
												idx[0] == ::tvm::IntImm(::tvm::DataType::Int(32), i),
												::tvm::IntImm(::tvm::DataType::Int(32), vals[i]),
												result
										);
									return result;
								},
								mdl::backend::tvm::elemName("cv")
						);
					},
			},
			node.value
	);
}
static tvm::te::Tensor translate(Arguments& args, Cache& cache, const mdl::InvocationNode& node, const mdl::ArgVals& argVals) {
	std::vector<std::any> translatedArgs;
	for (const auto& argRef : node.arguments)
		translatedArgs.push_back(translate(args, cache, argRef.get(), argVals));

	auto& fn = mdl::backend::tvm::functions[std::to_underlying(node.id)];
	assert(fn != nullptr); /** builtin not yet implemented for this backend **/
	return fn(translatedArgs);
}
static tvm::te::Tensor translate(Arguments& args, Cache&, const mdl::ErrorNode& node, const mdl::ArgVals&) {
	/** \todo output that this is an internal error; compilation cannot be run when there are error nodes present **/
	std::abort();
}
static tvm::te::Tensor translate(Arguments& args, Cache&, const mdl::ParameterNode& node, const mdl::ArgVals& argVals) {
	auto arg = toPlaceholder(mdl::getName(node), mdl::getDataType(node), argVals);
	args.emplace_back(arg);
	return arg;
}
static tvm::te::Tensor translate(Arguments& args, Cache&, const mdl::HyperparameterNode& node, const mdl::ArgVals& argVals) {
	if (auto val = argVals.lookup(node.name)) {
		tvm::PrimExpr e = tvm::IntImm(tvm::DataType::Int(32), *val);
		return tvm::te::compute(
			{1}, [e](tvm::ffi::Array<tvm::tir::Var>) { return e; },
			mdl::backend::tvm::elemName("hp")
		);
	}
	auto logger = mdl::log::getLogger("compiler");
	logger->error("No value for hyperparameter '{}' in argvals; all hyperparameter values must be known at compile time", node.name);
	std::abort();
}

static tvm::te::Tensor translate(Arguments& args, Cache& cache, const mdl::Node& node, const mdl::ArgVals& argVals) {
	if (auto it = cache.find(&node); it != cache.end())
		return it->second;
	auto result = std::visit([&](const auto& n) { return translate(args, cache, n, argVals); }, node);
	cache[&node] = result;
	return result;
}

static ::tvm::BaseFunc translateFunc(const mdl::FunctionDecl& func, const mdl::ArgVals& argVals) {
	// Reset block-name counter so every te::compute in this function gets a unique name.
	// Unique names are required for computeInlineElemwise to call GetBlock unambiguously.
	mdl::backend::tvm::resetBlockNames();

	std::string arguments;
	auto r = func.parameters | std::views::transform([](const auto& pair) {
				 return std::to_string(mdl::getDataType(pair.second.get())) + ": " + pair.first;
			 }) |
			 std::views::join_with(std::string_view{", "});
	std::ranges::copy(std::begin(r), std::end(r), std::back_inserter(arguments));
	auto logger = mdl::log::getLogger("compiler");
	logger->info("Translating {}({})", func.name, arguments);

	Arguments args;
	Cache cache;
	auto retnode = translate(args, cache, func.output, argVals);
	args.emplace_back(retnode);

	// Log the ordered parameter list so callers know the expected argument order.
	for (size_t i = 0; i < args.size(); ++i) {
		std::string shape = "[";
		for (size_t j = 0; j < args[i]->shape.size(); ++j) {
			if (j) shape += ", ";
			std::ostringstream ss;
			ss << args[i]->shape[j];
			shape += ss.str();
		}
		shape += "]";
		bool isOutput = (i + 1 == args.size());
		logger->info("  arg[{}]{} = {} {}", i, isOutput ? " [output]" : "", args[i]->op->name, shape);
	}

	// https://github.com/apache/tvm/blob/c8515e1ddfaf4d1afff916c484e68e1513631dd6/python/tvm/tir/build.py#L97C57-L99
	// set tvm::attr::kCallingConv to CallingConv::kDeviceKernelLaunch
	auto pf = ::tvm::tir::CreatePrimFunc(args, {});

	// Fuse all pure-elementwise chains (GELU, residuals, layer-norm scale/shift, …)
	// into single loops via compute_inline, eliminating intermediate buffer allocations.
	// https://tvm.apache.org/docs/reference/api/python/tir.html#tvm.tir.Schedule.compute_inline
	auto [inlinedPf, nInlined, nDup, nFail] = computeInlineElemwise(pf, func.name);
	pf = inlinedPf;
	logger->info("  compute_inline: {} fused, {} dup skipped, {} failed", nInlined, nDup, nFail);

	return tvm::WithAttrs(pf, {{tvm::attr::kGlobalSymbol, tvm::ffi::String{func.name}}});
}

void mdl::backends::tvm::compileToFile(
		const mdl::CompilationUnit& unit, const std::filesystem::path& output, const mdl::ArgVals& argvals
) {
	// Problems: tvm::tir::transform::MakePackedAPI() in the pipeline makes the callingconv packed; CUDA build does not like this
	::tvm::ffi::Map<::tvm::GlobalVar, ::tvm::BaseFunc> funcs;
	for (const auto& [name, member] : unit.definitions) {
		if (auto func = std::get_if<FunctionDecl>(&member)) {
			if (!func->templateParameters.empty())
				continue; // standalone template functions are only used via inlining
			funcs.Set(::tvm::GlobalVar{{name}}, translateFunc(*func, ArgVals{}));
		} else if (auto* module = std::get_if<ModuleType>(&member)) {
			for (const auto& [funcName, funcDecl] : module->functions) {
				if (!funcDecl.templateParameters.empty())
					continue; // template functions are only used via inlining
				FunctionDecl mangledDecl = funcDecl;
				mangledDecl.name = mdl::mangledName(name, funcName);
				funcs.Set(::tvm::GlobalVar{{mangledDecl.name}}, translateFunc(mangledDecl, argvals));
			}
		}
	}

	// Ask TVM's LLVM backend for the actual host CPU name (avoids -mcpu=native
	// which is rejected by some LLVM triple/container combinations).
	std::string cpuName = "generic";
	if (std::optional<::tvm::ffi::Function> fn = ::tvm::ffi::Function::GetGlobal("tvm.codegen.llvm.GetHostCPUName"))
		cpuName = (*fn)().cast<std::string>();
	std::string targetStr = "llvm -mcpu=" + cpuName;

	::tvm::Target target{targetStr};
	// tvm::Target target{"cuda"};
	target = ::tvm::Target::WithHost(target, ::tvm::Target{targetStr});

	// https://github.com/apache/tvm/blob/a393b4756368d26db927085cb1de028b567e78c0/python/tvm/tir/build.py#L156
	auto irmod = ::tvm::tir::transform::BindTarget(target)(::tvm::IRModule{funcs});
	irmod = getDefaultPipeline()(irmod);

	auto hostmod = finalizeHostPasses()(irmod);
	auto devicemod = finalizeDevicePasses()(irmod);

	auto mod = ::tvm::codegen::Build(devicemod, target);

	auto logger = mdl::log::getLogger("compiler");

	// WriteToFile only supports object/IR formats, not shared libraries directly.
	// For .so output: write a temp object file then link with the system C compiler.
	std::string outStr = output.string();
	if (outStr.size() >= 3 && outStr.substr(outStr.size() - 3) == ".so") {
		std::string objPath = outStr + ".o";
		mod->WriteToFile(objPath, "o");
		std::string cmd = "gcc -shared -fPIC -o " + outStr + " " + objPath;
		int rc = std::system(cmd.c_str());
		if (rc != 0)
			logger->warn("Linker exited with code {}; .so may be incomplete", rc);
		std::filesystem::remove(objPath);
	} else {
		mod->WriteToFile(outStr, "");
	}

	logger->info("Done");
}