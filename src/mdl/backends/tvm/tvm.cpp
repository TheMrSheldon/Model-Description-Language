#include "./tvm.hpp"

#include "../../logging.hpp"
#include "./details/builtins.hpp"
#include "./details/utils.hpp"

#include <mdl/computegraphv2/datatype.hpp>
#include <mdl/computegraphv2/graph.hpp>
#include <mdl/computegraphv2/node.hpp>
#include <mdl/parserv2/compilation_unit.hpp>
#include <mdl/utils/utils.hpp>

#include <tvm/target/codegen.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/transform.h>

#include <cassert>
#include <ranges>

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
					[](const mdl::BoundModuleType&) {
						std::abort();
						/** Should not happen; at this point the modules should be resolved and their member functions inlined into the compute graph. **/
						return tvm::runtime::DataType{};
					}
			},
			dtype
	);
}

static tvm::ffi::Array<tvm::PrimExpr> getShape(mdl::DataType dtype) {
	return std::visit(
			mdl::overloaded{
					[](const mdl::PrimitiveType& ptype) { return tvm::ffi::Array<tvm::PrimExpr>{1}; },
					[](const mdl::TensorType& ttype) {
						tvm::ffi::Array<tvm::PrimExpr> shape;
						for (const auto i : ttype.shape) {
							// assert(std::holds_alternative<unsigned>(i)); /** \todo support otherwise **/
							auto expr = std::visit(
									mdl::overloaded{
											[](mdl::Elipsis) { return tvm::PrimExpr{tvm::te::var("B")}; },
											[](unsigned u) {
												return tvm::PrimExpr{
														static_cast<int32_t>(u)
												}; /* May overflow since primexpr uses int32_t */
											},
											[](const mdl::Node* node) {
												assert(mdl::isCompiletimeEvalable(*node));
												assert(false);
												/** \todo compute node value and return it as a constant PrimExpr **/
												return tvm::PrimExpr{};
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
					[](const mdl::BoundModuleType&) {
						std::abort();
						/** Should not happen; at this point the modules should be resolved and their member functions inlined into the compute graph. **/
						return tvm::ffi::Array<tvm::PrimExpr>{};
					}
			},
			dtype
	);
}

static tvm::te::Tensor toPlaceholder(const std::string& name, mdl::DataType dtype) {
	return tvm::te::placeholder(getShape(dtype), toUnderlyingTVMType(dtype), name);
}

static tvm::te::Tensor translate(Arguments& args, const mdl::Node& node);

static tvm::te::Tensor translate(Arguments& args, const mdl::ConstNode& node) {
	auto dtype = node.datatype();
	assert(std::holds_alternative<mdl::PrimitiveType>(dtype)); /** ConstNode values are always scalars **/
	auto tvmtype = toUnderlyingTVMType(dtype);

	tvm::PrimExpr constExpr = std::visit(
			mdl::overloaded{
					[&](int v) -> tvm::PrimExpr {
						return tvm::cast(tvmtype, tvm::IntImm(tvm::DataType::Int(32), static_cast<int64_t>(v)));
					},
					[&](double v) -> tvm::PrimExpr {
						return tvm::cast(tvmtype, tvm::FloatImm(tvm::DataType::Float(64), v));
					}
			},
			node.value
	);

	return tvm::te::compute({1}, [constExpr](tvm::ffi::Array<tvm::tir::Var>) { return constExpr; }, "const");
}
static tvm::te::Tensor translate(Arguments& args, const mdl::InvocationNode& node) {
	std::vector<tvm::te::Tensor> translatedArgs;
	for (const auto& argRef : node.arguments)
		translatedArgs.push_back(translate(args, argRef.get()));

	auto& fn = mdl::backend::tvm::functions[std::to_underlying(node.id)];
	assert(fn != nullptr); /** builtin not yet implemented for this backend **/
	return fn(translatedArgs);
}
static tvm::te::Tensor translate(Arguments& args, const mdl::ErrorNode& node) {
	/** \todo output that this is an internal error; compilation cannot be run when there are error nodes present **/
	std::abort();
}
static tvm::te::Tensor translate(Arguments& args, const mdl::ParameterNode& node) {
	auto arg = toPlaceholder(mdl::getName(node), mdl::getDataType(node));
	args.emplace_back(arg);
	return arg;
}
static tvm::te::Tensor translate(Arguments& args, const mdl::HyperparameterNode& node) {
	/** \todo implement **/
	std::abort();
}

static tvm::te::Tensor translate(Arguments& args, const mdl::Node& node) {
	/** \todo store the tensor in a cache since the same node could be translated multiple times and should evaluate to the same tensor-node **/
	return std::visit([&](const auto& n) { return translate(args, n); }, node);
}

static ::tvm::BaseFunc translateFunc(const mdl::FunctionDecl& func) {
	assert(func.templateParameters.size() == 0); /** \todo support template params **/
	std::string arguments;
	auto r = func.parameters | std::views::transform([](const auto& pair) {
				 return std::to_string(mdl::getDataType(pair.second.get())) + ": " + pair.first;
			 }) |
			 std::views::join_with(std::string_view{", "});
	std::ranges::copy(std::begin(r), std::end(r), std::back_inserter(arguments));
	auto logger = mdl::log::getLogger("compiler");
	logger->info("Translating {}({})", func.name, arguments);

	Arguments args;
	auto retnode = translate(args, func.output);
	args.emplace_back(retnode);

	// https://github.com/apache/tvm/blob/c8515e1ddfaf4d1afff916c484e68e1513631dd6/python/tvm/tir/build.py#L97C57-L99
	// set tvm::attr::kCallingConv to CallingConv::kDeviceKernelLaunch
	return tvm::WithAttrs(
			tvm::tir::CreatePrimFunc(args, {}), {{tvm::attr::kGlobalSymbol, tvm::ffi::String{func.name}}}
	);
}

void mdl::backends::tvm::compileToFile(const mdl::CompilationUnit& unit, const std::filesystem::path& output) {
	// Problems: tvm::tir::transform::MakePackedAPI() in the pipeline makes the callingconv packed; CUDA build does not like this
	::tvm::ffi::Map<::tvm::GlobalVar, ::tvm::BaseFunc> funcs;
	for (const auto& [name, member] : unit.definitions) {
		if (auto func = std::get_if<FunctionDecl>(&member))
			funcs.Set(::tvm::GlobalVar{{name}}, translateFunc(*func));
	}

	::tvm::Target target{"llvm"};
	// tvm::Target target{"cuda"};
	target = ::tvm::Target::WithHost(target, ::tvm::Target{"llvm"});

	// https://github.com/apache/tvm/blob/a393b4756368d26db927085cb1de028b567e78c0/python/tvm/tir/build.py#L156
	auto irmod = ::tvm::tir::transform::BindTarget(target)(::tvm::IRModule{funcs});
	irmod = getDefaultPipeline()(irmod);

	auto hostmod = finalizeHostPasses()(irmod);
	auto devicemod = finalizeDevicePasses()(irmod);

	auto mod = ::tvm::codegen::Build(devicemod, target);
	mod->WriteToFile(output.c_str(), "");

	auto logger = mdl::log::getLogger("compiler");
	logger->info("Done");
}