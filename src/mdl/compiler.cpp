#include <mdl/compiler.hpp>

#include <mdl/computegraph.hpp>
#include <mdl/parser/definitions.hpp>

#include <tvm/target/codegen.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/transform.h>

#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <ranges>

#include <mdl/utils/overload.hpp>
#include <mdl/utils/utils.hpp>

#include "./backends/tvm/utils.hpp"
#include "logging.hpp"

namespace tvm {
	namespace tir {
		PrimFunc CreatePrimFunc(
				const ffi::Array<te::Tensor>& arg_list, std::optional<DataType> index_dtype_override = std::nullopt
		);
	}
} // namespace tvm

using Arguments = tvm::ffi::Array<tvm::te::Tensor>;
using TranslateFn = std::function<tvm::te::Tensor(Arguments&, const mdl::Computegraph::Node&)>;

static tvm::te::Tensor translate(/*[out]*/ Arguments& args, const mdl::Computegraph::Node& node);

using tmp = std::function<tvm::te::Tensor(const tvm::te::Tensor&, const tvm::te::Tensor&)>;

static tvm::runtime::DataType toTVMType(mdl::Primitive p) {
	switch (p) {
	case mdl::Primitive::boolean:
		return tvm::runtime::DataType::Bool();
	case mdl::Primitive::floating:
		return tvm::runtime::DataType::Float(32); /** \todo make default floating point type configurable **/
	case mdl::Primitive::uint:
		return tvm::runtime::DataType::UInt(32); /** \todo make default int width configurable **/
	case mdl::Primitive::integer:
		return tvm::runtime::DataType::Int(32); /** \todo make default int width configurable **/
	}
	assert(false); // Should never be reached
}

static tvm::runtime::DataType toUnderlyingTVMType(mdl::DataType dtype) {
	return std::visit(
			mdl::overloaded{
					[](const mdl::Primitive& ptype) { return toTVMType(ptype); },
					[](const mdl::TensorType& ttype) { return toTVMType(ttype.type); },
					[](const mdl::AggregateType&) {
						assert(false); /** \todo implement **/
						return tvm::runtime::DataType{};
					}
			},
			dtype
	);
}

static tvm::ffi::Array<tvm::PrimExpr> getShape(mdl::DataType dtype) {
	return std::visit(
			mdl::overloaded{
					[](const mdl::Primitive& ptype) { return tvm::ffi::Array<tvm::PrimExpr>{1}; },
					[](const mdl::TensorType& ttype) {
						tvm::ffi::Array<tvm::PrimExpr> shape;
						for (const auto i : ttype.shape) {
							// assert(std::holds_alternative<unsigned>(i)); /** \todo support otherwise **/
							if (std::holds_alternative<mdl::Elipsis>(i)) {
								shape.emplace_back(tvm::te::var("B"));
							} else {
								shape.emplace_back(
										tvm::PrimExpr{static_cast<int32_t>(std::get<unsigned>(i))}
								); // May overflow since primexpr uses int32_t
							}
						}
						return shape;
					},
					[](const mdl::AggregateType&) {
						assert(false); /** \todo implement **/
						return tvm::ffi::Array<tvm::PrimExpr>{};
					}
			},
			dtype
	);
}

static tvm::te::Tensor toPlaceholder(const std::string& name, mdl::DataType dtype) {
	return tvm::te::placeholder(getShape(dtype), toUnderlyingTVMType(dtype), name);
}

static const std::map<mdl::Computegraph::Operation, TranslateFn> operations = {
		{mdl::Computegraph::Operation::Argument,
		 [](Arguments& outargs, const mdl::Computegraph::Node& value) {
			 auto arg = toPlaceholder(mdl::getName(value), mdl::getDataType(value));
			 outargs.emplace_back(arg);
			 return arg;
		 }},
		{mdl::Computegraph::Operation::Input,
		 [](Arguments& outargs, const mdl::Computegraph::Node& value) {
			 auto arg = toPlaceholder(mdl::getName(value), mdl::getDataType(value));
			 outargs.emplace_back(arg);
			 return arg;
		 }},
		/** \todo add mdl::Computegraph::Operation::Const */
		{mdl::Computegraph::Operation::ADD,
		 [](Arguments& outargs, const mdl::Computegraph::Node& value) {
			 const auto& args = mdl::getInputs(value);
			 assert(args.size() == 2);
			 return tvm::topi::add(translate(outargs, *args[0]), translate(outargs, *args[1]));
		 }},
		{mdl::Computegraph::Operation::SUB,
		 [](Arguments& outargs, const mdl::Computegraph::Node& value) {
			 const auto& args = mdl::getInputs(value);
			 assert(args.size() == 2);
			 return tvm::topi::subtract(translate(outargs, *args[0]), translate(outargs, *args[1]));
		 }},
		{mdl::Computegraph::Operation::MULT,
		 [](Arguments& outargs, const mdl::Computegraph::Node& value) {
			 const auto& args = mdl::getInputs(value);
			 assert(args.size() == 2);
			 return tvm::topi::multiply(translate(outargs, *args[0]), translate(outargs, *args[1]));
		 }},
		{mdl::Computegraph::Operation::DIV,
		 [](Arguments& outargs, const mdl::Computegraph::Node& value) {
			 const auto& args = mdl::getInputs(value);
			 assert(args.size() == 2);
			 return tvm::topi::divide(translate(outargs, *args[0]), translate(outargs, *args[1]));
		 }},
		{mdl::Computegraph::Operation::MATMULT, [](Arguments& outargs, const mdl::Computegraph::Node& value) {
			 const auto& args = mdl::getInputs(value);
			 assert(args.size() == 2);
			 auto left = translate(outargs, *args[0]);
			 auto right = translate(outargs, *args[1]);
			 return tvm::topi::tensordot(
					 left, right, tvm::ffi::Array<tvm::PrimExpr>{static_cast<int32_t>(left.ndim() - 1)},
					 tvm::ffi::Array<tvm::PrimExpr>{static_cast<int32_t>((right.ndim() <= 1) ? 0 : (right.ndim() - 2))}
			 );
		 }},
};

static tvm::te::Tensor translate(Arguments& args, const mdl::Computegraph::Node& node) {
	auto it = operations.find(mdl::getOperation(node));
	assert(it != operations.end());
	return it->second(args, node);
}

static tvm::BaseFunc translateFunc(const FuncDesc& func) {
	std::string arguments;
	auto r =
			func.arguments |
			std::views::transform([](VarDesc d) { return std::to_string(mdl::getDataType(*d.node)) + ": " + d.name; }) |
			std::views::join_with(std::string_view{", "});
	std::ranges::copy(std::begin(r), std::end(r), std::back_inserter(arguments));
	auto logger = mdl::log::getLogger("compiler");
	logger->info("Translating {}({})", func.name, arguments);

	Arguments args;
	auto retnode = translate(args, *func.graph->getOutputNode());
	args.emplace_back(retnode);

	// https://github.com/apache/tvm/blob/c8515e1ddfaf4d1afff916c484e68e1513631dd6/python/tvm/tir/build.py#L97C57-L99
	// set tvm::attr::kCallingConv to CallingConv::kDeviceKernelLaunch
	return tvm::WithAttrs(
			tvm::tir::CreatePrimFunc(args, {}), {{tvm::attr::kGlobalSymbol, tvm::ffi::String{func.name}}}
	);
}

void mdl::compileToFile(const Module& module, const std::filesystem::path& output) {
	// Problems: tvm::tir::transform::MakePackedAPI() in the pipeline makes the callingconv packed; CUDA build does not like this
	tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> funcs;
	for (auto func : module.functions)
		funcs.Set(tvm::GlobalVar{{func.name}}, translateFunc(func));

	tvm::Target target{"llvm"};
	// tvm::Target target{"cuda"};
	target = tvm::Target::WithHost(target, tvm::Target{"llvm"});

	// https://github.com/apache/tvm/blob/a393b4756368d26db927085cb1de028b567e78c0/python/tvm/tir/build.py#L156
	auto irmod = tvm::tir::transform::BindTarget(target)(tvm::IRModule{funcs});
	irmod = getDefaultPipeline()(irmod);

	auto hostmod = finalizeHostPasses()(irmod);
	auto devicemod = finalizeDevicePasses()(irmod);

	auto mod = tvm::codegen::Build(devicemod, target);
	mod->WriteToFile(output.c_str(), "");

	auto logger = mdl::log::getLogger("compiler");
	logger->info("Done");
}