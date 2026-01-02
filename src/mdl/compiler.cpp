#include <mdl/compiler.hpp>

#include <mdl/computegraph.hpp>
#include <mdl/parser/definitions.hpp>

#include <tvm/ir/attrs.h>
#include <tvm/ir/module.h>
#include <tvm/relax/backend.h>
#include <tvm/relax/exec_builder.h>
#include <tvm/runtime/module.h>
#include <tvm/target/codegen.h>
#include <tvm/target/tag.h>
#include <tvm/target/target.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <tvm/topi/elemwise.h>
#include <tvm/topi/transform.h>

#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <optional>

#include <mdl/utils/overload.hpp>

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
							assert(std::holds_alternative<unsigned>(i)); /** \todo support otherwise **/
							shape.emplace_back(
									tvm::PrimExpr{static_cast<int32_t>(std::get<unsigned>(i))}
							); // May overflow since primexpr uses int32_t
						}
						//for (auto x : shape)
						//	std::cout << x << std::endl;
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
			 auto arg = toPlaceholder(value.name(), value.dtype);
			 outargs.emplace_back(arg);
			 return arg;
		 }},
		{mdl::Computegraph::Operation::Input,
		 [](Arguments& outargs, const mdl::Computegraph::Node& value) {
			 auto arg = toPlaceholder(value.name(), value.dtype);
			 outargs.emplace_back(arg);
			 return arg;
		 }},
		/** \todo add mdl::Computegraph::Operation::Const */
		{mdl::Computegraph::Operation::ADD,
		 [](Arguments& outargs, const mdl::Computegraph::Node& value) {
			 const auto& args = value.inputs();
			 assert(args.size() == 2);
			 return tvm::topi::add(translate(outargs, *args[0]), translate(outargs, *args[1]));
		 }},
		{mdl::Computegraph::Operation::SUB,
		 [](Arguments& outargs, const mdl::Computegraph::Node& value) {
			 const auto& args = value.inputs();
			 assert(args.size() == 2);
			 return tvm::topi::subtract(translate(outargs, *args[0]), translate(outargs, *args[1]));
		 }},
		{mdl::Computegraph::Operation::MULT,
		 [](Arguments& outargs, const mdl::Computegraph::Node& value) {
			 const auto& args = value.inputs();
			 assert(args.size() == 2);
			 return tvm::topi::multiply(translate(outargs, *args[0]), translate(outargs, *args[1]));
		 }},
		{mdl::Computegraph::Operation::DIV,
		 [](Arguments& outargs, const mdl::Computegraph::Node& value) {
			 const auto& args = value.inputs();
			 assert(args.size() == 2);
			 return tvm::topi::divide(translate(outargs, *args[0]), translate(outargs, *args[1]));
		 }},
		{mdl::Computegraph::Operation::MATMULT, [](Arguments& outargs, const mdl::Computegraph::Node& value) {
			 const auto& args = value.inputs();
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
	auto it = operations.find(node.operation);
	assert(it != operations.end());
	return it->second(args, node);
}

static tvm::BaseFunc translateFunc(const FuncDesc& func) {
	Arguments args;
	auto retnode = translate(args, *func.graph->getOutputNode());
	args.emplace_back(retnode);
	std::cout << args.size() << std::endl;

	// https://github.com/apache/tvm/blob/c8515e1ddfaf4d1afff916c484e68e1513631dd6/python/tvm/tir/build.py#L97C57-L99
	// set tvm::attr::kCallingConv to CallingConv::kDeviceKernelLaunch
	auto prim = tvm::WithAttrs(
			tvm::tir::CreatePrimFunc(args, {}),
			{{tvm::attr::kGlobalSymbol, tvm::ffi::String{func.name}},
			 {tvm::attr::kCallingConv, /*tvm::CallingConv::kDeviceKernelLaunch*/ tvm::CallingConv::kDefault}}
	);
	return prim;
}

void mdl::compileToFile(const Module& module, const std::filesystem::path& output) {
	// Problems: tvm::tir::transform::MakePackedAPI() in the pipeline makes the callingconv packed; CUDA build does not like this
	tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> funcs;
	for (auto func : module.functions) {
		funcs.Set(tvm::GlobalVar{{func.name}}, translateFunc(func));
	}

	// https://github.com/apache/tvm/blob/6dc71f3dc4a54571e3dc0165630cd9a1a0bb97b5/python/tvm/tir/pipeline.py#L33-L123
	auto pass = tvm::transform::Sequential(
			{tvm::transform::PrintIR(),
			 tvm::tir::transform::LowerCrossThreadReduction(),
			 tvm::tir::transform::LowerInitBlock(),
			 tvm::tir::transform::PlanAndUpdateBufferAllocationLocation(),
			 tvm::tir::transform::ConvertBlocksToOpaque(),
			 tvm::tir::transform::LiftThreadBinding(),
			 tvm::tir::transform::ManifestSharedMemoryLocalStage(),
			 tvm::tir::transform::CompactBufferAllocation(),
			 tvm::tir::transform::LowerAutoCopy(),
			 tvm::tir::transform::UnifyThreadBinding(),
			 tvm::tir::transform::LowerMatchBuffer(),
			 tvm::tir::transform::Simplify(),
			 tvm::tir::transform::InjectPermutedLayout(),
			 //tvm::tir::transform::AnnotateIrregularLoop(),
			 tvm::tir::transform::InjectSoftwarePipeline(),
			 tvm::tir::transform::TransformMmaBufferLayout(),
			 tvm::tir::transform::LowerOpaqueBlock(),
			 tvm::tir::transform::FlattenBuffer(),
			 tvm::tir::transform::BF16ComputeLegalize(),
			 tvm::tir::transform::NarrowDataType(32),
			 tvm::tir::transform::LoopPartition(),
			 tvm::tir::transform::VectorizeLoop(true),
			 tvm::tir::transform::InjectVirtualThread(),
			 tvm::tir::transform::InjectDoubleBuffer(),
			 tvm::tir::transform::StorageRewrite(),
			 tvm::tir::transform::LowerAsyncDMA(),
			 tvm::tir::transform::HoistIfThenElse(),
			 tvm::tir::transform::UnrollLoop(),
			 tvm::tir::transform::RenormalizeSplitPattern(),
			 tvm::tir::transform::Simplify(),
			 tvm::tir::transform::RemoveNoOp(),
			 tvm::tir::transform::RewriteUnsafeSelect(),
			 tvm::tir::transform::InstrumentBoundCheckers(),
			 tvm::tir::transform::InjectPTXLDG32(),
			 tvm::tir::transform::CommonSubexprElimTIR(),
			 tvm::tir::transform::InstrumentProfileIntrinsics(),
			 tvm::tir::transform::FP8ComputeLegalize(), // tvm::tir::transform::VerifyVTCMLimit(),
			 tvm::tir::transform::LowerVtcmAlloc(),		// tvm::tir::transform::VerifyMemory(),
			 tvm::tir::transform::AnnotateEntryFunc(),
			 tvm::tir::transform::AnnotateDeviceRegions(),
			 tvm::tir::transform::SplitHostDevice(),
			 tvm::tir::transform::MergeSharedMemoryAllocations(),
			 tvm::tir::transform::MakePackedAPI(),
			 tvm::tir::transform::FP8StorageLegalize(),
			 tvm::tir::transform::BF16StorageLegalize(),
			 tvm::tir::transform::LowerDeviceKernelLaunch(),
			 // tvm::tir::transform::SkipAssert(),
			 /*tvm::transform::PrintIR()*/}
	);

	tvm::Target target{"llvm"};
	// tvm::Target target{"cuda"};
	std::cout << target << std::endl;
	target = tvm::Target::WithHost(target, tvm::Target{"llvm"});
	std::cout << target << std::endl;

	auto tvmmodule = pass(tvm::tir::transform::BindTarget(target)(tvm::IRModule{funcs}));

	// std::cout << "Target Kinds:" << std::endl;
	// for (auto&& name : tvm::TargetKindRegEntry::ListTargetKinds())
	// 	std::cout << "  - " << name /*<< (tvm::runtime::RuntimeEnabled(name) ? "enabled" : "disabled")*/
	// 			  << std::endl;

	/*std::cout << "Target Tags:" << std::endl;
	for (auto&& [name, target] : tvm::TargetTag::ListTags()) {
		std::cout << "  - " << name << std::endl;
	}*/

	auto mod = tvm::codegen::Build(tvmmodule, target);
	std::cout << "######## " << mod->GetTypeKey() << std::endl;
	for (auto& format : mod->GetWriteFormats())
		std::cout << " - " << format.c_str() << std::endl;
	mod->WriteToFile(output.c_str(), "");

	auto logger = mdl::log::getLogger("compiler");
	logger->info("Done");
}