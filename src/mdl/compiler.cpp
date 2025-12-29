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

#include <cassert>
#include <map>
#include <optional>
#include <functional>
#include <iostream>

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

static const std::map<mdl::Computegraph::Operation, TranslateFn> operations = {
	{mdl::Computegraph::Operation::Argument, [](Arguments& outargs, const mdl::Computegraph::Node& value) {
		/** \todo implement other types **/
		/** \todo handle name better **/
		auto arg = tvm::te::placeholder({128, 128}, tvm::runtime::DataType::Float(32), std::to_string('A'+outargs.size()));
		outargs.emplace_back(arg);
		return arg;
	}},
	{mdl::Computegraph::Operation::Input, [](Arguments& outargs, const mdl::Computegraph::Node& value) {
		/** \todo implement other types **/
		/** \todo handle name better **/
		auto arg = tvm::te::placeholder({128, 128}, tvm::runtime::DataType::Float(32), std::to_string('A'+outargs.size()));
		outargs.emplace_back(arg);
		return arg;
	}},
	{mdl::Computegraph::Operation::ADD,  [](Arguments& outargs, const mdl::Computegraph::Node& value) {
		const auto& args = value.inputs();
		assert(args.size() == 2);
		return tvm::topi::add(translate(outargs, *args[0]), translate(outargs, *args[1]));
	}},
	{mdl::Computegraph::Operation::MATMULT,  [](Arguments& outargs, const mdl::Computegraph::Node& value) {
		const auto& args = value.inputs();
		assert(args.size() == 2);
		auto left = translate(outargs, *args[0]).as<tvm::te::Tensor>().value();
		auto right = translate(outargs, *args[1]).as<tvm::te::Tensor>().value();
		auto k = tvm::te::reduce_axis({0, 128}, "k");
		return tvm::te::compute({128, 128}, [&](auto i, auto j) { return tvm::sum(left(i, k) * right(k, j), {k}); });
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
			{tvm::transform::PrintIR(), tvm::tir::transform::LowerCrossThreadReduction(),
			 tvm::tir::transform::LowerInitBlock(), tvm::tir::transform::PlanAndUpdateBufferAllocationLocation(),
			 tvm::tir::transform::ConvertBlocksToOpaque(), tvm::tir::transform::LiftThreadBinding(),
			 tvm::tir::transform::ManifestSharedMemoryLocalStage(), tvm::tir::transform::CompactBufferAllocation(),
			 tvm::tir::transform::LowerAutoCopy(), tvm::tir::transform::UnifyThreadBinding(),
			 tvm::tir::transform::LowerMatchBuffer(), tvm::tir::transform::Simplify(),
			 tvm::tir::transform::InjectPermutedLayout(),
			 //tvm::tir::transform::AnnotateIrregularLoop(),
			 tvm::tir::transform::InjectSoftwarePipeline(), tvm::tir::transform::TransformMmaBufferLayout(),
			 tvm::tir::transform::LowerOpaqueBlock(), tvm::tir::transform::FlattenBuffer(),
			 tvm::tir::transform::BF16ComputeLegalize(), tvm::tir::transform::NarrowDataType(32),
			 tvm::tir::transform::LoopPartition(), tvm::tir::transform::VectorizeLoop(true),
			 tvm::tir::transform::InjectVirtualThread(), tvm::tir::transform::InjectDoubleBuffer(),
			 tvm::tir::transform::StorageRewrite(), tvm::tir::transform::LowerAsyncDMA(),
			 tvm::tir::transform::HoistIfThenElse(), tvm::tir::transform::UnrollLoop(),
			 tvm::tir::transform::RenormalizeSplitPattern(), tvm::tir::transform::Simplify(),
			 tvm::tir::transform::RemoveNoOp(), tvm::tir::transform::RewriteUnsafeSelect(),
			 tvm::tir::transform::InstrumentBoundCheckers(), tvm::tir::transform::InjectPTXLDG32(),
			 tvm::tir::transform::CommonSubexprElimTIR(), tvm::tir::transform::InstrumentProfileIntrinsics(),
			 tvm::tir::transform::FP8ComputeLegalize(), // tvm::tir::transform::VerifyVTCMLimit(),
			 tvm::tir::transform::LowerVtcmAlloc(),		// tvm::tir::transform::VerifyMemory(),
			 tvm::tir::transform::AnnotateEntryFunc(), tvm::tir::transform::AnnotateDeviceRegions(),
			 tvm::tir::transform::SplitHostDevice(), tvm::tir::transform::MergeSharedMemoryAllocations(),
			 tvm::tir::transform::MakePackedAPI(), tvm::tir::transform::FP8StorageLegalize(),
			 tvm::tir::transform::BF16StorageLegalize(), tvm::tir::transform::LowerDeviceKernelLaunch(),
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