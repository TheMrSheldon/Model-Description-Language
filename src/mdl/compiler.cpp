#include <mdl/compiler.hpp>

#include <mdl/computegraph.hpp>

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

#include <iostream>

#include "logging.hpp"


namespace tvm {
	namespace tir {
		PrimFunc CreatePrimFunc(
				const ffi::Array<te::Tensor>& arg_list, std::optional<DataType> index_dtype_override = std::nullopt
		);
	}
} // namespace tvm


void mdl::compileToFile(const mdl::Computegraph& graph, const std::filesystem::path& output) {
    auto a = tvm::te::placeholder({128, 128}, tvm::runtime::DataType::Float(32), "A");
	auto b = tvm::te::placeholder({128, 128}, tvm::runtime::DataType::Float(32), "B");
	std::cout << a->shape << std::endl;
	std::cout << a << std::endl;
	auto k = tvm::te::reduce_axis({0, 128}, "k");
	auto y = tvm::te::compute({128, 128}, [&](auto i, auto j) { return tvm::sum(a(i, k) * b(k, j), {k}); }, "Y");
	auto c = tvm::te::compute({128, 128}, [&](auto i, auto j) { return tvm::max(y(i, j), 0); }, "C");

	// https://github.com/apache/tvm/blob/c8515e1ddfaf4d1afff916c484e68e1513631dd6/python/tvm/tir/build.py#L97C57-L99
	// set tvm::attr::kCallingConv to CallingConv::kDeviceKernelLaunch
	auto prim = tvm::WithAttrs(
			tvm::tir::CreatePrimFunc({a, b, c}, {}),
			{{tvm::attr::kGlobalSymbol, tvm::ffi::String{"test"}},
			 {tvm::attr::kCallingConv, /*tvm::CallingConv::kDeviceKernelLaunch*/ tvm::CallingConv::kDefault}}
	);
	// Problems: tvm::tir::transform::MakePackedAPI() in the pipeline makes the callingconv packed; CUDA build does not like this
	tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> funcs{{tvm::GlobalVar{{"test"}}, prim}};

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

	auto module = pass(tvm::tir::transform::BindTarget(target)(tvm::IRModule{funcs}));

	// std::cout << "Target Kinds:" << std::endl;
	// for (auto&& name : tvm::TargetKindRegEntry::ListTargetKinds())
	// 	std::cout << "  - " << name /*<< (tvm::runtime::RuntimeEnabled(name) ? "enabled" : "disabled")*/
	// 			  << std::endl;

	/*std::cout << "Target Tags:" << std::endl;
	for (auto&& [name, target] : tvm::TargetTag::ListTags()) {
		std::cout << "  - " << name << std::endl;
	}*/

	auto mod = tvm::codegen::Build(module, target);
	mod->WriteToFile(output.c_str(), "");
	auto logger = mdl::log::getLogger("compiler");
	logger->info("Done");
}