#ifndef MDL_BACKENDS_TVM_UTILS_HPP
#define MDL_BACKENDS_TVM_UTILS_HPP

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace {

// Marks the outermost serial for-loop of each compute stage as parallel when
// its extent is either symbolic (unknown at compile time) or >= threshold.
// Inner loops are left sequential so VectorizeLoop can still SIMD-ify them.
// Relies on TVM's thread-pool runtime (tvm_parallel_launch), which is provided
// by the TVM Python package already loaded when the .so is used.
struct OuterParallelMutator : ::tvm::tir::StmtMutator {
	int threshold;
	explicit OuterParallelMutator(int t = 32) : threshold(t) {}

	::tvm::tir::Stmt VisitStmt_(const ::tvm::tir::ForNode* op) final {
		if (op->kind == ::tvm::tir::ForKind::kSerial) {
			const auto* imm = op->extent.as<::tvm::IntImmNode>();
			if (!imm || imm->value >= threshold) {
				return ::tvm::tir::For(
					op->loop_var, op->min, op->extent,
					::tvm::tir::ForKind::kParallel,
					op->body,  // leave body unchanged; inner loops stay serial for VectorizeLoop
					op->thread_binding, op->annotations
				);
			}
		}
		return StmtMutator::VisitStmt_(op);
	}
};

} // namespace

inline ::tvm::transform::Pass ParallelizeOuterLoops(int threshold = 32) {
	return ::tvm::tir::transform::CreatePrimFuncPass(
		[threshold](::tvm::tir::PrimFunc f, ::tvm::IRModule, ::tvm::transform::PassContext) {
			auto* n = f.CopyOnWrite();
			n->body = OuterParallelMutator{threshold}(std::move(n->body));
			return f;
		},
		0, "mdl.ParallelizeOuterLoops", {}
	);
}

inline tvm::transform::Sequential getDefaultPipeline() {
	// https://github.com/apache/tvm/blob/6dc71f3dc4a54571e3dc0165630cd9a1a0bb97b5/python/tvm/tir/pipeline.py#L29-L126
	auto passCtx = tvm::transform::PassContext::Current();
	auto getEntry = [passCtx](tvm::ffi::String name, auto def) {
		return passCtx->config.Get(name).and_then(
												[](tvm::ffi::Any any) { return any.as<decltype(def)>(); }
		).value_or(def);
	};

	tvm::ffi::Array<tvm::transform::Pass> passes = {
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
			ParallelizeOuterLoops(),
			tvm::tir::transform::VectorizeLoop(true),
			tvm::tir::transform::InjectVirtualThread(),
			tvm::tir::transform::InjectDoubleBuffer(),
	};
	if (!getEntry("tir.disable_storage_rewrite", false))
		passes.emplace_back(tvm::tir::transform::StorageRewrite());
	if (getEntry("tir.use_async_copy", false))
		passes.emplace_back(tvm::tir::transform::LowerAsyncDMA());
	passes.emplace_back(tvm::tir::transform::HoistIfThenElse());
	passes.emplace_back(tvm::tir::transform::UnrollLoop());
	passes.emplace_back(tvm::tir::transform::RenormalizeSplitPattern());
	passes.emplace_back(tvm::tir::transform::Simplify());
	passes.emplace_back(tvm::tir::transform::RemoveNoOp());
	passes.emplace_back(tvm::tir::transform::RewriteUnsafeSelect());
	if (getEntry("tir.instrument_bound_checkers", false))
		passes.emplace_back(tvm::tir::transform::InstrumentBoundCheckers());
	if (getEntry("tir.ptx_ldg32", false))
		passes.emplace_back(tvm::tir::transform::InjectPTXLDG32(true));
	passes.emplace_back(
			tvm::tir::transform::CommonSubexprElimTIR(
					!getEntry("tir.disable_cse_tir", false), getEntry("tir.enable_equiv_terms_in_cse_tir", false)
			)
	);
	if (getEntry("tir.instrument_lwp", false))
		passes.emplace_back(tvm::tir::transform::InstrumentProfileIntrinsics());

	passes.emplace_back(tvm::tir::transform::FP8ComputeLegalize());
	//passes.emplace_back(tvm::tir::transform::VerifyVTCMLimit());
	passes.emplace_back(tvm::tir::transform::LowerVtcmAlloc());
	//passes.emplace_back(tvm::tir::transform::VerifyMemory());
	passes.emplace_back(tvm::tir::transform::AnnotateEntryFunc());

	if (getEntry("tir.detect_global_barrier", false))
		passes.emplace_back(tvm::tir::transform::ThreadSync("global"));

	passes.emplace_back(tvm::tir::transform::ThreadSync("shared"));
	passes.emplace_back(tvm::tir::transform::ThreadSync("shared.dyn"));
	passes.emplace_back(tvm::tir::transform::ThreadSync("warp"));
	passes.emplace_back(tvm::tir::transform::InferFragment());
	passes.emplace_back(tvm::tir::transform::LowerThreadAllreduce());

	if (getEntry("tir.use_async_copy", false))
		passes.emplace_back(tvm::tir::transform::InjectPTXAsyncCopy());
	if (getEntry("tir.ptx_ldg32", false))
		passes.emplace_back(tvm::tir::transform::InjectPTXLDG32());

	passes.emplace_back(tvm::tir::transform::AnnotateDeviceRegions());
	passes.emplace_back(tvm::tir::transform::SplitHostDevice());
	passes.emplace_back(tvm::tir::transform::MergeSharedMemoryAllocations());
	passes.emplace_back(tvm::tir::transform::MakePackedAPI());
	passes.emplace_back(tvm::tir::transform::FP8StorageLegalize());
	passes.emplace_back(tvm::tir::transform::BF16StorageLegalize());
	passes.emplace_back(tvm::tir::transform::LowerDeviceKernelLaunch());
	return {passes};
}

inline tvm::transform::Sequential finalizeHostPasses() {
	// https://github.com/apache/tvm/blob/main/python/tvm/tir/pipeline.py#L130C1-L139C55
	return tvm::tir::transform::Sequential{{
			tvm::tir::transform::LowerTVMBuiltin(),
			tvm::tir::transform::LowerCustomDatatypes(),
			tvm::tir::transform::LowerIntrin(),
			tvm::tir::transform::LowerDeviceStorageAccessInfo(),
			tvm::tir::transform::CombineContextCall(),
	}};
}

inline tvm::transform::Sequential finalizeDevicePasses() {
	// https://github.com/apache/tvm/blob/main/python/tvm/tir/pipeline.py#L142C5-L151
	// LowerTVMBuiltin is needed here too: for CPU (LLVM) the device module contains
	// the actual compute kernels, and dynamic allocations must be converted to
	// workspace alloc calls before reaching LLVM codegen.
	return tvm::tir::transform::Sequential{{
			tvm::tir::transform::LowerWarpMemory(),
			tvm::tir::transform::Simplify(),
			tvm::tir::transform::LowerTVMBuiltin(),
			tvm::tir::transform::LowerCustomDatatypes(),
			tvm::tir::transform::LowerDeviceStorageAccessInfo(),
			tvm::tir::transform::LowerIntrin(),
	}};
}

#endif