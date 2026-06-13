#ifndef MDL_BACKENDS_TVM_UTILS_HPP
#define MDL_BACKENDS_TVM_UTILS_HPP

#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/var.h>
#include <mdl/logging.hpp>

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

// Collects name_hint and iter_var info for every Block in a PrimFunc body.
// Used to identify pure-elementwise blocks that are candidates for ComputeInline.
namespace {
struct ElemwiseBlockCollector : ::tvm::tir::StmtVisitor {
	// name → number of times it appears as a pure-elementwise (non-reduction) block
	std::map<std::string, int> counts;

	void VisitStmt_(const ::tvm::tir::BlockNode* op) final {
		if (!op->name_hint.empty()) {
			bool isReduction = false;
			for (const auto& iv : op->iter_vars) {
				if (iv->iter_type == ::tvm::tir::kCommReduce) {
					isReduction = true;
					break;
				}
			}
			if (!isReduction)
				counts[std::string(op->name_hint)]++;
		}
		StmtVisitor::VisitStmt_(op);
	}
};
} // namespace

// https://tvm.apache.org/docs/reference/api/python/tir.html#tvm.tir.Schedule.compute_inline
// Equivalent to iterating sch.compute_inline(block) over all pure-elementwise
// intermediate blocks in Python:  sch.compute_inline(sch.get_block("name"))
//
// Fuses elementwise chains (e.g. GELU's 5 ops → 1 loop) so that
// intermediate buffers are never materialised and vectorisation covers the full
// expression.  Only uniquely-named blocks are inlined; duplicates would make
// GetBlock throw.  Failures (output block, ExternOp consumers) are silently skipped.
//
// Uses a SINGLE Schedule for all blocks.  TVM's ComputeInline throws ScheduleError
// (not InternalError) for invalid targets (output blocks, ExternOp consumers),
// which leaves the Schedule in a consistent state — subsequent blocks can still be
// processed.  Using one schedule avoids quadratic re-analysis overhead: the forward
// function went from 66 s to ~1 s compile time with this change.
//
// Returns {modified PrimFunc, nInlined, nDupName, nFailed}.
inline std::tuple<::tvm::tir::PrimFunc, int, int, int> computeInlineElemwise(
	::tvm::tir::PrimFunc pf, const std::string& funcName)
{
	ElemwiseBlockCollector collector;
	collector(pf->body); // StmtVisitor public interface is operator()

	std::vector<std::string> targets;
	int nDup = 0;
	for (const auto& [name, cnt] : collector.counts) {
		if (cnt == 1) targets.push_back(name);
		else ++nDup;
	}
	if (targets.empty()) return {pf, 0, nDup, 0};

	::tvm::ffi::Map<::tvm::GlobalVar, ::tvm::BaseFunc> tmp;
	tmp.Set(::tvm::GlobalVar{funcName}, pf);
	// https://tvm.apache.org/docs/reference/api/python/tir.html#tvm.tir.Schedule
	auto sch = ::tvm::tir::Schedule::Concrete(
		::tvm::IRModule{tmp}, /*seed=*/0, /*debug_mask=*/0,
		::tvm::tir::ScheduleErrorRenderLevel::kNone
	);
	// https://tvm.apache.org/docs/reference/api/python/tir.html#tvm.tir.Schedule.work_on
	sch->WorkOn(::tvm::ffi::String(funcName));

	auto log = mdl::log::getLogger("compiler");
	int nInlined = 0, nFail = 0;
	for (const auto& name : targets) {
		try {
			// https://tvm.apache.org/docs/reference/api/python/tir.html#tvm.tir.Schedule.get_block
			// https://tvm.apache.org/docs/reference/api/python/tir.html#tvm.tir.Schedule.compute_inline
			sch->ComputeInline(sch->GetBlock(::tvm::ffi::String(name)));
			log->debug("    compute_inline ok: {}", name);
			++nInlined;
		} catch (const std::exception& e) {
			// ScheduleError (output block, ExternOp consumer) — schedule state intact.
			log->debug("    compute_inline skip: {} — {}", name, e.what());
			++nFail;
		}
	}
	if (nInlined == 0) return {pf, 0, nDup, nFail};
	return {
		::tvm::Downcast<::tvm::tir::PrimFunc>(sch->mod()->Lookup(::tvm::ffi::String(funcName))),
		nInlined, nDup, nFail
	};
}

// Collects reduction blocks (blocks with at least one kCommReduce iter var).
// Also detects extern-op blocks (e.g. tvm_contrib_cblas_*) whose presence
// prevents rfactor from working on any block in the same function — TVM's
// BlockReplacer uses raw-pointer identity to find targets in the IR, and
// the IR reconstruction triggered by extern-op bookkeeping invalidates those
// pointers for sibling blocks.
namespace {
struct ReductionBlockCollector : ::tvm::tir::StmtVisitor {
	std::map<std::string, int> counts;
	bool hasExternOp = false;

	void VisitStmt_(const ::tvm::tir::BlockNode* op) final {
		if (!op->name_hint.empty()) {
			// Extern-op blocks have names like "tvm_contrib_cblas_matmul".
			if (std::string(op->name_hint).find("tvm_contrib") != std::string::npos)
				hasExternOp = true;
			for (const auto& iv : op->iter_vars) {
				if (iv->iter_type == ::tvm::tir::kCommReduce) {
					counts[std::string(op->name_hint)]++;
					break;
				}
			}
		}
		StmtVisitor::VisitStmt_(op);
	}
};
} // namespace

// https://tvm.apache.org/docs/reference/api/python/tir.html#tvm.tir.Schedule.rfactor
// https://tvm.apache.org/docs/reference/api/python/tir.html#tvm.tir.Schedule.split
// https://tvm.apache.org/docs/reference/api/python/tir.html#tvm.tir.Schedule.vectorize
//
// For each uniquely-named reduction block (mean/var in LayerNorm, max/sum in Softmax),
// split the innermost reduction loop into (outer, inner=vecWidth) then rfactor on the
// inner split, promoting it from a reduction iter to a spatial iter so it can be
// vectorized.  Equivalent to the MetaSchedule CPU schedule for row-wise reductions:
//
//   k_outer, k_inner = sch.split(loops[-1], [None, vec_width])
//   block_rf = sch.rfactor(k_inner, factor_axis=0)
//   sch.vectorize(sch.get_loops(block_rf)[0])
//
// rfactor creates two new blocks: block_rf (partial reduction over k_inner, now spatial
// and vectorizable) and a merge block (reduces the vec_width partial sums).
// ParallelizeOuterLoops still parallelises the outer batch/seq axis of the merge block.
//
// Uses a single Schedule.  A ScheduleError marks corrupted=true and skips the rest.
// Returns {modified PrimFunc, nScheduled, nFailed}.
inline std::tuple<::tvm::tir::PrimFunc, int, int> scheduleRfactorReductions(
	::tvm::tir::PrimFunc pf, const std::string& funcName, int vecWidth = 8)
{
	ReductionBlockCollector col;
	col(pf->body);

	std::vector<std::string> targets;
	for (const auto& [name, cnt] : col.counts) {
		if (cnt != 1) continue;
		// Skip matmul blocks: rfactoring a large k-reduction (k=768, k=3072) creates
		// a C_rf[k,i,j] workspace that overwhelms LLVM codegen (10+ min compile time).
		// Matmuls are instead scheduled by ParallelizeOuterLoops + VectorizeLoop.
		if (name.find("gemm") != std::string::npos) continue;
		targets.push_back(name);
	}
	if (targets.empty()) return {pf, 0, 0};

	auto log = mdl::log::getLogger("compiler");
	if (col.hasExternOp) {
		log->debug("    rfactor: skipping {} — contains extern ops (BlockReplacer incompatibility)", funcName);
		return {pf, 0, 0};
	}

	// Per-block fresh schedule: Split followed by RFactor causes an InternalError
	// in TVM's BlockReplacer when applied via a shared schedule on large functions.
	// Root cause: Split reconstructs ForNode trees, changing BlockRealizeNode addresses;
	// BlockReplacer then fails its pointer-identity check.  A fresh Schedule per block
	// starts from an unmodified AST each time so the check always passes.
	// After each successful rfactor, `pf` is updated so the next block's fresh schedule
	// reflects the accumulated rfactored state.
	int nSched = 0, nFail = 0;

	for (const auto& name : targets) {
		::tvm::ffi::Map<::tvm::GlobalVar, ::tvm::BaseFunc> tmp;
		tmp.Set(::tvm::GlobalVar{funcName}, pf);
		auto sch = ::tvm::tir::Schedule::Concrete(
			::tvm::IRModule{tmp}, /*seed=*/0, /*debug_mask=*/0,
			::tvm::tir::ScheduleErrorRenderLevel::kNone
		);
		sch->WorkOn(::tvm::ffi::String(funcName));

		try {
			auto block = sch->GetBlock(::tvm::ffi::String(name));
			auto loops = sch->GetLoops(block);
			if (loops.size() < 2) {
				log->debug("    rfactor skip (too few loops): {}", name);
				++nFail; continue;
			}
			// RFactor the FULL reduction loop first (no Split before RFactor).
			// Split-before-RFactor causes TVM's BlockReplacer to fail: Split
			// reconstructs the surrounding SeqStmt, giving the BlockRealizeNode
			// a new address; BlockReplacer then can't match it by raw pointer.
			// By rfactoring the unreduced loop, we avoid any prior IR mutation.
			auto blockRf = sch->RFactor(loops.back(), 0);
			// The rfactor block now has a spatial loop of full extent (e.g. 768).
			// Split that loop and vectorize the inner part.
			auto rfLoops = sch->GetLoops(blockRf);
			if (rfLoops.size() < 2) {
				log->debug("    rfactor skip (rf too few loops): {}", name);
				++nFail; continue;
			}
			// rfLoops.back() is the factored spatial axis (full extent)
			auto rfSplits = sch->Split(rfLoops.back(), {std::nullopt, ::tvm::Integer(vecWidth)});
			sch->Vectorize(rfSplits.back());
			pf = ::tvm::Downcast<::tvm::tir::PrimFunc>(
				sch->mod()->Lookup(::tvm::ffi::String(funcName))
			);
			log->debug("    rfactor ok: {}", name);
			++nSched;
		} catch (const std::exception& e) {
			log->debug("    rfactor fail: {} — {}", name, e.what());
			++nFail;
		}
	}
	return {pf, nSched, nFail};
}

// https://tvm.apache.org/docs/reference/api/python/tir.html#tvm.tir.Schedule.split
// https://tvm.apache.org/docs/reference/api/python/tir.html#tvm.tir.Schedule.vectorize
//
// For each surviving non-reduction block (i.e. blocks that computeInlineElemwise
// left in place — typically the output block after inlining all intermediates),
// split its innermost loop into (outer, inner=vecWidth) and mark the inner loop
// as vectorized so that the TIR VectorizeLoop pass emits SIMD instructions:
//
//   # Python equivalent for the output block of bertGELU:
//   loops = sch.get_loops(output_block)   # [ax0] (batch*seq)
//   ax0_o, ax0_i = sch.split(loops[-1], [None, 8])
//   sch.vectorize(ax0_i)
//
// vecWidth=8 targets AVX2 (8 × float32 per SIMD lane).
//
// All blocks are processed in a SINGLE Schedule so the analysis is done once and
// the modified PrimFunc is only materialised once at the end.  A failed Split or
// Vectorize marks the schedule as corrupted; subsequent blocks are skipped.
// Returns {modified PrimFunc, nVectorized, nFailed}.
inline std::tuple<::tvm::tir::PrimFunc, int, int> scheduleVectorizeBlocks(
	::tvm::tir::PrimFunc pf, const std::string& funcName, int vecWidth = 8)
{
	// Collect surviving non-reduction blocks from the post-inline PrimFunc.
	ElemwiseBlockCollector col;
	col(pf->body);

	std::vector<std::string> targets;
	for (const auto& [name, cnt] : col.counts)
		if (cnt == 1) targets.push_back(name);
	if (targets.empty()) return {pf, 0, 0};

	// Single schedule — creating a new Schedule per block caused kVectorized ForNodes
	// to accumulate in the "current" PrimFunc fed into each new schedule, confusing
	// TVM's analysis and causing 77-second compile times on the forward function.
	::tvm::ffi::Map<::tvm::GlobalVar, ::tvm::BaseFunc> tmp;
	tmp.Set(::tvm::GlobalVar{funcName}, pf);
	auto sch = ::tvm::tir::Schedule::Concrete(
		::tvm::IRModule{tmp}, /*seed=*/0, /*debug_mask=*/0,
		::tvm::tir::ScheduleErrorRenderLevel::kNone
	);
	sch->WorkOn(::tvm::ffi::String(funcName));

	int nVec = 0, nFail = 0;
	bool corrupted = false;

	for (const auto& name : targets) {
		if (corrupted) { ++nFail; continue; }
		try {
			auto block = sch->GetBlock(::tvm::ffi::String(name));
			auto loops = sch->GetLoops(block);
			if (loops.empty()) { ++nFail; continue; }
			auto splits = sch->Split(loops.back(), {std::nullopt, ::tvm::Integer(vecWidth)});
			sch->Vectorize(splits.back());
			++nVec;
		} catch (...) {
			corrupted = true; // schedule state may be invalid from here
			++nFail;
		}
	}
	if (nVec == 0) return {pf, 0, nFail};
	return {
		::tvm::Downcast<::tvm::tir::PrimFunc>(sch->mod()->Lookup(::tvm::ffi::String(funcName))),
		nVec, nFail
	};
}

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
			// LoopPartition omitted: splits loops at bounds boundaries, which dramatically
			// increases IR size for large inlined CPU functions without runtime benefit.
			// tvm::tir::transform::LoopPartition(),
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