#include "./row_reduction.hpp"

#include <tvm_ported/s_tir/dlight/analysis/common_analysis.hpp>
#include <tvm_ported/s_tir/dlight/base/common_schedules.hpp>
#include <tvm_ported/s_tir/dlight/base/utils.hpp>

#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>

#include <exception>
#include <unordered_set>

namespace mdl::backend::tvm::gpu {

	namespace {
		namespace dl = mdl::tvm_ported::dlight;
		using ::tvm::PrimExpr;
		using ::tvm::ffi::Array;
		using ::tvm::s_tir::LoopRV;
		using ::tvm::s_tir::SBlockRV;

		std::optional<int64_t> constant(const PrimExpr& e) {
			if (const auto* imm = e.as<::tvm::IntImmNode>())
				return static_cast<int64_t>(imm->value);
			return std::nullopt;
		}

		/** @brief The product of @p doms if constant. */
		std::optional<int64_t> product(const std::vector<PrimExpr>& doms) {
			int64_t p = 1;
			for (const auto& d : doms) {
				const auto c = constant(d);
				if (!c)
					return std::nullopt;
				p *= *c;
			}
			return p;
		}

		Array<::tvm::ffi::Optional<PrimExpr>> factors(std::optional<int64_t> outer, int64_t inner) {
			return {outer ? ::tvm::ffi::Optional<PrimExpr>(::tvm::IntImm::Int32(*outer)) : ::tvm::ffi::Optional<PrimExpr>(),
					::tvm::IntImm::Int32(inner)};
		}
	} // namespace

	std::optional<::tvm::s_tir::Schedule>
	RowReduction::apply(const ::tvm::tirx::PrimFunc& func, const ::tvm::Target& target, bool /*tunable*/) const {
		if (!is_target_available(target))
			return std::nullopt;
		auto sch = dl::schedule_of(func);
		auto infos = dl::try_inline_contiguous_spatial(sch, dl::analysis::normalize_prim_func(sch));
		if (!infos || infos->empty())
			return std::nullopt;

		// The form: reductions over the same rows (dom kind S..SR..R), then optionally an injective epilogue over the
		// same rows (S..S of the same rank, e.g. the normalization of softmax).
		const std::string kind = (*infos)[0].dom_kind();
		const size_t numLeadingS = kind.find_first_not_of('S');
		if (numLeadingS == std::string::npos || numLeadingS == 0 ||
			kind.find_first_not_of('R', numLeadingS) != std::string::npos)
			return std::nullopt;
		const size_t numTrailingR = kind.size() - numLeadingS;
		const bool epilogue = (*infos).back().is_injective();
		const size_t numReductions = infos->size() - (epilogue ? 1 : 0);
		if (numReductions == 0)
			return std::nullopt;
		::tvm::sym::Analyzer analyzer;
		std::vector<PrimExpr> rowDoms, rowLengthDoms;
		for (size_t i = 0; i < kind.size(); ++i)
			(i < numLeadingS ? rowDoms : rowLengthDoms).push_back((*infos)[0].iters[i].dom);
		for (size_t b = 0; b < infos->size(); ++b) {
			const auto& info = (*infos)[b];
			if (b < numReductions ? info.dom_kind() != kind : info.dom_kind().size() != kind.size())
				return std::nullopt;
			for (size_t i = 0; i < numLeadingS; ++i)
				if (!analyzer->CanProveEqual(info.iters[i].dom, rowDoms[i]))
					return std::nullopt;
		}

		// Only where a warp per row fits: rows of a symbolic length or of at most kMaxRowLength elements, and enough
		// rows to occupy the GPU (DLight's rules handle the others).
		const auto rowLength = product(rowLengthDoms);
		if (rowLength && *rowLength > kMaxRowLength)
			return std::nullopt;
		const auto rows = product(rowDoms);
		if (rows && *rows < kMinRows)
			return std::nullopt;

		// ... and where a row is contiguous in memory (coalesced reads of the warp): the innermost index of every input of
		// the reductions is a reduction iterator (not e.g. the sums over the leading dimensions of a bias gradient).
		for (size_t b = 0; b < numReductions; ++b) {
			const auto block = sch->Get((*infos)[b].block_rv);
			std::unordered_set<const void*> reductionVars;
			for (const auto& iv : block->iter_vars)
				if (iv->iter_type == ::tvm::tirx::kCommReduce)
					reductionVars.insert(iv->var.get());
			for (const auto& read : block->reads) {
				if (read->region.empty())
					continue;
				bool innermostReduced = false;
				for (const auto& v : ::tvm::tirx::UndefinedVars(read->region.back()->min))
					innermostReduced |= reductionVars.contains(v.get());
				bool reads = false; // (a read that does not depend on the reduction, e.g. the maximum in softmax's sum)
				for (const auto& range : read->region)
					for (const auto& v : ::tvm::tirx::UndefinedVars(range->min))
						reads |= reductionVars.contains(v.get());
				if (reads && !innermostReduced)
					return std::nullopt;
			}
		}

		try {
			schedule(sch, *infos, numLeadingS, numTrailingR, numReductions);
		} catch (const std::exception&) {
			return std::nullopt; // (DLight's rules schedule it)
		}
		return sch;
	}

	void RowReduction::schedule(
			const ::tvm::s_tir::Schedule& sch, const std::vector<dl::analysis::SBlockInfo>& infos_, size_t numLeadingS,
			size_t numTrailingR, size_t numReductions
	) {
		const auto* infos = &infos_;
		// The last block: rows over blocks, the row over the warp (threadIdx.x). Several rows per block (threadIdx.y)
		// only for a single reduction: the intermediate results of several (e.g. the maximum and the sum of softmax)
		// are computed per row, whose region compute_at can only determine exactly for one row per block.
		const int64_t rowsPerBlock = infos->size() == 1 ? kRowsPerBlock : 1;
		const SBlockRV last = infos->back().block_rv;
		auto loops = sch->GetLoops(last);
		const LoopRV row = sch->Fuse(Array<LoopRV>(loops.begin(), loops.begin() + static_cast<std::ptrdiff_t>(numLeadingS)));
		const LoopRV col = loops.size() - numLeadingS == 1
								   ? loops.back()
								   : sch->Fuse(Array<LoopRV>(loops.begin() + static_cast<std::ptrdiff_t>(numLeadingS), loops.end()));
		LoopRV bx = row, at = row;
		if (rowsPerBlock > 1) {
			const auto split = sch->Split(row, factors(std::nullopt, rowsPerBlock));
			bx = split[0];
			at = split[1];
			sch->Bind(at, "threadIdx.y");
		}
		const auto split = sch->Split(col, factors(std::nullopt, kWarp));
		const LoopRV colOuter = split[0], tx = split[1];
		sch->Reorder(rowsPerBlock > 1 ? Array<LoopRV>{bx, at, tx, colOuter} : Array<LoopRV>{bx, tx, colOuter});
		sch->Bind(bx, "blockIdx.x");
		sch->Bind(tx, "threadIdx.x");

		// The reductions before it: per row (computed at the row loop), the row over the warp; their results in shared
		// memory for the threads of the row.
		for (size_t b = numReductions; b-- > 0;) {
			if (b == infos->size() - 1)
				continue; // (the last block, if it is a reduction: scheduled above)
			const SBlockRV block = (*infos)[b].block_rv;
			for (size_t i = 0; i < sch->Get(block)->writes.size(); ++i)
				sch->SetScope(block, static_cast<int>(i), "shared");
			sch->ComputeAt(block, at, /*preserve_unit_loops=*/true);
			const auto blockLoops = sch->GetLoops(block);
			const LoopRV r = sch->Fuse(Array<LoopRV>(blockLoops.end() - static_cast<std::ptrdiff_t>(numTrailingR), blockLoops.end()));
			const auto parts = sch->Split(r, factors(std::nullopt, kWarp));
			sch->Reorder({parts[1], parts[0]});
			sch->Bind(parts[1], "threadIdx.x");
		}
	}

} // namespace mdl::backend::tvm::gpu
