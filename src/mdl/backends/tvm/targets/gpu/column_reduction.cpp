#include "./column_reduction.hpp"

#include <tvm_ported/s_tir/dlight/analysis/common_analysis.hpp>
#include <tvm_ported/s_tir/dlight/base/common_schedules.hpp>
#include <tvm_ported/s_tir/dlight/base/utils.hpp>

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

		Array<::tvm::ffi::Optional<PrimExpr>> factors(int64_t inner) {
			return {::tvm::ffi::Optional<PrimExpr>(), ::tvm::IntImm::Int32(inner)};
		}

		std::optional<int64_t> product(const std::vector<PrimExpr>& doms) {
			int64_t p = 1;
			for (const auto& d : doms) {
				const auto* imm = d.as<::tvm::IntImmNode>();
				if (imm == nullptr)
					return std::nullopt;
				p *= static_cast<int64_t>(imm->value);
			}
			return p;
		}
	} // namespace

	std::optional<::tvm::s_tir::Schedule>
	ColumnReduction::apply(const ::tvm::tirx::PrimFunc& func, const ::tvm::Target& target, bool /*tunable*/) const {
		if (!is_target_available(target))
			return std::nullopt;
		auto sch = dl::schedule_of(func);
		auto infos = dl::try_inline_contiguous_spatial(sch, dl::analysis::normalize_prim_func(sch));
		if (!infos || infos->empty())
			return std::nullopt;
		// Reductions over the same columns (dom kind S..SR..R, the same domains), and elementwise epilogues per column.
		std::vector<SBlockRV> reductions, epilogues;
		const dl::analysis::SBlockInfo* first = nullptr;
		::tvm::sym::Analyzer analyzer;
		for (const auto& info : *infos) {
			if (!info.is_reduction()) {
				if (!info.is_injective())
					return std::nullopt;
				epilogues.push_back(info.block_rv);
				continue;
			}
			if (first == nullptr)
				first = &info;
			else if (info.dom_kind() != first->dom_kind())
				return std::nullopt;
			for (size_t i = 0; i < info.iters.size(); ++i)
				if (!analyzer->CanProveEqual(info.iters[i].dom, first->iters[i].dom))
					return std::nullopt;
			reductions.push_back(info.block_rv);
		}
		if (first == nullptr || reductions.size() > kMaxReductions)
			return std::nullopt;
		const std::string kind = first->dom_kind();
		const size_t numS = kind.find_first_not_of('S');
		if (numS == std::string::npos || numS == 0 || kind.find_first_not_of('R', numS) != std::string::npos)
			return std::nullopt;
		for (const auto& e : epilogues)
			if (sch->GetLoops(e).size() > numS)
				return std::nullopt; // (only per column)
		std::vector<PrimExpr> rowDoms;
		for (size_t i = numS; i < kind.size(); ++i)
			rowDoms.push_back(first->iters[i].dom);
		if (const auto rows = product(rowDoms); rows && *rows < kMinRows)
			return std::nullopt;
		std::vector<PrimExpr> columnDoms;
		for (size_t i = 0; i < numS; ++i)
			columnDoms.push_back(first->iters[i].dom);
		const auto columns = product(columnDoms);

		// Columns: every read that depends on a reduction has a spatial (not reduced) innermost index.
		for (const auto& red : reductions) {
			const auto block = sch->Get(red);
			std::unordered_set<const void*> reductionVars;
			for (const auto& iv : block->iter_vars)
				if (iv->iter_type == ::tvm::tirx::kCommReduce)
					reductionVars.insert(iv->var.get());
			bool reads = false;
			for (const auto& read : block->reads) {
				if (read->region.empty())
					continue;
				bool dependsOnReduction = false;
				for (const auto& range : read->region)
					for (const auto& v : ::tvm::tirx::UndefinedVars(range->min))
						dependsOnReduction |= reductionVars.contains(v.get());
				if (!dependsOnReduction)
					continue;
				reads = true;
				for (const auto& v : ::tvm::tirx::UndefinedVars(read->region.back()->min))
					if (reductionVars.contains(v.get()))
						return std::nullopt; // (a row reduction: RowReduction, DLight)
			}
			if (!reads)
				return std::nullopt;
		}

		// The threads of the second pass: a divisor of the (static) number of columns.
		int64_t threads = kThreads;
		if (columns) {
			while (threads > kWarp && *columns % threads != 0)
				threads /= 2;
			if (*columns % threads != 0)
				threads = *columns <= kThreads ? *columns : 0;
			if (threads == 0)
				return std::nullopt; // (DLight's rules)
		}
		// kVector adjacent columns per thread in the first pass (vector loads) if the columns divide evenly.
		const int64_t vector = columns && *columns % (kWarp * kVector) == 0 ? kVector : 1;
		try {
			// Each reduction: the rows in chunks of kChunk, partial sums [chunks, columns] (rfactor), the loops of both
			// passes split alike: the partial sums [columns / (warp · vector), chunks / per block, per block, warp, row in
			// chunk, vector], the write-back [columns / threads, threads].
			// The epilogues of each reduction (those that read its result; one reduction each).
			std::vector<std::vector<SBlockRV>> epiloguesOf(reductions.size());
			for (const auto& e : epilogues) {
				std::optional<size_t> of;
				for (size_t k = 0; k < reductions.size(); ++k)
					for (const auto& r : sch->Get(e)->reads)
						for (const auto& w : sch->Get(reductions[k])->writes) // (several: merged sibling sums)
							if (r->source.get() == w->source.get()) {
								if (of && *of != k)
									return std::nullopt;
								of = k;
							}
				if (!of)
					return std::nullopt;
				epiloguesOf[*of].push_back(e);
			}
			std::vector<std::vector<LoopRV>> partial, writeBack;
			std::vector<SBlockRV> partialSums;
			for (size_t k = 0; k < reductions.size(); ++k) {
				const SBlockRV red = reductions[k];
				auto loops = sch->GetLoops(red);
				const auto split = static_cast<std::ptrdiff_t>(numS);
				const LoopRV s = split == 1 ? loops[0] : sch->Fuse(Array<LoopRV>(loops.begin(), loops.begin() + split));
				const LoopRV r = loops.size() - numS == 1
										 ? loops.back()
										 : sch->Fuse(Array<LoopRV>(loops.begin() + split, loops.end()));
				const auto chunks = sch->Split(r, factors(kChunk));
				const SBlockRV rf = sch->RFactor(chunks[0], 0);
				partialSums.push_back(rf);
				auto l = sch->GetLoops(rf); // [s, chunk, row]
				if (l.size() != 3)
					return std::nullopt;
				const auto cols = sch->Split(
						l[0],
						{::tvm::ffi::Optional<PrimExpr>(), ::tvm::IntImm::Int32(kWarp), ::tvm::IntImm::Int32(vector)}
				);
				const auto chs = sch->Split(l[1], factors(kChunksPerBlock));
				sch->Reorder({cols[0], chs[0], chs[1], cols[1], l[2], cols[2]});
				partial.push_back({cols[0], chs[0], chs[1], cols[1], l[2], cols[2]});
				// The write-back: a thread per column sums the chunks, in a register of the reduction's type (e.g.
				// float32; local bfloat16 buffers are computed in float32), which its epilogues then read per column.
				// (threads that divide the columns: the write-back after the loop is not predicated)
				const auto back = sch->Split(sch->GetLoops(red)[0], factors(threads));
				// (the chunks unrolled by kUnroll: their loads in flight together, not one after the other)
				sch->Unroll(sch->Split(sch->GetLoops(red).back(), factors(kUnroll))[1]);
				const int results = static_cast<int>(sch->Get(red)->writes.size());
				if (epiloguesOf[k].empty()) {
					// (the caches of all results first: each is created next to the block, then moved)
					std::vector<SBlockRV> caches;
					for (int i = 0; i < results; ++i)
						caches.push_back(sch->CacheWrite(red, i, "local"));
					for (const auto& cache : caches)
						sch->ReverseComputeAt(cache, back[1], /*preserve_unit_loops=*/false);
				} else {
					for (int i = 0; i < results; ++i)
						sch->SetScope(red, i, "local");
					for (const auto& e : epiloguesOf[k])
						sch->ReverseComputeAt(e, back[1], /*preserve_unit_loops=*/false);
				}
				writeBack.push_back({back[0], back[1]});
			}
			// One kernel per pass: the reductions' loops merged level by level (they are independent).
			auto merged = [&](std::vector<std::vector<LoopRV>>& nests, size_t level) {
				if (nests.size() == 1)
					return nests[0][level];
				Array<LoopRV> loops;
				for (const auto& n : nests)
					loops.push_back(n[level]);
				return sch->Merge(loops);
			};
			// The partial sums: threadIdx.x on adjacent columns, threadIdx.y on chunks; each thread sums its chunk.
			std::vector<LoopRV> p;
			for (size_t level = 0; level < partial[0].size(); ++level)
				p.push_back(merged(partial, level));
			sch->Bind(p[0], "blockIdx.x");
			sch->Bind(p[1], "blockIdx.y");
			sch->Bind(p[2], "threadIdx.y");
			sch->Bind(p[3], "threadIdx.x");
			if (vector > 1)
				sch->Vectorize(p[5]);
			// (each thread's partial sums in registers, stored once after its chunk)
			for (const auto& rf : partialSums) {
				std::vector<SBlockRV> caches; // (all first: each is created next to the block, then moved)
				for (int i = 0; i < static_cast<int>(sch->Get(rf)->writes.size()); ++i)
					caches.push_back(sch->CacheWrite(rf, i, "local"));
				for (const auto& cache : caches)
					sch->ReverseComputeAt(cache, p[3], /*preserve_unit_loops=*/false);
			}
			sch->Bind(merged(writeBack, 0), "blockIdx.x");
			sch->Bind(merged(writeBack, 1), "threadIdx.x");
		} catch (const std::exception&) {
			return std::nullopt; // (DLight's rules schedule it)
		}
		return sch;
	}

} // namespace mdl::backend::tvm::gpu
