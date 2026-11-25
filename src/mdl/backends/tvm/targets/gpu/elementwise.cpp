#include "./elementwise.hpp"

#include <tvm_ported/s_tir/dlight/analysis/common_analysis.hpp>
#include <tvm_ported/s_tir/dlight/base/common_schedules.hpp>
#include <tvm_ported/s_tir/dlight/base/utils.hpp>

#include <tvm/sym/analyzer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>

#include <algorithm>
#include <exception>

namespace mdl::backend::tvm::gpu {

	std::optional<::tvm::s_tir::Schedule>
	Elementwise::apply(const ::tvm::tirx::PrimFunc& func, const ::tvm::Target& target, bool /*tunable*/) const {
		namespace dl = mdl::tvm_ported::dlight;
		using ::tvm::PrimExpr;
		if (!is_target_available(target))
			return std::nullopt;
		auto sch = dl::schedule_of(func);
		auto infos = dl::analysis::normalize_prim_func(sch);
		if (!infos || infos->size() != 1)
			return std::nullopt;
		const auto& info = (*infos)[0];
		const std::string kind = info.dom_kind();
		if (kind.empty() || kind.find_first_not_of('S') != std::string::npos)
			return std::nullopt;
		const auto block = sch->Get(info.block_rv);
		const auto* last = block->iter_vars.back()->var.get();
		// Every buffer's innermost index is the innermost iterator (so the lanes of a vector are adjacent elements of
		// every buffer); the widest element type decides the vector width.
		int64_t bytes = 0;
		auto innermost = [&](const auto& regions) {
			for (const auto& r : regions) {
				if (r->region.empty() || r->region.back()->min.get() != last)
					return false;
				const auto element = r->source.template as_or_throw<::tvm::tirx::BufferVar>().ElementType();
				bytes = std::max<int64_t>(bytes, (element.bits() * element.lanes() + 7) / 8);
			}
			return true;
		};
		if (block->reads.empty() || !innermost(block->reads) || !innermost(block->writes) || bytes == 0)
			return std::nullopt;
		const int64_t vector = std::max<int64_t>(kBytes / bytes, 1);
		const PrimExpr extent = info.iters.back().dom; // (the innermost dimension: the lanes stay within it)
		::tvm::sym::Analyzer analyzer;
		if (vector == 1 ||
			!analyzer->CanProve(::tvm::floormod(extent, ::tvm::IntImm(extent.ty(), vector)) == ::tvm::IntImm(extent.ty(), 0)))
			return std::nullopt;
		try {
			auto loops = sch->GetLoops(info.block_rv);
			if (loops.size() != kind.size())
				return std::nullopt;
			const auto all = loops.size() == 1 ? loops[0] : sch->Fuse(loops);
			// Exactly into vectors (no predicate on the lanes), then the vectors over the threads.
			const auto lanes = sch->Split(all, {::tvm::ffi::Optional<PrimExpr>(), ::tvm::IntImm::Int32(vector)});
			const auto threads = sch->Split(lanes[0], {::tvm::ffi::Optional<PrimExpr>(), ::tvm::IntImm::Int32(kThreads)});
			sch->Bind(threads[0], "blockIdx.x");
			sch->Bind(threads[1], "threadIdx.x");
			sch->Vectorize(lanes[1]);
		} catch (const std::exception&) {
			return std::nullopt;
		}
		return sch;
	}

} // namespace mdl::backend::tvm::gpu
