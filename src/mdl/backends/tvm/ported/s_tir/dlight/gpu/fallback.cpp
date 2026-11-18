// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/gpu/fallback.py
// (conventions: src/mdl/backends/tvm/ported/README.md)
// A fallback schedule rule for GPU operators.
#include "./gpu.hpp"

#include "../base/common_schedules.hpp"
#include "../base/utils.hpp"

#include <tvm/ffi/extra/structural_visit.h>

namespace mdl::tvm_ported::dlight::gpu {

	namespace {
		bool _has_internal_thread_env(const tirx::Stmt& stmt) {
			bool found = false;
			auto visit_attr = [&](const tirx::AttrStmt& node) -> ffi::Expected<ffi::WalkResult> {
				if (node->attr_key == "thread_extent" || node->attr_key == "virtual_thread")
					found = true;
				return ffi::WalkResult::Advance();
			};
			auto visit_for = [&](const tirx::For& node) -> ffi::Expected<ffi::WalkResult> {
				if (node->kind == tirx::ForKind::kThreadBinding)
					found = true;
				return ffi::WalkResult::Advance();
			};
			ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(stmt, visit_attr, visit_for);
			return found;
		}
	} // namespace

	std::optional<Schedule> Fallback::apply(const PrimFunc& func, const Target& target, bool /*tunable*/) const {
		if (!is_target_available(target))
			return std::nullopt;
		const int64_t max_threads_per_block = dlight::max_threads_per_block(target);

		Schedule sch = schedule_of(func);
		auto block_infos = analysis::normalize_prim_func(sch);

		if (!block_infos)
			return std::nullopt;

		auto remaining = try_inline(sch, *block_infos);
		std::vector<std::pair<SBlockRV, LoopRV>> reduction_blocks;
		for (const auto& info : remaining) {
			Array<LoopRV> s_loops, r_loops, o_loops;
			const std::string dom_kind = info.dom_kind();
			const SBlockRV block = info.block_rv;

			bool bound = false;
			for (const auto& loop_rv : sch->GetLoops(block))
				bound |= sch->Get(loop_rv)->thread_binding.has_value();
			if (bound)
				continue;

			if (sch->GetLoops(block).empty() && _has_internal_thread_env(sch->Get(block)->body))
				continue;

			const auto loops = sch->GetLoops(block);
			for (size_t i = 0; i < loops.size() && i < dom_kind.size(); ++i)
				(dom_kind[i] == 'S' ? s_loops : dom_kind[i] == 'R' ? r_loops : o_loops).push_back(loops[i]);

			if (s_loops.empty())
				s_loops.push_back(sch->AddUnitLoop(block));

			Array<LoopRV> order = s_loops;
			for (const auto& l : r_loops)
				order.push_back(l);
			for (const auto& l : o_loops)
				order.push_back(l);
			sch->Reorder(order);
			const auto split = sch->Split(sch->Fuse(s_loops), factors({std::nullopt, max_threads_per_block}));
			const LoopRV bx = split[0], tx = split[1];
			sch->Bind(bx, "blockIdx.x");
			sch->Bind(tx, "threadIdx.x");

			if (!r_loops.empty())
				reduction_blocks.emplace_back(block, r_loops[0]);
		}

		for (const auto& [block, r_loop] : reduction_blocks)
			sch->DecomposeReduction(block, r_loop);

		return sch;
	}

} // namespace mdl::tvm_ported::dlight::gpu
