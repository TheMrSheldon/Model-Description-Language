// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/gpu/general_reduction.py
// (conventions: src/mdl/backends/tvm/ported/README.md)
// Reduction rule for operators including softmax, layer norm, RMS norm, etc.
#include "./gpu.hpp"

#include "../base/common_schedules.hpp"
#include "../base/utils.hpp"

#include <tvm/ffi/extra/structural_visit.h>

#include <unordered_map>

namespace mdl::tvm_ported::dlight::gpu {

	std::optional<Schedule>
	GeneralReduction::apply(const PrimFunc& func, const Target& target, bool /*tunable*/) const {
		if (!is_target_available(target))
			return std::nullopt;

		int64_t len_tx, unroll_depth;
		if (target->kind->name == "cuda") {
			len_tx = 256;
			unroll_depth = 256;
		} else if (target->kind->name == "opencl") {
			len_tx = 256;
			unroll_depth = 64;
		} else {
			len_tx = 64;
			unroll_depth = 64;
		}

		Schedule sch = schedule_of(func);
		auto normalized = analysis::normalize_prim_func(sch);
		auto inlined = try_inline_contiguous_spatial(sch, normalized);
		if (!inlined || inlined->empty())
			return std::nullopt;
		std::vector<analysis::SBlockInfo>& block_infos = *inlined;

		const std::string dom_kind = block_infos[0].dom_kind();
		const size_t num_leading_s = dom_kind.find_first_not_of('S') == std::string::npos
											 ? dom_kind.size()
											 : dom_kind.find_first_not_of('S');
		const size_t num_trailing_r = dom_kind.find_last_not_of('R') == std::string::npos
											  ? dom_kind.size()
											  : dom_kind.size() - 1 - dom_kind.find_last_not_of('R');

		// Align the number of block iters of the last block.
		const size_t num_last_block_iter = block_infos.back().dom_kind().size();
		if (num_last_block_iter < dom_kind.size()) {
			// If the last block is a scalar value, there is nothing left to tile/parallelise, and the `iters` list
			// would be empty. Bind every block to a unit loop instead.
			if (num_last_block_iter == 0) {
				std::vector<s_tir::SBlock> blocks;
				for (const auto& info : block_infos)
					blocks.push_back(sch->Get(info.block_rv));
				const auto alloc_buffers = sch->Get(analysis::get_root_block(sch))->alloc_buffers;
				::tvm::sym::Analyzer analyzer;
				for (size_t block_index = 0; block_index + 1 < block_infos.size(); ++block_index) {
					const auto& block = blocks[block_index];
					bool unit = true;
					for (const auto& loop : sch->GetLoops(block_infos[block_index].block_rv))
						unit &= analyzer->CanProveEqual(sch->Get(loop)->extent, 1);
					if (!unit)
						continue;
					std::vector<::tvm::Expr> other_block_buffers;
					for (size_t other_index = 0; other_index < blocks.size(); ++other_index) {
						if (other_index == block_index)
							continue;
						for (const auto& region : blocks[other_index]->reads)
							other_block_buffers.push_back(region->source);
						for (const auto& region : blocks[other_index]->writes)
							other_block_buffers.push_back(region->source);
					}
					for (size_t buffer_index = 0; buffer_index < block->writes.size(); ++buffer_index) {
						const auto buffer = source_of(block->writes[buffer_index]);
						bool is_allocated = false, is_cross_block = false;
						for (const auto& other : alloc_buffers)
							is_allocated |= buffer.same_as(other);
						for (const auto& other : other_block_buffers)
							is_cross_block |= buffer.same_as(other);
						if (scope_of(buffer) == "global" && is_allocated && !is_cross_block)
							sch->SetScope(block_infos[block_index].block_rv, static_cast<int>(buffer_index), "local");
					}
				}
				for (size_t i = 0; i < block_infos.size(); ++i) {
					const LoopRV loop_rv = sch->AddUnitLoop(block_infos[i].block_rv);
					sch->Bind(loop_rv, i == 0 ? "blockIdx.x" : "threadIdx.x");
				}
				return sch;
			}

			// f_layout_mapping
			const auto& last_iters = block_infos.back().iters;
			Array<PrimVar> iters;
			for (size_t i = 0; i < num_last_block_iter; ++i)
				iters.push_back(PrimVar("i" + std::to_string(i), last_iters[i].dom.ty()));
			const PrimExpr zero = ::tvm::IntImm(iters[0].ty(), 0);
			::tvm::sym::Analyzer analyzer;
			size_t num_matched = 0;
			Array<PrimExpr> target_layout_iters;
			for (const auto& block_iter : block_infos[0].iters) {
				if (num_matched < iters.size() &&
					analyzer->CanProveEqual(block_iter.dom, block_infos.back().iters[num_matched].dom)) {
					target_layout_iters.push_back(iters[num_matched]);
					num_matched += 1;
				} else {
					target_layout_iters.push_back(zero);
				}
			}
			Array<PrimExpr> mapped;
			if (num_matched == iters.size()) {
				mapped = target_layout_iters;
			} else {
				for (size_t i = 0; i < dom_kind.size() - num_last_block_iter; ++i)
					mapped.push_back(zero);
				for (const auto& v : iters)
					mapped.push_back(v);
			}
			const tirx::IndexMap index_map(iters, mapped);
			sch->TransformBlockLayout(block_infos.back().block_rv, index_map);
		}

		// Check the assumptions.
		if (num_trailing_r == 0)
			return std::nullopt;
		for (size_t i = 1; i + 1 < block_infos.size(); ++i)
			if (block_infos[i].dom_kind() != dom_kind)
				return std::nullopt;
		if (!block_infos.back().is_injective() || block_infos.back().dom_kind().size() > dom_kind.size())
			return std::nullopt;

		// Reorder the loops of the last block: the spatial loops it reads the reduced buffers with come first.
		if (block_infos.back().dom_kind().find('R') == std::string::npos) {
			std::vector<::tvm::Expr> reduced_buffers;
			for (size_t i = 0; i + 1 < block_infos.size(); ++i)
				for (const auto& buffer_write : sch->Get(block_infos[i].block_rv)->writes)
					reduced_buffers.push_back(buffer_write->source);
			const auto spatial_block = sch->Get(block_infos.back().block_rv);
			std::unordered_map<const void*, const void*> block_var_to_loop_var;
			std::unordered_set<const void*> spatial_loops;
			const auto loops = sch->GetLoops(block_infos.back().block_rv);
			for (size_t i = 0; i < spatial_block->iter_vars.size() && i < loops.size(); ++i)
				block_var_to_loop_var[spatial_block->iter_vars[i]->var.get()] = sch->Get(loops[i])->loop_var.get();

			auto _visit_expr = [&](const ::tvm::Var& e) -> ffi::Expected<ffi::WalkResult> {
				if (auto it = block_var_to_loop_var.find(e.get()); it != block_var_to_loop_var.end())
					spatial_loops.insert(it->second);
				return ffi::WalkResult::Advance();
			};
			for (const auto& buffer_read : spatial_block->reads) {
				bool reduced = false;
				for (const auto& b : reduced_buffers)
					reduced |= b.same_as(buffer_read->source);
				if (!reduced)
					continue;
				for (const auto& read_range : buffer_read->region) {
					ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(read_range->min, _visit_expr);
					ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(read_range->extent, _visit_expr);
				}
			}
			Array<LoopRV> s_loops, other_loops;
			for (const auto& loop_rv : loops) {
				const tirx::For loop = sch->Get(loop_rv);
				if (spatial_loops.contains(loop->loop_var.get()) || as_int(loop->extent) == 1)
					s_loops.push_back(loop_rv);
				else
					other_loops.push_back(loop_rv);
			}
			for (const auto& l : other_loops)
				s_loops.push_back(l);
			sch->Reorder(s_loops);
		}

		auto loops = sch->GetLoops(block_infos.back().block_rv);
		const LoopRV bx = sch->Fuse(Array<LoopRV>(loops.begin(), loops.begin() + num_leading_s));
		auto split = sch->Split(loops.back(), factors({std::nullopt, len_tx}));
		LoopRV r_loop = split[0], tx = split[1];
		sch->Reorder({tx, r_loop});
		sch->Bind(bx, "blockIdx.x");
		sch->Bind(tx, "threadIdx.x");
		sch->Annotate(r_loop, "pragma_auto_unroll_max_step", ::tvm::IntImm::Int32(unroll_depth));
		sch->Annotate(r_loop, "pragma_unroll_explicit", ::tvm::IntImm::Int32(1));

		for (auto it = block_infos.rbegin() + 1; it != block_infos.rend(); ++it) {
			const SBlockRV block = it->block_rv;
			for (size_t i = 0; i < sch->Get(block)->writes.size(); ++i)
				sch->SetScope(block, static_cast<int>(i), "shared");
			sch->ComputeAt(block, bx, /*preserve_unit_loops=*/true);
			const auto block_loops = sch->GetLoops(block);
			r_loop = sch->Fuse(Array<LoopRV>(block_loops.end() - static_cast<std::ptrdiff_t>(num_trailing_r), block_loops.end()));
			split = sch->Split(r_loop, factors({std::nullopt, len_tx}));
			r_loop = split[0];
			tx = split[1];
			sch->Reorder({tx, r_loop});
			sch->Bind(tx, "threadIdx.x");
			sch->Annotate(r_loop, "pragma_auto_unroll_max_step", ::tvm::IntImm::Int32(unroll_depth));
			sch->Annotate(r_loop, "pragma_unroll_explicit", ::tvm::IntImm::Int32(1));
		}

		// TODO: It's just a workaround to avoid unroll spatial loops, because of the bug of
		// the pass lower-thread-allreduce. We should fix it in the future.
		// sch.annotate(bx, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
		// sch.annotate(bx, ann_key="pragma_unroll_explicit", ann_val=1)
		return sch;
	}

} // namespace mdl::tvm_ported::dlight::gpu
