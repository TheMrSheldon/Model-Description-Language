// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/gpu/transpose.py
// (conventions: src/tvm_ported/README.md)
// Schedule rule for transpose.
#include "./gpu.hpp"

#include "../base/common_schedules.hpp"
#include "../base/utils.hpp"

#include <tvm/sym/iter_affine_map.h>

#include <stdexcept>
#include <unordered_set>

namespace mdl::tvm_ported::dlight::gpu {

	namespace {
		bool is_transpose(const Schedule& sch, const SBlockRV& block_rv) {
			const s_tir::SBlock block = sch->Get(block_rv);
			if (const auto* store = block->body.as<tirx::BufferStoreNode>()) {
				if (const auto* rhs = store->value.as<::tvm::TensorLoadNode>()) {
					const auto &lhs_indices = store->indices, &rhs_indices = rhs->indices;
					// list(lhs_indices) != list(rhs_indices) and set(lhs_indices) == set(rhs_indices)
					bool same_list = lhs_indices.size() == rhs_indices.size();
					for (size_t i = 0; same_list && i < lhs_indices.size(); ++i)
						same_list = lhs_indices[i].same_as(rhs_indices[i]);
					std::unordered_set<const void*> lhs, rhs_set;
					for (const auto& e : lhs_indices)
						lhs.insert(e.get());
					for (const auto& e : rhs_indices)
						rhs_set.insert(e.get());
					if (!same_list && lhs == rhs_set)
						return true;
				}
			}
			return false;
		}
	} // namespace

	std::optional<Schedule> Transpose::apply(const PrimFunc& func, const Target& target, bool /*tunable*/) const {
		// pylint: disable=invalid-name
		if (!is_target_available(target))
			return std::nullopt;
		int64_t len_tx, len_ty, unroll_depth;
		if (target->kind->name == "cuda") {
			len_tx = 16;
			len_ty = 8;
			unroll_depth = 256;
		} else if (target->kind->name == "opencl") {
			len_tx = 16;
			len_ty = 8;
			unroll_depth = 64;
		} else {
			len_tx = 8;
			len_ty = 4;
			unroll_depth = 64;
		}
		int64_t len_vec = 4;

		Schedule sch = schedule_of(func);
		const auto normalized = analysis::normalize_prim_func(sch);
		if (!normalized)
			return std::nullopt; // (TVM iterates over None and fails)
		const auto& blocks = *normalized;
		std::ptrdiff_t transpose_block_idx = -1;
		for (std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(blocks.size()) - 1; idx >= 0; --idx) {
			if (is_transpose(sch, blocks[idx].block_rv)) {
				transpose_block_idx = idx;
				break;
			}
			if (!blocks[idx].is_injective())
				return std::nullopt;
		}
		if (transpose_block_idx == -1)
			return std::nullopt;
		const SBlockRV transpose_block = blocks[transpose_block_idx].block_rv;

		std::optional<SBlockRV> prologue; // the optional decoding block
		if (transpose_block_idx > 0) {
			const auto spatials = try_inline_contiguous_spatial(
					sch, std::vector<analysis::SBlockInfo>(blocks.begin(), blocks.begin() + transpose_block_idx - 1)
			);
			if (!spatials || !spatials->empty())
				throw std::logic_error("Transpose: the blocks before the prologue are not inlined");
			prologue = blocks[transpose_block_idx - 1].block_rv;
		}

		auto loops = sch->GetLoops(transpose_block);
		if (loops.size() != 2)
			// transpose with more than 2 axes is not supported
			return std::nullopt;

		int64_t c_factor = 1;
		if (prologue) {
			const s_tir::SBlock block_stmt = sch->Get(*prologue);
			ffi::Map<PrimVar, Range> input_iters;
			for (const auto& i : block_stmt->iter_vars)
				input_iters.Set(i->var, i->dom);
			::tvm::sym::Analyzer analyzer;
			const auto result = ::tvm::sym::NormalizeToIterSum(
					analysis::detect_dominant_read(block_stmt), input_iters, analyzer
			);
			if (!result->args.empty())
				c_factor = as_int(result->args[0]->lower_factor).value();
		}

		LoopRV i = loops[0], j = loops[1];
		auto split = sch->Split(i, factors({std::nullopt, c_factor}), /*preserve_unit_iters=*/true);
		i = split[0];
		LoopRV vi = split[1];
		split = sch->Split(i, factors({std::nullopt, len_ty}), /*preserve_unit_iters=*/true);
		const LoopRV bi = split[0], ti = split[1];
		split = sch->Split(j, factors({std::nullopt, len_tx}), /*preserve_unit_iters=*/true);
		const LoopRV bj = split[0], tj = split[1];
		sch->Reorder({bi, bj, ti, tj, vi});
		sch->Bind(bi, "blockIdx.y");
		sch->Bind(bj, "blockIdx.x");
		sch->Bind(ti, "threadIdx.y");
		sch->Bind(tj, "threadIdx.x");
		len_vec = std::min(len_vec, c_factor);
		vi = sch->Split(vi, factors({std::nullopt, len_vec}))[1];
		if (len_vec > 1)
			sch->Vectorize(vi);

		const SBlockRV cache_read = sch->CacheRead(transpose_block, /*read_buffer_index=*/0, "shared");
		sch->ComputeAt(cache_read, bj, /*preserve_unit_loops=*/false);
		loops = sch->GetLoops(cache_read);
		const LoopRV fused = sch->Fuse(Array<LoopRV>(loops.begin() + 2, loops.end()));
		split = sch->Split(fused, factors({std::nullopt, len_ty, len_tx, c_factor}));
		sch->Bind(split[1], "threadIdx.y");
		sch->Bind(split[2], "threadIdx.x");
		sch->Unroll(split[3]);
		sch->StorageAlign(cache_read, /*buffer_index=*/0, /*axis=*/0, /*factor=*/32, /*offset=*/1);

		sch->Annotate(bi, "pragma_auto_unroll_max_step", ::tvm::IntImm::Int32(unroll_depth));
		sch->Annotate(bi, "pragma_unroll_explicit", ::tvm::IntImm::Int32(1));

		if (prologue)
			sch->ComputeInline(*prologue);
		return sch;
	}

} // namespace mdl::tvm_ported::dlight::gpu
