// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/cpu/reduction.py
// (conventions: src/tvm_ported/README.md)
// CPU reduction rule for operators including softmax, layer norm, RMS norm, etc.
#include "./cpu.hpp"

#include "../analysis/common_analysis.hpp"
#include "../base/utils.hpp"

#include <tvm/ffi/function.h>

#include <algorithm>

namespace mdl::tvm_ported::dlight::cpu {

	namespace {
		/** @brief Count leading spatial ('S') axes in a dom_kind string. */
		size_t _get_num_leading_s(const std::string& dom_kind) {
			const size_t n = dom_kind.find_first_not_of('S');
			return n == std::string::npos ? dom_kind.size() : n;
		}

		int64_t llvm_get_vector_width(const Target& target) {
			static const auto fn = ffi::Function::GetGlobalRequired("target.llvm_get_vector_width");
			return fn(target).cast<int64_t>();
		}

		/** @brief Split the innermost loop to vec_lanes and vectorize. */
		void _vectorize_inner(const Schedule& sch, const SBlockRV& block_rv, int64_t vec_lanes) {
			const auto block_loops = sch->GetLoops(block_rv);
			if (block_loops.size() <= 1)
				return;
			const auto inner = block_loops.back();
			const auto extent = get_extent(sch, inner);
			if (const auto* imm = extent.as<::tvm::IntImmNode>()) {
				if (imm->value > vec_lanes) {
					const auto split = sch->Split(inner, factors({std::nullopt, vec_lanes}));
					sch->Vectorize(split[1]);
				} else if (imm->value >= 2) {
					sch->Vectorize(inner);
				}
			} else {
				const auto split = sch->Split(inner, factors({std::nullopt, vec_lanes}));
				sch->Vectorize(split[1]);
			}
		}

		/** @brief Split the reduction inner loop and annotate for unrolling. */
		void _unroll_reduction_inner(const Schedule& sch, const SBlockRV& block_rv, int64_t vec_lanes) {
			const auto block_loops = sch->GetLoops(block_rv);
			if (block_loops.size() <= 1)
				return;
			const auto inner = block_loops.back();
			const auto extent = get_extent(sch, inner);
			if (const auto* imm = extent.as<::tvm::IntImmNode>(); imm && imm->value <= vec_lanes)
				return;
			const auto split = sch->Split(inner, factors({std::nullopt, vec_lanes}));
			sch->Annotate(split[1], "pragma_auto_unroll_max_step", ::tvm::IntImm::Int32(vec_lanes));
			sch->Annotate(split[1], "pragma_unroll_explicit", ::tvm::IntImm::Int32(1));
		}
	} // namespace

	std::optional<Schedule> Reduction::apply(const PrimFunc& func, const Target& target, bool /*tunable*/) const {
		if (!is_target_available(target))
			return std::nullopt;

		auto sch = schedule_of(func);
		const auto block_infos = analysis::normalize_prim_func(sch);
		if (!block_infos || block_infos->size() < 2)
			return std::nullopt;

		// Must have at least one reduction block and last block must be injective.
		if (std::all_of(block_infos->begin(), block_infos->end(), [](const auto& bi) { return bi.is_injective(); }))
			return std::nullopt;
		if (!block_infos->back().is_injective())
			return std::nullopt;

		// Every block must start with at least one spatial axis, and all blocks must agree on the minimum number of
		// leading spatial axes.
		std::optional<size_t> num_leading_s;
		for (const auto& bi : *block_infos) {
			const auto dk = bi.dom_kind();
			if (dk.empty() || dk[0] != 'S')
				return std::nullopt;
			const size_t n = _get_num_leading_s(dk);
			num_leading_s = num_leading_s ? std::min(*num_leading_s, n) : n;
		}
		if (!num_leading_s || *num_leading_s == 0)
			return std::nullopt;

		// Infer dtype from the last block's write buffer.
		const auto last_block_stmt = sch->Get(block_infos->back().block_rv);
		const int64_t dtype_bits =
				last_block_stmt->writes.empty()
						? 32
						: last_block_stmt->writes[0]->source.as_or_throw<tirx::BufferVar>()->dtype.bits();

		// Determine vector lanes from target VLEN.
		int64_t vlen_bits = llvm_get_vector_width(target);
		if (vlen_bits <= 0)
			vlen_bits = 128;
		const int64_t vec_lanes = std::max<int64_t>(vlen_bits / dtype_bits, 2);

		// --- Phase 1: Parallelize spatial on the last block ---
		const auto& last_block = block_infos->back();
		const auto loops = sch->GetLoops(last_block.block_rv);
		const LoopRV spatial =
				*num_leading_s > 1 ? sch->Fuse(Array<LoopRV>(
											 loops.begin(), loops.begin() + static_cast<std::ptrdiff_t>(*num_leading_s)
									 ))
								   : loops[0];
		sch->Parallel(spatial);

		// --- Phase 2: Vectorize the last (injective) block ---
		_vectorize_inner(sch, last_block.block_rv, vec_lanes);

		// --- Phase 3: compute_at all preceding blocks under spatial ---
		for (auto it = block_infos->rbegin() + 1; it != block_infos->rend(); ++it)
			sch->ComputeAt(it->block_rv, spatial, /*preserve_unit_loops=*/true);

		// --- Phase 4: Vectorize injective, split+unroll reduction blocks ---
		for (size_t i = 0; i + 1 < block_infos->size(); ++i) {
			const auto& block_info = (*block_infos)[i];
			if (block_info.is_injective())
				_vectorize_inner(sch, block_info.block_rv, vec_lanes);
			else
				_unroll_reduction_inner(sch, block_info.block_rv, vec_lanes);
		}

		return sch;
	}

} // namespace mdl::tvm_ported::dlight::cpu
