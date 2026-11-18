// A rule for reduction (`Out[...] += In[...]` with an optional elementwise epilogue).
// Port of https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/gpu/reduction.py
// (see ../README.md).
#include "./gpu.hpp"

#include "../base/common_schedules.hpp"
#include "../base/utils.hpp"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/sym/iter_affine_map.h>

#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace mdl::dlight::gpu {

	namespace {
		using analysis::SBlockInfo;

		/** @brief The value added by the reduction (`Out = Out + <value>`), if the block has that form. */
		std::optional<PrimExpr> _get_reduction_expr(const s_tir::SBlock& block) {
			// Detect and return `Y` in `X[...] = X[...] + Y`
			const auto* buffer_store = block->body.as<tirx::BufferStoreNode>();
			if (buffer_store == nullptr)
				return std::nullopt;
			const auto* add = buffer_store->value.as<::tvm::prim::AddNode>();
			if (add == nullptr)
				return std::nullopt;
			if (!ffi::StructuralEqual::Equal(
						add->a, tirx::BufferLoad(buffer_store->buffer, buffer_store->indices), /*map_free_vars=*/true
				))
				return std::nullopt;
			return add->b;
		}

		bool _has_reduction_loop(const SBlockInfo& block_info) {
			for (const auto& info : block_info.iters)
				if (info.kind == 'R')
					return true;
			return false;
		}

		/** @brief A thread extent for the inner spatial dimension: the largest divisor of `s_factor` up to 16. */
		int64_t _suggest_inner_spatial_tx(const PrimExpr& s_factor) {
			const auto factor = as_int(s_factor);
			if (!factor)
				return 1;
			int64_t len_tx = 16;
			while (len_tx > 1 && *factor % len_tx != 0)
				len_tx -= 1;
			return len_tx;
		}

		/** @brief The spatial domains in the order of the dominant read; std::nullopt if not a simple access. */
		std::optional<std::vector<PrimExpr>>
		_get_spatial_domains_in_access_order(const SBlockInfo& block_info, const ::tvm::sym::IterSumExpr& access) {
			std::unordered_map<const void*, const analysis::IterInfo*> iter_to_info;
			for (const auto& info : block_info.iters)
				iter_to_info[info.var.get()] = &info;
			std::vector<PrimExpr> spatial_domains;
			std::set<const void*> seen;
			for (const auto& split_expr : access->args) {
				const void* var = split_expr->source->source.get();
				auto it = iter_to_info.find(var);
				if (it == iter_to_info.end())
					return std::nullopt;
				const auto& info = *it->second;
				if (info.kind == 'S') {
					if (seen.contains(var))
						return std::nullopt;
					seen.insert(var);
					if (as_int(split_expr->lower_factor).value_or(1) > 1)
						spatial_domains.push_back(split_expr->extent);
					else
						spatial_domains.push_back(info.dom);
				}
			}
			for (const auto& info : block_info.iters) {
				if (info.kind == 'S' && !seen.contains(info.var.get())) {
					if (as_int(info.dom) != 1)
						return std::nullopt;
					spatial_domains.push_back(info.dom);
				}
			}
			size_t num_spatial = 0;
			for (const auto& info : block_info.iters)
				num_spatial += info.kind == 'S';
			if (spatial_domains.size() != num_spatial)
				return std::nullopt;
			return spatial_domains;
		}

		bool _inner_spatial_write_back_is_affine(const std::vector<PrimExpr>& spatial_domains) {
			if (spatial_domains.empty())
				return false;
			size_t components = 0;
			for (size_t i = 0; i + 1 < spatial_domains.size(); ++i) {
				const auto dom = as_int(spatial_domains[i]);
				components += !dom || *dom > 1;
			}
			const PrimExpr& innermost = spatial_domains.back();
			const auto inner = as_int(innermost);
			if (!inner || _suggest_inner_spatial_tx(innermost) < *inner)
				components += 1;
			return components < 3;
		}

		struct Normalized {
			std::optional<bool> is_inner_reduction;
			std::optional<int64_t> c_factor;
			std::map<size_t, size_t> loop_order;
			std::optional<size_t> s_split_index;
		};

		Normalized _normalize(const Schedule& sch, const SBlockInfo& block_info, const ::tvm::sym::IterSumExpr& access) {
			if (as_int(access->base) != 0)
				return {};
			std::unordered_map<const void*, const analysis::IterInfo*> iter_to_info;
			for (const auto& i : block_info.iters)
				iter_to_info[i.var.get()] = &i;
			Array<LoopRV> s_loops, r_loops, c_loops;
			std::optional<int64_t> c_factor;
			std::optional<LoopRV> s_split_loop;
			std::optional<size_t> s_split_index;
			bool is_inner_reduction = false;
			for (const auto& split_expr : access->args) {
				const void* var = split_expr->source->source.get();
				auto it = iter_to_info.find(var);
				if (it == iter_to_info.end())
					return {};
				const auto info = *it->second;
				iter_to_info.erase(it);
				LoopRV loop = info.loop_rv;
				is_inner_reduction = info.kind == 'R';
				const int64_t lower_factor = as_int(split_expr->lower_factor).value_or(1);
				if (lower_factor > 1) {
					if (!c_loops.empty())
						return {};
					s_split_loop = loop;
					s_split_index = s_loops.size();
					const auto split = sch->Split(loop, factors({std::nullopt, lower_factor}));
					loop = split[0];
					c_loops.push_back(split[1]);
					if (!is_inner_reduction)
						c_factor = lower_factor;
				}
				if (is_inner_reduction)
					r_loops.push_back(loop);
				else
					s_loops.push_back(loop);
			}

			if (!iter_to_info.empty()) {
				for (const auto& [var, info] : iter_to_info) {
					if (info->kind == 'S' && as_int(info->dom) == 1)
						s_loops.push_back(info->loop_rv);
					else
						return {};
				}
			}

			std::map<size_t, size_t> loop_order;
			std::vector<LoopRV> s_block_var_loops;
			auto in_s_loops = [&](const LoopRV& l) {
				for (const auto& s : s_loops)
					if (s.same_as(l))
						return true;
				return false;
			};
			for (const auto& i : block_info.iters)
				if (in_s_loops(i.loop_rv) || (s_split_loop && i.loop_rv.same_as(*s_split_loop)))
					s_block_var_loops.push_back(i.loop_rv);

			for (size_t i = 0; i < s_block_var_loops.size(); ++i) {
				for (size_t j = 0; j < s_loops.size(); ++j) {
					if (s_block_var_loops[i].same_as(s_loops[j])) {
						loop_order[i] = j;
						break;
					}
					if (s_split_loop && s_block_var_loops[i].same_as(*s_split_loop)) {
						loop_order[i] = *s_split_index;
						break;
					}
				}
			}

			if (s_loops.empty() || r_loops.empty())
				throw std::logic_error("Reduction._normalize: no spatial or no reduction loops");
			size_t num_spatial = 0;
			for (const auto& i : block_info.iters)
				num_spatial += i.kind == 'S';
			if (s_loops.size() != num_spatial)
				return {};
			if (c_loops.empty())
				c_loops.push_back(sch->AddUnitLoop(block_info.block_rv));
			Array<LoopRV> order = s_loops;
			for (const auto& l : r_loops)
				order.push_back(l);
			for (const auto& l : c_loops)
				order.push_back(l);
			sch->Reorder(order);
			sch->Fuse(s_loops);
			sch->Fuse(r_loops);
			return {is_inner_reduction, c_factor, loop_order, s_split_index};
		}

		/** @brief Reorders the spatial loops @p s by @p loop_order and splits off the unrolled spatial factor. */
		std::pair<LoopRV, LoopRV> reorder_unrolled_spatial(
				const Schedule& sch, const Array<LoopRV>& s, int64_t unroll_spatial_factor,
				const std::map<size_t, size_t>& loop_order, size_t s_split_index
		) {
			if (s.size() != loop_order.size())
				throw std::logic_error("Reduction: unexpected spatial loops");
			Array<LoopRV> new_order_s;
			for (size_t i = 0; i < s.size(); ++i)
				new_order_s.push_back(s[loop_order.at(i)]);
			sch->Reorder(new_order_s);
			const auto split = sch->Split(new_order_s[s_split_index], factors({std::nullopt, unroll_spatial_factor}));
			new_order_s.Set(s_split_index, split[0]);
			const LoopRV c = split[1];
			Array<LoopRV> order = new_order_s;
			order.push_back(c);
			sch->Reorder(order);
			return {sch->Fuse(new_order_s), c};
		}

		void _sch_inner_reduction(
				const Schedule& sch, const Target& target, const SBlockRV& block,
				std::optional<int64_t> unroll_spatial_factor, const std::optional<SBlockInfo>& epilogue_info,
				const std::map<size_t, size_t>& loop_order, std::optional<size_t> s_split_index
		) {
			// pylint: disable=invalid-name
			auto loops = sch->GetLoops(block);
			LoopRV r = loops[1];
			const int64_t len_tx = suggest_threads_per_block(target, {sch->Get(r)})[0];

			const LoopRV tx0 = sch->Split(r, factors({std::nullopt, len_tx}))[1];
			// Schedule the RF block
			const SBlockRV rf = sch->RFactor(tx0, 0);
			loops = sch->GetLoops(rf);
			const LoopRV bx = loops[0];
			r = loops[1];
			LoopRV tx = loops[2];
			sch->Reorder({bx, tx, r});
			sch->Bind(bx, "blockIdx.x");
			sch->Bind(tx, "threadIdx.x");
			sch->Annotate(tx, "pragma_auto_unroll_max_step", ::tvm::IntImm::Int32(256));
			sch->Annotate(tx, "pragma_unroll_explicit", ::tvm::IntImm::Int32(1));
			sch->SetScope(rf, 0, "local");
			sch->DecomposeReduction(rf, r);
			// Schedule the write back block
			sch->ReverseComputeAt(block, bx, /*preserve_unit_loops=*/true);
			loops = sch->GetLoops(block);
			tx = loops[1];
			const Array<LoopRV> s(loops.begin() + 2, loops.end());

			if (unroll_spatial_factor) {
				auto [fused, c] = reorder_unrolled_spatial(sch, s, *unroll_spatial_factor, loop_order, *s_split_index);
				sch->Reorder({fused, tx, c});
			} else {
				const LoopRV fused = sch->Fuse(s);
				sch->Reorder({fused, tx});
			}
			sch->Bind(tx, "threadIdx.x");
			// Schedule epilogue
			if (epilogue_info) {
				const SBlockRV epilogue = epilogue_info->block_rv;
				sch->ReverseComputeAt(epilogue, bx, /*preserve_unit_loops=*/true);
				if (analysis::is_broadcast_epilogue(sch, block, epilogue)) {
					sch->SetScope(block, 0, "shared");
					const auto e_loops = sch->GetLoops(epilogue); // pylint: disable=invalid-name
					const Array<LoopRV> es(e_loops.begin() + 1, e_loops.end());
					const LoopRV etx = sch->Split(sch->Fuse(es), factors({std::nullopt, len_tx}))[1];
					sch->Bind(etx, "threadIdx.x");
				} else {
					sch->SetScope(block, 0, "local");
				}
			}
			// pylint: enable=invalid-name
		}

		void _sch_inner_spatial(
				const Schedule& sch, const Target& /*target*/, const SBlockRV& block, const PrimExpr& s_factor,
				std::optional<int64_t> unroll_spatial_factor, const std::optional<SBlockInfo>& epilogue_info,
				const std::map<size_t, size_t>& loop_order, std::optional<size_t> s_split_index
		) {
			// pylint: disable=invalid-name
			auto loops = sch->GetLoops(block);
			LoopRV s = loops[0], r = loops[1];
			const int64_t len_ty = 16;
			const int64_t len_tx = _suggest_inner_spatial_tx(s_factor);
			sch->Split(s, factors({std::nullopt, len_tx}));
			const LoopRV ty0 = sch->Split(r, factors({std::nullopt, len_ty}))[1];
			// Schedule the RF block
			const SBlockRV rf = sch->RFactor(ty0, 0);
			loops = sch->GetLoops(rf);
			const LoopRV bx = loops[0], tx = loops[1];
			r = loops[2];
			const LoopRV ty = loops[3];
			sch->Reorder({bx, tx, ty, r});
			sch->Bind(tx, "threadIdx.x");
			sch->Bind(ty, "threadIdx.y");
			sch->Bind(bx, "blockIdx.x");
			sch->SetScope(rf, 0, "local");
			sch->DecomposeReduction(rf, r);
			// Schedule the write back block
			sch->ReverseComputeAt(block, bx, /*preserve_unit_loops=*/true);
			loops = sch->GetLoops(block);
			r = loops[1];
			const Array<LoopRV> ss(loops.begin() + 2, loops.end());

			LoopRV fused;
			if (unroll_spatial_factor) {
				auto [f, c] = reorder_unrolled_spatial(sch, ss, *unroll_spatial_factor, loop_order, *s_split_index);
				fused = f;
				sch->Reorder({fused, c, r});
			} else {
				fused = sch->Fuse(ss);
				sch->Reorder({fused, r});
			}
			sch->Bind(fused, "threadIdx.x");
			sch->Bind(r, "threadIdx.y");

			// Schedule epilogue
			if (epilogue_info) {
				const SBlockRV epilogue = epilogue_info->block_rv;
				sch->ReverseComputeAt(epilogue, bx, /*preserve_unit_loops=*/true);
				const auto e_loops = sch->GetLoops(epilogue); // pylint: disable=invalid-name
				const Array<LoopRV> es(e_loops.begin() + 1, e_loops.end());
				if (analysis::is_broadcast_epilogue(sch, block, epilogue)) {
					sch->SetScope(block, 0, "shared");
					const auto split = sch->Split(sch->Fuse(es), factors({std::nullopt, len_tx, len_ty}));
					sch->Bind(split[1], "threadIdx.x");
					sch->Bind(split[2], "threadIdx.y");
				} else {
					// The epilogue is element-wise without broadcasting.
					// Thus the remaining spatial part should be bind to tx.
					sch->SetScope(block, 0, "local");
					const auto split = sch->Split(sch->Fuse(es), factors({len_tx, std::nullopt}));
					sch->Bind(split[0], "threadIdx.x");
				}
			}
			// pylint: enable=invalid-name
		}
	} // namespace

	std::optional<Schedule> Reduction::apply(const PrimFunc& func, const Target& target, bool /*tunable*/) const {
		if (!is_target_available(target))
			return std::nullopt;
		Schedule sch = schedule_of(func);
		auto normalized = analysis::normalize_prim_func(sch);
		if (!normalized)
			return std::nullopt;
		auto block_infos = *try_inline_contiguous_spatial(sch, normalized);
		std::optional<SBlockInfo> epilogue;
		if (block_infos.size() == 1) {
			epilogue = std::nullopt;
		} else if (block_infos.size() == 2) {
			epilogue = block_infos[1];
			if (!epilogue->is_injective())
				return std::nullopt;
		} else {
			return std::nullopt;
		}

		const SBlockInfo block_info = block_infos[0];
		const SBlockRV block = block_info.block_rv;
		const s_tir::SBlock block_stmt = sch->Get(block);

		// Step 1. Check reduction block
		if (!block_info.is_reduction() || !_has_reduction_loop(block_info) || block_stmt->writes.size() != 1 ||
			!_get_reduction_expr(block_stmt))
			return std::nullopt;
		// Step 2. Normalize the block, merge spatial and reduction iters
		ffi::Map<PrimVar, Range> input_iters;
		for (const auto& i : block_stmt->iter_vars)
			input_iters.Set(i->var, i->dom);
		::tvm::sym::Analyzer analyzer;
		const auto access =
				::tvm::sym::NormalizeToIterSum(analysis::detect_dominant_read(block_stmt), input_iters, analyzer);
		const auto spatial_domains = _get_spatial_domains_in_access_order(block_info, access);
		if (!spatial_domains)
			return std::nullopt;
		const auto [is_inner_reduction, c_factor, loop_order, s_split_index] = _normalize(sch, block_info, access);
		if (!is_inner_reduction && !c_factor)
			return std::nullopt;
		// Step 3. Do the scheduling
		if (is_inner_reduction.value_or(false)) {
			_sch_inner_reduction(sch, target, block, c_factor, epilogue, loop_order, s_split_index);
		} else {
			if (!_inner_spatial_write_back_is_affine(*spatial_domains))
				return std::nullopt;
			_sch_inner_spatial(
					sch, target, block, spatial_domains->back(), c_factor, epilogue, loop_order, s_split_index
			);
		}
		return sch;
	}

} // namespace mdl::dlight::gpu
