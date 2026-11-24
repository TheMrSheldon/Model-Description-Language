#include "./fallback.hpp"

#include <tvm_ported/s_tir/dlight/base/utils.hpp>

#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/tirx/stmt.h>

#include <exception>
#include <optional>
#include <utility>
#include <vector>

namespace mdl::backend::tvm::cpu {

	namespace {
		using ::tvm::IntImmNode;
		using ::tvm::ffi::Array;
		using ::tvm::ffi::Optional;
		using ::tvm::s_tir::LoopRV;
		using ::tvm::s_tir::SBlockRV;
		using ::tvm::s_tir::Schedule;
		namespace tirx = ::tvm::tirx;

		/** @brief The constant extent of a loop, or -1 if symbolic. */
		int64_t extentOf(const Schedule& sch, const LoopRV& loop) {
			const auto* imm = sch->Get(loop)->extent.as<IntImmNode>();
			return imm ? static_cast<int64_t>(imm->value) : -1;
		}

		/** @brief The split factor for vectorization. */
		Optional<::tvm::PrimExpr> lanes() { return ::tvm::IntImm(::tvm::PrimType::Int(32), Fallback::kVectorLanes); }

		bool largeEnough(int64_t work) { return work < 0 || work >= Fallback::kMinParallelWork; }

		/** @brief Injective block (only spatial iterators): fuse, vectorize the inner part, parallelize the rest. */
		void scheduleInjective(Schedule& sch, const SBlockRV& block) {
			auto loops = sch->GetLoops(block);
			if (loops.empty())
				return;
			LoopRV fused = loops.size() == 1 ? loops[0] : sch->Fuse(loops);
			const int64_t extent = extentOf(sch, fused);
			if (extent >= 0 && extent < Fallback::kVectorLanes)
				return;
			auto split = sch->Split(fused, {std::nullopt, lanes()});
			sch->Vectorize(split[1]);
			if (largeEnough(extent))
				sch->Parallel(split[0]);
		}

		/**
		 * @brief Block with reduction iterators (sums, maxima, TE matmuls): split the innermost spatial loop for
		 * vectorization and move it inside the reductions; parallelize the outer spatial loops.
		 */
		void scheduleReduction(Schedule& sch, const SBlockRV& block, size_t numSpatial) {
			auto loops = sch->GetLoops(block);
			const size_t numReduction = loops.size() - numSpatial;
			std::vector<LoopRV> spatial(loops.begin(), loops.begin() + numSpatial);
			std::vector<LoopRV> reduction(loops.begin() + numSpatial, loops.end());
			if (spatial.empty() || numReduction == 0)
				return;

			std::optional<LoopRV> outer;
			const int64_t innerExtent = extentOf(sch, spatial.back());
			if (innerExtent > 0 && innerExtent % Fallback::kVectorLanes == 0) {
				auto split = sch->Split(spatial.back(), {std::nullopt, lanes()});
				Array<LoopRV> order(spatial.begin(), spatial.end() - 1);
				order.push_back(split[0]);
				for (const auto& r : reduction)
					order.push_back(r);
				order.push_back(split[1]);
				sch->Reorder(order);
				sch->Vectorize(split[1]);
				spatial.back() = split[0];
			}
			// Parallelize over the outer spatial loops (all spatial loops outside the reductions).
			int64_t work = 1;
			for (const auto& s : spatial) {
				int64_t e = extentOf(sch, s);
				work = (e < 0 || work < 0) ? -1 : work * e;
			}
			if (!largeEnough(work))
				return;
			LoopRV par = spatial.size() == 1 ? spatial[0] : sch->Fuse(Array<LoopRV>(spatial.begin(), spatial.end()));
			sch->Parallel(par);
		}

		/**
		 * @brief Fused kernels consist of one block per operation: inlines the elementwise producers into their
		 * consumers, so that intermediates are not materialized (one pass over the data instead of one per operation).
		 */
		void inlineElementwise(Schedule& sch, const SBlockRV& root) {
			for (const auto& block : sch->GetChildBlocks(root)) {
				bool spatialOnly = true;
				for (const auto& iv : sch->Get(block)->iter_vars)
					spatialOnly &= iv->iter_type == tirx::kDataPar;
				if (!spatialOnly)
					continue;
				try {
					sch->ComputeInline(block); // fails (and leaves the schedule unchanged) e.g. for the output block
				} catch (const std::exception&) {
				}
			}
		}

		/**
		 * @brief (number of spatial iterators, whether there are reductions) of a simple loop nest: one loop per block
		 * iterator, spatial iterators first; nullopt otherwise.
		 */
		std::optional<std::pair<size_t, bool>> shapeOf(const Schedule& sch, const SBlockRV& block) {
			size_t numSpatial = 0;
			bool reductionSeen = false, spatialAfterReduction = false, other = false;
			for (const auto& iv : sch->Get(block)->iter_vars) {
				if (iv->iter_type == tirx::kDataPar) {
					++numSpatial;
					spatialAfterReduction |= reductionSeen;
				} else if (iv->iter_type == tirx::kCommReduce) {
					reductionSeen = true;
				} else {
					other = true;
				}
			}
			if (other || spatialAfterReduction || sch->GetLoops(block).size() != sch->Get(block)->iter_vars.size())
				return std::nullopt;
			return std::pair{numSpatial, reductionSeen};
		}

	} // namespace

	std::optional<Schedule> Fallback::apply(const tirx::PrimFunc& func, const ::tvm::Target& target, bool) const {
		if (!is_target_available(target))
			return std::nullopt;
		try {
			Schedule sch = mdl::tvm_ported::dlight::schedule_of(func);
			const SBlockRV root = sch->GetSBlock("root", ::tvm::ffi::String("main"));
			inlineElementwise(sch, root);
			for (const auto& block : sch->GetChildBlocks(root))
				if (const auto shape = shapeOf(sch, block); shape && shape->second)
					scheduleReduction(sch, block, shape->first);
				else if (shape)
					scheduleInjective(sch, block);
			return sch;
		} catch (const std::exception&) {
			return std::nullopt; // (e.g. no root block: left as it is)
		}
	}

} // namespace mdl::backend::tvm::cpu
