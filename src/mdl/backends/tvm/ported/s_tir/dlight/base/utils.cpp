// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/base/utils.py
// (conventions: src/mdl/backends/tvm/ported/README.md)
#include "./utils.hpp"

#include <stdexcept>

namespace mdl::tvm_ported::dlight {

	Schedule schedule_of(const PrimFunc& func) {
		// Python's s_tir.Schedule defaults: a traced schedule, random seed, no debug checks, detailed errors.
		return Schedule::Traced(
				IRModule({{::tvm::GlobalVar("main"), func}}), /*seed=*/-1, /*debug_mask=*/0,
				s_tir::ScheduleErrorRenderLevel::kDetail
		);
	}

	int64_t get_bytes(const ::tvm::PrimType& dtype) { return (dtype.bits() * dtype.lanes() + 7) / 8; }

	PrimExpr get_extent(const Schedule& sch, const LoopRV& loop_rv) { return sch->Get(loop_rv)->extent; }

	void auto_vectorize(const Schedule& sch, const LoopRV& loop, int64_t max_vec) {
		const auto extent = as_int(get_extent(sch, loop));
		if (!extent)
			return;
		const LoopRV v = *extent <= max_vec ? loop
											: sch->Split(loop, {std::nullopt, ::tvm::IntImm::Int32(max_vec)}).back();
		sch->Vectorize(v);
	}

	int64_t max_threads_per_block(const Target& target) {
		for (const char* name : {"max_threads_per_block", "max_num_threads"})
			if (auto result = target->GetAttr<int64_t>(name))
				return result.value();
		if (target->kind->name == "cuda")
			return 1024;
		return 256;
	}

	std::vector<int64_t> suggest_threads_per_block(
			const Target& target, const std::vector<tirx::For>& loops, int64_t max_threads_for_dynamic_loop
	) {
		int64_t threads;
		const std::string kind = target->kind->name;
		if (kind == "cuda")
			threads = 1024;
		else if (kind == "rocm" || kind == "metal" || kind == "opencl")
			threads = 256;
		else
			threads = 64;
		std::vector<int64_t> results;
		std::vector<size_t> dynamic;
		for (size_t i = 0; i < loops.size(); ++i) {
			if (auto loop_extent = as_int(loops[i]->extent)) {
				int64_t extent = 1;
				while (extent <= *loop_extent && extent <= threads)
					extent *= 2;
				extent /= 2;
				if (extent < 1 || threads % extent != 0)
					throw std::logic_error("suggest_threads_per_block");
				threads /= extent;
				results.push_back(extent);
			} else {
				results.push_back(0); // None
				dynamic.push_back(i);
			}
		}
		for (size_t i : dynamic) {
			int64_t extent = 1;
			while (extent <= max_threads_for_dynamic_loop && extent <= threads)
				extent *= 2;
			extent /= 2;
			if (extent < 1 || threads % extent != 0)
				throw std::logic_error("suggest_threads_per_block");
			threads /= extent;
			results[i] = extent;
		}
		if (!dynamic.empty())
			results[dynamic[0]] *= threads;
		return results;
	}

} // namespace mdl::tvm_ported::dlight
