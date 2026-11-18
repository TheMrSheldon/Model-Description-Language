// Port of https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/analysis/common_analysis.py
// (see ../README.md).
#include "./common_analysis.hpp"

#include <tvm/tirx/analysis.h>

#include <exception>
#include <map>
#include <stdexcept>

// TVM's C++ implementation behind Python's `get_global_func("s_tir.schedule.NormalizePrimFunc")`. It is not in TVM's
// public headers (src/s_tir/schedule/transform.cc), but exported (TVM is built with HIDE_PRIVATE_SYMBOLS OFF).
namespace tvm::s_tir {
	ffi::Optional<ffi::ObjectRef> NormalizePrimFunc(Schedule sch);
} // namespace tvm::s_tir

namespace mdl::dlight::analysis {

	namespace {
		char iter_kind(const tirx::IterVar& i) {
			return i->iter_type == tirx::kDataPar ? 'S' : i->iter_type == tirx::kCommReduce ? 'R' : 'O';
		}

		void assert_gpu_target(const Target& target) {
			for (const auto& key : target->keys)
				if (key == "gpu")
					return;
			throw std::invalid_argument("Expect a GPU target, but got " + std::string(target->str()));
		}
	} // namespace

	std::vector<PrimExpr> SBlockInfo::dom() const {
		std::vector<PrimExpr> result;
		for (const auto& i : iters)
			result.push_back(i.dom);
		return result;
	}

	std::string SBlockInfo::dom_kind() const {
		std::string result;
		for (const auto& i : iters)
			result += i.kind;
		return result;
	}

	bool SBlockInfo::is_injective() const {
		for (const auto& i : iters)
			if (i.kind != 'S')
				return false;
		return true;
	}

	bool SBlockInfo::is_elementwise(const Schedule& sch) const {
		auto check_unit_var_range = [](const Range& dom, const tirx::IterVar& var) {
			return dom->min.same_as(var->var) && as_int(dom->extent) == 1;
		};
		if (!is_injective())
			return false;
		const s_tir::SBlock block = sch->Get(block_rv);
		if (block->reads.size() != 1 || block->writes.size() != 1)
			return false;
		const auto &r_region = block->reads[0]->region, &w_region = block->writes[0]->region;
		if (r_region.size() != w_region.size())
			return false;
		for (size_t i = 0; i < block->iter_vars.size() && i < r_region.size(); ++i)
			if (!check_unit_var_range(r_region[i], block->iter_vars[i]) ||
				!check_unit_var_range(w_region[i], block->iter_vars[i]))
				return false;
		return true;
	}

	std::vector<LoopRV> SBlockInfo::get_loops() const {
		std::vector<LoopRV> result;
		for (const auto& i : iters)
			result.push_back(i.loop_rv);
		return result;
	}

	std::optional<std::vector<SBlockInfo>> normalize_prim_func(const Schedule& sch) {
		ffi::Optional<ffi::ObjectRef> result;
		try {
			result = ::tvm::s_tir::NormalizePrimFunc(sch);
			if (!result)
				return std::nullopt;
		} catch (const std::exception&) {
			return std::nullopt;
		}
		const auto parts = ffi::Any(result.value()).cast<Array<ffi::Any>>();
		const auto blocks_rv = parts[0].cast<Array<SBlockRV>>();
		const auto loops = parts[1].cast<Array<Array<LoopRV>>>();
		const auto iters = parts[2].cast<Array<Array<tirx::IterVar>>>();
		const auto is_reduction = parts[3].cast<Array<::tvm::IntImm>>();
		std::vector<SBlockInfo> blocks;
		for (size_t b = 0; b < blocks_rv.size(); ++b) {
			SBlockInfo info{
					.name = sch->Get(blocks_rv[b])->name_hint, .iters = {}, .block_rv = blocks_rv[b],
					.reduction_block = static_cast<int64_t>(is_reduction[b]->value) != 0
			};
			for (size_t i = 0; i < loops[b].size() && i < iters[b].size(); ++i)
				info.iters.push_back(IterInfo{
						.kind = iter_kind(iters[b][i]), .var = iters[b][i]->var, .dom = iters[b][i]->dom->extent,
						.loop_rv = loops[b][i]
				});
			blocks.push_back(std::move(info));
		}
		return blocks;
	}

	SBlockInfo get_sblock_info(const Schedule& sch, const SBlockRV& block) {
		const s_tir::SBlock stmt = sch->Get(block);
		bool reduction = false;
		for (const auto& iter_var : stmt->iter_vars)
			reduction |= iter_kind(iter_var) == 'R';
		SBlockInfo info{.name = stmt->name_hint, .iters = {}, .block_rv = block, .reduction_block = reduction};
		const auto loops = sch->GetLoops(block);
		for (size_t i = 0; i < loops.size() && i < stmt->iter_vars.size(); ++i)
			info.iters.push_back(IterInfo{
					.kind = iter_kind(stmt->iter_vars[i]), .var = stmt->iter_vars[i]->var,
					.dom = stmt->iter_vars[i]->dom->extent, .loop_rv = loops[i]
			});
		return info;
	}

	int64_t get_max_threads_per_block(const Target& target) {
		assert_gpu_target(target);
		for (const char* name : {"max_threads_per_block", "max_num_threads"})
			if (auto value = target->GetAttr<int64_t>(name))
				return value.value();
		return 64;
	}

	int64_t get_max_shared_memory_per_block(const Target& target) {
		assert_gpu_target(target);
		if (auto value = target->GetAttr<int64_t>("max_shared_memory_per_block"))
			return value.value();
		static const std::map<std::string, int64_t> TARGET_KIND_TO_DEFAULT_MAX_SMEM{
				{"cuda", 49152}, {"rocm", 65536}, {"metal", 32768}, {"opencl", 16384}, {"vulkan", 16384}
		};
		auto it = TARGET_KIND_TO_DEFAULT_MAX_SMEM.find(std::string(target->kind->name));
		return it == TARGET_KIND_TO_DEFAULT_MAX_SMEM.end() ? 16384 : it->second;
	}

	SBlockRV get_root_block(const Schedule& sch, const std::string& func_name) {
		const auto func = sch->mod()->Lookup(func_name).as_or_throw<PrimFunc>();
		const auto* realize = func->body.as<s_tir::SBlockRealizeNode>();
		if (realize == nullptr)
			throw std::invalid_argument("The function body is expected to be the root block");
		return sch->GetSBlock(realize->block->name_hint, ffi::String(func_name));
	}

	VarSet collect_block_iter_vars_used_in_access_region(const s_tir::SBlock& block, const Array<Range>& region) {
		VarSet tir_vars;
		for (const auto& expr : region) {
			if (as_int(expr->extent) != 1)
				throw std::logic_error("collect_block_iter_vars_used_in_access_region: extent != 1");
			for (const void* v : collect_vars_used_in_prim_expr(expr->min))
				tir_vars.insert(v);
		}
		VarSet result;
		for (const auto& iter_var : block->iter_vars)
			if (tir_vars.contains(iter_var->var.get()))
				result.insert(iter_var->var.get());
		return result;
	}

	VarSet collect_vars_used_in_prim_expr(const PrimExpr& expr) {
		VarSet tir_vars;
		for (const auto& v : tirx::UndefinedVars(expr))
			tir_vars.insert(v.get());
		return tir_vars;
	}

	PrimExpr detect_dominant_read(const s_tir::SBlock& block) {
		std::optional<TensorRegion> dominant_read;
		int64_t num_read_iters = -1;
		for (const auto& buffer_region : block->reads) {
			const auto tir_vars = collect_block_iter_vars_used_in_access_region(block, buffer_region->region);
			if (num_read_iters < static_cast<int64_t>(tir_vars.size())) {
				num_read_iters = static_cast<int64_t>(tir_vars.size());
				dominant_read = buffer_region;
			}
		}
		if (!dominant_read)
			throw std::logic_error("detect_dominant_read: no read");
		Array<PrimExpr> mins;
		for (const auto& e : dominant_read.value()->region)
			mins.push_back(e->min);
		return source_of(dominant_read.value()).OffsetOf(mins)[0];
	}

	bool is_broadcast_epilogue(const Schedule& sch, const SBlockRV& block, const SBlockRV& epilogue) {
		VarSet write_buffers;
		for (const auto& r : sch->Get(block)->writes)
			write_buffers.insert(r->source.get());
		// TVM: `{i.var: i for i in ... iter_vars if i.dom != 1}`; `i.dom` is a Range, which never equals 1, so all
		// iterators count.
		const size_t epilogue_iters = sch->Get(epilogue)->iter_vars.size();
		for (const auto& buffer_region : sch->Get(epilogue)->reads) {
			if (!write_buffers.contains(buffer_region->source.get()))
				continue;
			const auto tir_vars =
					collect_block_iter_vars_used_in_access_region(sch->Get(epilogue), buffer_region->region);
			if (tir_vars.size() < epilogue_iters)
				return true;
		}
		return false;
	}

} // namespace mdl::dlight::analysis
