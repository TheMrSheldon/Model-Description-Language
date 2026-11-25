#include "./reduction.hpp"

#include <tvm_ported/s_tir/dlight/base/utils.hpp>

namespace mdl::backend::tvm::cpu {

	std::optional<::tvm::s_tir::Schedule>
	Reduction::apply(const ::tvm::tirx::PrimFunc& func, const ::tvm::Target& target, bool tunable) const {
		const auto sch = mdl::tvm_ported::dlight::schedule_of(func);
		if (sch->GetOutputBlocks(sch->GetSBlock("root", ::tvm::ffi::String("main"))).size() != 1)
			return std::nullopt;
		return mdl::tvm_ported::dlight::cpu::Reduction().apply(func, target, tunable);
	}

} // namespace mdl::backend::tvm::cpu
