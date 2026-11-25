// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/base/transform.py
// (conventions: src/tvm_ported/README.md)
#include "./transform.hpp"

#include <tvm/ir/attrs.h>

#include <stdexcept>

namespace mdl::tvm_ported::dlight {

	namespace {
		bool is_scheduled(const PrimFunc& func) {
			auto scheduled = func->GetAttr<::tvm::IntImm>("tirx.is_scheduled");
			return scheduled && static_cast<int64_t>(scheduled.value()->value) == 1;
		}

		Target get_target(const PrimFunc& func) {
			if (auto target = func->GetAttr<Target>(::tvm::attr::kTarget))
				return target.value();
			auto current = Target::Current(/*allow_not_defined=*/false);
			return current;
		}

		std::optional<Schedule> apply_rules(
				const PrimFunc& func, const Target& target,
				const std::vector<std::shared_ptr<const ScheduleRule>>& rules, bool tunable
		) {
			for (const auto& rule : rules) {
				auto space = rule->apply(func, target, tunable);
				if (!space)
					continue;
				return space;
			}
			return std::nullopt;
		}
	} // namespace

	::tvm::transform::Pass ApplyDefaultSchedule(std::vector<std::shared_ptr<const ScheduleRule>> rules) {
		return ::tvm::transform::CreateModulePass(
				[rules](IRModule mod, ::tvm::transform::PassContext) {
					std::vector<std::pair<::tvm::GlobalVar, PrimFunc>> updated_functions;
					for (const auto& [g_var, base_func] : mod->functions) {
						const auto* func = base_func.as<tirx::PrimFuncNode>();
						if (func == nullptr || is_scheduled(ffi::GetRef<PrimFunc>(func)))
							continue;
						const PrimFunc prim = ffi::GetRef<PrimFunc>(func);
						const Target target = get_target(prim);
						auto sch = apply_rules(prim, target, rules, /*tunable=*/false);
						if (sch)
							updated_functions.emplace_back(
									g_var, ::tvm::WithAttr(
												   sch.value()->mod()->Lookup("main").as_or_throw<PrimFunc>(),
												   "tirx.is_scheduled", ::tvm::IntImm::Bool(true)
										   )
							);
					}
					for (const auto& [g_var, func] : updated_functions)
						mod->Update(g_var, func);
					return mod;
				},
				0, "ApplyDefaultSchedule", {}
		);
	}

} // namespace mdl::tvm_ported::dlight
