#include "./fusion.hpp"

#include "./duplicate_producers.hpp"
#include "./materialize.hpp"
#include "./planner.hpp"

#include <tvm/relax/transform.h>

namespace mdl::backend::tvm::fusion {

	namespace {
		namespace relax = ::tvm::relax;

		::tvm::transform::Pass planAndGroup() {
			return ::tvm::transform::CreateModulePass(
					[](::tvm::IRModule mod, ::tvm::transform::PassContext) {
						Partition groups;
						for (const auto& [gv, f] : mod->functions)
							if (const auto* fn = f.as<relax::FunctionNode>();
								fn && !fn->HasNonzeroAttr(relax::attr::kPrimitive))
								for (auto& g : plan(mod, ::tvm::ffi::GetRef<relax::Function>(fn)))
									groups.push_back(std::move(g));
						return materialize(mod, groups);
					},
					0, "mdl.PlanAndGroup", {}
			);
		}
	} // namespace

	::tvm::transform::Pass fuseKernels() {
		return ::tvm::transform::Sequential(
				{duplicateProducers(), relax::transform::DeadCodeElimination({}), planAndGroup()}, "mdl.FuseKernels"
		);
	}

} // namespace mdl::backend::tvm::fusion
