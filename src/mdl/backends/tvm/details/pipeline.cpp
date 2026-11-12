#include "./pipeline.hpp"

#include "../../../logging.hpp"
#include "./aot.hpp"
#include "./legalize.hpp"
#include "./pattern_registry.hpp"
#include "./schedule.hpp"

#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/target/codegen.h>
#include <tvm/target/target.h>
#include <tvm/tirx/transform.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace mdl::backend::tvm {

	namespace RT = ::tvm::relax::transform;
	using ::tvm::IRModule;
	using ::tvm::ffi::Array;
	using ::tvm::transform::Pass;

	namespace {
		/** @brief Gets a pass from TVM's registry (many TIR passes are only exposed through the FFI). */
		template <typename... Args>
		Pass registeredPass(const std::string& name, Args&&... args) {
			auto factory = ::tvm::ffi::Function::GetGlobal(name);
			if (!factory)
				throw std::runtime_error("TVM pass not found: " + name);
			return (*factory)(std::forward<Args>(args)...).template cast<Pass>();
		}

		IRModule runPasses(IRModule mod, const std::vector<Pass>& passes, const ::tvm::Target& target) {
			auto ctx = ::tvm::transform::PassContext::Create();
			::tvm::With<::tvm::transform::PassContext> ctxGuard(ctx);
			::tvm::With<::tvm::Target> targetGuard(target);
			return ::tvm::transform::Sequential(Array<Pass>(passes.begin(), passes.end()))(std::move(mod));
		}

		/**
		 * @brief Replaces scalar float constants in elementwise arithmetic by PrimValues.
		 * @details Legalization then inlines them into the generated kernels instead of passing them as (0-d) tensor
		 * arguments (which would require constant data in the compiled object).
		 */
		class InlineScalarConstants : public ::tvm::relax::ExprMutator {
		public:
			using ExprMutator::VisitExpr_;

			::tvm::relax::Expr VisitExpr_(const ::tvm::relax::CallNode* op) final {
				auto call = ::tvm::Downcast<::tvm::relax::Call>(ExprMutator::VisitExpr_(op));
				static const std::vector<::tvm::Op> elementwise{
						::tvm::Op::Get("relax.add"), ::tvm::Op::Get("relax.subtract"),
						::tvm::Op::Get("relax.multiply"), ::tvm::Op::Get("relax.divide")
				};
				const auto* callee = call->op.as<::tvm::OpNode>();
				if (!callee || std::none_of(elementwise.begin(), elementwise.end(), [&](const auto& o) {
						return o.get() == callee;
					}))
					return call;
				Array<::tvm::relax::Expr> args;
				bool changed = false;
				for (const auto& arg : call->args) {
					const auto* c = arg.as<::tvm::relax::ConstantNode>();
					if (c && c->data.Shape().Product() == 1 && c->data->dtype == DLDataType{kDLFloat, 32, 1}) {
						float value = static_cast<const float*>(c->data->data)[0];
						args.push_back(::tvm::relax::PrimValue(::tvm::FloatImm(::tvm::DataType::Float(32), value)));
						changed = true;
					} else {
						args.push_back(arg);
					}
				}
				if (!changed)
					return call;
				// Only valid if the other operand is a tensor (scalar-scalar arithmetic stays as it is).
				bool hasTensor = std::any_of(args.begin(), args.end(), [](const auto& a) {
					return ::tvm::relax::GetStructInfoAs<::tvm::relax::TensorStructInfoNode>(a) != nullptr;
				});
				if (!hasTensor)
					return call;
				return ::tvm::relax::Call(call->op, args, call->attrs, call->sinfo_args);
			}
		};

		Pass inlineScalarConstantsPass() {
			return RT::CreateFunctionPass(
					[](::tvm::relax::Function f, IRModule, ::tvm::transform::PassContext) {
						return ::tvm::Downcast<::tvm::relax::Function>(InlineScalarConstants().VisitExpr(f));
					},
					0, "mdl.InlineScalarConstants", {}
			);
		}
	} // namespace

	::tvm::Target makeTarget(const mdl::CompileOptions& opts) {
		std::string cpu = "generic";
		if (auto fn = ::tvm::ffi::Function::GetGlobal("tvm.codegen.llvm.GetHostCPUName"))
			cpu = (*fn)().cast<std::string>();
		std::string json = R"({"kind": "llvm", "mcpu": ")" + cpu + "\"";
		if (opts.fastMath)
			json += R"(, "fast-math": true)";
		::tvm::Target target(json + "}");
		// The host of a CPU target is the CPU itself (MakePackedAPI only lowers functions with a host target).
		return ::tvm::Target(target, target);
	}

	IRModule runRelaxPipeline(IRModule mod, const ::tvm::Target& target, const mdl::CompileOptions& opts) {
		auto logger = mdl::log::getLogger("compiler");

		// Constant folding first (e.g. sqrt(64) → 8) so that patterns can check constant values.
		mod = runPasses(mod, {RT::Normalize(), RT::FoldConstant()}, target);
		mod = applyPatterns(mod, defaultPatterns());
		mod = runPasses(
				mod,
				{
						RT::Normalize(),
						RT::DeadCodeElimination({}),
						inlineScalarConstantsPass(),
						RT::LegalizeOps(legalizeMap(opts), std::nullopt, /*enable_warning=*/true),
						RT::AnnotateTIROpPattern(),
						// No FoldConstant after legalization: evaluating call_tir on constants builds TIR through
						// Python in TVM 0.24 (constants were folded before the patterns).
						RT::FuseOps(),
						RT::FuseTIR(),
						RT::DeadCodeElimination({}),
				},
				target
		);
		logger->info("Relax legalization and fusion done");
		mod = runPasses(
				mod,
				{
						RT::RewriteDataflowReshape(),
						RT::ToNonDataflow(),
						RT::RemovePurityChecking(),
						RT::CallTIRRewrite(),
						RT::StaticPlanBlockMemory(),
				},
				target
		);
		logger->info("Relax memory planning done");
		return mod;
	}

	void buildObject(
			IRModule mod, const ::tvm::Target& target, const std::string& systemLibPrefix,
			const std::filesystem::path& output
	) {
		auto logger = mdl::log::getLogger("compiler");
		mod = ::tvm::WithAttr(mod, ::tvm::attr::kSystemLibPrefix, ::tvm::ffi::String(systemLibPrefix));

		// CPU subset of TVM's default TIR pipeline (python/tvm/s_tir/pipeline.py), followed by the host finalization.
		// The block-lowering passes only apply to the kernels; the AOT entry functions are already flat.
		IRModule kernels = IRModule(::tvm::ffi::Map<::tvm::GlobalVar, ::tvm::BaseFunc>(), mod->source_map, mod->attrs);
		std::vector<std::pair<::tvm::GlobalVar, ::tvm::BaseFunc>> entries;
		for (const auto& [gv, func] : mod->functions) {
			if (func->HasNonzeroAttr(kAOTEntryAttr))
				entries.emplace_back(gv, func);
			else
				kernels->Add(gv, func);
		}
		kernels = applyDefaultCPUSchedule(kernels);
		mod = runPasses(
				kernels,
				{
						::tvm::tirx::transform::BindTarget(target),
						registeredPass("s_tir.transform.CanonicalizeLoop"),
						registeredPass("s_tir.transform.LowerCrossThreadReduction"),
						registeredPass("s_tir.transform.LowerInitBlock"),
						registeredPass("s_tir.transform.PlanAndUpdateBufferAllocationLocation"),
						registeredPass("s_tir.transform.ConvertBlocksToOpaque"),
						registeredPass("s_tir.transform.CompactBufferAllocation", true),
						registeredPass("s_tir.transform.LowerMatchBuffer"),
						registeredPass("tirx.transform.Simplify"),
						registeredPass("s_tir.transform.LowerOpaqueBlock"),
				},
				target
		);
		for (const auto& [gv, func] : entries)
			mod->Add(gv, func);
		mod = runPasses(
				mod,
				{
						::tvm::tirx::transform::BindTarget(target),
						registeredPass("tirx.transform.FlattenBuffer"),
						registeredPass("tirx.transform.BF16ComputeLegalize"),
						registeredPass("tirx.transform.NarrowDataType", 32),
						registeredPass("s_tir.transform.LoopPartition"),
						registeredPass("tirx.transform.VectorizeLoop", true),
						registeredPass("tirx.transform.StorageRewrite"),
						registeredPass("s_tir.transform.HoistIfThenElse"),
						registeredPass("tirx.transform.UnrollLoop"),
						registeredPass("s_tir.transform.RenormalizeSplitPattern"),
						registeredPass("tirx.transform.Simplify"),
						registeredPass("tirx.transform.RemoveNoOp"),
						registeredPass("s_tir.transform.RewriteUnsafeSelect"),
						registeredPass("tirx.transform.CommonSubexprElim"),
						registeredPass("tirx.transform.FP8ComputeLegalize", ::tvm::ffi::String("float32")),
						registeredPass("tirx.transform.VerifyMemory"),
						registeredPass("tirx.transform.AnnotateDeviceRegions"),
						registeredPass("tirx.transform.SplitHostDevice"),
						registeredPass("tirx.transform.MakePackedAPI"),
						registeredPass("tirx.transform.FP8StorageLegalize"),
						registeredPass("tirx.transform.BF16StorageLegalize"),
						registeredPass("tirx.transform.LowerDeviceKernelLaunch"),
						registeredPass("tirx.transform.LowerTVMBuiltin"),
						registeredPass("tirx.transform.LowerIntrin"),
				},
				target
		);
		logger->info("TIR lowering done ({} functions)", mod->functions.size());

		auto module = ::tvm::codegen::Build(mod, target);
		module->WriteToFile(output.string(), "o");
		logger->info("Wrote {}", output.string());
	}

} // namespace mdl::backend::tvm
