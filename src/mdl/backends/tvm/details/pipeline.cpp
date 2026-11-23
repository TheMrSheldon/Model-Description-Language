#include "./pipeline.hpp"
#include "./tvm_compat.hpp"

#include "../../../logging.hpp"
#include "./aot.hpp"
#include "./flatten_injective.hpp"
#include "./fuse_rows.hpp"
#include "./fuse_siblings.hpp"
#include "./stack_sums.hpp"
#include "./attention.hpp"
#include "./kernel_match.hpp"
#include "./legalize.hpp"
#include "./pattern_registry.hpp"

#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/target/codegen.h>
#include <tvm/target/target.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mdl::backend::tvm {

	namespace RT = ::tvm::relax::transform;
	using ::tvm::IRModule;
	using ::tvm::ffi::Array;
	using ::tvm::transform::Pass;

	namespace {
		/**
		 * @brief Prepares bfloat16 stores for tirx.transform.BF16StorageLegalize, which only converts stored values that
		 * are reinterpretations: a bfloat16 value built from constants (e.g. a vectorized zero, `broadcast(reinterpret(
		 * bfloat16, bits))`) becomes `reinterpret(bfloat16, <its uint16 bits>)`.
		 */
		/**
		 * @brief TIR's common subexpression elimination for the kernels only: the AOT entries (host code that calls the
		 * kernels) gain nothing from it, and the pass misplaces its variables around their let bindings (the offsets of
		 * the workspace arena, see aot.cpp).
		 */
		Pass kernelSubexprElim() {
			return ::tvm::transform::CreateModulePass(
					[](IRModule mod, ::tvm::transform::PassContext ctx) {
						IRModule kernels = mod;
						kernels.CopyOnWrite();
						for (const auto& [gv, func] : mod->functions)
							if (const auto* f = func.as<::tvm::tirx::PrimFuncNode>(); f && f->HasNonzeroAttr(kAOTEntryAttr))
								kernels->Remove(gv);
						kernels = registeredPass("tirx.transform.CommonSubexprElim")(kernels);
						IRModule result = mod; // (with its attributes, e.g. the system library prefix)
						result.CopyOnWrite();
						for (const auto& [gv, func] : kernels->functions)
							result->Update(gv, func);
						return result;
					},
					0, "mdl.KernelSubexprElim", {}
			);
		}

		Pass wrapBF16Stores() {
			namespace tirx = ::tvm::tirx;
			using ::tvm::PrimType;
			struct Wrap : tirx::StmtExprMutator {
				using StmtExprMutator::Mutate_;
				/**
				 * @brief The uint16 bits of a bfloat16 value built from constants: a constant, a reinterpretation of bits,
				 * or a broadcast of these (TVM's LLVM code generation has no bfloat16 vectors); std::nullopt for other
				 * values (e.g. loads, which the legalization converts itself).
				 */
				static std::optional<::tvm::PrimExpr> bitsOf(const ::tvm::PrimExpr& value) {
					if (const auto* c = value.as<::tvm::FloatImmNode>()) {
						const float f = static_cast<float>(c->value);
						uint32_t u;
						std::memcpy(&u, &f, sizeof(u));
						const auto bits = static_cast<int64_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); // round to even
						return ::tvm::IntImm(PrimType::UInt(16), bits);
					}
					if (const auto* call = value.as<::tvm::CallNode>();
						call && call->op.same_as(tirx::builtin::reinterpret()) && call->args.size() == 1) {
						const auto arg = call->args[0].as_or_throw<::tvm::PrimExpr>();
						if (arg.ty().MatchesElementType(kDLUInt, 16))
							return arg;
					}
					if (const auto* b = value.as<::tvm::prim::BroadcastNode>())
						if (auto bits = bitsOf(b->value))
							return ::tvm::prim::Broadcast(*bits, b->lanes);
					return std::nullopt;
				}
				::tvm::UnchangedOr<tirx::Stmt> Mutate_(const tirx::BufferStoreNode* op, ::tvm::InplaceMode mode) final {
					auto result = StmtExprMutator::Mutate_(op, mode);
					if (!result.IsUnchanged())
						op = ::tvm::ffi::AnyView(result).as<tirx::BufferStoreNode>();
					const PrimType type = op->value.ty();
					const auto* call = op->value.as<::tvm::CallNode>();
					if (!isBFloat16(op->buffer->dtype) || !type.MatchesElementType(kDLBfloat, 16) ||
						(call != nullptr && call->op.same_as(tirx::builtin::reinterpret())))
						return result;
					auto bits = bitsOf(op->value);
					if (!bits)
						return result;
					return tirx::BufferStore(op->buffer, ::tvm::prim::reinterpret(type, *bits), op->indices);
				}
			};
			return tirx::transform::CreatePrimFuncPass(
					[](tirx::PrimFunc f, IRModule, ::tvm::transform::PassContext) {
						Wrap wrap;
						auto body = wrap.Mutate(f->body).ValueOrUnchanged(f->body);
						if (!body.same_as(f->body))
							f.CopyOnWrite()->body = std::move(body);
						return f;
					},
					0, "mdl.WrapBF16Stores", {}
			);
		}

		/**
		 * @brief Restores the bfloat16 element type of the tensors that entry functions pass to kernels, after
		 * tirx.transform.BF16StorageLegalize: the AOT lowering marks them with a uint16 zero (see typeMarker, aot.cpp).
		 */
		Pass restoreBF16ArrayTypes() {
			namespace tirx = ::tvm::tirx;
			using ::tvm::PrimType;
			struct Restore : tirx::StmtExprMutator {
				using StmtExprMutator::Mutate_;
				::tvm::UnchangedOr<::tvm::Expr> Mutate_(const ::tvm::CallNode* op, ::tvm::InplaceMode mode) final {
					auto result = StmtExprMutator::Mutate_(op, mode);
					if (!result.IsUnchanged())
						op = ::tvm::ffi::AnyView(result).as<::tvm::CallNode>();
					if (!op->op.same_as(tirx::builtin::tvm_stack_make_array()) || op->args.size() < 5 ||
						op->args[4]->ty.as<::tvm::PrimTypeNode>() == nullptr ||
						op->args[4]->ty.as_or_throw<PrimType>() != PrimType::UInt(16))
						return result;
					auto copy = ::tvm::ffi::make_object<::tvm::CallNode>(*op);
					copy->args.Set(4, ::tvm::prim::MakeConst(PrimType::BFloat(16), 0));
					return ::tvm::Expr(std::move(copy));
				}
			};
			return tirx::transform::CreatePrimFuncPass(
					[](tirx::PrimFunc f, IRModule, ::tvm::transform::PassContext) {
						Restore restore;
						auto body = restore.Mutate(f->body).ValueOrUnchanged(f->body);
						if (body.same_as(f->body))
							return f;
						f.CopyOnWrite()->body = std::move(body);
						return f;
					},
					0, "mdl.RestoreBF16ArrayTypes", {}
			);
		}

		/**
		 * @brief Replaces scalar float constants in elementwise arithmetic by scalars (FloatImm).
		 * @details Legalization then inlines them into the generated kernels instead of passing them as (0-d) tensor
		 * arguments (which would require constant data in the compiled object).
		 */
		class InlineScalarConstants : public ::tvm::relax::ExprMutator {
		public:
			using ExprMutator::VisitExpr_;

			::tvm::relax::Expr VisitExpr_(const ::tvm::CallNode* op) final {
				auto call = (ExprMutator::VisitExpr_(op)).as_or_throw<::tvm::Call>();
				static const std::vector<::tvm::Op> elementwise{
						::tvm::Op::Get("relax.add"), ::tvm::Op::Get("relax.subtract"), ::tvm::Op::Get("relax.multiply"),
						::tvm::Op::Get("relax.divide")
				};
				const auto* callee = call->op.as<::tvm::OpNode>();
				if (!callee || std::none_of(elementwise.begin(), elementwise.end(), [&](const auto& o) {
						return o.get() == callee;
					}))
					return call;
				Array<::tvm::relax::Expr> args;
				bool changed = false;
				for (const auto& arg : call->args) {
					const auto c = tensorConstant(arg);
					std::optional<double> value;
					if (c && c->Shape().Product() == 1) {
						const DLDataType t = (*c)->dtype;
						uint16_t bits = 0;
						if (t.bits == 16)
							std::memcpy(&bits, (*c)->data, sizeof(bits));
						if (t == DLDataType{kDLFloat, 32, 1}) {
							value = static_cast<const float*>((*c)->data)[0];
						} else if (t == DLDataType{kDLBfloat, 16, 1}) {
							const uint32_t wide = static_cast<uint32_t>(bits) << 16;
							float f;
							std::memcpy(&f, &wide, sizeof(f));
							value = f;
						} else if (t == DLDataType{kDLFloat, 16, 1}) {
							_Float16 half;
							std::memcpy(&half, &bits, sizeof(half));
							value = static_cast<double>(half);
						}
					}
					if (value) {
						args.push_back(::tvm::FloatImm(::tvm::PrimType((*c)->dtype), *value));
						changed = true;
					} else {
						args.push_back(arg);
					}
				}
				if (!changed)
					return call;
				// Only valid if the other operand is a tensor (scalar-scalar arithmetic stays as it is).
				bool hasTensor = std::any_of(args.begin(), args.end(), [](const auto& a) {
					return ::tvm::relax::GetTypeAs<::tvm::relax::TensorTypeNode>(a) != nullptr;
				});
				if (!hasTensor)
					return call;
				return ::tvm::Call(call->ty, call->op, args, call->attrs, call->ty_args);
			}
		};

		/**
		 * @brief Sums of 16-bit floats accumulate in float32: `astype(sum(astype(x, float32)), T)` (an internal
		 * precision; the result keeps its type). Runs after the pattern rewrites (whose fused kernels compute in float32
		 * anyway) and before the legalization; FuseOps fuses the conversions into the reduction.
		 */
		class WidenSums : public ::tvm::relax::ExprMutator {
		public:
			using ExprMutator::VisitExpr_;

			::tvm::relax::Expr VisitExpr_(const ::tvm::CallNode* op) final {
				auto call = (ExprMutator::VisitExpr_(op)).as_or_throw<::tvm::Call>();
				static const ::tvm::Op sum = ::tvm::Op::Get("relax.sum");
				if (!call->op.same_as(sum))
					return call;
				const auto* info = ::tvm::relax::GetTypeAs<::tvm::relax::TensorTypeNode>(call->args[0]);
				if (info == nullptr || info->IsUnknownDtype() || info->dtype.value().bits() != 16 ||
					!(info->dtype.value().MatchesCode(kDLFloat) || info->dtype.value().MatchesElementType(kDLBfloat, 16)
					))
					return call;
				static const auto astype = ::tvm::ffi::Function::GetGlobalRequired("relax.op.astype");
				auto wide = builder_->Emit(
						astype(call->args[0], ::tvm::PrimType::Float(32)->dtype).cast<::tvm::relax::Expr>(), "wide"
				);
				auto reduced = builder_->Emit(
						::tvm::Call(::tvm::Type::Missing(), call->op, {wide}, call->attrs, call->ty_args), "sum32"
				);
				return astype(reduced, info->dtype.value()->dtype).cast<::tvm::relax::Expr>();
			}
		};

		Pass widenSumsPass() {
			return RT::CreateFunctionPass(
					[](::tvm::relax::Function f, IRModule, ::tvm::transform::PassContext) {
						return (WidenSums().VisitExpr(f)).as_or_throw<::tvm::relax::Function>();
					},
					0, "mdl.WidenSums", {}
			);
		}

		Pass inlineScalarConstantsPass() {
			return RT::CreateFunctionPass(
					[](::tvm::relax::Function f, IRModule, ::tvm::transform::PassContext) {
						return (InlineScalarConstants().VisitExpr(f)).as_or_throw<::tvm::relax::Function>();
					},
					0, "mdl.InlineScalarConstants", {}
			);
		}
	} // namespace

	IRModule canonicalize(IRModule mod, const ::tvm::Target& target) {
		// Constant folding first (e.g. sqrt(64) → 8) so that patterns can check constant values.
		mod = runPasses(mod, {RT::Normalize(), RT::FoldConstant()}, target);
		mod = applyPatterns(mod, defaultPatterns());
		return runPasses(
				mod,
				{
						// (e.g. the calls of mdl.attention of an attention's result and its statistics are one)
						RT::EliminateCommonSubexpr(),
						foldPermutesIntoAttention(),
						foldPermutesIntoMatmul(),
						RT::Normalize(),
						RT::DeadCodeElimination({}),
						inlineScalarConstantsPass(),
						widenSumsPass(),
						RT::Normalize(),
				},
				target
		);
	}

	IRModule expandUnmatched(IRModule mod, const ::tvm::Target& target) {
		return runPasses(
				mod,
				{expandAttention(), RT::Normalize(), RT::EliminateCommonSubexpr(), RT::CanonicalizeBindings(),
				 RT::DeadCodeElimination({}), RT::Normalize()},
				target
		);
	}

	IRModule afterPruning(IRModule mod, const ::tvm::Target& target) {
		// (foldPermutesIntoAttention: whether the attention's row statistics are still used)
		return runPasses(
				mod, {RT::DeadCodeElimination({}), foldPermutesIntoAttention(), RT::Normalize()}, target
		);
	}

	IRModule lowerRelax(IRModule mod, const Target& device, const mdl::CompileOptions& opts) {
		auto logger = mdl::log::getLogger("compiler");
		const auto& target = device.tvm();
		auto handlers = legalizeMap(opts);
		device.overrideLegalizations(handlers);
		mod = runPasses(
				mod,
				{
						// Sums of tensors computed from a common one: one sum (reads it once).
						stackSiblingSums(),
						RT::LegalizeOps(handlers, std::nullopt, /*enable_warning=*/true),
						RT::AnnotateTIROpPattern(),
						// No FoldConstant after legalization: evaluating call_tir on constants builds TIR through
						// Python in TVM 0.24 (constants were folded before the patterns).
						RT::FuseOps(),
						// Reductions over the same rows and their epilogue: one group (FuseOps: one reduction each).
						fuseRows(),
						RT::FuseTIR(),
						// The fallbacks of runtime guards, now fused: back into their branches.
						inlineGuardFallbacks(),
						// Independent elementwise kernels reading the same inputs: one kernel (FuseOps fuses only
						// producers into consumers).
						fuseSiblings(),
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

	IRModule runRelaxPipeline(IRModule mod, const Target& target, const mdl::CompileOptions& opts) {
		return lowerRelax(canonicalize(std::move(mod), target.tvm()), target, opts);
	}

	void buildObject(IRModule mod, const Target& target, const std::string& systemLibPrefix, const std::filesystem::path& output) {
		auto logger = mdl::log::getLogger("compiler");
		const ::tvm::Target& tvmTarget = target.tvm();
		mod = ::tvm::WithAttr(mod, ::tvm::attr::kSystemLibPrefix, ::tvm::ffi::String(systemLibPrefix));

		// TVM's default TIR pipeline (python/tvm/s_tir/pipeline.py) with the device-specific parts from the target,
		// followed by the host finalization. The block-lowering passes only apply to the kernels; the AOT entry
		// functions are already flat.
		IRModule kernels = IRModule(::tvm::ffi::Map<::tvm::GlobalVar, ::tvm::BaseFunc>(), mod->source_map, mod->attrs);
		std::vector<std::pair<::tvm::GlobalVar, ::tvm::BaseFunc>> entries;
		for (const auto& [gv, func] : mod->functions) {
			if (func->HasNonzeroAttr(kAOTEntryAttr))
				entries.emplace_back(gv, func);
			else
				kernels->Add(gv, func);
		}
		kernels = runPasses(kernels, {::tvm::tirx::transform::BindTarget(tvmTarget)}, tvmTarget);
		kernels = flattenInjective(kernels, tvmTarget); // (target-independent: flat loops for injective kernels)
		kernels = target.schedule(kernels);
		mod = runPasses(kernels, target.blockLowering(), tvmTarget);
		for (const auto& [gv, func] : entries)
			mod->Add(gv, func);
		std::vector<Pass> passes{
				::tvm::tirx::transform::BindTarget(tvmTarget),
				registeredPass("tirx.transform.FlattenBuffer"),
				registeredPass("tirx.transform.BF16ComputeLegalize"),
				registeredPass("tirx.transform.NarrowDataType", 32),
				registeredPass("s_tir.transform.LoopPartition"),
				registeredPass("tirx.transform.VectorizeLoop", true),
		};
		for (auto& p : target.afterVectorize())
			passes.push_back(std::move(p));
		for (auto p : {
					 registeredPass("tirx.transform.StorageRewrite"),
				registeredPass("s_tir.transform.HoistIfThenElse"),
				registeredPass("tirx.transform.UnrollLoop"),
				registeredPass("s_tir.transform.RenormalizeSplitPattern"),
				registeredPass("tirx.transform.StmtSimplify"),
				registeredPass("tirx.transform.RemoveNoOp"),
				registeredPass("s_tir.transform.RewriteUnsafeSelect"),
				kernelSubexprElim(),
				registeredPass("tirx.transform.FP8ComputeLegalize", ::tvm::ffi::String("float32")),
				registeredPass("tirx.transform.VerifyMemory"),
				registeredPass("tirx.transform.AnnotateEntryFunc"),
			 })
			passes.push_back(std::move(p));
		for (auto& p : target.beforeSplit())
			passes.push_back(std::move(p));
		for (auto p : {
					 registeredPass("tirx.transform.SplitHostDevice"),
					 registeredPass("tirx.transform.MakePackedAPI"),
					 registeredPass("tirx.transform.FP8StorageLegalize"),
					 wrapBF16Stores(),
					 registeredPass("tirx.transform.BF16StorageLegalize"),
					 restoreBF16ArrayTypes(),
			 })
			passes.push_back(std::move(p));
		mod = runPasses(mod, passes, tvmTarget);
		logger->info("TIR lowering done ({} functions)", mod->functions.size());

		target.emit(mod, output);
		logger->info("Wrote {}", output.string());
	}

} // namespace mdl::backend::tvm
