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
		/** @brief Gets a pass from TVM's registry (many TIR passes are only exposed through the FFI). */
		template <typename... Args>
		Pass registeredPass(const std::string& name, Args&&... args) {
			auto factory = ::tvm::ffi::Function::GetGlobal(name);
			if (!factory)
				throw std::runtime_error("TVM pass not found: " + name);
			return (*factory)(std::forward<Args>(args)...).template cast<Pass>();
		}

		/**
		 * @brief Stores bfloat16 as uint16 (for targets without native bfloat16, after tirx.transform.BF16ComputeLegalize,
		 * which leaves bfloat16 only in buffers, accessed through reinterpret).
		 * @details Replaces TVM 0.24's tirx.transform.BF16StorageLegalize, which fails on bfloat16 buffers whose data
		 * variable it has not seen defined (AllocBuffer statements, constants of the AOT lowering, views created by
		 * StorageRewrite: "Cannot find var remap"). Every bfloat16 buffer becomes a uint16 buffer over the same data
		 * (variables of type bfloat16* become uint16*), and the reinterpretations between bfloat16 and uint16 vanish.
		 * The tensors that entry functions pass to kernels keep their bfloat16 element type: packedArg (aot.cpp) marks
		 * them with a uint16 zero, which becomes a bfloat16 one here (MDL tensors are never uint16).
		 */
		Pass bf16Storage() {
			namespace tirx = ::tvm::tirx;
			using ::tvm::DataType;
			struct Rewriter : tirx::StmtExprMutator {
				std::unordered_map<const tirx::VarNode*, tirx::Var> vars;
				std::unordered_map<const tirx::BufferNode*, tirx::Buffer> buffers;

				static bool isBF16Pointer(const tirx::Var& v) {
					const auto* ptr = v->type_annotation.as<::tvm::PointerTypeNode>();
					const auto* prim = ptr ? ptr->element_type.as<::tvm::PrimTypeNode>() : nullptr;
					return prim != nullptr && prim->dtype.is_bfloat16();
				}
				tirx::Var var(const tirx::Var& v) {
					if (auto it = vars.find(v.get()); it != vars.end())
						return it->second;
					if (!isBF16Pointer(v))
						return v;
					const auto* ptr = v->type_annotation.as<::tvm::PointerTypeNode>();
					const auto lanes = ptr->element_type.as<::tvm::PrimTypeNode>()->dtype.lanes();
					tirx::Var u(v->name_hint, ::tvm::PointerType(::tvm::PrimType(DataType::UInt(16, lanes)), ptr->storage_scope));
					vars.emplace(v.get(), u);
					return u;
				}
				tirx::Buffer buffer(const tirx::Buffer& b) {
					if (auto it = buffers.find(b.get()); it != buffers.end())
						return it->second;
					tirx::Var data = var(b->data);
					if (!b->dtype.is_bfloat16() && data.same_as(b->data))
						return b;
					const DataType dtype = b->dtype.is_bfloat16() ? DataType::UInt(16, b->dtype.lanes()) : b->dtype;
					tirx::Buffer u(
							data, dtype, b->shape, b->strides, b->elem_offset, b->name, b->data_alignment, b->offset_factor,
							b->buffer_type, b->axis_separators, b->span
					);
					buffers.emplace(b.get(), u);
					return u;
				}

				::tvm::PrimExpr VisitExpr_(const tirx::VarNode* op) final { return var(::tvm::ffi::GetRef<tirx::Var>(op)); }
				::tvm::PrimExpr VisitExpr_(const tirx::BufferLoadNode* op) final {
					auto load = ::tvm::Downcast<tirx::BufferLoad>(tirx::StmtExprMutator::VisitExpr_(op));
					tirx::Buffer b = buffer(load->buffer);
					if (b.same_as(load->buffer))
						return load;
					return tirx::BufferLoad(b, load->indices, load->predicate);
				}
				tirx::Stmt VisitStmt_(const tirx::BufferStoreNode* op) final {
					auto store = ::tvm::Downcast<tirx::BufferStore>(tirx::StmtExprMutator::VisitStmt_(op));
					tirx::Buffer b = buffer(store->buffer);
					::tvm::PrimExpr value = store->value;
					if (b->dtype != store->buffer->dtype && value.dtype().is_bfloat16())
						value = ::tvm::reinterpret(b->dtype, value); // (collapsed below if it reinterprets a uint16)
					if (b.same_as(store->buffer) && value.same_as(store->value))
						return store;
					return tirx::BufferStore(b, VisitExpr(value), store->indices, store->predicate);
				}
				::tvm::PrimExpr VisitExpr_(const tirx::CallNode* op) final {
					auto call = ::tvm::Downcast<tirx::Call>(tirx::StmtExprMutator::VisitExpr_(op));
					if (call->op.same_as(tirx::builtin::reinterpret())) {
						const ::tvm::PrimExpr& x = call->args[0];
						const DataType to = call->dtype.is_bfloat16() ? DataType::UInt(16, call->dtype.lanes()) : call->dtype;
						if (x.dtype() == to)
							return x;
						if (const auto* inner = x.as<tirx::CallNode>(); inner && inner->op.same_as(tirx::builtin::reinterpret()) &&
																	 inner->args[0].dtype() == to)
							return inner->args[0];
						if (to != call->dtype)
							return ::tvm::reinterpret(to, x);
						return call;
					}
					if (call->op.same_as(tirx::builtin::tvm_stack_make_array()) && call->args.size() == 6 &&
						call->args[4].dtype() == DataType::UInt(16)) {
						auto args = call->args;
						args.Set(4, ::tvm::FloatImm(DataType::BFloat(16), 0.0));
						return tirx::Call(call->dtype, call->op, args);
					}
					return call;
				}
				tirx::Stmt VisitStmt_(const tirx::AllocBufferNode* op) final {
					auto stmt = ::tvm::Downcast<tirx::AllocBuffer>(tirx::StmtExprMutator::VisitStmt_(op));
					tirx::Buffer b = buffer(stmt->buffer);
					return b.same_as(stmt->buffer) ? stmt : tirx::AllocBuffer(b, stmt->annotations);
				}
				tirx::Stmt VisitStmt_(const tirx::DeclBufferNode* op) final {
					auto stmt = ::tvm::Downcast<tirx::DeclBuffer>(tirx::StmtExprMutator::VisitStmt_(op));
					tirx::Buffer b = buffer(stmt->buffer);
					return b.same_as(stmt->buffer) ? stmt : tirx::DeclBuffer(b);
				}
				tirx::Stmt VisitStmt_(const tirx::BindNode* op) final {
					auto stmt = ::tvm::Downcast<tirx::Bind>(tirx::StmtExprMutator::VisitStmt_(op));
					tirx::Var v = var(stmt->var);
					return v.same_as(stmt->var) ? stmt : tirx::Bind(v, stmt->value);
				}
			};
			return ::tvm::tirx::transform::CreatePrimFuncPass(
					[](tirx::PrimFunc f, IRModule, ::tvm::transform::PassContext) {
						Rewriter rewriter;
						auto* n = f.CopyOnWrite();
						Array<tirx::Var> params;
						for (const auto& p : n->params)
							params.push_back(rewriter.var(p));
						n->params = params;
						n->body = rewriter(std::move(n->body));
						return f;
					},
					0, "mdl.BF16Storage", {}
			);
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
					std::optional<double> value;
					if (c && c->data.Shape().Product() == 1) {
						const DLDataType t = c->data->dtype;
						uint16_t bits = 0;
						if (t.bits == 16)
							std::memcpy(&bits, c->data->data, sizeof(bits));
						if (t == DLDataType{kDLFloat, 32, 1}) {
							value = static_cast<const float*>(c->data->data)[0];
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
						args.push_back(::tvm::relax::PrimValue(::tvm::FloatImm(::tvm::DataType(c->data->dtype), *value)));
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

		/**
		 * @brief Sums of 16-bit floats accumulate in float32: `astype(sum(astype(x, float32)), T)` (an internal
		 * precision; the result keeps its type). Runs after the pattern rewrites (whose fused kernels compute in float32
		 * anyway) and before the legalization; FuseOps fuses the conversions into the reduction.
		 */
		class WidenSums : public ::tvm::relax::ExprMutator {
		public:
			using ExprMutator::VisitExpr_;

			::tvm::relax::Expr VisitExpr_(const ::tvm::relax::CallNode* op) final {
				auto call = ::tvm::Downcast<::tvm::relax::Call>(ExprMutator::VisitExpr_(op));
				static const ::tvm::Op sum = ::tvm::Op::Get("relax.sum");
				if (!call->op.same_as(sum))
					return call;
				const auto* info = ::tvm::relax::GetStructInfoAs<::tvm::relax::TensorStructInfoNode>(call->args[0]);
				if (info == nullptr || info->IsUnknownDtype() || info->dtype.bits() != 16 ||
					!(info->dtype.is_float() || info->dtype.is_bfloat16()))
					return call;
				static const auto astype = ::tvm::ffi::Function::GetGlobalRequired("relax.op.astype");
				auto wide = builder_->Emit(astype(call->args[0], ::tvm::DataType::Float(32)).cast<::tvm::relax::Expr>(), "wide");
				auto reduced = builder_->Emit(::tvm::relax::Call(call->op, {wide}, call->attrs, call->sinfo_args), "sum32");
				return astype(reduced, info->dtype).cast<::tvm::relax::Expr>();
			}
		};

		Pass widenSumsPass() {
			return RT::CreateFunctionPass(
					[](::tvm::relax::Function f, IRModule, ::tvm::transform::PassContext) {
						return ::tvm::Downcast<::tvm::relax::Function>(WidenSums().VisitExpr(f));
					},
					0, "mdl.WidenSums", {}
			);
		}

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
						foldPermutesIntoMatmul(),
						RT::Normalize(),
						RT::DeadCodeElimination({}),
						inlineScalarConstantsPass(),
						widenSumsPass(),
						RT::Normalize(),
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
						bf16Storage(),
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
