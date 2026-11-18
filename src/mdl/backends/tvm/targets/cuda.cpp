#include "./cuda.hpp"

#include "./llvm.hpp"

#include "../details/schedule.hpp"

#include <tvm/ir/attrs.h>
#include <tvm/target/codegen.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/op.h>

#include <cstdlib>
#include <stdexcept>
#include <unordered_map>

namespace mdl::backend::tvm {

	namespace {
		/**
		 * @brief CUDA's math library (exp, erf, ...) for the NVPTX code generator: MDL_CUDA_LIBDEVICE (environment), else
		 * the one of the CUDA toolkit mdlc was built with.
		 */
		void registerLibdevice() {
			if (::tvm::ffi::Function::GetGlobal("tvm_callback_libdevice_path"))
				return;
			::tvm::ffi::Function::SetGlobal(
					"tvm_callback_libdevice_path", ::tvm::ffi::Function::FromTyped([](int /*computeVersion*/) {
						const char* env = std::getenv("MDL_CUDA_LIBDEVICE");
						const std::string path = env != nullptr ? env : MDL_CUDA_LIBDEVICE;
						if (!std::filesystem::is_regular_file(path))
							throw std::runtime_error(
									"CUDA's libdevice not found at " + path + " (set MDL_CUDA_LIBDEVICE)"
							);
						return path;
					})
			);
		}

		/**
		 * @brief 32-bit thread indices: threads bound to loops with 64-bit variables (the kernels' shapes are int64)
		 * get an int32 variable, and the loop variable becomes its cast. (TVM's NVPTX code generation reads the
		 * indices as i32 and does not convert them.)
		 */
		::tvm::transform::Pass int32ThreadIndices() {
			namespace tirx = ::tvm::tirx;
			using DataType = ::tvm::PrimType;
			struct Narrow : tirx::StmtExprMutator {
				using StmtExprMutator::Mutate_;
				std::unordered_map<const void*, ::tvm::PrimExpr> replaced;

				::tvm::UnchangedOr<tirx::Stmt> Mutate_(const tirx::AttrStmtNode* op, ::tvm::InplaceMode mode) final {
					auto iv = op->node.try_cast<tirx::IterVar>();
					if (op->attr_key != tirx::attr::thread_extent || !iv || iv.value()->var.ty() != ::tvm::PrimType::Int(64))
						return StmtExprMutator::Mutate_(op, mode);
					const auto& old = iv.value();
					::tvm::PrimVar narrow(std::string(old->var->name) , DataType::Int(32));
					replaced[old->var.get()] = ::tvm::prim::cast(DataType::Int(64), narrow);
					const ::tvm::PrimExpr extent = ::tvm::prim::cast(DataType::Int(32), op->value.as_or_throw<::tvm::PrimExpr>());
					tirx::IterVar thread(
							::tvm::Range::FromMinExtent(::tvm::IntImm(DataType::Int(32), 0), extent), narrow, old->iter_type,
							old->thread_tag
					);
					auto body = Mutate(op->body).ValueOrUnchanged(op->body);
					return tirx::AttrStmt(thread, op->attr_key, extent, body);
				}

				::tvm::UnchangedOr<::tvm::Expr> Mutate_(const ::tvm::VarNode* op, ::tvm::InplaceMode mode) final {
					if (auto it = replaced.find(op); it != replaced.end())
						return ::tvm::Expr(it->second);
					return StmtExprMutator::Mutate_(op, mode);
				}
			};
			return tirx::transform::CreatePrimFuncPass(
					[](tirx::PrimFunc f, ::tvm::IRModule, ::tvm::transform::PassContext) {
						Narrow narrow;
						auto body = narrow.Mutate(f->body).ValueOrUnchanged(f->body);
						if (!body.same_as(f->body))
							f.CopyOnWrite()->body = std::move(body);
						return f;
					},
					0, "mdl.Int32ThreadIndices", {}
			);
		}
	} // namespace

	CUDATarget::CUDATarget(const mdl::CompileOptions& opts) {
		// The architecture must be chosen (TVM falls back to sm_50 for a bare `cuda`; the compiler may run without a
		// GPU). `arch` (TVM's cuda target, tags) or `mcpu`.
		::tvm::Target spec = parseTarget(opts.target);
		auto a = spec->GetAttr<::tvm::ffi::String>("arch");
		if (!a || opts.target == "cuda")
			a = spec->GetAttr<::tvm::ffi::String>("mcpu");
		if (!a || std::string(a.value()).empty() || opts.target == "cuda")
			throw std::runtime_error(
					"target '" + opts.target + "': the GPU architecture is required, e.g. --target 'cuda -arch=sm_80'"
			);
		arch = a.value();
		// Device code through TVM's LLVM NVPTX backend: PTX, which the driver also compiles for newer GPUs. (TVM's CUDA C
		// backend needs Python callbacks for its header, intrinsics, and NVRTC.)
		::tvm::Target device(R"({"kind": "nvptx", "mcpu": ")" + arch + "\"}");
		target = ::tvm::Target(device, LLVMTarget::host(opts));
		registerLibdevice();
	}

	kernels::Platform CUDATarget::kernelPlatform() const {
		auto p = kernels::hostPlatform();
		p.deviceArch = arch;
		return p;
	}

	::tvm::IRModule CUDATarget::schedule(::tvm::IRModule kernels) const {
		// Our own schedule: TVM's DLight rules are Python-only; heavy operators (GEMMs) are expected from kernel
		// packages.
		return applyDefaultGPUSchedule(std::move(kernels));
	}

	std::vector<::tvm::transform::Pass> CUDATarget::blockLowering() const {
		// TVM's default TIR pipeline up to the opaque blocks (python/tvm/s_tir/pipeline.py).
		return {
				registeredPass("s_tir.transform.CanonicalizeLoop"),
				registeredPass("s_tir.transform.LowerCrossThreadReduction"),
				registeredPass("s_tir.transform.LowerInitBlock"),
				registeredPass("s_tir.transform.PlanAndUpdateBufferAllocationLocation"),
				registeredPass("s_tir.transform.ConvertBlocksToOpaque"),
				registeredPass("s_tir.transform.LiftThreadBinding"),
				registeredPass("s_tir.transform.ManifestSharedMemoryLocalStage"),
				registeredPass("s_tir.transform.CompactBufferAllocation", true),
				registeredPass("s_tir.transform.LowerAutoCopy"),
				registeredPass("s_tir.transform.UnifyThreadBinding"),
				registeredPass("s_tir.transform.LowerMatchBuffer"),
				registeredPass("tirx.transform.StmtSimplify"),
				registeredPass("s_tir.transform.InjectPermutedLayout"),
				registeredPass("s_tir.transform.InjectSoftwarePipeline"),
				registeredPass("s_tir.transform.TransformMmaBufferLayout"),
				registeredPass("s_tir.transform.LowerOpaqueBlock"),
		};
	}

	std::vector<::tvm::transform::Pass> CUDATarget::beforeSplit() const {
		return {
				registeredPass("s_tir.transform.ThreadSync", ::tvm::ffi::String("shared")),
				registeredPass("s_tir.transform.ThreadSync", ::tvm::ffi::String("shared.dyn")),
				registeredPass("s_tir.transform.ThreadSync", ::tvm::ffi::String("warp")),
				registeredPass("s_tir.transform.InferFragment"),
				registeredPass("s_tir.transform.LowerThreadAllreduce"),
				registeredPass("s_tir.transform.MergeSharedMemoryAllocations"),
				int32ThreadIndices(), // after LowerThreadAllreduce, which needs the thread variables themselves
		};
	}

	void CUDATarget::emit(::tvm::IRModule mod, const std::filesystem::path& output) const {
		const ::tvm::Target host = target->GetHost().value();
		::tvm::IRModule hostMod(::tvm::ffi::Map<::tvm::GlobalVar, ::tvm::BaseFunc>(), mod->source_map, mod->attrs);
		::tvm::IRModule deviceMod(::tvm::ffi::Map<::tvm::GlobalVar, ::tvm::BaseFunc>(), mod->source_map, mod->attrs);
		for (const auto& [gv, func] : mod->functions) {
			auto t = func->GetAttr<::tvm::Target>(::tvm::attr::kTarget);
			const bool onHost = !t || t.value()->kind->name == "llvm" || t.value()->kind->name == "c";
			(onHost ? hostMod : deviceMod)->Add(gv, func);
		}
		hostMod = runPasses(hostMod, LLVMTarget::finalizeHost(), host);
		deviceMod = runPasses(
				deviceMod,
				{registeredPass("tirx.transform.LowerWarpMemory"), registeredPass("tirx.transform.StmtSimplify"),
				 registeredPass("tirx.transform.LowerIntrin")},
				target
		);
		auto code = ::tvm::codegen::Build(hostMod, host);
		if (!deviceMod->functions.empty())
			code->ImportModule(::tvm::codegen::Build(deviceMod, target));

		// One object file: the host code, and the device code as a blob that the system library registers (the same
		// as TVM's export_library with a system-lib prefix), combined by a relocatable link.
		const auto prefix = mod->GetAttr<::tvm::ffi::String>(::tvm::attr::kSystemLibPrefix).value_or("");
		const auto hostObj = output.string() + ".host.o", blobObj = output.string() + ".devc.o";
		code->WriteToFile(hostObj, "o");
		auto triple = host->GetAttr<::tvm::ffi::String>("mtriple");
		const std::string llvmTarget =
				triple ? R"({"kind": "llvm", "mtriple": ")" + std::string(triple.value()) + "\"}" : std::string("llvm");
		::tvm::codegen::PackImportsToLLVM(code, /*system_lib=*/true, llvmTarget, prefix)->WriteToFile(blobObj, "o");
		const std::string link = std::string(MDL_LINKER) + " -r -o '" + output.string() + "' '" + hostObj + "' '" +
								 blobObj + "'";
		const int status = std::system(link.c_str());
		std::filesystem::remove(hostObj);
		std::filesystem::remove(blobObj);
		if (status != 0)
			throw std::runtime_error("combining the host and device code failed: " + link);
	}

} // namespace mdl::backend::tvm
