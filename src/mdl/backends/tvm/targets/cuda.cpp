#include "./cuda.hpp"

#include "./gpu/column_reduction.hpp"
#include "./gpu/row_reduction.hpp"
#include "./gpu/scatter_add_rows.hpp"
#include "./llvm.hpp"

#include <tvm_ported/globals.hpp>
#include <tvm_ported/s_tir/dlight/base/transform.hpp>
#include <tvm_ported/s_tir/dlight/gpu/gpu.hpp>
#include <tvm_ported/support/nvcc.hpp>

#include <tvm/ir/attrs.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/target/codegen.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

// TVM internals (src/; pinned TVM version): the CUDA module of the code generation without a CUDA runtime.
#include <backend/cuda/codegen/cuda_fallback_module.h>
#include <target/build_common.h>

#include <cstdlib>
#include <format>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mdl::backend::tvm {

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
		dumpDir = opts.dumpDir;
		// Device code through TVM's CUDA C code generation, compiled with NVRTC (cubin). Its Python callbacks (the
		// CUDA header, intrinsic codegens, the compiler) and the tensor intrinsics are ported to C++ (src/tvm_ported).
		target = ::tvm::Target(::tvm::Target(R"({"kind": "cuda", "arch": ")" + arch + "\"}"), LLVMTarget::host(opts));
		mdl::tvm_ported::register_cuda_globals();
	}

	kernels::Platform CUDATarget::kernelPlatform() const {
		auto p = kernels::hostPlatform();
		p.deviceArch = arch;
		return p;
	}

	void CUDATarget::overrideLegalizations(::tvm::ffi::Map<::tvm::ffi::String, ::tvm::ffi::Function>& handlers) const {
		handlers.Set("mdl.scatter_add_rows", gpu::scatterAddRows());
	}

	::tvm::IRModule CUDATarget::schedule(::tvm::IRModule kernels) const {
		// TVM's default GPU schedules (DLight), as TVM's own pipelines apply them; ported to C++ (src/tvm_ported).
		// The calls of kernel packages (a block without loops around a call_extern) are host code: marked as
		// scheduled, which ApplyDefaultSchedule skips (in TVM's pipelines, such calls are not PrimFuncs).
		std::vector<std::pair<::tvm::GlobalVar, ::tvm::BaseFunc>> externs;
		for (const auto& [gv, func] : kernels->functions) {
			const auto* prim = func.as<::tvm::tirx::PrimFuncNode>();
			if (prim == nullptr)
				continue;
			bool loops = false;
			::tvm::ffi::StructuralWalk<::tvm::ffi::WalkOrder::kPreOrder>(
					prim->body, [&](const ::tvm::tirx::For&) -> ::tvm::ffi::Expected<::tvm::ffi::WalkResult> {
						loops = true;
						return ::tvm::ffi::WalkResult::Advance();
					}
			);
			if (!loops)
				externs.emplace_back(gv, ::tvm::WithAttr(func, "tirx.is_scheduled", ::tvm::IntImm::Bool(true)));
		}
		for (const auto& [gv, func] : externs)
			kernels->Update(gv, func);
		namespace gpu = mdl::tvm_ported::dlight::gpu;
		const auto apply = mdl::tvm_ported::dlight::ApplyDefaultSchedule({
				// (RowReduction, ColumnReduction: ours, before DLight's reductions.)
				std::make_shared<gpu::Matmul>(), std::make_shared<mdl::backend::tvm::gpu::RowReduction>(),
				std::make_shared<mdl::backend::tvm::gpu::ColumnReduction>(),
				std::make_shared<gpu::Reduction>(), std::make_shared<gpu::GeneralReduction>(),
				std::make_shared<gpu::Transpose>(), std::make_shared<gpu::Fallback>()
		});
		try {
			return runPasses(kernels, {apply}, target);
		} catch (const std::exception& e) {
			// Name the kernel that cannot be scheduled.
			for (const auto& [gv, func] : kernels->functions) {
				try {
					runPasses(::tvm::IRModule({{gv, func}}), {apply}, target);
				} catch (const std::exception& inner) {
					throw std::runtime_error(
							std::format("cannot schedule the kernel {} for CUDA: {}\n{}", std::string(gv->name_hint),
										inner.what(), [&] { std::ostringstream o; o << func; return o.str(); }())
					);
				}
			}
			throw;
		}
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

	std::vector<::tvm::transform::Pass> CUDATarget::afterVectorize() const {
		// As in TVM's default TIR pipeline (python/tvm/s_tir/pipeline.py): DLight's schedules bind virtual threads.
		return {registeredPass("s_tir.transform.InjectVirtualThread"), registeredPass("s_tir.transform.InjectDoubleBuffer")};
	}

	std::vector<::tvm::transform::Pass> CUDATarget::beforeSplit() const {
		return {
				registeredPass("s_tir.transform.ThreadSync", ::tvm::ffi::String("shared")),
				registeredPass("s_tir.transform.ThreadSync", ::tvm::ffi::String("shared.dyn")),
				registeredPass("s_tir.transform.ThreadSync", ::tvm::ffi::String("warp")),
				registeredPass("s_tir.transform.InferFragment"),
				registeredPass("s_tir.transform.LowerThreadAllreduce"),
				registeredPass("s_tir.transform.MergeSharedMemoryAllocations"),
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
		if (!deviceMod->functions.empty()) {
			// (The compiler callback reads the architecture from the current target.)
			::tvm::With<::tvm::Target> scope(target);
			auto device = ::tvm::codegen::Build(deviceMod, target);
			// Without TVM's CUDA runtime in the compiler (mdlc does not need a GPU), TVM keeps the CUDA source for a
			// compilation when the model is loaded: compile it now (NVRTC, cubin), so that models load machine code.
			if (!::tvm::ffi::Function::GetGlobal("ffi.Module.create.cuda")) {
				const std::string source = device->InspectSource("cuda");
				if (!dumpDir.empty()) {
					std::filesystem::create_directories(dumpDir);
					std::ofstream(dumpDir / (output.stem().string() + ".cu")) << source;
				}
				const std::string cubin = mdl::tvm_ported::support::nvcc::compile_cuda(source, "cubin", arch);
				device = ::tvm::target::CUDAFallbackModuleCreate(
						::tvm::ffi::Bytes(cubin.data(), cubin.size()), "cubin",
						::tvm::codegen::ExtractFuncInfo(deviceMod), {{"cuda", ::tvm::ffi::String(source)}}
				);
			}
			code->ImportModule(device);
		}

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
