#include "./llvm.hpp"

#include "../details/schedule.hpp"

#include <tvm/target/codegen.h>

namespace mdl::backend::tvm {

	::tvm::Target LLVMTarget::host(const mdl::CompileOptions& opts) {
		std::string cpu = "generic";
		if (auto fn = ::tvm::ffi::Function::GetGlobal("tvm.codegen.llvm.GetHostCPUName"))
			cpu = (*fn)().cast<std::string>();
		std::string json = R"({"kind": "llvm", "mcpu": ")" + cpu + "\"";
		if (opts.fastMath)
			json += R"(, "fast-math": true)";
		return ::tvm::Target(json + "}");
	}

	LLVMTarget::LLVMTarget(const mdl::CompileOptions& opts) {
		// `llvm` is the host CPU; `llvm -mcpu=...` (or JSON) another CPU.
		::tvm::Target cpu = opts.target == "llvm" ? host(opts) : parseTarget(opts.target);
		// The host of a CPU target is the CPU itself (MakePackedAPI only lowers functions with a host target).
		target = ::tvm::Target(cpu, cpu);
	}

	::tvm::IRModule LLVMTarget::schedule(::tvm::IRModule kernels) const {
		return applyDefaultCPUSchedule(std::move(kernels));
	}

	std::vector<::tvm::transform::Pass> LLVMTarget::blockLowering() const {
		// The CPU subset of TVM's default TIR pipeline (python/tvm/s_tir/pipeline.py).
		return {
				registeredPass("s_tir.transform.CanonicalizeLoop"),
				registeredPass("s_tir.transform.LowerCrossThreadReduction"),
				registeredPass("s_tir.transform.LowerInitBlock"),
				registeredPass("s_tir.transform.PlanAndUpdateBufferAllocationLocation"),
				registeredPass("s_tir.transform.ConvertBlocksToOpaque"),
				registeredPass("s_tir.transform.CompactBufferAllocation", true),
				registeredPass("s_tir.transform.LowerMatchBuffer"),
				registeredPass("tirx.transform.StmtSimplify"),
				registeredPass("s_tir.transform.LowerOpaqueBlock"),
		};
	}

	std::vector<::tvm::transform::Pass> LLVMTarget::finalizeHost() {
		return {registeredPass("tirx.transform.LowerTVMBuiltin"), registeredPass("tirx.transform.LowerIntrin")};
	}

	void LLVMTarget::emit(::tvm::IRModule mod, const std::filesystem::path& output) const {
		mod = runPasses(std::move(mod), finalizeHost(), target);
		::tvm::codegen::Build(mod, target)->WriteToFile(output.string(), "o");
	}

} // namespace mdl::backend::tvm
