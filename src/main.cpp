#include "mdl/logging.hpp"

#include <mdl/parser.hpp>

#include <CLI/CLI.hpp>

#include <iostream>
#include <optional>

struct LoggerConf final {
	bool quiet = false;
	/**
	 * @brief Sets the verbosity of the logger. Per default (verbosity=0) only critical, error and warnings are printed.
	 * If verbosity is...
	 *  - 1: info and above is printed
	 *  - 2: debug and above is printed
	 *  - 3+: trace and above is printed
	 */
	int verbosity = 0;

	mdl::log::Verbosity getVerbosity() const noexcept {
		if (quiet)
			return mdl::log::Verbosity::Off;
		auto verb = static_cast<int>(mdl::log::Verbosity::Warning) + verbosity;
		verb = std::min(verb, static_cast<int>(mdl::log::Verbosity::Trace));
		return static_cast<mdl::log::Verbosity>(verb);
	}
};

void setupLoggerArgs(CLI::App& app, LoggerConf& conf) {
	app.add_flag(
			"-v,--verbose", conf.verbosity,
			"Sets the logger's verbosity. Passing it multiple times increases verbosity."
	);
	app.add_flag("-q,--quiet", conf.quiet, "Supresses all outputs");
}

struct CompileArgs {
	std::vector<std::filesystem::path> inputs;
	std::optional<std::filesystem::path> output;
	LoggerConf logConf;
};

template <class... Ts>
struct overloaded : Ts... {
	using Ts::operator()...;
};

static void print(const Struct& s) { std::cout << "Struct " << s.name << std::endl; }

static void print(const Module& m) {
	std::cout << "Module " << m.name << std::endl;
	std::cout << "Arguments:" << std::endl;
	for (const auto& a : m.arguments) {
		std::cout << a.name << std::endl;
	}
	std::cout << "Variables:" << std::endl;
	for (const auto& v : m.variables) {
		std::cout << v.name << std::endl;
	}
	std::cout << "Functions:" << std::endl;
	for (const auto& f : m.functions) {
		std::cout << f.name << std::endl;
	}
}

#include <iostream>
#include <tvm/ir/attrs.h>
#include <tvm/ir/module.h>
#include <tvm/relax/backend.h>
#include <tvm/relax/exec_builder.h>
#include <tvm/runtime/module.h>
#include <tvm/target/codegen.h>
#include <tvm/target/tag.h>
#include <tvm/target/target.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
	namespace tir {
		PrimFunc CreatePrimFunc(
				const ffi::Array<te::Tensor>& arg_list, std::optional<DataType> index_dtype_override = std::nullopt
		);
	}
} // namespace tvm

static void runCompileCommand(const CompileArgs& args) {
	mdl::log::setVerbosity(args.logConf.getVerbosity());
	std::filesystem::path output = args.output.value_or("a.o");

	auto a = tvm::te::placeholder({128, 128}, tvm::runtime::DataType::Float(32), "A");
	auto b = tvm::te::placeholder({128, 128}, tvm::runtime::DataType::Float(32), "B");
	std::cout << a->shape << std::endl;
	std::cout << a << std::endl;
	auto k = tvm::te::reduce_axis({0, 128}, "k");
	auto y = tvm::te::compute({128, 128}, [&](auto i, auto j) { return tvm::sum(a(i, k) * b(k, j), {k}); }, "Y");
	auto c = tvm::te::compute({128, 128}, [&](auto i, auto j) { return tvm::max(y(i, j), 0); }, "C");

	auto prim = tvm::WithAttrs(
			tvm::tir::CreatePrimFunc({a, b, c}, {}),
			{{tvm::ffi::String{tvm::attr::kGlobalSymbol}, tvm::ffi::Any{tvm::ffi::String{"test"}}}}
	);
	tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> funcs{{tvm::GlobalVar{{"test"}}, prim}};

	// https://github.com/apache/tvm/blob/6dc71f3dc4a54571e3dc0165630cd9a1a0bb97b5/python/tvm/tir/pipeline.py#L33-L123
	auto pass = tvm::transform::Sequential(
			{tvm::transform::PrintIR(), tvm::tir::transform::LowerCrossThreadReduction(),
			 tvm::tir::transform::LowerInitBlock(), tvm::tir::transform::PlanAndUpdateBufferAllocationLocation(),
			 tvm::tir::transform::ConvertBlocksToOpaque(), tvm::tir::transform::LiftThreadBinding(),
			 tvm::tir::transform::ManifestSharedMemoryLocalStage(), tvm::tir::transform::CompactBufferAllocation(),
			 tvm::tir::transform::LowerAutoCopy(), tvm::tir::transform::UnifyThreadBinding(),
			 tvm::tir::transform::LowerMatchBuffer(), tvm::tir::transform::Simplify(),
			 tvm::tir::transform::InjectPermutedLayout(),
			 //tvm::tir::transform::AnnotateIrregularLoop(),
			 tvm::tir::transform::InjectSoftwarePipeline(), tvm::tir::transform::TransformMmaBufferLayout(),
			 tvm::tir::transform::LowerOpaqueBlock(), tvm::tir::transform::FlattenBuffer(),
			 tvm::tir::transform::BF16ComputeLegalize(), tvm::tir::transform::NarrowDataType(32),
			 tvm::tir::transform::LoopPartition(), tvm::tir::transform::VectorizeLoop(true),
			 tvm::tir::transform::InjectVirtualThread(), tvm::tir::transform::InjectDoubleBuffer(),
			 tvm::tir::transform::StorageRewrite(), tvm::tir::transform::LowerAsyncDMA(),
			 tvm::tir::transform::HoistIfThenElse(), tvm::tir::transform::UnrollLoop(),
			 tvm::tir::transform::RenormalizeSplitPattern(), tvm::tir::transform::Simplify(),
			 tvm::tir::transform::RemoveNoOp(), tvm::tir::transform::RewriteUnsafeSelect(),
			 tvm::tir::transform::InstrumentBoundCheckers(), tvm::tir::transform::InjectPTXLDG32(),
			 tvm::tir::transform::CommonSubexprElimTIR(), tvm::tir::transform::InstrumentProfileIntrinsics(),
			 tvm::tir::transform::FP8ComputeLegalize(), // tvm::tir::transform::VerifyVTCMLimit(),
			 tvm::tir::transform::LowerVtcmAlloc(),		// tvm::tir::transform::VerifyMemory(),
			 tvm::tir::transform::AnnotateEntryFunc(), tvm::tir::transform::AnnotateDeviceRegions(),
			 tvm::tir::transform::SplitHostDevice(), tvm::tir::transform::MergeSharedMemoryAllocations(),
			 tvm::tir::transform::MakePackedAPI(), tvm::tir::transform::FP8StorageLegalize(),
			 tvm::tir::transform::BF16StorageLegalize(), tvm::tir::transform::LowerDeviceKernelLaunch(),
			 // tvm::tir::transform::SkipAssert(),
			 tvm::transform::PrintIR()}
	);

	tvm::Target target{"llvm"};
	target = target.WithHost(target, target);

	auto module = pass(tvm::tir::transform::BindTarget(target)(tvm::IRModule{funcs}));

	/*std::cout << "Target Kinds:" << std::endl;
	for (auto&& name : tvm::TargetKindRegEntry::ListTargetKinds())
		std::cout << "  - " << name << std::endl;

	std::cout << "Target Tags:" << std::endl;
	for (auto&& [name, target] : tvm::TargetTag::ListTags()) {
		std::cout << "  - " << name << std::endl;
	}*/

	auto mod = tvm::codegen::Build(module, target);
	mod->WriteToFile(output.c_str(), "");

	return;
#if 0
	/** \todo: handle multiple input files */
	auto program = mdl::parser::parse(args.inputs[0]);
	for (const auto& [key, value] : program.definitions) {
		std::visit(overloaded{[](const Struct& s) { print(s); }, [](const Module& m) { print(m); }}, value);
	}
#endif
}

int main(int argc, char* argv[]) {
	CLI::App app("TODO");

	CompileArgs args;
	setupLoggerArgs(app, args.logConf);
	app.add_option("input", args.inputs, "The files to be compiled")->check(CLI::ExistingFile)->required();
	app.add_option("-o,--out", args.output, "The output path.");
	app.callback([&args]() { runCompileCommand(args); });

	CLI11_PARSE(app, argc, argv);
	return 0;
}