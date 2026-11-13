#include "commands/common.hpp"
#include "mdl/argvals.hpp"

#include <mdl/bindings/generator.hpp>
#include <mdl/compiler.hpp>
#include <mdl/exports.hpp>
#include <mdl/utils/pathpattern.hpp>
#include <mdl/computegraphv2/function.hpp>
#include <mdl/parser.hpp>
#include <mdl/utils/utils.hpp>

#include <fstream>
#include <iostream>
#include <optional>

/** @brief A requested binding: `TYPE` or `TYPE=PATH` (see `--binding`). */
struct BindingRequest {
	mdl::bindings::GeneratorFactory factory;
	std::optional<std::filesystem::path> path; ///< Defaults to a file next to the output
};

namespace CLI::detail {
	template <>
	bool lexical_cast<BindingRequest>(const std::string& input, BindingRequest& output) {
		auto eq = input.find('=');
		auto generators = mdl::bindings::getGenerators();
		auto it = generators.find(input.substr(0, eq));
		if (it == generators.end())
			return false;
		output.factory = it->second;
		if (eq != std::string::npos)
			output.path = input.substr(eq + 1);
		return true;
	}
} // namespace CLI::detail

struct CompileArgs {
	std::vector<std::filesystem::path> inputs;
	std::vector<std::filesystem::path> importdirs;
	std::optional<std::filesystem::path> output;
	std::optional<std::filesystem::path> argvals;
	std::optional<std::filesystem::path> depfile;
	std::vector<std::string> gradFunctions;
	std::vector<std::string> trainable;
	std::optional<std::filesystem::path> trainableFile;
	bool inputGrads = false;
	bool gradFused = false;
	bool gradOverwrite = false;
	std::vector<std::string> maskFunctions;
	std::vector<std::string> ellipsisRanks;
	std::vector<std::string> checkpoint;
	mdl::PrimitiveType defaultFloat = mdl::PrimitiveType::float32;
	std::vector<std::string> warnings;
	LoggerConf logConf;
	std::vector<BindingRequest> bindings;
	mdl::CompileOptions compileOpts;
};

/** @brief Writes a Makefile-style dependency file (supported by CMake's DEPFILE for Ninja and Makefiles). */
static void writeDepfile(
		const std::filesystem::path& depfile, const std::filesystem::path& target,
		const std::vector<std::filesystem::path>& sources
) {
	auto escape = [](std::string path) {
		std::string result;
		for (char c : path) {
			if (c == ' ' || c == '#')
				result += '\\';
			if (c == '$')
				result += '$';
			result += c;
		}
		return result;
	};
	std::ofstream out{depfile};
	out << escape(target.string()) << ":";
	for (const auto& source : sources)
		out << " \\\n  " << escape(source.string());
	out << "\n";
}

static void runCompileCommand(const CompileArgs& args) {
	mdl::log::setVerbosity(args.logConf.getVerbosity());
	auto logger = mdl::log::getLogger("compiler");

	mdl::ArgVals argvals = args.argvals.transform(mdl::ArgVals::fromYAML).value_or(mdl::ArgVals{});

	std::filesystem::path output = args.output.value_or("a.o");
	if (output.has_parent_path())
		std::filesystem::create_directories(output.parent_path());
	/** \todo: handle multiple input files */
	mdl::parser::ParseOptions parseOptions{.importDirs = args.importdirs, .defaultFloat = args.defaultFloat};
	// -W<name>: warning, -Wno-<name>: ignored, -Werror=<name>: error.
	for (const auto& w : args.warnings) {
		auto level = mdl::parser::Diagnostic::warning;
		std::string name = w;
		if (name.starts_with("no-")) {
			level = mdl::parser::Diagnostic::ignored;
			name = name.substr(3);
		} else if (name.starts_with("error=")) {
			level = mdl::parser::Diagnostic::error;
			name = name.substr(6);
		}
		if (name == "float-widening")
			parseOptions.floatWidening = level;
		else if (name == "float-narrowing")
			parseOptions.floatNarrowing = level;
		else if (name == "unknown-annotation")
			parseOptions.unknownAnnotation = level;
		else
			throw std::runtime_error(
					"unknown diagnostic -W" + w + " (known: float-widening, float-narrowing, unknown-annotation)"
			);
	}
	auto result = mdl::parser::parse(args.inputs[0], parseOptions);
	if (!result.has_value())
		exit(1);
	auto& program = result.value();
	logger->debug("Computegraph has {} nodes", program.graph().getNumNodes());

	mdl::GradientRequest gradients{
			.functions = args.gradFunctions, .options = {}, .fused = args.gradFused, .accumulate = !args.gradOverwrite,
			.checkpoint = args.checkpoint
	};
	{
		std::vector<std::string> patterns = args.trainable;
		if (args.trainableFile) {
			std::ifstream file{*args.trainableFile};
			for (std::string line; std::getline(file, line);)
				patterns.push_back(line);
		}
		gradients.options.trainable = mdl::PathPatterns{std::move(patterns)};
		gradients.options.inputs = args.inputGrads;
	}
	mdl::CompileOptions compileOpts = args.compileOpts;
	for (const auto& value : args.ellipsisRanks) {
		const auto eq = value.find('=');
		const std::string number = eq == std::string::npos ? value : value.substr(eq + 1);
		unsigned rank = 0;
		try {
			rank = static_cast<unsigned>(std::stoul(number));
		} catch (const std::exception&) {
			throw std::runtime_error("--ellipsis-rank: expected N or NAME=K, got " + value);
		}
		if (eq == std::string::npos) {
			if (rank < 1 || rank > 8)
				throw std::runtime_error("--ellipsis-rank: the rank must be in 1..8");
			compileOpts.ellipsisRank = rank;
		} else {
			compileOpts.inputEllipsisRanks[value.substr(0, eq)] = rank;
		}
	}
	for (const auto& [name, rank] : compileOpts.inputEllipsisRanks)
		if (rank > compileOpts.ellipsisRank)
			throw std::runtime_error("--ellipsis-rank " + name + "=" + std::to_string(rank) + " exceeds the rank of `...`");
	auto exports = mdl::exportedFunctions(
			program, argvals, compileOpts.ellipsisRank, gradients, args.maskFunctions, compileOpts.inputEllipsisRanks
	);
	if (exports.empty())
		logger->warn("Nothing to compile: no module has all of its hyperparameters set (see --arg-vals)");
	for (const auto& fn : exports)
		logger->info(
				"Exporting {} ({} input(s), {} weight(s), {} output(s))", fn.symbol, fn.inputs.size(), fn.weights.size(),
				fn.outputs.size()
		);

	mdl::compileToFile(exports, output, argvals, compileOpts);

	for (const auto& binding : args.bindings) {
		auto generator = binding.factory();
		auto path = binding.path.value_or(generator->inferBindingsFilename(output));
		if (path.has_parent_path())
			std::filesystem::create_directories(path.parent_path());
		generator->generate(exports, path);
	}
	if (args.depfile)
		writeDepfile(*args.depfile, output, program.sources);
}

int main(int argc, char* argv[]) {
	CLI::App app("mdlc -- compiler for the model description language");

	CompileArgs args;
	setupLoggerArgs(app, args.logConf);
	app.add_option("input", args.inputs, "The files to be compiled")->check(CLI::ExistingFile)->required();
	app.add_option("-o,--out", args.output, "The output path (an object file).");
	app.add_option("-i,--import-dir", args.importdirs, "Search paths to search through for import statements.")
			->check(CLI::ExistingDirectory);
	app.add_option("-a,--arg-vals", args.argvals, "Argument values.")->check(CLI::ExistingFile);
	app.add_option(
			   "--binding", args.bindings,
			   "Generate bindings for the compiled functions: TYPE or TYPE=PATH, where TYPE is one of cpp (C++ header), "
			   "python, python-torch, python-tensorflow, json (metadata). Without PATH, the file is placed next to the output. Repeatable."
	)
			->type_name("TYPE[=PATH]");
	app.add_option("--depfile", args.depfile, "Write a dependency file listing all (imported) source files.");
	app.add_option(
			"--grad", args.gradFunctions,
			"Gradient build: replace this member function by its gradient (compile without --grad for inference). "
			"A function with a scalar output (an objective) returns it and the gradients of the trainable weights. "
			"Otherwise <function> also saves the values that <function>_backward(d_<output>...) needs in the "
			"context buffer <function>_ctx. Repeatable."
	);
	app.add_flag(
			"--grad-fused", args.gradFused,
			"For functions without scalar output: compute the outputs and gradients in one call "
			"<function>(inputs..., d_<output>...) instead of <function> + <function>_backward."
	);
	app.add_option(
			"--mask", args.maskFunctions,
			"Make this member function masked: it takes an additional input mask: bool[...] (true: valid), and masked "
			"positions in the last dimension of `...` (e.g. the padding of shorter sequences) do not influence the "
			"valid results; outputs are zero at masked positions. Repeatable."
	);
	app.add_option(
			"--checkpoint", args.checkpoint,
			"Gradient checkpointing: the module instances matching this gitignore-style pattern over their paths "
			"(e.g. 'encoder.layers.*') save only their inputs; the backward pass recomputes their computations. "
			"Repeatable."
	);
	app.add_flag(
			"--grad-overwrite", args.gradOverwrite,
			"Write the gradients to the output buffers instead of adding them (the default accumulates)."
	);
	app.add_option(
			"--trainable", args.trainable,
			"gitignore-style pattern selecting the weights to differentiate (e.g. 'layers.**', '!layers.0.**'); "
			"all weights if none is given. Repeatable; later patterns take precedence."
	);
	app.add_option("--trainable-file", args.trainableFile, "File with --trainable patterns (one per line).")
			->check(CLI::ExistingFile);
	app.add_flag("--input-grads", args.inputGrads, "Also compute the gradients with respect to the inputs.");
	app.add_option(
			"-O,--optimize", args.compileOpts.optimization,
			"Optimization level. Use 'metaschedule' to enable MetaSchedule auto-tuning "
			"(slow compile, near-optimal runtime)."
	);
	app.add_option("--backend", args.compileOpts.backend, "Compiler backend: tvm (default), iree")
			->transform(CLI::CheckedTransformer(std::map<std::string, mdl::Backend>{
					{"tvm", mdl::Backend::Tvm},
					{"iree", mdl::Backend::Iree},
			}));
	app.add_option(
			   "--kernels", args.compileOpts.kernels,
			   "Kernels for recognized subgraphs: onednn (default; link against mdl_kernels), tvm (no dependency)"
	)
			->transform(CLI::CheckedTransformer(std::map<std::string, mdl::Kernels>{
					{"onednn", mdl::Kernels::OneDnn},
					{"tvm", mdl::Kernels::Tvm},
			}));
	app.add_option(
			   "--ellipsis-rank", args.ellipsisRanks,
			   "Number of dimensions that `...` stands for in compiled signatures (default 1; e.g. 2 for [batch, seq]). "
			   "NAME=K gives the input NAME a smaller rank: its `...` are the leading K dimensions (e.g. labels=1 for "
			   "labels [batch] next to token ids [batch, seq]). Repeatable."
	)
			->type_name("N|NAME=K");
	app.add_option(
			   "--float", args.defaultFloat,
			   "The type that `float` stands for: f32 (default), bf16, or f16. The explicit types float32, bfloat16 and "
			   "float16 are not affected; conversions between float types are explicit (e.g. float32(x))."
	)
			->transform(CLI::CheckedTransformer(
					std::map<std::string, mdl::PrimitiveType>{
							{"f32", mdl::PrimitiveType::float32},
							{"bf16", mdl::PrimitiveType::bfloat16},
							{"f16", mdl::PrimitiveType::float16}},
					CLI::ignore_case
			));
	app.add_option(
			   "-W", args.warnings,
			   "Diagnostics like g++: -W<name> warns, -Wno-<name> ignores, -Werror=<name> makes it an error. "
			   "float-widening (implicit bfloat16/float16 to float32) and float-narrowing (float32 to 16 bits, or between "
			   "bfloat16 and float16) are errors by default; as a warning or ignored, the conversion is inserted. "
			   "unknown-annotation (an annotation the compiler does not know) is a warning by default. Repeatable."
	)
			->type_name("[no-|error=]NAME");
	app.add_flag("--fast-math", args.compileOpts.fastMath, "Use fast (approximate) math functions.");
	app.add_flag(
			"--profile", args.compileOpts.profile,
			"Time every kernel call; the times per kernel are printed at program exit (requires mdl_kernels)."
	);
	app.add_option("--dump-dir", args.compileOpts.dumpDir, "Write intermediate IR dumps into this directory.");
	app.callback([&args]() { runCompileCommand(args); });

	CLI11_PARSE(app, argc, argv);
	return 0;
}
