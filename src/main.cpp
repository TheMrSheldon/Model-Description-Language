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
	auto result = mdl::parser::parse(args.inputs[0], {.importDirs = args.importdirs});
	if (!result.has_value())
		exit(1);
	auto& program = result.value();
	logger->debug("Computegraph has {} nodes", program.graph().getNumNodes());

	mdl::GradientRequest gradients{.functions = args.gradFunctions, .options = {}};
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
	auto exports = mdl::exportedFunctions(program, argvals, args.compileOpts.ellipsisRank, gradients);
	if (exports.empty())
		logger->warn("Nothing to compile: no module has all of its hyperparameters set (see --arg-vals)");
	for (const auto& fn : exports)
		logger->info(
				"Exporting {} ({} input(s), {} weight(s), {} output(s))", fn.symbol, fn.inputs.size(), fn.weights.size(),
				fn.outputs.size()
		);

	mdl::compileToFile(exports, output, argvals, args.compileOpts);

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
			   "python, python-torch, python-tensorflow. Without PATH, the file is placed next to the output. Repeatable."
	)
			->type_name("TYPE[=PATH]");
	app.add_option("--depfile", args.depfile, "Write a dependency file listing all (imported) source files.");
	app.add_option(
			"--grad", args.gradFunctions,
			"Also compile the gradient (vector-Jacobian product) of this member function, exported as <function>_vjp. "
			"Repeatable."
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
			   "--ellipsis-rank", args.compileOpts.ellipsisRank,
			   "Number of dimensions that `...` stands for in compiled signatures (default 1; e.g. 2 for [batch, seq])"
	)
			->check(CLI::Range(1u, 8u));
	app.add_flag("--fast-math", args.compileOpts.fastMath, "Use fast (approximate) math functions.");
	app.add_option("--dump-dir", args.compileOpts.dumpDir, "Write intermediate IR dumps into this directory.");
	app.callback([&args]() { runCompileCommand(args); });

	CLI11_PARSE(app, argc, argv);
	return 0;
}
