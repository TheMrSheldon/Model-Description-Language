// mdl-kernels: inspects the kernel packages of the kernel directories (see docs/kernels.md).
#include "compile_cli.hpp"

#include <mdl/kernels.hpp>
#include <mdl/version.hpp>

#include <cstdio>
#include <format>
#include <iostream>
#include <string>
#include <vector>

namespace {
	using namespace mdl::kernels;

	std::string kindName(Artifact::Kind kind) {
		switch (kind) {
		case Artifact::Kind::staticLibrary:
			return "static-library";
		case Artifact::Kind::sharedLibrary:
			return "shared-library";
		case Artifact::Kind::cubin:
			return "cubin";
		case Artifact::Kind::ptx:
			return "ptx";
		}
		return "";
	}

	/** @brief Whether @p artifact can be used on the host, and why not. */
	std::string usability(const Artifact& artifact, const Platform& host) {
		if (!artifact.supported())
			return "not supported yet (" + kindName(artifact.kind) + ")";
		if (!compatible(artifact.platform, host))
			return "not for this host";
		std::error_code ec;
		if (!std::filesystem::is_regular_file(artifact.path, ec))
			return "missing file " + artifact.path.string();
		return "usable";
	}

	int
	list(const std::vector<std::filesystem::path>& explicitDirs, const std::optional<std::filesystem::path>& source) {
		const auto host = hostPlatform();
		const auto dirs = kernelDirs(explicitDirs, source);
		const auto found = discover(dirs);

		std::cout << std::format("Host: {}\n\nKernel directories (most specific first):\n", host.str());
		if (dirs.empty())
			std::cout << "  (none)\n";
		for (const auto& d : dirs)
			std::cout << std::format("  {:<14} {}\n", str(d.origin), d.path.string());

		std::cout << "\nPackages:\n";
		if (found.packages.empty())
			std::cout << "  (none)\n";
		for (const auto& p : found.packages) {
			std::cout << std::format("  {} {}  ({})\n", p.name, p.version, p.directory.string());
			if (!p.description.empty())
				std::cout << std::format("    {}\n", p.description);
			for (const auto& a : p.artifacts)
				std::cout << std::format(
						"    artifact {} [{}, {}]: {}\n", a.id, kindName(a.kind),
						a.supported() ? a.platform.str() : "device " + a.deviceArch, usability(a, host)
				);
			for (const auto& impl : p.implementations)
				std::cout << std::format(
						"    implementation {} (pattern {}, priority {}, symbol {})\n", impl.id, impl.pattern,
						impl.priority, impl.symbol
				);
		}
		for (const auto& [hidden, by] : found.hidden)
			std::cout << std::format(
					"  hidden: {} {} in {} (by the package in {})\n", hidden.name, hidden.version,
					hidden.directory.string(), by.string()
			);

		if (!found.problems.empty()) {
			std::cout << "\nProblems (these packages are skipped):\n";
			for (const auto& problem : found.problems)
				std::cout << "  " << problem.str() << "\n";
		}
		return found.problems.empty() ? 0 : 1;
	}
} // namespace

int main(int argc, char* argv[]) {
	CLI::App app{"mdl-kernels -- inspects the kernel packages of the kernel directories"};
	app.set_version_flag("--version", std::string{mdl::version});
	app.require_subcommand(1);

	std::vector<std::filesystem::path> explicitDirs;
	app.add_option(
			"--kernels-dir", explicitDirs,
			"A kernel directory (searched first; repeatable). Then: the project's .mdl/kernels, the user's directory, "
			"MDL_KERNELS_PATH, and the installation's share/mdl/kernels."
	);

	std::optional<std::filesystem::path> source;
	auto* listCommand = app.add_subcommand("list", "Lists the kernel directories, packages, and their problems");
	listCommand->add_option("source", source, "A source file: its project's .mdl/kernels is searched too");

	// check and dump compile the model like mdlc (with the same options) up to the matching boundary.
	CompileArgs checkArgs, dumpArgs;
	bool explainAll = false;
	auto* checkCommand = app.add_subcommand(
			"check", "Matches the kernel packages against a model: the selected implementations (no code is generated)"
	);
	auto* checkConfig = addCompileOptions(*checkCommand, checkArgs);
	checkCommand->add_flag(
			"--explain", explainAll,
			"Show every candidate with the verdict of each constraint and why it is (not) selected"
	);
	auto* dumpCommand = app.add_subcommand(
			"dump", "Prints the model's Relax module at the matching boundary (what patterns match)"
	);
	auto* dumpConfig = addCompileOptions(*dumpCommand, dumpArgs);

	try {
		app.parse(argc, argv);
	} catch (const CLI::ParseError& e) {
		return app.exit(e);
	}
	if (listCommand->parsed())
		return list(explicitDirs, source);
	auto inspect = [&](CompileArgs& args, CLI::Option* config, mdl::CompileOptions::KernelInspection mode) {
		if (config->count() > 0)
			args.argvals = config->as<std::string>();
		args.compileOpts.kernelDirs.insert(
				args.compileOpts.kernelDirs.begin(), explicitDirs.begin(), explicitDirs.end()
		);
		args.compileOpts.inspectKernels = mode;
		args.compileOpts.inspectOut = &std::cout;
		try {
			runCompileCommand(args);
		} catch (const std::exception& e) {
			std::cerr << "mdl-kernels: error: " << e.what() << "\n";
			return 1;
		}
		return 0;
	};
	using Inspection = mdl::CompileOptions::KernelInspection;
	if (checkCommand->parsed())
		return inspect(checkArgs, checkConfig, explainAll ? Inspection::explain : Inspection::brief);
	if (dumpCommand->parsed())
		return inspect(dumpArgs, dumpConfig, Inspection::dump);
	return 0;
}
