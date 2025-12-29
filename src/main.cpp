#include "mdl/logging.hpp"

#include <mdl/parser.hpp>
#include <mdl/compiler.hpp>

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
	bool generateHeaders;
};

template <class... Ts>
struct overloaded : Ts... {
	using Ts::operator()...;
};

static void print(const Struct& s) { std::cout << "Struct " << s.name << std::endl; }

static void print(const Node* node, std::string prefix = "") {
	if (node == nullptr)
		std::cout << "Null" << std::endl;
	else {
		std::cout << prefix << std::to_underlying(node->operation) << std::endl;
		std::visit(overloaded{
			[prefix](const std::variant<double, int>& c){std::visit([prefix](auto x){ std::cout << prefix << ':' << x << std::endl;}, c);},
			[prefix](const std::vector<const Node*>& nodes){
				for (const auto& n : nodes)
					print(n, prefix+" ");
			}
		}, node->value);
	}
}

static void print(const Module& m) {
	std::cout << "== Module " << m.name << " ==" << std::endl;
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
		for (const auto& arg : f.arguments)
			std::cout << arg.name << std::endl;
		std::cout << "Num Nodes: " << f.graph->getNumNodes() << std::endl;
		print(f.graph->getOutputNode());
	}
	std::cout << "=============" << std::endl;
}

static void runCompileCommand(const CompileArgs& args) {
	mdl::log::setVerbosity(args.logConf.getVerbosity());
	std::filesystem::path output = args.output.value_or("a.o"); /** \todo should be a shared object? **/
	/** \todo: handle multiple input files */
	auto program = mdl::parser::parse(args.inputs[0]);
	for (const auto& [key, value] : program.definitions) {
		std::visit(overloaded{[](const Struct& s) { print(s); }, [](const Module& m) { print(m); }}, value);
	}

	const Module& module = std::get<Module>(program.definitions["Linear"]); /** \todo from module **/
	
	mdl::compileToFile(module, output);
}

int main(int argc, char* argv[]) {
	CLI::App app("TODO");

	CompileArgs args;
	setupLoggerArgs(app, args.logConf);
	app.add_option("input", args.inputs, "The files to be compiled")->check(CLI::ExistingFile)->required();
	app.add_option("-o,--out", args.output, "The output path.");
	app.add_flag("--generate-headers", args.generateHeaders, "If set, a header file will be generated.");
	app.callback([&args]() { runCompileCommand(args); });

	CLI11_PARSE(app, argc, argv);
	return 0;
}