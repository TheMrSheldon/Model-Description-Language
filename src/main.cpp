#include "commands/common.hpp"
#include "mdl/argvals.hpp"

#include <mdl/compiler.hpp>
#include <mdl/parser.hpp>
#include <mdl/utils/utils.hpp>

#include <iostream>
#include <optional>

struct CompileArgs {
	std::vector<std::filesystem::path> inputs;
	std::vector<std::filesystem::path> importdirs;
	std::optional<std::filesystem::path> output;
	std::optional<std::filesystem::path> argvals;
	LoggerConf logConf;
	bool generateHeaders;
};

static void print(std::string_view name, const mdl::AggregateType& s) { std::cout << "Struct " << name << std::endl; }

static void print(const mdl::Node* node, std::string prefix = "") {
	if (node == nullptr)
		std::cout << "Null" << std::endl;
	else {
		/*std::cout << prefix << std::to_underlying(mdl::getOperation(*node)) << std::endl;
		switch (mdl::getOperation(*node)) {
		case mdl::Operation::Input:
		case mdl::Operation::Argument:
			std::cout << prefix << "Input/Arg: " << mdl::getName(*node) << ": "
					  << std::to_string(mdl::getDataType(*node)) << std::endl;
			break;
		case mdl::Operation::Const:
			std::visit([prefix](auto x) { std::cout << prefix << ':' << x << std::endl; }, mdl::getConst(*node));
			break;
		default:
			for (const auto& n : mdl::getInputs(*node))
				print(n, prefix + " ");
			break;
		}*/
		/** \todo **/
		std::cout << "TODO" << std::endl;
	}
}

static void print(std::string_view name, const mdl::ModuleType& m) {
	std::cout << "== Module " << name << " ==" << std::endl;
	std::cout << "Arguments:" << std::endl;
	for (const auto& [name, a] : m.typeargs) {
		std::cout << name << std::endl;
	}
	std::cout << "Variables:" << std::endl;
	for (const auto& [name, v] : m.variables) {
		std::cout << name << std::endl;
	}
	std::cout << "Functions:" << std::endl;
	for (const auto& [name, f] : m.functions) {
		std::cout << name << std::endl;
		std::cout << "Template parameters" << std::endl;
		for (const auto& [argname, arg] : f.templateParameters)
			std::cout << argname << std::endl;
		std::cout << "Parameters" << std::endl;
		for (const auto& [argname, arg] : f.parameters)
			std::cout << argname << std::endl;
		print(&f.output.get());
	}
	std::cout << "=============" << std::endl;
}

static void runCompileCommand(const CompileArgs& args) {
	mdl::log::setVerbosity(args.logConf.getVerbosity());

	auto argvals = args.argvals.transform(mdl::ArgVals::fromYAML);

	std::filesystem::path output = args.output.value_or("a.o");
	/** \todo: handle multiple input files */
	auto result = mdl::parser::parse(args.inputs[0]);
	if (!result.has_value())
		exit(1);
	auto& program = result.value();
	for (const auto& [key, value] : program.definitions) {
		std::visit(
				mdl::overloaded{
						[&](const mdl::AggregateType& s) { print(key, s); },
						[&](const mdl::ModuleType& m) { print(key, m); }
				},
				value
		);
	}
	std::cout << "Num Nodes: " << program.graph.getNumNodes() << std::endl;

	const mdl::Module& module = std::get<mdl::Module>(program.definitions["Linear"]); /** \todo from module **/

	mdl::compileToFile(module, output);
}

int main(int argc, char* argv[]) {
	CLI::App app("TODO");

	CompileArgs args;
	setupLoggerArgs(app, args.logConf);
	app.add_option("input", args.inputs, "The files to be compiled")->check(CLI::ExistingFile)->required();
	app.add_option("-o,--out", args.output, "The output path.");
	app.add_option("-i,--import-dir", args.importdirs, "Search paths to search through for import statements.")
			->check(CLI::ExistingDirectory);
	app.add_option("-a,--arg-vals", args.argvals, "Argument values.")->check(CLI::ExistingFile);
	/** \todo possible alternative name: --generate-bindings=<type>; with possible values: cpp, python, ... */
	app.add_flag("--generate-headers", args.generateHeaders, "If set, a header file will be generated.");
	app.callback([&args]() { runCompileCommand(args); });

	CLI11_PARSE(app, argc, argv);
	return 0;
}