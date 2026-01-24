#include "commands/common.hpp"

#include <mdl/compiler.hpp>
#include <mdl/parser.hpp>
#include <mdl/utils/utils.hpp>

#include <iostream>
#include <optional>

struct CompileArgs {
	std::vector<std::filesystem::path> inputs;
	std::optional<std::filesystem::path> output;
	std::optional<std::filesystem::path> argvals;
	LoggerConf logConf;
	bool generateHeaders;
};

static void print(const Struct& s) { std::cout << "Struct " << s.name << std::endl; }

static void print(const Node* node, std::string prefix = "") {
	if (node == nullptr)
		std::cout << "Null" << std::endl;
	else {
		std::cout << prefix << std::to_underlying(mdl::getOperation(*node)) << std::endl;
		switch (mdl::getOperation(*node)) {
		case Operation::Input:
		case Operation::Argument:
			std::cout << prefix << "Input/Arg: " << mdl::getName(*node) << ": "
					  << std::to_string(mdl::getDataType(*node)) << std::endl;
			break;
		case Operation::Const:
			std::visit([prefix](auto x) { std::cout << prefix << ':' << x << std::endl; }, mdl::getConst(*node));
			break;
		default:
			for (const auto& n : mdl::getInputs(*node))
				print(n, prefix + " ");
			break;
		}
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
	std::filesystem::path output = args.output.value_or("a.o");
	/** \todo: handle multiple input files */
	auto program = mdl::parser::parse(args.inputs[0]);
	for (const auto& [key, value] : program.definitions) {
		std::visit(mdl::overloaded{[](const Struct& s) { print(s); }, [](const Module& m) { print(m); }}, value);
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
	app.add_option("-a,--arg-vals", args.argvals, "Argument values.")->check(CLI::ExistingFile);
	app.add_flag("--generate-headers", args.generateHeaders, "If set, a header file will be generated.");
	app.callback([&args]() { runCompileCommand(args); });

	CLI11_PARSE(app, argc, argv);
	return 0;
}