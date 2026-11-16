// mdlc: the compiler for the model description language.
#include "compile_cli.hpp"

int main(int argc, char* argv[]) {
	CLI::App app("mdlc -- compiler for the model description language");
	app.set_version_flag("--version", std::string{mdl::version});

	CompileArgs args;
	auto* config = addCompileOptions(app, args);
	app.callback([&args, config]() {
		if (config->count() > 0)
			args.argvals = config->as<std::string>();
		try {
			runCompileCommand(args);
		} catch (const std::exception& e) { // errors of the compilation (not of the command line)
			std::cerr << "mdlc: error: " << e.what() << "\n";
			std::exit(1);
		}
	});

	CLI11_PARSE(app, argc, argv);
	return 0;
}
