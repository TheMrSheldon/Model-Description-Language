// mdls: the language server of the model description language (Language Server Protocol over stdin/stdout). The
// editor starts it as a child process; it exits when the editor sends `exit` or closes stdin.
#include "mdls/server.hpp"

#include <CLI/CLI.hpp>

#include <fstream>
#include <iostream>
#include <memory>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

int main(int argc, char* argv[]) {
	CLI::App app("mdls -- language server for the model description language (LSP over stdin/stdout)");
	std::string logFile;
	app.add_option("--log", logFile, "Write a log to this file.");
	app.add_flag("--stdio", "Communicate over stdin/stdout (the default; accepted for compatibility with clients).");
	CLI11_PARSE(app, argc, argv);
#if defined(_WIN32)
	// The protocol counts bytes: no newline translation.
	_setmode(_fileno(stdin), _O_BINARY);
	_setmode(_fileno(stdout), _O_BINARY);
#endif
	std::ios::sync_with_stdio(false);
	std::unique_ptr<std::ofstream> log;
	if (!logFile.empty())
		log = std::make_unique<std::ofstream>(logFile);
	mdls::Server server{std::cin, std::cout, log.get()};
	return server.run();
}
