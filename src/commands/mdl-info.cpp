#include "common.hpp"
#include <mdl/version.hpp>

#include <tvm/target/tag.h>
#include <tvm/target/target_kind.h>

struct InfoArgs {
	LoggerConf logConf;
};

static void runInfoCommand(const InfoArgs& args) {
	mdl::log::setVerbosity(args.logConf.getVerbosity());
	std::cout << "Target Kinds:" << std::endl;
	for (auto&& name : tvm::TargetKindRegEntry::ListTargetKinds())
		std::cout << "  - " << name << /*(tvm::runtime::RuntimeEnabled(name) ? "enabled" : "disabled") <<*/ std::endl;
	std::cout << "Target Tags:" << std::endl;
	for (auto&& [name, target] : tvm::TargetTag::ListTags()) {
		std::cout << "  - " << name << std::endl;
	}
}

int main(int argc, char* argv[]) {
	CLI::App app("TODO");
	app.set_version_flag("--version", std::string{mdl::version});

	InfoArgs args;
	setupLoggerArgs(app, args.logConf);
	app.callback([&args]() { runInfoCommand(args); });

	CLI11_PARSE(app, argc, argv);
	return 0;
}