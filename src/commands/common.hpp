#ifndef COMMANDS_COMMON_HPP
#define COMMANDS_COMMON_HPP

#include "../mdl/logging.hpp"

#include <CLI/CLI.hpp>

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
		auto verb = std::to_underlying(mdl::log::Verbosity::Warning) + verbosity;
		verb = std::min(verb, std::to_underlying(mdl::log::Verbosity::Trace));
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

#endif