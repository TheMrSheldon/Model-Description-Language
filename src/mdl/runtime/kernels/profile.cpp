// Kernel profiling for models compiled with `mdlc --profile`: the generated entry functions call
// mdl_profile_begin/end around every kernel call. The times are summed per kernel kind (the kernel name without its
// numeric suffix) and printed to stderr when the program exits.
#include <mdl/runtime/kernels.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {
	using Clock = std::chrono::steady_clock;

	struct Entry {
		long long calls = 0;
		double seconds = 0;
	};

	struct Profile {
		std::mutex mutex;
		std::map<std::string, Entry> entries;
		Clock::time_point start;

		static Profile& instance() {
			// Never destroyed: the summary is printed by an exit handler, after static destructors may have run.
			static Profile* profile = [] {
				auto* p = new Profile;
				std::atexit([] { instance().print(); });
				return p;
			}();
			return *profile;
		}

		void print() {
			std::vector<std::pair<std::string, Entry>> sorted(entries.begin(), entries.end());
			std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second.seconds > b.second.seconds; });
			double total = 0;
			for (const auto& [name, e] : sorted)
				total += e.seconds;
			std::fprintf(stderr, "\nmdl kernel profile (total %.3f s):\n%10s %7s %8s  %s\n", total, "seconds", "share", "calls", "kernel");
			for (const auto& [name, e] : sorted)
				std::fprintf(stderr, "%10.3f %6.1f%% %8lld  %s\n", e.seconds, total > 0 ? 100 * e.seconds / total : 0.0, e.calls, name.c_str());
		}
	};

	/** @brief The kernel kind: the name without the symbol prefix (`mdl_<model>_k_`) and the numeric suffix. */
	std::string kind(const char* name) {
		std::string s = name;
		if (auto k = s.find("_k_"); k != std::string::npos)
			s = s.substr(k + 3);
		while (!s.empty() && std::isdigit(static_cast<unsigned char>(s.back())))
			s.pop_back();
		return s;
	}
} // namespace

extern "C" int mdl_profile_begin(const char* /*name*/) {
	Profile::instance().start = Clock::now();
	return 0;
}

extern "C" int mdl_profile_end(const char* name) {
	auto& profile = Profile::instance();
	const double seconds = std::chrono::duration<double>(Clock::now() - profile.start).count();
	std::lock_guard lock{profile.mutex};
	auto& e = profile.entries[kind(name)];
	++e.calls;
	e.seconds += seconds;
	return 0;
}
