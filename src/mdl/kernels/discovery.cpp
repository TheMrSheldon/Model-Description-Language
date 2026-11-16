// Kernel directories, package discovery, and platform compatibility (see docs/kernels.md).
#include <mdl/kernels.hpp>
#include <mdl/utils/platform.hpp>

#include <algorithm>
#include <cstdlib>
#include <format>
#include <ranges>
#include <set>
#include <sstream>

#if defined(__GLIBC__)
#include <gnu/libc-version.h>
#endif

namespace mdl::kernels {

	std::string str(DirOrigin origin) {
		switch (origin) {
		case DirOrigin::explicitDir:
			return "--kernels-dir";
		case DirOrigin::project:
			return "project";
		case DirOrigin::user:
			return "user";
		case DirOrigin::global:
			return "global";
		}
		return "";
	}

	namespace {
		std::optional<std::filesystem::path> environment(const char* name) {
			const char* value = std::getenv(name);
			if (value == nullptr || *value == '\0')
				return std::nullopt;
			return std::filesystem::path{value};
		}

		std::vector<std::filesystem::path> searchPath(const char* name) {
#ifdef _WIN32
			constexpr char separator = ';';
#else
			constexpr char separator = ':';
#endif
			std::vector<std::filesystem::path> paths;
			if (const char* value = std::getenv(name)) {
				std::stringstream list{value};
				std::string entry;
				while (std::getline(list, entry, separator))
					if (!entry.empty())
						paths.emplace_back(entry);
			}
			return paths;
		}

		/** @brief The project's `.mdl/kernels`, searched upward from the directory of @p source. */
		std::optional<std::filesystem::path> projectKernelDir(const std::filesystem::path& source) {
			std::error_code ec;
			auto dir = std::filesystem::absolute(source, ec).parent_path();
			while (!ec && !dir.empty()) {
				auto candidate = dir / ".mdl" / "kernels";
				std::error_code missing; // (a candidate that does not exist is no error)
				if (std::filesystem::is_directory(candidate, missing))
					return candidate;
				if (dir == dir.root_path())
					break;
				dir = dir.parent_path();
			}
			return std::nullopt;
		}
	} // namespace

	std::optional<std::filesystem::path> userKernelDir() {
#if defined(_WIN32)
		if (auto local = environment("LOCALAPPDATA"))
			return *local / "mdl" / "kernels";
		return std::nullopt;
#elif defined(__APPLE__)
		if (auto home = environment("HOME"))
			return *home / "Library" / "Application Support" / "mdl" / "kernels";
		return std::nullopt;
#else
		if (auto data = environment("XDG_DATA_HOME"))
			return *data / "mdl" / "kernels";
		if (auto home = environment("HOME"))
			return *home / ".local" / "share" / "mdl" / "kernels";
		return std::nullopt;
#endif
	}

	std::vector<KernelDir> kernelDirs(
			const std::vector<std::filesystem::path>& explicitDirs, const std::optional<std::filesystem::path>& source
	) {
		std::vector<KernelDir> candidates;
		for (const auto& d : explicitDirs)
			candidates.push_back({d, DirOrigin::explicitDir});
		if (source)
			if (auto project = projectKernelDir(*source))
				candidates.push_back({*project, DirOrigin::project});
		if (auto user = userKernelDir())
			candidates.push_back({*user, DirOrigin::user});
		for (const auto& d : searchPath("MDL_KERNELS_PATH"))
			candidates.push_back({d, DirOrigin::global});
		// An installation: <prefix>/bin/<tool> and <prefix>/share/mdl/kernels.
		if (auto exe = utils::executablePath(); !exe.empty())
			candidates.push_back({exe.parent_path().parent_path() / "share" / "mdl" / "kernels", DirOrigin::global});
#ifdef MDL_DEFAULT_KERNELS_DIR
		candidates.push_back({MDL_DEFAULT_KERNELS_DIR, DirOrigin::global}); // the source tree (running from the build)
#endif
		// Only existing directories, each once (the first occurrence counts).
		std::vector<KernelDir> dirs;
		std::set<std::filesystem::path> seen;
		for (auto& c : candidates) {
			std::error_code ec;
			if (!std::filesystem::is_directory(c.path, ec))
				continue;
			auto canonical = std::filesystem::weakly_canonical(c.path, ec);
			if (ec || !seen.insert(canonical).second)
				continue;
			dirs.push_back({canonical, c.origin});
		}
		return dirs;
	}

	Discovery discover(const std::vector<KernelDir>& dirs) {
		Discovery result{.dirs = dirs, .packages = {}, .hidden = {}, .problems = {}};
		for (const auto& dir : dirs) {
			// The packages of one directory, in a deterministic order.
			std::vector<std::filesystem::path> manifests;
			std::error_code ec;
			for (const auto& entry : std::filesystem::directory_iterator(dir.path, ec))
				if (entry.is_directory(ec) && std::filesystem::is_regular_file(entry.path() / "package.yml", ec))
					manifests.push_back(entry.path() / "package.yml");
			std::ranges::sort(manifests);
			for (const auto& manifest : manifests) {
				auto package = loadPackage(manifest, result.problems);
				if (!package)
					continue;
				if (package->name != manifest.parent_path().filename().string()) {
					result.problems.push_back(
							Problem{.file = manifest,
									.line = 0,
									.column = 0,
									.message = std::format(
											"the package '{}' must be in a directory of the same name (not '{}')",
											package->name, manifest.parent_path().filename().string()
									)}
					);
					continue;
				}
				auto earlier = std::ranges::find(result.packages, package->name, &Package::name);
				if (earlier != result.packages.end())
					result.hidden.emplace_back(std::move(*package), earlier->directory);
				else
					result.packages.push_back(std::move(*package));
			}
		}
		return result;
	}

	Platform hostPlatform() {
		Platform p;
#if defined(_WIN32)
		p.os = "windows";
#elif defined(__APPLE__)
		p.os = "darwin";
#else
		p.os = "linux";
#endif
#if defined(__x86_64__) || defined(_M_X64)
		p.arch = "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
		p.arch = "aarch64";
#endif
#if defined(_MSC_VER)
		p.abi = "msvc";
#elif defined(__APPLE__)
		p.abi = "darwin";
#else
		p.abi = "gnu";
#endif
#if defined(__GLIBC__)
		p.libc = gnu_get_libc_version();
#endif
		return p;
	}

	namespace {
		/** @brief `2.28` -> {2, 28}. */
		std::vector<int> versionParts(std::string_view version) {
			std::vector<int> parts;
			for (auto part : std::views::split(version, '.')) {
				int value = 0;
				for (char c : part) {
					if (c < '0' || c > '9')
						break;
					value = value * 10 + (c - '0');
				}
				parts.push_back(value);
			}
			return parts;
		}
	} // namespace

	bool compatible(const Platform& required, const Platform& host) {
		if (required.os != host.os || required.arch != host.arch)
			return false;
		if (!required.abi.empty() && required.abi != host.abi)
			return false;
		if (required.libc.starts_with(">=")) {
			if (host.libc.empty())
				return false; // a glibc requirement on a system without glibc
			if (versionParts(host.libc) < versionParts(std::string_view{required.libc}.substr(2)))
				return false;
		}
		return true;
	}

} // namespace mdl::kernels
