// Operating-system specific helpers (kept in their own file: <windows.h> defines many macros).
#include <mdl/utils/platform.hpp>

#include <cstdint>
#include <string>
#include <system_error>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

std::filesystem::path mdl::utils::executablePath() {
#if defined(_WIN32)
	std::wstring buffer(32768, L'\0');
	const DWORD n = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
	return n > 0 ? std::filesystem::path{buffer.substr(0, n)} : std::filesystem::path{};
#elif defined(__APPLE__)
	uint32_t size = 0;
	_NSGetExecutablePath(nullptr, &size);
	std::string buffer(size, '\0');
	if (_NSGetExecutablePath(buffer.data(), &size) != 0)
		return {};
	return std::filesystem::path{buffer.c_str()};
#else
	std::error_code ec;
	auto path = std::filesystem::read_symlink("/proc/self/exe", ec);
	return ec ? std::filesystem::path{} : path;
#endif
}
