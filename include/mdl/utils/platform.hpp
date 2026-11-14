#ifndef MDL_UTILS_PLATFORM_HPP
#define MDL_UTILS_PLATFORM_HPP

#include <filesystem>

namespace mdl::utils {
	/** @brief The path of the running executable (empty if it cannot be determined). */
	std::filesystem::path executablePath();
} // namespace mdl::utils

#endif
