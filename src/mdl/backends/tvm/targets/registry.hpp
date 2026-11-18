#pragma once

/**
 * @file registry.hpp
 * @brief The targets by name: the kind (the first word of a target string, e.g. `cuda` in `cuda -arch=sm_80`)
 * selects the implementation, which is created from the whole string.
 */

#include "./target.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mdl::backend::tvm {

	/**
	 * @brief Creates the target of @p opts (CompileOptions::target).
	 * @throws std::runtime_error if the kind is unknown, or the target is not supported by this build (UnsupportedTarget).
	 */
	std::unique_ptr<Target> makeTarget(const mdl::CompileOptions& opts);

	/** @brief The registered kinds (including those this build does not support). */
	std::vector<std::string> targetKinds();

} // namespace mdl::backend::tvm
