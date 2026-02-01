#ifndef MDL_ARGVALS_HPP
#define MDL_ARGVALS_HPP

#include <filesystem>

namespace mdl {
	/**
     * Stores the values for model arguments (hyperparameters).
     */
	class ArgVals {
	private:
		ArgVals() = default;

	public:
		static ArgVals fromYAML(const std::filesystem::path& path);
	};
} // namespace mdl

#endif