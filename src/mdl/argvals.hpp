#ifndef MDL_ARGVALS_HPP
#define MDL_ARGVALS_HPP

#include <expected>
#include <filesystem>
#include <string_view>
#include <variant>

#include <yaml-cpp/yaml.h>

namespace mdl {
	/**
	 * Stores the values for model arguments (hyperparameters).
	 */
	class ArgVals {
	public:
		enum class ErrorCode { Ok, NotFound, InvalidDataType };

	private:
		ArgVals(YAML::Node&& data);

	public:
		ArgVals() = default;
		ArgVals(const ArgVals&) = default;
		ArgVals(ArgVals&&) = default;

		ArgVals& operator=(const ArgVals&) = default;
		ArgVals& operator=(ArgVals&&) = default;

		static ArgVals fromYAML(const std::filesystem::path& path);

		/** Looks up a dotted path (e.g. "TransformerEncoder.conf.dims.model"); nullopt if absent or non-integer. */
		std::expected<int64_t, ErrorCode> lookup(std::string_view path) const;

		using Scalar = std::variant<int64_t, double, bool>;
		/** Looks up a scalar (integer, floating-point, or boolean) at a dotted path. */
		std::expected<Scalar, ErrorCode> lookupScalar(std::string_view path) const;

	private:
		YAML::Node data;
	};
} // namespace mdl

#endif