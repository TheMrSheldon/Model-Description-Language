#include "./argvals.hpp"

#include <yaml-cpp/yaml.h>

using mdl::ArgVals;
using ErrorCode = mdl::ArgVals::ErrorCode;

ArgVals::ArgVals(YAML::Node&& data) : data(std::move(data)) {}

ArgVals ArgVals::fromYAML(const std::filesystem::path& path) { return ArgVals{YAML::LoadFile(path.string())}; }

std::expected<int64_t, ErrorCode> ArgVals::lookup(std::string_view path) const {
	if (!data.IsDefined() || path.empty())
		return std::unexpected{ErrorCode::NotFound};
	YAML::Node node = data;
	while (!path.empty()) {
		auto dot = path.find('.');
		std::string key{path.substr(0, dot)};
		YAML::Node child = std::as_const(node)[key];
		if (!node.IsMap() || !child)
			return std::unexpected{ErrorCode::NotFound};
		// Use reset() to navigate without mutating the underlying tree.
		// operator= calls AssignNode which calls set_ref, corrupting the shared YAML tree.
		node.reset(child);
		if (dot == std::string_view::npos)
			break;
		path = path.substr(dot + 1);
	}
	if (!node.IsScalar())
		return std::unexpected{ErrorCode::InvalidDataType};
	return node.as<int64_t>();
}