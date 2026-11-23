#include "./argvals.hpp"

#include <yaml-cpp/yaml.h>

#include <charconv>

using mdl::ArgVals;
using ErrorCode = mdl::ArgVals::ErrorCode;

ArgVals::ArgVals(YAML::Node&& data) : data(std::move(data)) {}

ArgVals ArgVals::fromYAML(const std::filesystem::path& path) {
	// The hyperparameters are in `args:` (or, in the older layout, at the top level next to the build settings).
	YAML::Node root = YAML::LoadFile(path.string());
	if (root.IsMap() && root["args"])
		return ArgVals{YAML::Node(root["args"])};
	return ArgVals{std::move(root)};
}

YAML::Node ArgVals::navigate(std::string_view path) const {
	if (!data.IsDefined() || path.empty())
		return YAML::Node{YAML::NodeType::Undefined};
	YAML::Node node = data;
	while (!path.empty()) {
		auto dot = path.find('.');
		const std::string_view key = path.substr(0, dot);
		// (yaml-cpp: operator= assigns through the reference, reset() rebinds but throws for invalid nodes, and a
		// missing key yields an invalid node, which copying and operator! handle.)
		size_t index = 0;
		bool isIndex = false;
		if (node.IsSequence()) {
			auto [end, ec] = std::from_chars(key.data(), key.data() + key.size(), index);
			isIndex = ec == std::errc{} && end == key.data() + key.size() && index < node.size();
			if (!isIndex)
				return YAML::Node{YAML::NodeType::Undefined};
		} else if (!node.IsMap()) {
			return YAML::Node{YAML::NodeType::Undefined};
		}
		const YAML::Node child = isIndex ? std::as_const(node)[index] : std::as_const(node)[std::string{key}];
		if (!child)
			return YAML::Node{YAML::NodeType::Undefined};
		// Use reset() to navigate without mutating the underlying tree (operator= would corrupt the shared tree).
		node.reset(child);
		if (dot == std::string_view::npos)
			break;
		path = path.substr(dot + 1);
	}
	return node;
}

bool ArgVals::contains(std::string_view path) const { return navigate(path).IsDefined(); }

std::expected<size_t, ErrorCode> ArgVals::listSize(std::string_view path) const {
	YAML::Node node = navigate(path);
	if (!node.IsDefined())
		return std::unexpected{ErrorCode::NotFound};
	if (!node.IsSequence())
		return std::unexpected{ErrorCode::InvalidDataType};
	return node.size();
}

std::expected<ArgVals::Scalar, ErrorCode> ArgVals::lookupScalar(std::string_view path) const {
	YAML::Node node = navigate(path);
	if (!node.IsDefined())
		return std::unexpected{ErrorCode::NotFound};
	if (!node.IsScalar())
		return std::unexpected{ErrorCode::InvalidDataType};
	try {
		return Scalar{node.as<int64_t>()};
	} catch (const YAML::Exception&) {
	}
	try {
		return Scalar{node.as<double>()};
	} catch (const YAML::Exception&) {
	}
	try {
		return Scalar{node.as<bool>()};
	} catch (const YAML::Exception&) {
	}
	return std::unexpected{ErrorCode::InvalidDataType};
}

std::expected<int64_t, ErrorCode> ArgVals::lookup(std::string_view path) const {
	YAML::Node node = navigate(path);
	if (!node.IsDefined())
		return std::unexpected{ErrorCode::NotFound};
	if (!node.IsScalar())
		return std::unexpected{ErrorCode::InvalidDataType};
	try {
		return node.as<int64_t>();
	} catch (const YAML::Exception&) {
		return std::unexpected{ErrorCode::InvalidDataType};
	}
}
