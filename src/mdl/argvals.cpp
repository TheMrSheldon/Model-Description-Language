#include "./argvals.hpp"

#include <yaml-cpp/yaml.h>

#include <iostream>

using mdl::ArgVals;

ArgVals ArgVals::fromYAML(const std::filesystem::path& path) {
	YAML::Node config = YAML::LoadFile(path.string());
	// std::cout << config << std::endl;
	/** \todo implement **/
	return ArgVals{};
}