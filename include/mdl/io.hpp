#ifndef MDL_IO_HPP
#define MDL_IO_HPP

#include "computegraph.hpp"

#include <filesystem>
#include <iostream>

namespace mdl::io {
	bool writeToFile(const std::filesystem::path& path, const mdl::Computegraph& graph);
	bool writeToStream(const std::ostream& stream, const mdl::Computegraph& graph);

	bool readFromFile(const std::filesystem::path& path, mdl::Computegraph& graph);
	bool readFromStream(const std::ostream& stream, mdl::Computegraph& graph);
} // namespace mdl::io

#endif