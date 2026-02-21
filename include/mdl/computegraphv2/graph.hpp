#ifndef MDL_COMPUTEGRAPH_GRAPH_HPP
#define MDL_COMPUTEGRAPH_GRAPH_HPP

#include "./datatype.hpp"

#include <string>
#include <vector>

namespace mdl {
	struct FunctionDef final {
		std::string name;
		DataType returnType;
		std::vector<std::pair<std::string, DataType>> templateParameters;
		std::vector<std::pair<std::string, DataType>> parameters;
		Computegraph* body;
	};

	struct ObjectRef final {};

	class Computegraph final {
	private:
	public:
	};
} // namespace mdl

#endif