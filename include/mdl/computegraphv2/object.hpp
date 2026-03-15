#ifndef MDL_COMPUTEGRAPH_OBJECT_HPP
#define MDL_COMPUTEGRAPH_OBJECT_HPP

#include "./function.hpp"
#include "./node_decl.hpp"

#include <optional>
#include <string_view>
#include <variant>
#include <vector>

namespace mdl {
	using Object = std::variant<const Node*, FunctionDecl>;

	std::optional<std::reference_wrapper<const FunctionDecl>>
	getMember(const Object& obj, std::string_view name, const std::vector<DataType>& argtypes);
}; // namespace mdl

#endif