#ifndef MDL_COMPUTEGRAPH_OBJECTDECL_HPP
#define MDL_COMPUTEGRAPH_OBJECTDECL_HPP

#include "./node_decl.hpp"

#include <variant>

namespace mdl {
	struct AggregateInstance;
	struct ModuleInstance;
	struct FunctionDecl;

	/**
	 * @brief A first-class value in the MDL language.
	 * @details Either a compute-graph node (const Node*), a function definition
	 *          (FunctionDecl), or a struct instance (AggregateInstance).
	 */
	using Object = std::variant<const Node*, FunctionDecl, AggregateInstance, ModuleInstance>;
} // namespace mdl

#endif
