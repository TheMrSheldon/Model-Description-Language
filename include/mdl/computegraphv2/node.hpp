#ifndef MDL_COMPUTEGRAPH_NODE_HPP
#define MDL_COMPUTEGRAPH_NODE_HPP

#include "./builtins.hpp"
#include "./datatype.hpp"
#include "./node_decl.hpp"

#include <optional>

namespace mdl {
	/**
     * @brief "Const nodes" represent constant values in the compute graph.
     * @details The value is an Atomic: a scalar (int/double) or a compile-time integer
     *          tensor (std::vector<int>, e.g. permutation indices).
     * @see Atomic
     */
	struct ConstNode {
		mdl::Atomic value;
		mutable std::optional<DataType> _cachedType; ///< for non-scalar Atomic variants

		/**
		 * @brief Get the data type of this node.
		 */
		const DataType& datatype() const {
			static DataType inttype = PrimitiveType::integer;
			static DataType floattype = PrimitiveType::floating;
			return std::visit(
					mdl::overloaded{
							[&](const int&) -> const DataType& { return inttype; },
							[&](const double&) -> const DataType& { return floattype; },
							[&](const std::vector<int>& v) -> const DataType& {
								if (!_cachedType)
									_cachedType = TensorType{
											PrimitiveType::integer,
											{static_cast<unsigned>(v.size())}};
								return *_cachedType;
							},
					},
					value
			);
		}

		/**
		 * @brief Whether this node can be evaluated at compile time.
		 * @details For ConstNode this is always true as it represents a compiletime-constant.
		 */
		bool isConsteval() const { return true; }
	};
	/**
     * @brief "Invocation nodes" represent function calls in the compute graph. They consist of a function reference and a list of argument nodes.
     * @details Member functions and operators are represented as invocation nodes with a function reference to the corresponding function and the object on which the member function is called as the first argument.
     */
	struct InvocationNode {
		DataType type;
		bool constevalable;
		mdl::builtins::FunctionIDs id;
		std::vector<std::reference_wrapper<const Node>> arguments;

		/**
		 * @brief Get the data type of this node.
		 */
		const DataType& datatype() const { return type; }

		/**
		 * @brief Whether this node can be evaluated at compile time.
		 */
		bool isConsteval() const { return constevalable; }
	};
	/**
     * @brief "Error nodes" represent invalid nodes in the compute graph (e.g., due to semantic errors).
     * @details Computegraphs with these nodes should not be processed further and should be rejected by the compiler. These are used to simplify error handling within the compiler, as they can be generated at any point during graph construction.
     */
	struct ErrorNode {
		/**
		 * @brief Get the data type of this node.
		 */
		const DataType& datatype() const {
			static DataType error{mdl::PrimitiveType::error};
			return error;
		}

		/**
		 * @brief Whether this node can be evaluated at compile time.
		 */
		bool isConsteval() const { return true; }
	};
	/**
     * @brief "Parameter nodes" represent member variables.
     */
	struct ParameterNode {
		DataType type;
		bool constevalable;
		std::string name;

		/**
		 * @brief Get the data type of this node.
		 */
		const DataType& datatype() const { return type; }

		/**
		 * @brief Whether this node can be evaluated at compile time.
		 */
		bool isConsteval() const { return constevalable; }
	};
	/**
     * @brief "Hyperparameter nodes" repreesent "template parameters", i.e., compile-time constant variables (e.g., module arguments and function template arguments).
     */
	struct HyperparameterNode {
		DataType type;
		bool constevalable;
		std::string name;

		/**
		 * @brief Get the data type of this node.
		 */
		const DataType& datatype() const { return type; }

		/**
		 * @brief Whether this node can be evaluated at compile time.
		 */
		bool isConsteval() const { return constevalable; }
	};

	const mdl::DataType& getDataType(const mdl::ConstNode& node);
	const mdl::DataType& getDataType(const mdl::InvocationNode& node);
	const mdl::DataType& getDataType(const mdl::ErrorNode& node);
	const mdl::DataType& getDataType(const mdl::ParameterNode& node);
	const mdl::DataType& getDataType(const mdl::HyperparameterNode& node);
	/**
     * @brief Get the data type of a node.
     */
	const mdl::DataType& getDataType(const Node& node);
} // namespace mdl

#endif