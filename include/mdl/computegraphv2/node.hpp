#ifndef MDL_COMPUTEGRAPH_NODE_HPP
#define MDL_COMPUTEGRAPH_NODE_HPP

#include "./builtins.hpp"
#include "./datatype.hpp"
#include "./node_decl.hpp"
#include "./object.hpp"

namespace mdl {

	namespace details {
		struct BaseNode {
		private:
			DataType type;
			bool constevalable;

		protected:
			BaseNode() = default;

		public:
			/**
			 * @brief Get the data type of this node.
			 */
			const DataType& datatype() const { return type; }

			/**
			 * @brief Whether this node can be evaluated at compile time.
			 */
			bool isConsteval() const { return constevalable; }
		};
	} // namespace details

	/**
     * @brief "Const nodes" represent constant values in the compute graph. They consist of a data type and an atomic value.
     * @see Atomic
     */
	struct ConstNode {
		mdl::ObjectRef value;

		/**
		 * @brief Get the data type of this node.
		 */
		const DataType& datatype() const { return value.type; }

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
	struct InvocationNode : details::BaseNode {
		mdl::builtins::FunctionIDs id;
		std::vector<std::reference_wrapper<Node>> arguments;
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
	struct ParameterNode : details::BaseNode {
		std::string name;
	};
	/**
     * @brief "Hyperparameter nodes" repreesent "template parameters", i.e., compile-time constant variables (e.g., module arguments and function template arguments).
     */
	struct HyperparameterNode : details::BaseNode {
		std::string name;
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