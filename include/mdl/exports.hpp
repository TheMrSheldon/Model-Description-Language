#ifndef MDL_EXPORTS_HPP
#define MDL_EXPORTS_HPP

#include "./computegraphv2/datatype.hpp"
#include "./computegraphv2/node.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mdl {
	struct CompilationUnit;
	class ArgVals;

	/** @brief One tensor in the signature of an exported function. */
	struct TensorSignature {
		std::string name;
		PrimitiveType dtype;
		/** @brief Concrete shape; -1 marks a dimension that is only known at runtime. */
		std::vector<int64_t> shape;
		/** @brief The graph node (nullptr for an argument that the function does not use). */
		const Node* node;
	};

	/**
	 * @brief A function that is compiled and exported by the backend.
	 * @details The exported (DPS) calling convention is: inputs (in declaration order), then weights (in canonical
	 * order, see mdl::weightPaths), then preallocated outputs. Backends and binding generators must both use this
	 * description so that the generated bindings match the compiled code.
	 */
	struct ExportedFunction {
		std::string module;	  ///< Name of the module that defines the function
		std::string function; ///< Name of the function within the module
		std::string symbol;	  ///< Mangled name, see mdl::mangledName
		const FunctionDecl* decl;
		std::vector<TensorSignature> inputs;
		std::vector<TensorSignature> weights; ///< Only the weights that the function uses
		std::vector<TensorSignature> outputs;
		/** @brief All weights of the module (not only those used by this function) in canonical order. */
		std::vector<std::string> moduleWeights;
	};

	/**
	 * @brief Lists the functions that are compiled: all non-template member functions of modules whose
	 * hyperparameters are all given in @p argvals.
	 */
	std::vector<ExportedFunction> exportedFunctions(const CompilationUnit& unit, const ArgVals& argvals);

	/**
	 * @brief Evaluates a compile-time constant (integer) expression, looking up hyperparameters in @p argvals.
	 * @return the value or std::nullopt if the expression is not a constant integer expression.
	 */
	std::optional<int64_t> evalConstant(const Node& node, const ArgVals& argvals);
} // namespace mdl

#endif
