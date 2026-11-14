#ifndef MDL_COMPUTEGRAPH_DATATYPE_HPP
#define MDL_COMPUTEGRAPH_DATATYPE_HPP

#include "../utils/overload.hpp"
#include "node_decl.hpp"
#include <algorithm>

#include <expected>
#include <format>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mdl {
	struct Elipsis {};

	enum class PrimitiveType {
		error,	 /**< used to mark invalid nodes in the computegraph (e.g., due to semantic errors) **/
		float32, /**< IEEE 754 single precision (`float32`; `float` by default, see the option for the default float) */
		integer, /**< represents arbitrary signed integers. The precise datatype is compile-time configurable. */
		uint,	 /**< represents arbitrary unsigned integers. The precise datatype is compile-time configurable. */
		boolean, /**< represents booleans. */
		float16, /**< IEEE 754 half precision (`float16`) */
		bfloat16, /**< bfloat16 (`bfloat16`): float32's exponent range with 8 bits of precision */
		int8	  /**< 8-bit integers: the storage of quantized weights (see builtins::FunctionIDs::DEQUANT) */
	};

	/** @brief Whether @p type is a floating-point type. */
	constexpr bool isFloat(PrimitiveType type) {
		return type == PrimitiveType::float32 || type == PrimitiveType::float16 || type == PrimitiveType::bfloat16;
	}

	/** @brief The size of an element of @p type in bytes (booleans: 1). */
	constexpr unsigned byteSize(PrimitiveType type) {
		return type == PrimitiveType::float16 || type == PrimitiveType::bfloat16 ? 2
			   : type == PrimitiveType::boolean || type == PrimitiveType::int8	 ? 1
																				 : 4;
	}

	struct TensorType final {
		using Shape = std::vector<std::variant<unsigned, Elipsis, const Node*>>;
		PrimitiveType type;
		Shape shape;
		/**
		 * @brief Whether the tensor is masked (`masked float[...]`): dense values plus a mask of the valid elements
		 * (like torch.masked). Operations on masked tensors yield masked tensors; see the masking pass.
		 */
		bool masked = false;
	};

	/** @brief The definition of a function (opaque to the IR; owned by the frontend), see FunctionDecl::templ. */
	struct FunctionTemplate;

	/**
	 * @brief Represents a function definition.
	 * @details Each function is associated with a computegraph from which it references nodes. A function may have
	 * multiple outputs; each output is a separate node of the graph.
	 */
	struct FunctionDecl final {
		std::string name; /** \todo possibly don't name the function here but only in members; */
		std::vector<std::pair<std::string, std::reference_wrapper<const HyperparameterNode>>> templateParameters;
		std::vector<std::pair<std::string, std::reference_wrapper<const InputNode>>> parameters;
		std::vector<std::reference_wrapper<const Node>> outputs;
		/**
		 * @brief The declared return types (one per output). Defines the exported ABI.
		 * @details Functions return tensors or scalars only.
		 */
		std::vector<std::variant<PrimitiveType, TensorType>> returnTypes = {};
		/** @brief Names of the outputs in the exported ABI (optional; e.g. set for gradient functions). */
		std::vector<std::string> outputNames = {};
		/**
		 * @brief Other definitions with the same name (overloads). A call uses the one whose parameter types match the
		 * arguments best: a `masked` parameter takes only masked tensors, a plain one any tensor.
		 */
		std::vector<FunctionDecl> overloads = {};
		/**
		 * @brief The definition this function is elaborated from. A call with argument types that differ from the
		 * declared ones (e.g. a masked tensor for a plain parameter) elaborates the body again for these types, so that
		 * the calls in the body see them as well (and e.g. choose masked overloads).
		 */
		std::shared_ptr<FunctionTemplate> templ = nullptr;
	};

	/**
	 * @brief The definition of a module (opaque to the IR; owned by the frontend), from which instances are expanded.
	 */
	struct ModuleTemplate;

	/**
	 * @brief A module type.
	 * @details Modules are templates over their hyperparameters (typeargs): the variables and functions are expanded
	 * per instantiation with concrete hyperparameter values (see CompilationUnit::instantiate). A module type without
	 * expanded members (but with a template) is the uninstantiated definition.
	 */
	struct ModuleType final {
		std::vector<std::pair<std::string, std::reference_wrapper<const HyperparameterNode>>> typeargs;
		std::vector<std::pair<std::string, std::reference_wrapper<const ParameterNode>>> variables;
		std::map<std::string, FunctionDecl> functions;
		/** @brief The definition this type is (or can be) expanded from. */
		std::shared_ptr<const ModuleTemplate> templ = nullptr;
		/** @brief Whether the members are expanded (false for the uninstantiated definition of a template). */
		bool instantiated = false;
		/**
		 * @brief The hyperparameter values of an instance (including defaults), flattened to (path, number) pairs,
		 * e.g. ("conf.dims.model", 768) or ("ch.2", 256) for arrays.
		 */
		std::vector<std::pair<std::string, std::variant<int64_t, double>>> hyperparameterValues;

		static constexpr size_t InvalidIdx = -1;

		size_t variableIdxByName(std::string_view name) const {
			auto it = std::find_if(std::begin(variables), std::end(variables), [&](const auto& p) {
				return std::get<0>(p) == name;
			});
			if (it == std::end(variables))
				return ModuleType::InvalidIdx;
			return std::distance(std::begin(variables), it);
		}
	};

	struct BoundModuleType final {
		ModuleType definition;
		// Named bindings: must match definition.typeargs length (when fully bound)
		std::map<std::string, const Node*> hyperparamValues;
	};

	/** \todo handle member functions? **/
	using DataType = std::variant<
			PrimitiveType, TensorType, struct AggregateType, ModuleType, BoundModuleType, struct ArrayType>;

	struct AggregateType final {
		std::map<std::string, DataType> members;
	};

	/**
	 * @brief A compile-time array of @p size values of type @p element (`array[uint; 5]`), e.g. a hyperparameter; not a
	 * tensor (its values are an ArrayInstance).
	 */
	struct ArrayType final {
		std::shared_ptr<const DataType> element;
		unsigned size = 0;
	};

	using TypeArg = std::variant<Elipsis, const Node*>; /** \todo use mdl::Object instead of Node*? */

	std::expected<DataType, std::string> setTypeArgs(const PrimitiveType& prim, const std::vector<TypeArg>& typeargs);

	std::expected<DataType, std::string> setTypeArgs(const TensorType& tensor, const std::vector<TypeArg>& typeargs);

	std::expected<DataType, std::string> setTypeArgs(const AggregateType& agg, const std::vector<TypeArg>& typeargs);

	std::expected<DataType, std::string> setTypeArgs(const ModuleType& mod, const std::vector<TypeArg>& typeargs);

	std::expected<DataType, std::string> setTypeArgs(const BoundModuleType& mod, const std::vector<TypeArg>& typeargs);

	std::expected<DataType, std::string> setTypeArgs(const DataType& dtype, const std::vector<TypeArg>& typeargs);

	std::optional<std::reference_wrapper<const FunctionDecl>>
	getMember(const DataType& dtype, std::string_view name, const std::vector<DataType> argtypes);

	std::optional<std::reference_wrapper<const FunctionDecl>>
	getMember(const FunctionDecl& decl, std::string_view name, const std::vector<DataType>& argtypes);

	/**
	 * @brief Lists the fully qualified paths of all weights (tensor-typed member variables) of a module, including the
	 * weights of its submodules, in declaration order (depth first).
	 * @details This is the canonical (stable) order in which backends and binding generators expose weights.
	 * E.g., for `module A[] { x: float[2]  lin: Linear[...] }` this returns `{"x", "lin.W", "lin.b"}`.
	 */
	std::vector<std::string> weightPaths(const ModuleType& mod);
} // namespace mdl

#endif