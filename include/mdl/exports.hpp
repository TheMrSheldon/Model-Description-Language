#ifndef MDL_EXPORTS_HPP
#define MDL_EXPORTS_HPP

#include "./computegraphv2/datatype.hpp"
#include "./computegraphv2/node.hpp"
#include "./passes/autodiff.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mdl {
	struct CompilationUnit;
	class ArgVals;

	/** @brief One tensor in the signature of an exported function. */
	struct TensorSignature {
		std::string name;
		PrimitiveType dtype;
		/**
		 * @brief Concrete shape; -1 marks a dimension that is only known at runtime. Each `...` contributes
		 * ellipsisRank such dimensions (the same ones in all tensors of the function).
		 */
		std::vector<int64_t> shape;
		/** @brief The graph node (nullptr for an argument that the function does not use). */
		const Node* node;
		/**
		 * @brief Whether the tensor is stored in the function's context buffer (see ExportedFunction::context) instead
		 * of being a separate argument. Its shape is then only known to the backend (the layout is computed from the
		 * runtime dimensions).
		 */
		bool packed = false;
		/** @brief Whether the function adds to this (output) tensor instead of overwriting it (accumulated gradients). */
		bool accumulate = false;
		/** @brief For a gradient output: the weight path or input name it is the gradient of; else empty. */
		std::string gradientOf = {};
		/** @brief For an input: the number of dimensions of its `...` if smaller than the function's (else -1). */
		int ellipsisRank = -1;
	};

	/**
	 * @brief An opaque byte buffer that carries packed tensors from one exported function to another (e.g. the
	 * residuals from `forward_train` to `forward_backward`).
	 * @details The producer writes its packed outputs into the buffer, the consumer reads its packed inputs from it, in
	 * the same order. The backend lays the tensors out at offsets computed from the runtime dimensions (64-byte aligned)
	 * and exports a size function `size(d_0, ..., d_{ellipsisRank-1}) -> bytes` that takes the dimensions `...` stands
	 * for. The required size grows monotonically with the dimensions, so a buffer allocated for the largest dimensions
	 * can be reused for all smaller ones. The consumer must be called with the same dimensions as the producer.
	 */
	struct ContextBuffer {
		std::string name;		///< Name in the bindings (e.g. `forward_ctx`)
		std::string sizeSymbol; ///< Mangled name of the exported size function
		bool producer;			///< Whether the function writes (true) or reads (false) the buffer
		unsigned ellipsisRank;	///< Number of arguments of the size function
	};

	/**
	 * @brief The exported function that returns the workspace size in bytes of the function @p symbol: it takes the
	 * function's arguments without the workspace.
	 */
	inline std::string workspaceSizeSymbol(std::string_view symbol) { return std::string(symbol) + "_workspace_size"; }

	/**
	 * @brief A function that is compiled and exported by the backend.
	 * @details The exported (DPS) calling convention is: inputs (in declaration order), then weights (in canonical
	 * order, see mdl::weightPaths), then preallocated outputs, then the context buffer (if any; a 1-D uint8 tensor),
	 * then the workspace (a 1-D uint8 tensor of at least workspaceSizeSymbol's result, aligned to 64 bytes, on the
	 * tensors' device; owned by the caller's session, see docs/technical/runtime.md).
	 * Packed inputs and outputs are not arguments (they are stored in the context buffer). Backends and binding
	 * generators must both use this description so that the generated bindings match the compiled code.
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
		/**
		 * @brief The hyperparameter values the module was compiled with, by path (e.g. `conf.numEncoders`), in
		 * declaration order.
		 */
		std::vector<std::pair<std::string, std::variant<int64_t, double>>> hyperparameters;
		/** @brief The context buffer that carries the packed tensors, if any. */
		std::optional<ContextBuffer> context = std::nullopt;
	};

	/**
	 * @brief Lists the functions that are compiled: all non-template member functions of modules whose
	 * hyperparameters are all given in @p argvals.
	 */
	/**
	 * @brief The gradient pass: member functions that are replaced by their gradient (a gradient build; compile
	 * without it for inference).
	 * @details A function `f` with a scalar output (an objective) becomes `f(inputs...) -> (outputs..., d_<weight>...)`.
	 * Otherwise, the caller computes the objective and provides the cotangents of the outputs:
	 *  - by default, `f(inputs...) -> outputs` saves the values the backward pass needs in the context buffer `f_ctx`,
	 *    and `f_backward(d_<output>...) -> d_<weight>...` computes the gradients from them;
	 *  - with `fused`, `f(inputs..., d_<output>...) -> (outputs..., d_<weight>...)` does both in one call.
	 */
	struct GradientRequest {
		std::vector<std::string> functions;
		autodiff::Options options;
		bool fused = false;
		/** @brief Whether gradient outputs are added to the caller's buffers (true) or overwrite them. */
		bool accumulate = true;
		/**
		 * @brief Gradient checkpointing: module instances (patterns over their paths) whose computations the backward
		 * pass recomputes instead of saving them (see autodiff::split); empty for none.
		 */
		std::vector<std::string> checkpoint = {};
	};

	/**
	 * @brief Which weights are quantized and how (`mdlc --quantize`): weights whose paths @p weights selects. A weight
	 * W [N, K] (2-D, e.g. of a linear layer) becomes the weights `W.qweight`, `W.scales`, and (asymmetric) `W.zeros`
	 * (see builtins::QuantFormat); W itself is their dequantization. Quantized weights are frozen (not trainable).
	 * @details Only matrices are quantized: patterns that select modules skip their other weights (biases, norms). A
	 * pattern that selects no matrix is a warning; a pattern that is the exact path of another weight is an error.
	 */
	struct QuantizationRule {
		builtins::QuantFormat format = builtins::QuantFormat::int8;
		unsigned group = 0; ///< values of K per scale; 0: all of K (one scale per output, the default of int8)
		std::vector<std::string> weights; ///< patterns over weight paths (see PathPatterns); at least one not negated
	};

	/**
	 * @param masked member functions that become masked functions (see mdl::masking::apply): `name` or `name=element`
	 * take an additional input `mask: bool[...]` (true: valid), `name=suffix` an input `lengths: int[...]` over the
	 * dimensions of `...` but the last (the number of valid elements in the last one; the others are a masked suffix,
	 * e.g. padding). Masked elements in the last dimension of `...` do not influence the valid results. Masking is
	 * applied before differentiation.
	 * @param exportOnly the functions to export (`--export`: `forward` or `Module.forward`); empty: all. Only `public`
	 * member functions are exported; naming another function, or differentiating one that is not exported, is an
	 * error.
	 */
	std::vector<ExportedFunction> exportedFunctions(
			const CompilationUnit& unit, const ArgVals& argvals, unsigned ellipsisRank = 1,
			const GradientRequest& gradients = {}, const std::vector<std::string>& masked = {},
			const std::map<std::string, unsigned>& inputRanks = {}, const std::vector<QuantizationRule>& quantize = {},
			const std::vector<std::string>& exportOnly = {}
	);

	/**
	 * @brief Evaluates a compile-time constant (integer) expression, looking up hyperparameters in @p argvals.
	 * @return the value or std::nullopt if the expression is not a constant integer expression.
	 */
	std::optional<int64_t> evalConstant(const Node& node, const ArgVals& argvals);
} // namespace mdl

#endif
