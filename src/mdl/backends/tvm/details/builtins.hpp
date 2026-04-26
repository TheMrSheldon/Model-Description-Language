#include <mdl/computegraphv2/builtins.hpp>

#include <tvm/te/tensor.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/transform.h>

#include <any>
#include <array>
#include <cassert>
#include <functional>
#include <vector>

namespace mdl::backend::tvm {

	/**
	 * @brief Translation function for a builtin operation.
	 * @details Each argument is a std::any. Tensor-producing nodes are passed as
	 *          tvm::te::Tensor; ConstNodes are passed as their raw Atomic value
	 *          (int, double, or std::vector<int>). Builtins that consume compile-time
	 *          constants cast the relevant args directly — wrong casts are internal errors.
	 */
	using TranslateFunc = std::function<::tvm::te::Tensor(std::vector<std::any>)>;

	/** Type-checked cast helper — mismatched type is an internal compiler error. **/
	template <typename T>
	inline const T& tvmArg(const std::any& a) {
		assert(a.type() == typeid(T));
		return *std::any_cast<T>(&a);
	}

	std::array<TranslateFunc, std::to_underlying(mdl::builtins::FunctionIDs::COUNT)> functions{
			/** Tensor Operators **/
			/*[TENSOR_ADD] =*/
			[](auto ts) {
				return ::tvm::topi::add(tvmArg<::tvm::te::Tensor>(ts[0]), tvmArg<::tvm::te::Tensor>(ts[1]));
			},
			/*[TENSOR_SUB] =*/
			[](auto ts) {
				return ::tvm::topi::subtract(tvmArg<::tvm::te::Tensor>(ts[0]), tvmArg<::tvm::te::Tensor>(ts[1]));
			},
			/*[TENSOR_DOT] =*/
			[](auto ts) {
				return ::tvm::topi::matmul(tvmArg<::tvm::te::Tensor>(ts[0]), tvmArg<::tvm::te::Tensor>(ts[1]));
			},
			/*[TENSOR_PROD]=*/
			[](auto ts) {
				return ::tvm::topi::multiply(tvmArg<::tvm::te::Tensor>(ts[0]), tvmArg<::tvm::te::Tensor>(ts[1]));
			},
			/*[TENSOR_DIV] =*/
			[](auto ts) {
				return ::tvm::topi::divide(tvmArg<::tvm::te::Tensor>(ts[0]), tvmArg<::tvm::te::Tensor>(ts[1]));
			},
			/** Statistic Functions **/
			/*[MAX] =*/
			[](auto ts) {
				return ::tvm::topi::max(
						tvmArg<::tvm::te::Tensor>(ts[0]), ::tvm::ffi::Optional<::tvm::ffi::Array<::tvm::Integer>>{},
						false
				);
			},
			/*[MIN] =*/
			[](auto ts) {
				return ::tvm::topi::min(
						tvmArg<::tvm::te::Tensor>(ts[0]), ::tvm::ffi::Optional<::tvm::ffi::Array<::tvm::Integer>>{},
						false
				);
			},
			/*[MEAN] =*/nullptr, /** \todo no direct topi equivalent; implement as sum/count **/
			/*[SUM] =*/
			[](auto ts) {
				return ::tvm::topi::sum(
						tvmArg<::tvm::te::Tensor>(ts[0]), ::tvm::ffi::Optional<::tvm::ffi::Array<::tvm::Integer>>{},
						false
				);
			},
			/** Arithmetic Functions **/
			/*[COS] =*/[](auto ts) { return ::tvm::topi::cos(tvmArg<::tvm::te::Tensor>(ts[0])); },
			/*[SIN] =*/[](auto ts) { return ::tvm::topi::sin(tvmArg<::tvm::te::Tensor>(ts[0])); },
			/*[TAN] =*/[](auto ts) { return ::tvm::topi::tan(tvmArg<::tvm::te::Tensor>(ts[0])); },
			/*[EXP] =*/[](auto ts) { return ::tvm::topi::exp(tvmArg<::tvm::te::Tensor>(ts[0])); },
			/*[ERF] =*/[](auto ts) { return ::tvm::topi::erf(tvmArg<::tvm::te::Tensor>(ts[0])); },
			/*[SQRT] =*/[](auto ts) { return ::tvm::topi::sqrt(tvmArg<::tvm::te::Tensor>(ts[0])); },
			/*[RSQRT] =*/[](auto ts) { return ::tvm::topi::rsqrt(tvmArg<::tvm::te::Tensor>(ts[0])); },
			/*[VAR] =*/
			[](auto ts) -> ::tvm::te::Tensor {
				std::abort(); /** \todo implement */
			},
			/** Einops **/
			/*[RESHAPE] =*/
			[](auto ts) -> ::tvm::te::Tensor {
				// ts[0]: tensor; ts[1..]: int shape dims (-1 = infer dimension at runtime).
				::tvm::ffi::Array<::tvm::PrimExpr> shape;
				for (size_t i = 1; i < ts.size(); ++i)
					shape.push_back(::tvm::IntImm(::tvm::DataType::Int(32), tvmArg<int>(ts[i])));
				return ::tvm::topi::reshape(tvmArg<::tvm::te::Tensor>(ts[0]), shape);
			},
			/*[PERMUTE] =*/
			[](auto ts) -> ::tvm::te::Tensor {
				// ts[0]: tensor; ts[1]: std::vector<int> permutation (from ConstNode<vector<int>>).
				// Negative indices are resolved relative to the actual tensor rank.
				// \todo Support ellipsis expansion: when perm.size() < rank, one perm entry
				//       represents all batch dims and must be expanded to individual axes.
				//       This requires the batch rank to be known at compile time (constant folding).
				const auto& perm = tvmArg<std::vector<int>>(ts[1]);
				const auto& tensor = tvmArg<::tvm::te::Tensor>(ts[0]);
				int rank = static_cast<int>(tensor->shape.size());
				assert(static_cast<int>(perm.size()) == rank && "PERMUTE with ellipsis not yet supported");
				::tvm::ffi::Array<::tvm::Integer> axes;
				for (int idx : perm) {
					if (idx < 0)
						idx += rank;
					axes.push_back(::tvm::Integer(idx));
				}
				return ::tvm::topi::transpose(tensor, axes);
			},
			/*[EXPAND] =*/
			[](auto ts) -> ::tvm::te::Tensor {
				// ts[0]: tensor; ts[1..]: int target shape dims.
				::tvm::ffi::Array<::tvm::PrimExpr> shape;
				for (size_t i = 1; i < ts.size(); ++i)
					shape.push_back(::tvm::IntImm(::tvm::DataType::Int(32), tvmArg<int>(ts[i])));
				return ::tvm::topi::broadcast_to(tvmArg<::tvm::te::Tensor>(ts[0]), shape);
			},
	};

} // namespace mdl::backend::tvm
