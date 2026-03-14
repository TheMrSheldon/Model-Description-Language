#include <mdl/computegraphv2/builtins.hpp>

#include <tvm/te/tensor.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/transform.h>

#include <array>
#include <functional>
#include <vector>

namespace mdl::backend::tvm {

	using TranslateFunc = std::function<::tvm::te::Tensor(std::vector<::tvm::te::Tensor>)>;

	std::array<TranslateFunc, std::to_underlying(mdl::builtins::FunctionIDs::COUNT)> functions{
			/** Tensor Operators **/
			/*[TENSOR_ADD] =*/[](auto ts) { return ::tvm::topi::add(ts[0], ts[1]); },
			/*[TENSOR_SUB] =*/[](auto ts) { return ::tvm::topi::subtract(ts[0], ts[1]); },
			/*[TENSOR_DOT] =*/[](auto ts) { return ::tvm::topi::matmul(ts[0], ts[1]); },
			/*[TENSOR_PROD]=*/[](auto ts) { return ::tvm::topi::multiply(ts[0], ts[1]); },
			/*[TENSOR_DIV] =*/[](auto ts) { return ::tvm::topi::divide(ts[0], ts[1]); },
			/** Statistic Functions **/
			/*[MAX] =*/
			[](auto ts) {
				return ::tvm::topi::max(ts[0], ::tvm::ffi::Optional<::tvm::ffi::Array<::tvm::Integer>>{}, false);
			},
			/*[MIN] =*/
			[](auto ts) {
				return ::tvm::topi::min(ts[0], ::tvm::ffi::Optional<::tvm::ffi::Array<::tvm::Integer>>{}, false);
			},
			/*[AVG] =*/nullptr, /** \todo no direct topi equivalent; implement as sum/count **/
								/*[SUM] =*/
			[](auto ts) {
				return ::tvm::topi::sum(ts[0], ::tvm::ffi::Optional<::tvm::ffi::Array<::tvm::Integer>>{}, false);
			},
			/** Arithmetic Functions **/
			/*[COS] =*/[](auto ts) { return ::tvm::topi::cos(ts[0]); },
			/*[SIN] =*/[](auto ts) { return ::tvm::topi::sin(ts[0]); },
			/*[TAN] =*/[](auto ts) { return ::tvm::topi::tan(ts[0]); },
			/*[EXP] =*/[](auto ts) { return ::tvm::topi::exp(ts[0]); },
			/*[ERF] =*/[](auto ts) { return ::tvm::topi::erf(ts[0]); },
	};

} // namespace mdl::backend::tvm
