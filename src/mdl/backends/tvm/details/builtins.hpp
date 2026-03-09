#include <mdl/computegraphv2/builtins.hpp>

#include <array>
#include <functional>

namespace mdl::backend::tvm {

	std::array<std::function<void(void)>, std::to_underlying(mdl::builtins::FunctionIDs::COUNT)> functions{
			/** Tensor Operators **/
			/*[std::to_underlying(mdl::builtins::FunctionIDs::TENSOR_ADD)] =*/nullptr,
			/*[std::to_underlying(mdl::builtins::FunctionIDs::TENSOR_SUB)] =*/nullptr,
			/*[std::to_underlying(mdl::builtins::FunctionIDs::TENSOR_DOT)] =*/nullptr,
			/*[std::to_underlying(mdl::builtins::FunctionIDs::TENSOR_PROD)] =*/nullptr,
			/*[std::to_underlying(mdl::builtins::FunctionIDs::TENSOR_DIV)] =*/nullptr,
			/** Statistic Functions **/
			/*[std::to_underlying(mdl::builtins::FunctionIDs::MAX)] =*/nullptr,
			/*[std::to_underlying(mdl::builtins::FunctionIDs::MIN)] =*/nullptr,
			/*[std::to_underlying(mdl::builtins::FunctionIDs::AVG)] =*/nullptr,
			/*[std::to_underlying(mdl::builtins::FunctionIDs::SUM)] =*/nullptr,
	};

}