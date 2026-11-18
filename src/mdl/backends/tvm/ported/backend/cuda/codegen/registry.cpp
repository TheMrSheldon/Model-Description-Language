// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/backend/cuda/codegen/registry.py
// (conventions: src/mdl/backends/tvm/ported/README.md)
#include "./registry.hpp"

namespace mdl::tvm_ported::backend::cuda::codegen {

	std::map<std::string, ::tvm::ffi::Function>& CODEGEN_REGISTRY() {
		static std::map<std::string, ::tvm::ffi::Function> registry;
		return registry;
	}

	::tvm::ffi::Optional<::tvm::ffi::Function> get_codegen(const std::string& op) {
		auto it = CODEGEN_REGISTRY().find(op);
		if (it == CODEGEN_REGISTRY().end())
			return std::nullopt;
		return it->second;
	}

} // namespace mdl::tvm_ported::backend::cuda::codegen
