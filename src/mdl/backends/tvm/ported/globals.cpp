#include "./globals.hpp"

#include "./backend/cuda/codegen/header.hpp"
#include "./backend/cuda/codegen/registry.hpp"
#include "./s_tir/tensor_intrin/cuda.hpp"
#include "./support/nvcc.hpp"

#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include <mutex>
#include <string>
#include <vector>

namespace mdl::tvm_ported {

	namespace {
		namespace ffi = ::tvm::ffi;

		void set_global(const char* name, ffi::Function function) {
			if (!ffi::Function::GetGlobal(name))
				ffi::Function::SetGlobal(name, std::move(function));
		}
	} // namespace

	void register_cuda_globals() {
		static std::once_flag once;
		std::call_once(once, [] {
			set_global("tirx.intrinsics.cuda.header_generator", ffi::Function::FromTyped([](ffi::Array<ffi::String> tags) {
						   std::vector<std::string> t(tags.begin(), tags.end());
						   return ffi::String(backend::cuda::codegen::header_generator(t));
					   }));
			set_global("tirx.intrinsics.cuda.get_codegen", ffi::Function::FromTyped([](ffi::String op) {
						   return backend::cuda::codegen::get_codegen(op);
					   }));
			set_global("tvm_callback_cuda_compile", ffi::Function::FromTyped([](ffi::String code) {
						   const std::string binary = support::nvcc::tvm_callback_cuda_compile(code);
						   return ffi::Bytes(binary.data(), binary.size());
					   }));
			tensor_intrin::cuda::register_wmma_intrinsics();
		});
	}

} // namespace mdl::tvm_ported
