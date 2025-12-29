#include <tvm/ffi/c_api.h>

//#include <tvm/ffi/container/tensor.h>
//#include <tvm/ffi/extra/module.h>

#include <iostream>
#include <string_view>


extern "C" {
    // https://tvm.apache.org/ffi/get_started/stable_c_abi.html#tvm-ffi-c-abi
    int __tvm_ffi_forward(void* handle, const TVMFFIAny* args, int N, TVMFFIAny* result);
};


int main(int argc, char* argv[]) {
    //tvm::ffi::Module mod = tvm::ffi::Module::LoadFromFile("/workspaces/model-description-language/build/examples/01_ffn/libtest.so");
    // Look up `add_one_cpu` function
    //tvm::ffi::Function tmp = mod->GetFunction("forward").value();
    //tvm::ffi::TensorView a,b;
    //tmp(a,b);
    
    TVMFFIVersion version;
    TVMFFIGetVersion(&version);

    std::cout << "Running on TVM ABI v" << version.major << '.' << version.minor << '.' << version.patch << std::endl;

    TVMFFIAny none = {.type_index = kTVMFFINone};
    TVMFFIAny args[0];
    auto i = __tvm_ffi_forward(nullptr, args, 0, &none);

    std::cout << "Return code " << i << std::endl;

    TVMFFIObjectHandle err = NULL;
    TVMFFIErrorMoveFromRaised(&err);
    auto cell = reinterpret_cast<TVMFFIErrorCell*>(reinterpret_cast<uintptr_t>(err) + sizeof(TVMFFIObject));
    std::cout << cell->kind.data << ": " << std::string_view(cell->message.data, cell->message.size) << std::endl;
    return 0;
}