#include <mdl/bindings/generator.hpp>

#include "./impl/cpp_ffi.hpp"
#include "./impl/py_ffi.hpp"
#include "./impl/py_tensorflow.hpp"
#include "./impl/py_torch.hpp"

namespace mdl::bindings {

	std::map<std::string, GeneratorFactory> getGenerators() {
		return {
				{"cpp", [] { return std::make_unique<CppFfiGenerator>(); }},
				{"python", [] { return std::make_unique<PyFfiGenerator>(); }},
				{"python-torch", [] { return std::make_unique<PyTorchGenerator>(); }},
				{"python-tensorflow", [] { return std::make_unique<PyTensorflowGenerator>(); }},
		};
	}

} // namespace mdl::bindings
