#pragma once

#include "py_common.hpp"

namespace mdl::bindings {

	class PyTorchGenerator final : public PyBindingGenerator {
	public:
		std::filesystem::path inferBindingsFilename(const std::filesystem::path& outfile) override {
			// replace_extension cannot be used here: suffixes like "_pytorch.py" don't start with '.'
			// so replace_extension would prepend an extra dot, yielding "model._pytorch.py".
			return outfile.parent_path() / (outfile.stem().string() + "_pytorch.py");
		}

	protected:
		std::string imports() const override { return "import torch\n"; }
		// torch.Tensor implements the DLPack protocol, which tvm_ffi accepts directly.
		std::string tensorType() const override { return "torch.Tensor"; }
	};

} // namespace mdl::bindings
