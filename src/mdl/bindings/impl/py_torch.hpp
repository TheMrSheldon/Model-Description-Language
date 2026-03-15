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
		std::string imports() const override {
			return "import torch\n"
				   "import tvm.nd\n"
				   "import tvm.runtime\n";
		}

		std::string tensorInputType() const override { return "torch.Tensor"; }
		std::string tensorOutputType() const override { return "torch.Tensor"; }

		std::string toTvmArg(const std::string& paramName) const override {
			// torch.Tensor supports the DLPack protocol; tvm.nd.from_dlpack accepts it directly.
			return "tvm.nd.from_dlpack(" + paramName + ")";
		}
	};

} // namespace mdl::bindings
