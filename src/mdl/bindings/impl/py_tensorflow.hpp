#pragma once

#include "py_common.hpp"

namespace mdl::bindings {

	class PyTensorflowGenerator final : public PyBindingGenerator {
	public:
		std::filesystem::path inferBindingsFilename(const std::filesystem::path& outfile) override {
			// replace_extension cannot be used here: suffixes like "_tensorflow.py" don't start with '.'
			// so replace_extension would prepend an extra dot, yielding "model._tensorflow.py".
			return outfile.parent_path() / (outfile.stem().string() + "_tensorflow.py");
		}

	protected:
		std::string imports() const override {
			return "import tensorflow as tf\n"
				   "import tensorflow.experimental.dlpack as tf_dlpack\n";
		}

		std::string tensorType() const override { return "tf.Tensor"; }

		// Output must be a tf.Variable since tf.Tensor is immutable and cannot receive DLPack writes.
		std::string tensorOutputType() const override { return "tf.Variable"; }

		std::string toFfiArg(const std::string& expr) const override {
			return "tvm_ffi.from_dlpack(tf_dlpack.to_dlpack(" + expr + "))";
		}
	};

} // namespace mdl::bindings
