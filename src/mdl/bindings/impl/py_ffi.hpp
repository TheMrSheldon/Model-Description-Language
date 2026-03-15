#pragma once

#include "py_common.hpp"

namespace mdl::bindings {

	class PyFfiGenerator final : public PyBindingGenerator {
	public:
		std::filesystem::path inferBindingsFilename(const std::filesystem::path& outfile) override {
			return outfile.parent_path() / (outfile.stem().string() + ".py");
		}

	protected:
		std::string imports() const override {
			return "import tvm.nd\n"
				   "import tvm.runtime\n";
		}

		std::string tensorInputType() const override { return "tvm.nd.NDArray"; }
		std::string tensorOutputType() const override { return "tvm.nd.NDArray"; }

		std::string toTvmArg(const std::string& paramName) const override {
			return paramName; // already a TVM NDArray — no conversion needed
		}
	};

} // namespace mdl::bindings
