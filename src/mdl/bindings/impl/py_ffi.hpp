#pragma once

#include "py_common.hpp"

namespace mdl::bindings {

	/** @brief Python bindings that accept any DLPack-compatible tensor. */
	class PyFfiGenerator final : public PyBindingGenerator {
	public:
		std::filesystem::path inferBindingsFilename(const std::filesystem::path& outfile) override {
			return outfile.parent_path() / (outfile.stem().string() + ".py");
		}

	protected:
		std::string imports() const override { return "from typing import Any\n"; }
		std::string tensorType() const override { return "Any"; }
	};

} // namespace mdl::bindings
