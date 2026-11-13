#pragma once

#include <mdl/bindings/generator.hpp>
#include <mdl/exports.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

namespace mdl::bindings {

	/**
	 * @brief Writes a JSON description of the compiled functions (metadata for runtimes that load the library
	 * dynamically, e.g. tmm's model loader).
	 * @details Lists per module the weights (path, dtype, shape, whether a gradient is computed for them), the
	 * hyperparameters, and every exported function with its C symbol, its arguments in calling order (inputs, the
	 * weights it uses, outputs, and the context buffer), whether an output is accumulated, and what a gradient output is
	 * the gradient of. Shapes use -1 for dimensions that are only known at runtime (the dimensions of `...`).
	 */
	class JsonGenerator final : public BindingGenerator {
	public:
		std::filesystem::path inferBindingsFilename(const std::filesystem::path& outfile) override {
			return outfile.parent_path() / (outfile.stem().string() + ".json");
		}

		void generate(const std::vector<ExportedFunction>& exports, const std::filesystem::path& output) override;
	};

} // namespace mdl::bindings

namespace mdl::bindings::detail {
	inline std::string jsonString(const std::string& s) {
		std::string out = "\"";
		for (char c : s) {
			if (c == '"' || c == '\\') {
				out += '\\';
				out += c;
			} else if (static_cast<unsigned char>(c) < 0x20) {
				char buf[8];
				std::snprintf(buf, sizeof(buf), "\\u%04x", c);
				out += buf;
			} else {
				out += c;
			}
		}
		return out + "\"";
	}

	inline std::string jsonDtype(PrimitiveType p) {
		switch (p) {
		case PrimitiveType::float32:
			return "\"float32\"";
		case PrimitiveType::float16:
			return "\"float16\"";
		case PrimitiveType::bfloat16:
			return "\"bfloat16\"";
		case PrimitiveType::integer:
			return "\"int32\"";
		case PrimitiveType::uint:
			return "\"uint32\"";
		case PrimitiveType::boolean:
			return "\"bool\"";
		default:
			return "null";
		}
	}

	inline std::string jsonShape(const std::vector<int64_t>& shape) {
		std::string s = "[";
		for (size_t i = 0; i < shape.size(); ++i)
			s += (i > 0 ? ", " : "") + std::to_string(shape[i]);
		return s + "]";
	}

	inline std::string jsonTensor(const TensorSignature& t) {
		std::string s = "{\"name\": " + jsonString(t.name) + ", \"dtype\": " + jsonDtype(t.dtype) +
						", \"shape\": " + jsonShape(t.shape);
		if (t.accumulate)
			s += ", \"accumulate\": true";
		if (!t.gradientOf.empty())
			s += ", \"gradient_of\": " + jsonString(t.gradientOf);
		return s + "}";
	}
} // namespace mdl::bindings::detail

inline void mdl::bindings::JsonGenerator::generate(
		const std::vector<ExportedFunction>& exports, const std::filesystem::path& output
) {
	using namespace detail;
	std::ofstream out(output);
	std::vector<std::string> modules;
	for (const auto& fn : exports)
		if (std::find(modules.begin(), modules.end(), fn.module) == modules.end())
			modules.push_back(fn.module);

	out << "{\n  \"format\": \"mdl-metadata\",\n  \"version\": 1,\n  \"modules\": [";
	for (size_t m = 0; m < modules.size(); ++m) {
		std::vector<const ExportedFunction*> fns;
		for (const auto& fn : exports)
			if (fn.module == modules[m])
				fns.push_back(&fn);
		const auto& first = *fns.front();
		std::map<std::string, const TensorSignature*> weightSigs;
		std::map<std::string, bool> trainable;
		for (const auto* fn : fns) {
			for (const auto& w : fn->weights)
				weightSigs.emplace(w.name, &w);
			for (const auto& o : fn->outputs)
				if (!o.gradientOf.empty())
					trainable[o.gradientOf] = true;
		}

		out << (m > 0 ? "," : "") << "\n    {\n      \"name\": " << jsonString(modules[m]) << ",\n";
		out << "      \"hyperparameters\": {";
		for (size_t i = 0; i < first.hyperparameters.size(); ++i) {
			const auto& [path, value] = first.hyperparameters[i];
			out << (i > 0 ? ", " : "") << jsonString(path) << ": "
				<< std::visit([](auto v) { return std::to_string(v); }, value);
		}
		out << "},\n      \"weights\": [";
		for (size_t i = 0; i < first.moduleWeights.size(); ++i) {
			const auto& path = first.moduleWeights[i];
			auto sig = weightSigs.find(path);
			out << (i > 0 ? "," : "") << "\n        {\"path\": " << jsonString(path) << ", \"dtype\": "
				<< (sig != weightSigs.end() ? jsonDtype(sig->second->dtype) : "null") << ", \"shape\": "
				<< (sig != weightSigs.end() ? jsonShape(sig->second->shape) : "null")
				<< ", \"trainable\": " << (trainable.contains(path) ? "true" : "false") << "}";
		}
		out << "\n      ],\n      \"functions\": [";
		for (size_t f = 0; f < fns.size(); ++f) {
			const auto& fn = *fns[f];
			out << (f > 0 ? "," : "") << "\n        {\n          \"name\": " << jsonString(fn.function)
				<< ",\n          \"symbol\": " << jsonString("__tvm_ffi_" + fn.symbol) << ",\n          \"inputs\": [";
			bool firstEntry = true;
			for (const auto& in : fn.inputs)
				if (!in.packed) {
					out << (firstEntry ? "" : ", ") << jsonTensor(in);
					firstEntry = false;
				}
			out << "],\n          \"weights\": [";
			for (size_t i = 0; i < fn.weights.size(); ++i)
				out << (i > 0 ? ", " : "") << jsonString(fn.weights[i].name);
			out << "],\n          \"outputs\": [";
			firstEntry = true;
			for (const auto& o : fn.outputs)
				if (!o.packed) {
					out << (firstEntry ? "" : ", ") << jsonTensor(o);
					firstEntry = false;
				}
			out << "],\n          \"context\": ";
			if (fn.context)
				out << "{\"name\": " << jsonString(fn.context->name) << ", \"role\": \""
					<< (fn.context->producer ? "producer" : "consumer") << "\", \"size_symbol\": "
					<< jsonString("__tvm_ffi_" + fn.context->sizeSymbol)
					<< ", \"size_arguments\": " << fn.context->ellipsisRank << "}";
			else
				out << "null";
			out << "\n        }";
		}
		out << "\n      ]\n    }";
	}
	out << "\n  ]\n}\n";
}
