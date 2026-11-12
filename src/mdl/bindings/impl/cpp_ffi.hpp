#pragma once

#include "cpp_common.hpp"

#include <mdl/bindings/generator.hpp>
#include <mdl/exports.hpp>

#include <algorithm>
#include <cctype>
#include <format>
#include <fstream>
#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace mdl::bindings {

	/**
	 * @brief Generates a C++ header for the compiled object file.
	 * @details The header only depends on `<dlpack/dlpack.h>` and `<tvm/ffi/c_api.h>`. For every module it emits a
	 * struct with
	 *  - a nested `Weights` struct with one `const DLTensor*` slot per weight (mirroring the submodule hierarchy),
	 *    a `paths` table and a `find(path)` lookup to bind weights by name, and
	 *  - one static member function per exported function that calls the compiled code (destination passing style:
	 *    outputs are preallocated by the caller) and throws `std::runtime_error` on failure.
	 */
	class CppFfiGenerator final : public BindingGenerator {
	public:
		std::filesystem::path inferBindingsFilename(const std::filesystem::path& outfile) override {
			return outfile.parent_path() / (outfile.stem().string() + ".hpp");
		}

		void generate(const std::vector<ExportedFunction>& exports, const std::filesystem::path& output) override;
	};

} // namespace mdl::bindings

namespace mdl::bindings::detail {
	/** @brief Tree of weight paths (e.g., `weightQuery.W`) used to emit nested structs. */
	struct WeightTree {
		std::vector<std::pair<std::string, WeightTree>> children;
		std::string path; ///< set for leaves

		WeightTree& child(const std::string& name) {
			for (auto& [n, c] : children)
				if (n == name)
					return c;
			return children.emplace_back(name, WeightTree{}).second;
		}
	};

	inline std::string cppIdent(const std::string& name) {
		return (!name.empty() && std::isdigit(static_cast<unsigned char>(name[0]))) ? "_" + name : name;
	}

	/// `ndim, {d0, d1, ...}` initializer of mdl_detail::TensorInfo.
	inline std::string infoShape(const TensorSignature& sig) {
		std::string s = std::to_string(sig.shape.size()) + ", {";
		for (size_t i = 0; i < sig.shape.size(); ++i)
			s += (i > 0 ? ", " : "") + std::to_string(sig.shape[i]);
		return s + "}";
	}

	inline void emitWeightTree(
			std::ostream& out, const WeightTree& tree, const std::map<std::string, const TensorSignature*>& sigs,
			const std::string& indent
	) {
		for (const auto& [name, node] : tree.children) {
			if (node.children.empty()) {
				out << indent << "const DLTensor* " << cppIdent(name) << " = nullptr;";
				if (auto it = sigs.find(node.path); it != sigs.end())
					out << " ///< " << shapeStr(*it->second);
				out << "\n";
			} else {
				out << indent << "struct {\n";
				emitWeightTree(out, node, sigs, indent + "    ");
				out << indent << "} " << cppIdent(name) << ";\n";
			}
		}
	}

	inline std::string cppAccess(const std::string& path) {
		std::string result;
		size_t begin = 0;
		while (true) {
			auto end = path.find('.', begin);
			result += (result.empty() ? "" : ".") + cppIdent(path.substr(begin, end - begin));
			if (end == std::string::npos)
				return result;
			begin = end + 1;
		}
	}

	/** @brief Tree of hyperparameter paths (e.g. `conf.dims.model`) with the values at the leaves. */
	struct HyperparameterTree {
		std::vector<std::pair<std::string, HyperparameterTree>> children;
		std::variant<int64_t, double> value;

		HyperparameterTree& child(const std::string& name) {
			for (auto& [n, c] : children)
				if (n == name)
					return c;
			return children.emplace_back(name, HyperparameterTree{}).second;
		}
	};

	inline void emitHyperparameterFields(std::ostream& out, const HyperparameterTree& tree, const std::string& indent) {
		for (const auto& [name, node] : tree.children) {
			if (node.children.empty()) {
				out << indent << (std::holds_alternative<int64_t>(node.value) ? "int64_t " : "double ") << cppIdent(name)
					<< ";\n";
			} else {
				out << indent << "struct {\n"; // anonymous struct type (initialized by designated initializers)
				emitHyperparameterFields(out, node, indent + "    ");
				out << indent << "} " << cppIdent(name) << ";\n";
			}
		}
	}

	/** @brief Designated initializer of the hyperparameter values, e.g. `.conf = {.dims = {.model = 768}}`. */
	inline std::string hyperparameterInit(const HyperparameterTree& tree) {
		std::string init;
		for (const auto& [name, node] : tree.children) {
			init += (init.empty() ? "" : ", ") + ("." + cppIdent(name) + " = ");
			if (node.children.empty())
				init += std::visit([](auto v) { return std::format("{}", v); }, node.value);
			else
				init += "{" + hyperparameterInit(node) + "}";
		}
		return init;
	}

	inline void generateCppModule(std::ostream& out, const std::vector<const ExportedFunction*>& fns) {
		const auto& module = fns.front()->module;
		std::map<std::string, const TensorSignature*> sigs;
		for (const auto* fn : fns)
			for (const auto& w : fn->weights)
				sigs.emplace(w.name, &w);
		WeightTree tree;
		for (const auto& path : fns.front()->moduleWeights) {
			WeightTree* node = &tree;
			size_t begin = 0;
			while (true) {
				auto end = path.find('.', begin);
				node = &node->child(path.substr(begin, end - begin));
				if (end == std::string::npos)
					break;
				begin = end + 1;
			}
			node->path = path;
		}

		out << "/// Compiled functions of the MDL module `" << module << "`.\n"
			<< "struct " << module << " {\n"
			<< "    /// Weights of `" << module << "`; dimensions `B` are only known at runtime.\n"
			<< "    struct Weights {\n";
		emitWeightTree(out, tree, sigs, "        ");
		out << "\n        /// Paths and shapes of all weights in canonical order (-1: unknown dimension).\n"
			<< "        static constexpr std::array<mdl_detail::TensorInfo, " << fns.front()->moduleWeights.size()
			<< "> infos{{\n";
		for (const auto& path : fns.front()->moduleWeights) {
			out << "            {\"" << path << "\", ";
			if (auto it = sigs.find(path); it != sigs.end())
				out << infoShape(*it->second);
			else
				out << "-1, {}";
			out << "},\n";
		}
		out << "        }};\n\n"
			<< "        /// Returns the slot of the weight with the given path, or nullptr if there is no such weight.\n"
			<< "        const DLTensor** find(std::string_view path) {\n";
		for (const auto& path : fns.front()->moduleWeights)
			out << "            if (path == \"" << path << "\") return &" << cppAccess(path) << ";\n";
		out << "            return nullptr;\n"
			<< "        }\n"
			<< "    };\n";

		if (!fns.front()->hyperparameters.empty()) {
			// A struct type mirroring the hyperparameters (nested structs for struct-typed ones) and a constexpr
			// instance with the values the module was compiled with.
			HyperparameterTree tree;
			for (const auto& [path, value] : fns.front()->hyperparameters) {
				HyperparameterTree* node = &tree;
				for (size_t begin = 0;;) {
					auto end = path.find('.', begin);
					node = &node->child(path.substr(begin, end - begin));
					if (end == std::string::npos)
						break;
					begin = end + 1;
				}
				node->value = value;
			}
			out << "\n    /// Hyperparameters (see `hyperparameters` for the values the module was compiled with).\n"
				<< "    struct Hyperparameters {\n";
			emitHyperparameterFields(out, tree, "        ");
			out << "    };\n"
				<< "    static constexpr Hyperparameters hyperparameters{" << hyperparameterInit(tree) << "};\n";
		}

		for (const auto* fn : fns) {
			out << "\n    /// `" << fn->function << "(";
			for (size_t i = 0; i < fn->inputs.size(); ++i)
				out << (i > 0 ? ", " : "") << fn->inputs[i].name << ": " << shapeStr(fn->inputs[i]);
			out << ") -> ";
			for (size_t i = 0; i < fn->outputs.size(); ++i)
				out << (i > 0 ? ", " : "") << shapeStr(fn->outputs[i]);
			out << "`\n"
				<< "    /// Outputs must be preallocated by the caller (destination passing style). All tensors must be\n"
				<< "    /// contiguous float32 CPU tensors whose data is aligned to 64 bytes (e.g. mdl::runtime::Tensor).\n"
				<< "    /// `B`/`Bi` mark the dimensions of `...`, which are only known at runtime (the same in all tensors).\n"
				<< "    static void " << fn->function << "(";
			for (const auto& in : fn->inputs)
				out << "const DLTensor* " << in.name << ", ";
			out << "const Weights& weights";
			for (const auto& o : fn->outputs)
				out << ", DLTensor* " << o.name;
			out << ") {\n"
				<< "        TVMFFIAny args[] = {\n";
			for (const auto& in : fn->inputs)
				out << "            mdl_detail::tensorArg(" << in.name << "),\n";
			for (const auto& w : fn->weights)
				out << "            mdl_detail::tensorArg(weights." << cppAccess(w.name) << ", \"" << w.name << "\"),\n";
			for (const auto& o : fn->outputs)
				out << "            mdl_detail::tensorArg(" << o.name << "),\n";
			out << "        };\n"
				<< "        mdl_detail::call(__tvm_ffi_" << fn->symbol << ", args, \"" << fn->symbol << "\");\n"
				<< "    }\n";
			auto emitInfos = [&](const char* suffix, const std::vector<TensorSignature>& tensors) {
				out << "    static constexpr std::array<mdl_detail::TensorInfo, " << tensors.size() << "> " << fn->function
					<< "_" << suffix << "{{\n";
				for (const auto& t : tensors)
					out << "        {\"" << t.name << "\", " << infoShape(t) << "},\n";
				out << "    }};\n";
			};
			out << "    /// Names and shapes of the inputs and outputs of `" << fn->function << "`.\n";
			emitInfos("inputs", fn->inputs);
			emitInfos("outputs", fn->outputs);
		}
		out << "};\n\n";
	}
} // namespace mdl::bindings::detail

inline void mdl::bindings::CppFfiGenerator::generate(
		const std::vector<ExportedFunction>& exports, const std::filesystem::path& output
) {
	std::ofstream out(output);
	out << "// Generated by mdlc -- do not edit\n"
		<< "#pragma once\n\n"
		<< "#include <dlpack/dlpack.h>\n"
		<< "#include <tvm/ffi/c_api.h>\n\n"
		<< "#include <array>\n"
		<< "#include <cstdint>\n"
		<< "#include <stdexcept>\n"
		<< "#include <string>\n"
		<< "#include <string_view>\n\n"
		<< "extern \"C\" {\n";
	for (const auto& fn : exports)
		out << "int __tvm_ffi_" << fn.symbol
			<< "(void* self, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result);\n";
	out << "}\n\n"
		<< "namespace mdl::models {\n\n"
		<< "#ifndef MDL_MODELS_DETAIL_DEFINED\n"
		<< "#define MDL_MODELS_DETAIL_DEFINED\n"
		<< "namespace mdl_detail {\n"
		<< "    /// Path (or name) and shape of a tensor; -1 marks dimensions that are only known at runtime.\n"
		<< "    struct TensorInfo {\n"
		<< "        const char* name;\n"
		<< "        int32_t ndim;\n"
		<< "        std::array<int64_t, 8> shape;\n"
		<< "    };\n\n"
		<< "    /// Compiled functions require tensor data to be aligned to this many bytes.\n"
		<< "    inline constexpr size_t kTensorAlignment = 64;\n\n"
		<< "    inline TVMFFIAny tensorArg(const DLTensor* tensor, const char* name = nullptr) {\n"
		<< "        if (tensor == nullptr)\n"
		<< "            throw std::invalid_argument(std::string{\"missing tensor \"} + (name ? name : \"argument\"));\n"
		<< "        if ((reinterpret_cast<uintptr_t>(tensor->data) + tensor->byte_offset) % kTensorAlignment != 0)\n"
		<< "            throw std::invalid_argument(std::string{\"tensor \"} + (name ? name : \"argument\") +\n"
		<< "                                        \" is not aligned to 64 bytes\");\n"
		<< "        TVMFFIAny any{};\n"
		<< "        any.type_index = kTVMFFIDLTensorPtr;\n"
		<< "        any.v_ptr = const_cast<DLTensor*>(tensor);\n"
		<< "        return any;\n"
		<< "    }\n\n"
		<< "    template <size_t N>\n"
		<< "    inline void call(int (*fn)(void*, const TVMFFIAny*, int32_t, TVMFFIAny*), TVMFFIAny (&args)[N],\n"
		<< "                     const char* name) {\n"
		<< "        TVMFFIAny result{};\n"
		<< "        result.type_index = kTVMFFINone;\n"
		<< "        if (fn(nullptr, args, static_cast<int32_t>(N), &result) == 0)\n"
		<< "            return;\n"
		<< "        TVMFFIObjectHandle error = nullptr;\n"
		<< "        TVMFFIErrorMoveFromRaised(&error);\n"
		<< "        std::string message = std::string{name} + \" failed\";\n"
		<< "        if (error != nullptr) {\n"
		<< "            const TVMFFIErrorCell* cell = TVMFFIErrorGetCellPtr(error);\n"
		<< "            message += \": \" + std::string{cell->kind.data, cell->kind.size} + \": \" +\n"
		<< "                       std::string{cell->message.data, cell->message.size};\n"
		<< "            TVMFFIObjectDecRef(error);\n"
		<< "        }\n"
		<< "        throw std::runtime_error(message);\n"
		<< "    }\n"
		<< "} // namespace mdl_detail\n"
		<< "#endif\n\n";

	// Group the exported functions by module (preserving order).
	std::vector<std::vector<const ExportedFunction*>> modules;
	for (const auto& fn : exports) {
		auto it = std::find_if(modules.begin(), modules.end(), [&](const auto& m) { return m.front()->module == fn.module; });
		if (it == modules.end())
			modules.push_back({&fn});
		else
			it->push_back(&fn);
	}
	for (const auto& fns : modules)
		detail::generateCppModule(out, fns);

	out << "} // namespace mdl::models\n";
}
