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

	/// DLDataType initializer of a primitive type (float32/float16/bfloat16, int32, uint32, bool).
	inline std::string dlDtype(PrimitiveType p) {
		switch (p) {
		case PrimitiveType::float16:
			return "{kDLFloat, 16, 1}";
		case PrimitiveType::bfloat16:
			return "{kDLBfloat, 16, 1}";
		case PrimitiveType::integer:
			return "{kDLInt, 32, 1}";
		case PrimitiveType::uint:
			return "{kDLUInt, 32, 1}";
		case PrimitiveType::boolean:
			return "{kDLBool, 8, 1}";
		default:
			return "{kDLFloat, 32, 1}";
		}
	}

	/// `ndim, {d0, d1, ...}, dtype` initializer of mdl_detail::TensorInfo.
	inline std::string infoShape(const TensorSignature& sig) {
		std::string s = std::to_string(sig.shape.size()) + ", {";
		for (size_t i = 0; i < sig.shape.size(); ++i)
			s += (i > 0 ? ", " : "") + std::to_string(sig.shape[i]);
		return s + "}, " + dlDtype(sig.dtype);
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
			<< "    using Context = mdl_detail::Context;\n\n"
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
				out << "-1, {}, {kDLFloat, 32, 1}";
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
			// Packed tensors are not arguments: they are stored in the context buffer.
			std::vector<const TensorSignature*> inputs, outputs;
			for (const auto& in : fn->inputs)
				if (!in.packed)
					inputs.push_back(&in);
			for (const auto& o : fn->outputs)
				if (!o.packed)
					outputs.push_back(&o);
			const auto& context = fn->context;

			if (context && context->producer) {
				std::string dims, dimArgs;
				for (unsigned d = 0; d < context->ellipsisRank; ++d) {
					dims += (d > 0 ? ", " : "") + std::format("int64_t d{}", d);
					dimArgs += (d > 0 ? ", " : "") + std::format("d{}", d);
				}
				out << "\n    /// Size in bytes of the context buffer `" << context->name << "` (written by `" << fn->function
					<< "`) for the given\n"
					<< "    /// dimensions of `...`. It grows with the dimensions: a buffer for the largest dimensions can be\n"
					<< "    /// reused for all smaller ones.\n"
					<< "    static size_t " << context->name << "_size(" << dims << ") {\n";
				if (context->ellipsisRank > 0) {
					out << "        TVMFFIAny args[] = {";
					for (unsigned d = 0; d < context->ellipsisRank; ++d)
						out << (d > 0 ? ", " : "") << "mdl_detail::intArg(d" << d << ")";
					out << "};\n";
				} else {
					out << "        const TVMFFIAny* args = nullptr;\n";
				}
				out << "        return static_cast<size_t>(mdl_detail::callInt(__tvm_ffi_" << context->sizeSymbol << ", args, "
					<< context->ellipsisRank << ", \"" << context->sizeSymbol << "\"));\n"
					<< "    }\n"
					<< "    /// Allocates a context buffer for the given dimensions of `...` (free it with "
					<< context->name << "_free).\n"
					<< "    static Context " << context->name << "_alloc(" << dims << ") {\n"
					<< "        return mdl_detail::allocContext(" << context->name << "_size(" << dimArgs << "));\n"
					<< "    }\n"
					<< "    static void " << context->name << "_free(Context& ctx) { mdl_detail::freeContext(ctx); }\n";
			}

			out << "\n    /// `" << fn->function << "(";
			for (size_t i = 0; i < inputs.size(); ++i)
				out << (i > 0 ? ", " : "") << inputs[i]->name << ": " << shapeStr(*inputs[i]);
			out << ") -> ";
			for (size_t i = 0; i < outputs.size(); ++i)
				out << (i > 0 ? ", " : "") << shapeStr(*outputs[i]);
			out << "`\n"
				<< "    /// Outputs must be preallocated by the caller (destination passing style). All tensors must be\n"
				<< "    /// contiguous CPU tensors of the declared type (float32, or int32 for `int`) whose data is aligned to\n"
				<< "    /// 64 bytes (e.g. mdl::runtime::Tensor / mdl::runtime::IntTensor).\n"
				<< "    /// `B`/`Bi` mark the dimensions of `...`, which are only known at runtime (the same in all tensors).\n";
			if (context)
				out << "    /// " << (context->producer ? "Writes the values the backward pass needs into" : "Reads")
					<< " the context buffer `ctx` (see " << context->name << "_size)"
					<< (context->producer ? "" : ", which must have been written with the same dimensions") << ".\n";
			if (std::any_of(outputs.begin(), outputs.end(), [](const auto* o) { return o->accumulate; }))
				out << "    /// The gradients are added to the given buffers (accumulation): zero them before the first step.\n";
			out << "    static void " << fn->function << "(";
			for (const auto* in : inputs)
				out << "const DLTensor* " << in->name << ", ";
			out << "const Weights& weights";
			for (const auto* o : outputs)
				out << ", DLTensor* " << o->name;
			if (context)
				out << ", " << (context->producer ? "" : "const ") << "Context& ctx";
			out << ") {\n";
			if (context)
				out << "        int64_t ctx_shape[1] = {static_cast<int64_t>(ctx.bytes)};\n"
					<< "        DLTensor ctx_tensor{ctx.data, {kDLCPU, 0}, 1, {kDLUInt, 8, 1}, ctx_shape, nullptr, 0};\n";
			out << "        TVMFFIAny args[] = {\n";
			for (const auto* in : inputs)
				out << "            mdl_detail::tensorArg(" << in->name << ", \"" << in->name << "\", " << dlDtype(in->dtype)
					<< "),\n";
			for (const auto& w : fn->weights)
				out << "            mdl_detail::tensorArg(weights." << cppAccess(w.name) << ", \"" << w.name << "\", "
					<< dlDtype(w.dtype) << "),\n";
			for (const auto* o : outputs)
				out << "            mdl_detail::tensorArg(" << o->name << ", \"" << o->name << "\", " << dlDtype(o->dtype)
					<< "),\n";
			if (context)
				out << "            mdl_detail::tensorArg(&ctx_tensor, \"ctx\", {kDLUInt, 8, 1}),\n";
			out << "        };\n"
				<< "        mdl_detail::call(__tvm_ffi_" << fn->symbol << ", args, \"" << fn->symbol << "\");\n"
				<< "    }\n";
			auto emitInfos = [&](const char* suffix, const std::vector<const TensorSignature*>& tensors) {
				out << "    static constexpr std::array<mdl_detail::TensorInfo, " << tensors.size() << "> " << fn->function
					<< "_" << suffix << "{{\n";
				for (const auto* t : tensors)
					out << "        {\"" << t->name << "\", " << infoShape(*t) << "},\n";
				out << "    }};\n";
			};
			out << "    /// Names and shapes of the inputs and outputs of `" << fn->function << "`.\n";
			emitInfos("inputs", inputs);
			emitInfos("outputs", outputs);
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
		<< "#include <cstdlib>\n"
		<< "#include <new>\n"
		<< "#include <stdexcept>\n"
		<< "#include <string>\n"
		<< "#include <string_view>\n\n"
		<< "extern \"C\" {\n";
	for (const auto& fn : exports) {
		out << "int __tvm_ffi_" << fn.symbol
			<< "(void* self, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result);\n";
		if (fn.context && fn.context->producer)
			out << "int __tvm_ffi_" << fn.context->sizeSymbol
				<< "(void* self, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result);\n";
	}
	out << "}\n\n"
		<< "namespace mdl::models {\n\n"
		<< "#ifndef MDL_MODELS_DETAIL_DEFINED\n"
		<< "#define MDL_MODELS_DETAIL_DEFINED\n"
		<< "namespace mdl_detail {\n"
		<< "    /// Path (or name), shape, and element type of a tensor; -1 marks dimensions that are only known at runtime.\n"
		<< "    struct TensorInfo {\n"
		<< "        const char* name;\n"
		<< "        int32_t ndim;\n"
		<< "        std::array<int64_t, 8> shape;\n"
		<< "        DLDataType dtype;\n"
		<< "    };\n\n"
		<< "    /// Compiled functions require tensor data to be aligned to this many bytes.\n"
		<< "    inline constexpr size_t kTensorAlignment = 64;\n\n"
		<< "    inline TVMFFIAny tensorArg(const DLTensor* tensor, const char* name, DLDataType dtype) {\n"
		<< "        if (tensor == nullptr)\n"
		<< "            throw std::invalid_argument(std::string{\"missing tensor \"} + (name ? name : \"argument\"));\n"
		<< "        if (tensor->dtype.code != dtype.code || tensor->dtype.bits != dtype.bits || tensor->dtype.lanes != dtype.lanes)\n"
		<< "            throw std::invalid_argument(std::string{\"tensor \"} + name + \" has the wrong element type\");\n"
		<< "        if ((reinterpret_cast<uintptr_t>(tensor->data) + tensor->byte_offset) % kTensorAlignment != 0)\n"
		<< "            throw std::invalid_argument(std::string{\"tensor \"} + (name ? name : \"argument\") +\n"
		<< "                                        \" is not aligned to 64 bytes\");\n"
		<< "        TVMFFIAny any{};\n"
		<< "        any.type_index = kTVMFFIDLTensorPtr;\n"
		<< "        any.v_ptr = const_cast<DLTensor*>(tensor);\n"
		<< "        return any;\n"
		<< "    }\n\n"
		<< "    inline TVMFFIAny intArg(int64_t value) {\n"
		<< "        TVMFFIAny any{};\n"
		<< "        any.type_index = kTVMFFIInt;\n"
		<< "        any.v_int64 = value;\n"
		<< "        return any;\n"
		<< "    }\n\n"
		<< "    /// An opaque buffer that carries values from one compiled function to another (e.g. from\n"
		<< "    /// `forward_train` to `forward_backward`). `data` must be aligned to kTensorAlignment.\n"
		<< "    struct Context {\n"
		<< "        void* data = nullptr;\n"
		<< "        size_t bytes = 0;\n"
		<< "    };\n\n"
		<< "    inline Context allocContext(size_t bytes) {\n"
		<< "        const size_t rounded = (bytes + kTensorAlignment - 1) / kTensorAlignment * kTensorAlignment;\n"
		<< "        Context ctx{std::aligned_alloc(kTensorAlignment, rounded > 0 ? rounded : kTensorAlignment), bytes};\n"
		<< "        if (ctx.data == nullptr)\n"
		<< "            throw std::bad_alloc();\n"
		<< "        return ctx;\n"
		<< "    }\n\n"
		<< "    inline void freeContext(Context& ctx) {\n"
		<< "        std::free(ctx.data);\n"
		<< "        ctx = Context{};\n"
		<< "    }\n\n"
		<< "    using Function = int (*)(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);\n\n"
		<< "    inline TVMFFIAny callRaw(Function fn, const TVMFFIAny* args, int32_t num_args, const char* name) {\n"
		<< "        TVMFFIAny result{};\n"
		<< "        result.type_index = kTVMFFINone;\n"
		<< "        if (fn(nullptr, args, num_args, &result) == 0)\n"
		<< "            return result;\n"
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
		<< "    }\n\n"
		<< "    template <size_t N>\n"
		<< "    inline void call(Function fn, TVMFFIAny (&args)[N], const char* name) {\n"
		<< "        callRaw(fn, args, static_cast<int32_t>(N), name);\n"
		<< "    }\n\n"
		<< "    inline int64_t callInt(Function fn, const TVMFFIAny* args, int32_t num_args, const char* name) {\n"
		<< "        TVMFFIAny result = callRaw(fn, args, num_args, name);\n"
		<< "        if (result.type_index != kTVMFFIInt)\n"
		<< "            throw std::runtime_error(std::string{name} + \" did not return an integer\");\n"
		<< "        return result.v_int64;\n"
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
