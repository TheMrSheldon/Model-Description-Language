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
#include <sstream>
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
		case PrimitiveType::int8:
			return "{kDLInt, 8, 1}";
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

	/** @brief Whether the children of @p node are the elements of an array (named 0, 1, ...). */
	inline bool isArray(const HyperparameterTree& node) {
		if (node.children.empty())
			return false;
		for (size_t i = 0; i < node.children.size(); ++i)
			if (node.children[i].first != std::to_string(i))
				return false;
		return true;
	}

	/** @brief The C++ type of a leaf (`int64_t` or `double`). */
	inline std::string leafType(const HyperparameterTree& node) {
		return std::holds_alternative<int64_t>(node.value) ? "int64_t" : "double";
	}

	inline void emitHyperparameterFields(std::ostream& out, const HyperparameterTree& tree, const std::string& indent) {
		for (const auto& [name, node] : tree.children) {
			if (node.children.empty()) {
				out << indent << leafType(node) << " " << cppIdent(name) << ";\n";
			} else if (isArray(node)) {
				// Arrays (`array[T; n]`): std::array of the elements' type (a named struct for structs).
				const auto& first = node.children.front().second;
				std::string element;
				if (first.children.empty()) {
					element = "int64_t";
					for (const auto& [_, e] : node.children)
						if (!std::holds_alternative<int64_t>(e.value))
							element = "double";
				} else {
					element = cppIdent(name) + "_element";
					out << indent << "struct " << element << " {\n";
					emitHyperparameterFields(out, first, indent + "    ");
					out << indent << "};\n";
				}
				out << indent << "std::array<" << element << ", " << node.children.size() << "> " << cppIdent(name)
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
		auto value = [](const HyperparameterTree& node) {
			return std::visit([](auto v) { return std::format("{}", v); }, node.value);
		};
		std::string init;
		for (const auto& [name, node] : tree.children) {
			init += (init.empty() ? "" : ", ") + ("." + cppIdent(name) + " = ");
			if (node.children.empty()) {
				init += value(node);
			} else if (isArray(node)) {
				init += "{{";
				for (size_t i = 0; i < node.children.size(); ++i) {
					const auto& element = node.children[i].second;
					init += (i > 0 ? ", " : "") +
							(element.children.empty() ? value(element) : "{" + hyperparameterInit(element) + "}");
				}
				init += "}}";
			} else {
				init += "{" + hyperparameterInit(node) + "}";
			}
		}
		return init;
	}

	/** @param load the function that loads the object's system library (called before every compiled function). */
	inline void
	generateCppModule(std::ostream& out, const std::vector<const ExportedFunction*>& fns, const std::string& load) {
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
			<< "        /// Returns the slot of the weight with the given path, or nullptr if there is no such "
			   "weight.\n"
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

		std::ostringstream members; // (of Session)
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
				out << "\n    /// Size in bytes of the context buffer `" << context->name << "` (written by `"
					<< fn->function << "`) for the given\n"
					<< "    /// dimensions of `...`. It grows with the dimensions: a buffer for the largest dimensions "
					   "can be\n"
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
				out << "        " << load << "();\n"
					<< "        return static_cast<size_t>(mdl_detail::callInt(__tvm_ffi_" << context->sizeSymbol
					<< ", args, " << context->ellipsisRank << ", \"" << context->sizeSymbol << "\"));\n"
					<< "    }\n"
					<< "    /// Allocates a context buffer for the given dimensions of `...` (free it with "
					<< context->name << "_free).\n"
					<< "    /// `device`: the device of the tensors that will be passed (the CPU by default).\n"
					<< "    static Context " << context->name << "_alloc(" << dims << (dims.empty() ? "" : ", ")
					<< "DLDevice device = {kDLCPU, 0}) {\n"
					<< "        return mdl_detail::allocContext(" << context->name << "_size(" << dimArgs
					<< "), device);\n"
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
				<< "    /// Outputs must be preallocated by the caller (destination passing style). All tensors must "
				   "be\n"
				<< "    /// contiguous tensors of the declared type (float32, or int32 for `int`) on the device the "
				   "model was\n"
				<< "    /// compiled for, whose data is aligned to 64 bytes (e.g. mdl::runtime::Tensor on the CPU).\n"
				<< "    /// `B`/`Bi` mark the dimensions of `...`, which are only known at runtime (the same in all "
				   "tensors).\n";
			if (context)
				out << "    /// " << (context->producer ? "Writes the values the backward pass needs into" : "Reads")
					<< " the context buffer `ctx` (see " << context->name << "_size)"
					<< (context->producer ? "" : ", which must have been written with the same dimensions") << ".\n";
			if (std::any_of(outputs.begin(), outputs.end(), [](const auto* o) { return o->accumulate; }))
				out << "    /// The gradients are added to the given buffers (accumulation): zero them before the "
					   "first step.\n";
			// The function is a member of Session (members); the static function calls it on the default session.
			std::string params, names;
			for (const auto* in : inputs) {
				params += "const DLTensor* " + in->name + ", ";
				names += in->name + ", ";
			}
			params += "const Weights& weights";
			names += "weights";
			for (const auto* o : outputs) {
				params += ", DLTensor* " + o->name;
				names += ", " + o->name;
			}
			if (context) {
				params += std::string(", ") + (context->producer ? "" : "const ") + "Context& ctx";
				names += ", ctx";
			}
			out << "    static void " << fn->function << "(" << params << ") {\n"
				<< "        defaultSession()." << fn->function << "(" << names << ");\n"
				<< "    }\n";
			members << "        void " << fn->function << "(" << params << ") {\n";
			if (context)
				members << "            int64_t ctx_shape[1] = {static_cast<int64_t>(ctx.bytes)};\n"
						<< "            DLTensor ctx_tensor{ctx.data, ctx.device, 1, {kDLUInt, 8, 1}, ctx_shape, "
						   "nullptr, "
						   "0};\n";
			members << "            TVMFFIAny args[] = {\n";
			for (const auto* in : inputs)
				members << "                mdl_detail::tensorArg(" << in->name << ", \"" << in->name << "\", "
						<< dlDtype(in->dtype) << "),\n";
			for (const auto& w : fn->weights)
				members << "                mdl_detail::tensorArg(weights." << cppAccess(w.name) << ", \"" << w.name
						<< "\", " << dlDtype(w.dtype) << "),\n";
			for (const auto* o : outputs)
				members << "                mdl_detail::tensorArg(" << o->name << ", \"" << o->name << "\", "
						<< dlDtype(o->dtype) << "),\n";
			if (context)
				members << "                mdl_detail::tensorArg(&ctx_tensor, \"ctx\", {kDLUInt, 8, 1}),\n";
			const std::string sizeSymbol = workspaceSizeSymbol(fn->symbol);
			members << "                TVMFFIAny{}, // (the workspace)\n"
					<< "            };\n"
					<< "            " << load << "();\n"
					<< "            mdl_detail::callWithWorkspace(__tvm_ffi_" << fn->symbol << ", __tvm_ffi_"
					<< sizeSymbol << ", args, workspace, \"" << fn->symbol << "\", \"" << sizeSymbol << "\");\n"
					<< "        }\n";
			auto emitInfos = [&](const char* suffix, const std::vector<const TensorSignature*>& tensors) {
				out << "    static constexpr std::array<mdl_detail::TensorInfo, " << tensors.size() << "> "
					<< fn->function << "_" << suffix << "{{\n";
				for (const auto* t : tensors)
					out << "        {\"" << t->name << "\", " << infoShape(*t) << "},\n";
				out << "    }};\n";
			};
			out << "    /// Names and shapes of the inputs and outputs of `" << fn->function << "`.\n";
			emitInfos("inputs", inputs);
			emitInfos("outputs", outputs);
		}
		out << "\n    /// The state of calls (see docs/technical/runtime.md): the workspace, grown on demand. One per "
			   "thread (not\n"
			<< "    /// thread-safe); movable, not copyable. Its functions are those of `" << module
			<< "` (whose static functions use a\n"
			<< "    /// session per thread).\n"
			<< "    class Session {\n"
			<< "    public:\n"
			<< members.str() << "\n"
			<< "    private:\n"
			<< "        mdl_detail::Buffer workspace;\n"
			<< "    };\n\n"
			<< "    /// The session of the static functions (one per thread).\n"
			<< "    static Session& defaultSession() {\n"
			<< "        static thread_local Session session;\n"
			<< "        return session;\n"
			<< "    }\n"
			<< "};\n\n";
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
		<< "#include <string_view>\n"
		<< "#include <utility>\n\n"
		<< "extern \"C\" {\n"
		<< "// Device memory of context buffers (TVM runtime; see mdl_detail::allocContext).\n"
		<< "void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int dtype_code_hint,\n"
		<< "                               int dtype_bits_hint);\n"
		<< "int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr);\n";
	for (const auto& fn : exports) {
		out << "int __tvm_ffi_" << fn.symbol
			<< "(void* self, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result);\n";
		out << "int __tvm_ffi_" << workspaceSizeSymbol(fn.symbol)
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
		<< "    /// Path (or name), shape, and element type of a tensor; -1 marks dimensions that are only known at "
		   "runtime.\n"
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
		<< "        if (tensor->dtype.code != dtype.code || tensor->dtype.bits != dtype.bits || tensor->dtype.lanes != "
		   "dtype.lanes)\n"
		<< "            throw std::invalid_argument(std::string{\"tensor \"} + name + \" has the wrong element "
		   "type\");\n"
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
		<< "    /// `forward_train` to `forward_backward`), on the device of the tensors. `data` must be aligned to\n"
		<< "    /// kTensorAlignment.\n"
		<< "    struct Context {\n"
		<< "        void* data = nullptr;\n"
		<< "        size_t bytes = 0;\n"
		<< "        DLDevice device{kDLCPU, 0};\n"
		<< "    };\n\n"
		<< "    inline Context allocContext(size_t bytes, DLDevice device) {\n"
		<< "        const size_t rounded = (bytes + kTensorAlignment - 1) / kTensorAlignment * kTensorAlignment;\n"
		<< "        const size_t size = rounded > 0 ? rounded : kTensorAlignment;\n"
		<< "        Context ctx{nullptr, bytes, device};\n"
		<< "        // Host memory directly; device memory through the TVM runtime (which the model links anyway).\n"
		<< "        ctx.data = device.device_type == kDLCPU\n"
		<< "                       ? std::aligned_alloc(kTensorAlignment, size)\n"
		<< "                       : TVMBackendAllocWorkspace(device.device_type, device.device_id, size, kDLUInt, "
		   "8);\n"
		<< "        if (ctx.data == nullptr)\n"
		<< "            throw std::bad_alloc();\n"
		<< "        return ctx;\n"
		<< "    }\n\n"
		<< "    inline void freeContext(Context& ctx) {\n"
		<< "        if (ctx.device.device_type == kDLCPU)\n"
		<< "            std::free(ctx.data);\n"
		<< "        else if (ctx.data != nullptr)\n"
		<< "            TVMBackendFreeWorkspace(ctx.device.device_type, ctx.device.device_id, ctx.data);\n"
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
		<< "    }\n\n"
		<< R"(    /// A global function of the TVM runtime (looked up once by the caller).
    inline TVMFFIObjectHandle globalFunction(const char* name) {
        TVMFFIByteArray bytes{name, std::char_traits<char>::length(name)};
        TVMFFIObjectHandle fn = nullptr;
        if (TVMFFIFunctionGetGlobal(&bytes, &fn) != 0 || fn == nullptr)
            throw std::runtime_error(std::string{"the TVM runtime has no "} + name);
        return fn;
    }

    /// Memory on a device, freed on any thread: host memory directly; device memory of the TVM runtime (a tensor
    /// object, freed when its last reference goes; unlike TVM's workspace pools, which are per thread).
    class Buffer {
    public:
        Buffer() = default;
        /// `bytes` bytes on `device`, aligned to kTensorAlignment.
        Buffer(size_t bytes, DLDevice device) {
            const size_t rounded = (bytes + kTensorAlignment - 1) / kTensorAlignment * kTensorAlignment;
            shape = static_cast<int64_t>(rounded > 0 ? rounded : kTensorAlignment);
            if (device.device_type == kDLCPU) {
                host = {std::aligned_alloc(kTensorAlignment, static_cast<size_t>(shape)), device, 1,
                        {kDLUInt, 8, 1}, &shape, nullptr, 0};
                if (host.data == nullptr)
                    throw std::bad_alloc();
                tensor = &host;
                return;
            }
            static const TVMFFIObjectHandle shapeFn = globalFunction("ffi.Shape");
            static const TVMFFIObjectHandle emptyFn = globalFunction("runtime.TVMTensorAllocWithScope");
            TVMFFIAny size = intArg(shape);
            TVMFFIAny shape{};
            shape.type_index = kTVMFFINone;
            if (TVMFFIFunctionCall(shapeFn, &size, 1, &shape) != 0)
                throw std::runtime_error("creating a shape failed");
            TVMFFIAny args[4]{};
            args[0] = shape;
            args[1].type_index = kTVMFFIDataType;
            args[1].v_dtype = {kDLUInt, 8, 1};
            args[2].type_index = kTVMFFIDevice;
            args[2].v_device = device;
            args[3].type_index = kTVMFFINone;
            TVMFFIAny result{};
            result.type_index = kTVMFFINone;
            const int status = TVMFFIFunctionCall(emptyFn, args, 4, &result);
            TVMFFIObjectDecRef(shape.v_obj);
            if (status != 0)
                throw std::bad_alloc();
            handle = result.v_obj;
            tensor = TVMFFITensorGetDLTensorPtr(handle);
        }
        Buffer(Buffer&& other) noexcept { swap(other); }
        Buffer& operator=(Buffer&& other) noexcept {
            Buffer(std::move(other)).swap(*this);
            return *this;
        }
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        ~Buffer() {
            if (handle != nullptr)
                TVMFFIObjectDecRef(handle);
            else if (tensor != nullptr)
                std::free(tensor->data);
        }
        size_t bytes() const { return tensor != nullptr ? static_cast<size_t>(tensor->shape[0]) : 0; }
        const DLTensor* get() const { return tensor; }

    private:
        TVMFFIObjectHandle handle = nullptr; ///< (device memory)
        DLTensor host{};                     ///< (host memory)
        int64_t shape = 0;
        DLTensor* tensor = nullptr;
        void swap(Buffer& other) noexcept {
            std::swap(handle, other.handle);
            std::swap(host, other.host);
            std::swap(shape, other.shape);
            std::swap(tensor, other.tensor);
            // (pointers into the objects themselves)
            for (Buffer* b : {this, &other})
                if (b->handle == nullptr && b->tensor != nullptr) {
                    b->tensor = &b->host;
                    b->host.shape = &b->shape;
                }
        }
    };

    /// Calls `fn` with `args` and a workspace from `workspace` (grown to the size that `size` returns for `args`; on
    /// the device of the first tensor): `args` has one more element, for the workspace.
    template <size_t N>
    inline void callWithWorkspace(Function fn, Function size, TVMFFIAny (&args)[N], Buffer& workspace,
                                  const char* name, const char* sizeName) {
        const auto bytes = static_cast<size_t>(callInt(size, args, static_cast<int32_t>(N - 1), sizeName));
        const DLDevice device = static_cast<const DLTensor*>(args[0].v_ptr)->device;
        const DLTensor* current = workspace.get();
        if (current == nullptr || workspace.bytes() < bytes || current->device.device_type != device.device_type ||
            current->device.device_id != device.device_id)
            workspace = Buffer(bytes > 0 ? bytes : kTensorAlignment, device);
        args[N - 1] = tensorArg(workspace.get(), "workspace", {kDLUInt, 8, 1});
        call(fn, args, name);
    }

)"
		<< "    /// Loads the TVM system library `prefix` (the compiled object), which attaches its device code (e.g.\n"
		<< "    /// CUDA kernels) to its functions; nothing to do on the CPU, but harmless.\n"
		<< "    inline bool loadSystemLib(const char* prefix) {\n"
		<< "        static constexpr char kName[] = \"ffi.SystemLib\";\n"
		<< "        TVMFFIByteArray name{kName, sizeof(kName) - 1};\n"
		<< "        TVMFFIObjectHandle fn = nullptr;\n"
		<< "        if (TVMFFIFunctionGetGlobal(&name, &fn) != 0 || fn == nullptr)\n"
		<< "            throw std::runtime_error(\"the TVM runtime has no ffi.SystemLib\");\n"
		<< "        TVMFFIAny arg{};\n"
		<< "        arg.type_index = kTVMFFIRawStr;\n"
		<< "        arg.v_c_str = prefix;\n"
		<< "        TVMFFIAny result{};\n"
		<< "        result.type_index = kTVMFFINone;\n"
		<< "        const int status = TVMFFIFunctionCall(fn, &arg, 1, &result);\n"
		<< "        TVMFFIObjectDecRef(fn);\n"
		<< "        if (status != 0) {\n"
		<< "            std::string message = std::string{\"loading the system library \"} + prefix + \" failed\";\n"
		<< "            TVMFFIObjectHandle error = nullptr;\n"
		<< "            TVMFFIErrorMoveFromRaised(&error);\n"
		<< "            if (error != nullptr) {\n"
		<< "                const TVMFFIErrorCell* cell = TVMFFIErrorGetCellPtr(error);\n"
		<< "                message += \": \" + std::string{cell->message.data, cell->message.size};\n"
		<< "                TVMFFIObjectDecRef(error);\n"
		<< "            }\n"
		<< "            throw std::runtime_error(message);\n"
		<< "        }\n"
		<< "        if (result.type_index >= kTVMFFIStaticObjectBegin)\n"
		<< "            TVMFFIObjectDecRef(result.v_obj); // the runtime keeps the library alive\n"
		<< "        return true;\n"
		<< "    }\n"
		<< "} // namespace mdl_detail\n"
		<< "#endif\n\n";
	const std::string load = "mdl_lib_" + libraryPrefix + "load";
	out << "/// Loads the compiled object (its TVM system library), once.\n"
		<< "inline void " << load << "() {\n"
		<< "    static const bool loaded = mdl_detail::loadSystemLib(\"" << libraryPrefix << "\");\n"
		<< "    (void)loaded;\n"
		<< "}\n\n";

	// Group the exported functions by module (preserving order).
	std::vector<std::vector<const ExportedFunction*>> modules;
	for (const auto& fn : exports) {
		auto it = std::find_if(modules.begin(), modules.end(), [&](const auto& m) {
			return m.front()->module == fn.module;
		});
		if (it == modules.end())
			modules.push_back({&fn});
		else
			it->push_back(&fn);
	}
	for (const auto& fns : modules)
		detail::generateCppModule(out, fns, load);

	out << "} // namespace mdl::models\n";
}
