#pragma once

#include "py_common.hpp"

#include <algorithm>
#include <format>
#include <string>
#include <vector>

namespace mdl::bindings {

	/**
	 * @brief Python bindings for PyTorch.
	 * @details Besides the plain calls, every function of a gradient build that is split into a forward pass (which
	 * saves into a context buffer) and a backward pass gets a `torch.autograd.Function` and a method
	 * `<function>_autograd(inputs...)`: its output takes part in PyTorch's autograd, so that a loss computed in PyTorch
	 * can be backpropagated through the compiled model into the module's weights (e.g. for training with PyTorch
	 * optimizers). The weights must be tensors that require gradients (e.g. torch.nn.Parameter).
	 */
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

		/** @brief The pairs (forward pass, backward pass) of the split gradient functions of @p module. */
		static std::vector<std::pair<const ExportedFunction*, const ExportedFunction*>>
		splitGradients(const std::string& module, const std::vector<ExportedFunction>& exports) {
			std::vector<std::pair<const ExportedFunction*, const ExportedFunction*>> pairs;
			for (const auto& fn : exports)
				if (fn.module == module && fn.context && fn.context->producer)
					for (const auto& other : exports)
						if (other.module == module && other.context && !other.context->producer &&
							other.context->name == fn.context->name)
							pairs.emplace_back(&fn, &other);
			return pairs;
		}

		static std::string className(const ExportedFunction& fn) { return "_" + fn.module + "_" + fn.function + "_Function"; }

		static std::string torchDtype(PrimitiveType p) {
			switch (p) {
			case PrimitiveType::integer:
				return "torch.int32";
			case PrimitiveType::boolean:
				return "torch.bool";
			case PrimitiveType::float16:
				return "torch.float16";
			case PrimitiveType::bfloat16:
				return "torch.bfloat16";
			case PrimitiveType::int8:
				return "torch.int8";
			default:
				return "torch.float32";
			}
		}

		static std::string pyList(const std::vector<std::string>& names) {
			std::string s = "(";
			for (const auto& n : names)
				s += "\"" + n + "\", ";
			return s + ")";
		}

		void classExtras(std::ostream& out, const std::string& module, const std::vector<ExportedFunction>& exports) const override {
			for (const auto& [fw, bw] : splitGradients(module, exports)) {
				std::vector<std::string> inputs;
				for (const auto& in : fw->inputs)
					if (!in.packed)
						inputs.push_back(in.name);
				std::vector<std::string> trainable;
				for (const auto& o : bw->outputs)
					if (!o.gradientOf.empty() && std::find(inputs.begin(), inputs.end(), o.gradientOf) == inputs.end())
						trainable.push_back(o.gradientOf);
				out << "\n    def " << fw->function << "_autograd(self";
				for (const auto& in : inputs)
					out << ", " << in << ": torch.Tensor";
				out << "):\n"
					<< "        \"\"\"`" << fw->function << "` as part of PyTorch's autograd: the gradients flow into the "
					<< "weights " << pyList(trainable) << ".\"\"\"\n"
					<< "        return " << className(*fw) << ".apply(self";
				for (const auto& in : inputs)
					out << ", " << in;
				out << ", *[self.weights[p] for p in " << className(*fw) << ".trainable])\n";
			}
		}

		void fileExtras(std::ostream& out, const std::vector<ExportedFunction>& exports) const override {
			std::vector<std::string> modules;
			for (const auto& fn : exports)
				if (std::find(modules.begin(), modules.end(), fn.module) == modules.end())
					modules.push_back(fn.module);
			for (const auto& module : modules)
				for (const auto& [fw, bw] : splitGradients(module, exports))
					emitFunction(out, *fw, *bw);
		}

		void emitFunction(std::ostream& out, const ExportedFunction& fw, const ExportedFunction& bw) const {
			const unsigned rank = fw.context->ellipsisRank;
			std::vector<const TensorSignature*> inputs, outputs;
			for (const auto& in : fw.inputs)
				if (!in.packed)
					inputs.push_back(&in);
			for (const auto& o : fw.outputs)
				if (!o.packed)
					outputs.push_back(&o);
			// The runtime dimensions of `...`: the leading ones of the first input that has all of them.
			int dimsFrom = -1;
			for (size_t i = 0; i < inputs.size() && dimsFrom < 0; ++i)
				if (inputs[i]->ellipsisRank < 0 && inputs[i]->shape.size() >= rank &&
					std::all_of(inputs[i]->shape.begin(), inputs[i]->shape.begin() + rank, [](int64_t d) { return d < 0; }))
					dimsFrom = static_cast<int>(i);
			auto shapeExpr = [](const TensorSignature& t) {
				std::string s = "(";
				int k = 0;
				for (int64_t d : t.shape)
					s += (d < 0 ? std::format("dims[{}]", k++) : std::to_string(d)) + ", ";
				return s + ")";
			};
			std::vector<std::string> inputNames, fwWeights, bwWeights, trainable;
			for (const auto* in : inputs)
				inputNames.push_back(in->name);
			for (const auto& w : fw.weights)
				fwWeights.push_back(w.name);
			for (const auto& w : bw.weights)
				bwWeights.push_back(w.name);
			for (const auto& o : bw.outputs)
				if (!o.gradientOf.empty() && std::find(inputNames.begin(), inputNames.end(), o.gradientOf) == inputNames.end())
					trainable.push_back(o.gradientOf);

			out << "\n\nclass " << className(fw) << "(torch.autograd.Function):\n"
				<< "    \"\"\"`" << fw.module << "." << fw.function << "` (forward pass, saves the context) and `"
				<< bw.function << "` (backward pass).\"\"\"\n\n"
				<< "    forward_weights = " << pyList(fwWeights) << "\n"
				<< "    backward_weights = " << pyList(bwWeights) << "\n"
				<< "    trainable = " << pyList(trainable) << "\n\n"
				<< "    @staticmethod\n"
				<< "    def forward(ctx, owner, *args):\n"
				<< "        inputs = [a.contiguous() for a in args[:" << inputs.size() << "]]\n";
			if (dimsFrom >= 0)
				out << "        dims = [int(d) for d in inputs[" << dimsFrom << "].shape[:" << rank << "]]\n";
			else
				out << "        dims = []\n";
			out << "        buffer = torch.empty(owner._mod[\"" << fw.context->sizeSymbol << "\"](*dims), dtype=torch.uint8)\n"
				<< "        outputs = [";
			for (const auto* o : outputs)
				out << "torch.empty(" << shapeExpr(*o) << ", dtype=" << torchDtype(o->dtype) << "), ";
			out << "]\n"
				<< "        owner._mod[\"" << fw.symbol << "\"](*inputs, *[owner.weights[p] for p in " << className(fw)
				<< ".forward_weights], *outputs, buffer)\n"
				<< "        ctx.owner, ctx.buffer, ctx.dims = owner, buffer, dims\n"
				<< "        ctx.output_shapes = [o.shape for o in outputs]\n"
				<< "        ctx.output_dtypes = [o.dtype for o in outputs]\n"
				<< "        return outputs[0] if len(outputs) == 1 else tuple(outputs)\n\n"
				<< "    @staticmethod\n"
				<< "    def backward(ctx, *grads):\n"
				<< "        owner = ctx.owner\n"
				<< "        cotangents = [g.contiguous() if g is not None else torch.zeros(s, dtype=d) for g, s, d in zip(grads, ctx.output_shapes, ctx.output_dtypes)]\n"
				<< "        # The compiled backward pass adds the gradients to its output buffers: start from zero.\n"
				<< "        gradients = {p: torch.zeros_like(owner.weights[p]) for p in " << className(fw) << ".trainable}\n";
			// The outputs of the backward pass, in order: gradients of weights (by path) and of inputs (by name).
			out << "        outputs = [";
			for (const auto& o : bw.outputs) {
				if (o.packed)
					continue;
				const bool isInput = std::find(inputNames.begin(), inputNames.end(), o.gradientOf) != inputNames.end();
				if (isInput)
					out << "torch.zeros(" << shapeExpr(o) << ", dtype=" << torchDtype(o.dtype) << "), ";
				else
					out << "gradients[\"" << o.gradientOf << "\"], ";
			}
			out << "]\n"
				<< "        dims = ctx.dims  # noqa: F841 (shapes of input gradients)\n"
				<< "        owner._mod[\"" << bw.symbol << "\"](*cotangents, *[owner.weights[p] for p in " << className(fw)
				<< ".backward_weights], *outputs, ctx.buffer)\n";
			// Gradients of the inputs (if computed), then of the trainable weights.
			out << "        input_grads = [None] * " << inputs.size() << "\n";
			size_t k = 0;
			for (const auto& o : bw.outputs) {
				if (o.packed)
					continue;
				auto it = std::find(inputNames.begin(), inputNames.end(), o.gradientOf);
				if (it != inputNames.end())
					out << "        input_grads[" << (it - inputNames.begin()) << "] = outputs[" << k << "]\n";
				++k;
			}
			out << "        return (None, *input_grads, *[gradients[p] for p in " << className(fw) << ".trainable])\n";
		}
	};

} // namespace mdl::bindings
