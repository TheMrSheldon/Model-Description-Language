// The gradients of BERT (examples/02_transformer, `Bert` compiled with `--grad forward` for CUDA) against Hugging Face's
// autograd: forward (which saves the residuals in a context buffer on the GPU), then forward_backward with the
// cotangent dy; every weight gradient is compared with the reference of export_bert_reference.py --grad, relative to
// the largest reference value of the weight (at least 1% of the largest of all). With MDL_GRAD_FLOAT (bfloat16 or
// float16), the model is compiled with `--float bf16|f16` (attention then uses fused kernels, e.g. cuDNN's, forward and
// backward). With MDL_GRAD_MASKED, the model is compiled with `--mask forward` and the reference has padded sequences
// of different lengths (its cotangent is zero at the padding); with MDL_GRAD_SUFFIX as well, as a suffix mask
// (`--mask forward=suffix`: the sequences' lengths).
#include "model.hpp"

#include <mdl/runtime.hpp>
#include <tvm/runtime/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using Model = mdl::models::Bert;
#ifdef MDL_GRAD_FLOAT
using Float = mdl::runtime::MDL_GRAD_FLOAT;
constexpr auto kConversion = mdl::runtime::FloatConversion::allowed;
#else
using Float = float;
constexpr auto kConversion = mdl::runtime::FloatConversion::none;
#endif
constexpr double kTolerance = MDL_GRAD_TOLERANCE;

namespace {
	constexpr DLDevice device{kDLCUDA, 0};
	std::vector<tvm::runtime::Tensor> deviceTensors;

	/** @brief A copy of @p t on the device (alive until the end of main). */
	DLTensor* onDevice(const DLTensor* t) {
		auto& d = deviceTensors.emplace_back(
				tvm::runtime::Tensor::Empty(tvm::ffi::Shape(t->shape, t->shape + t->ndim), t->dtype, device)
		);
		d.CopyFrom(t);
		return const_cast<DLTensor*>(d.operator->());
	}
} // namespace

int main() {
	const std::string path = MDL_GRAD_REFERENCE;
	auto loaded = mdl::runtime::loadWeights<Model>(path, {}, kConversion);
	Model::Weights weights = loaded.weights;
	for (const auto& info : Model::Weights::infos)
		*weights.find(info.name) = onDevice(*weights.find(info.name));
	auto floats = mdl::runtime::loadSafetensorsAs<float>(path);
	auto ints = mdl::runtime::loadSafetensorsAs<int32_t>(path);
	const DLTensor* ids = onDevice(ints.at("input.ids"));
	const DLTensor* types = onDevice(ints.at("input.types"));
	const DLTensor* positions = onDevice(ints.at("input.positions"));
	const auto& reference = floats.at("reference.output");
	const int64_t batch = reference.shape()[0], seq = reference.shape()[1];

	mdl::runtime::BasicTensor<Float> dyHost(reference.shape());
	const auto& dyRef = floats.at("input.dy");
	for (size_t i = 0; i < dyHost.size(); ++i)
		dyHost.data()[i] = Float(dyRef.data()[i]);
	const DLTensor* dy = onDevice(dyHost);
	mdl::runtime::BasicTensor<Float> outHost(reference.shape());
	DLTensor* out = onDevice(outHost);

	// The gradients are added to the buffers: they start at kPrevious (checks the update in place).
	constexpr float kPrevious = 0.5f;
	std::vector<mdl::runtime::BasicTensor<Float>> gradsHost;
	std::vector<DLTensor*> grads;
	std::vector<size_t> gradIndices; // (into deviceTensors)
	for (const auto& info : Model::forward_backward_outputs) {
		std::vector<int64_t> shape(info.shape.begin(), info.shape.begin() + info.ndim);
		auto& g = gradsHost.emplace_back(shape);
		std::fill(g.values().begin(), g.values().end(), Float(kPrevious));
		grads.push_back(onDevice(g));
		gradIndices.push_back(deviceTensors.size() - 1);
	}
	auto ctx = Model::forward_ctx_alloc(batch, seq, device);
#ifdef MDL_GRAD_MASKED
	auto masks = mdl::runtime::loadSafetensorsAs<uint8_t>(path);
#ifdef MDL_GRAD_SUFFIX
	// A suffix mask (`--mask forward=suffix`): the number of valid positions per sequence.
	const auto& hostMask = masks.at("input.mask");
	mdl::runtime::IntTensor lengths({hostMask.shape()[0]});
	for (int64_t b = 0; b < hostMask.shape()[0]; ++b) {
		lengths.data()[b] = 0;
		for (int64_t p = 0; p < hostMask.shape()[1]; ++p)
			lengths.data()[b] += hostMask.data()[b * hostMask.shape()[1] + p] != 0;
	}
	const DLTensor* mask = onDevice(lengths);
#else
	const DLTensor* mask = onDevice(masks.at("input.mask"));
#endif
	Model::forward(ids, types, positions, mask, weights, out, ctx);
#else
	Model::forward(ids, types, positions, weights, out, ctx);
#endif
	[&]<size_t... I>(std::index_sequence<I...>) {
		Model::forward_backward(dy, weights, grads[I]..., ctx);
	}(std::make_index_sequence<Model::forward_backward_outputs.size()>{});

	// The reference gradients by the names of the outputs: `d_` and the weight path with `_` for `.`.
	std::map<std::string, const mdl::runtime::BasicTensor<float>*> references;
	for (const auto& [key, tensor] : floats)
		if (key.starts_with("grad.")) {
			std::string name = "d_" + key.substr(5);
			std::replace(name.begin(), name.end(), '.', '_');
			references.emplace(name, &tensor);
		}
	// Errors relative to the largest reference value of the weight, but at least 1% of the largest of all weights: a
	// gradient that is zero in exact arithmetic (e.g. of the keys' bias, to which softmax is invariant) is compared
	// with rounding noise otherwise.
	double globalMax = 0;
	for (const auto& [_, ref] : references)
		for (size_t i = 0; i < ref->size(); ++i)
			globalMax = std::max(globalMax, double(std::abs(ref->data()[i])));
	bool ok = true;
	double worst = 0;
	for (size_t p = 0; p < grads.size(); ++p) {
		const std::string name = Model::forward_backward_outputs[p].name;
		auto it = references.find(name);
		if (it == references.end()) {
			std::printf("FAILED: no reference gradient for %s\n", name.c_str());
			ok = false;
			continue;
		}
		auto& host = gradsHost[p];
		deviceTensors[gradIndices[p]].CopyTo(host.dl());
		double maxErr = 0, maxRef = 0;
		for (size_t i = 0; i < host.size(); ++i) {
			maxErr = std::max(maxErr, double(std::abs(float(host.data()[i]) - kPrevious - it->second->data()[i])));
			maxRef = std::max(maxRef, double(std::abs(it->second->data()[i])));
		}
		const double rel = maxErr / std::max(maxRef, 1e-2 * globalMax);
		worst = std::max(worst, rel);
		if (rel > kTolerance) {
			std::printf("mismatch %s: max abs error %.3g (max |reference| %.3g)\n", name.c_str(), maxErr, maxRef);
			ok = false;
		}
	}
	std::printf("%zu gradients, max relative error: %.3g\n", grads.size(), worst);
	std::puts(ok ? "OK" : "FAILED");
	deviceTensors.clear(); // before the device runtime is unloaded at exit
	return ok ? 0 : 1;
}
