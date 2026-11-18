// Compares the MDL model `Bert` (token, segment, and position embeddings plus the encoder) with Hugging Face's
// BertModel: the weights, inputs, and reference output come from a safetensors file written by
// export_bert_reference.py (default: data/bert_tiny.safetensors next to this file). With MDL_BERT_MASKED, the model is
// compiled with `--mask forward` and the file contains padded sequences and their mask: only the valid positions are
// compared (they must not depend on the padding), and the masked positions of the output must be zero.
// With MDL_BERT_FLOAT (bfloat16 or float16), the model is compiled with `--float bf16|f16`: the float32 reference
// weights are converted when loading, and the outputs are compared with the tolerance MDL_BERT_TOLERANCE (relative to
// the largest output). With MDL_BERT_DEVICE (a DLDeviceType, e.g. kDLCUDA), the model is compiled for that device
// (mdlc --target): the weights and inputs are copied there, the outputs back.
//
//     02_transformer_bert [file.safetensors]
#include "model.hpp"

#include <mdl/runtime.hpp>
#ifdef MDL_BERT_DEVICE
#include <tvm/runtime/tensor.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using Model = mdl::models::Bert;
#ifdef MDL_BERT_FLOAT
using Float = mdl::runtime::MDL_BERT_FLOAT;
constexpr auto kConversion = mdl::runtime::FloatConversion::allowed;
constexpr double kTolerance = MDL_BERT_TOLERANCE;
#else
using Float = float;
constexpr auto kConversion = mdl::runtime::FloatConversion::none;
#ifdef MDL_BERT_TOLERANCE
constexpr double kTolerance = MDL_BERT_TOLERANCE;
#else
constexpr double kTolerance = 1e-4;
#endif
#endif

#ifdef MDL_BERT_DEVICE
namespace {
	constexpr DLDevice device{MDL_BERT_DEVICE, 0};
	std::vector<tvm::runtime::Tensor> deviceTensors;

	/** @brief A copy of @p t on the device (alive until the end of main). */
	const DLTensor* onDevice(const DLTensor* t) {
		auto& d = deviceTensors.emplace_back(
				tvm::runtime::Tensor::Empty(tvm::ffi::Shape(t->shape, t->shape + t->ndim), t->dtype, device)
		);
		d.CopyFrom(t);
		return d.operator->();
	}

	/** @brief Calls @p f with a device buffer for @p host, then copies the buffer back into @p host. */
	template <typename T, typename F>
	void intoHost(mdl::runtime::BasicTensor<T>& host, F&& f) {
		auto* d = const_cast<DLTensor*>(onDevice(host));
		f(d);
		deviceTensors.back().CopyTo(host.dl());
	}
} // namespace
#else
namespace {
	const DLTensor* onDevice(const DLTensor* t) { return t; }
	template <typename T, typename F>
	void intoHost(mdl::runtime::BasicTensor<T>& host, F&& f) {
		f(host.dl());
	}
} // namespace
#endif

int main(int argc, char** argv) {
	const std::string path = argc > 1 ? argv[1] : MDL_BERT_REFERENCE;
	// The weights are loaded directly into Model::Weights (with dtype and shape checks); the file also contains the
	// inputs and the reference output, which are not weights of the model.
	auto loaded = mdl::runtime::loadWeights<Model>(path, {}, kConversion);
	Model::Weights weights = loaded.weights;
	for (const auto& info : Model::Weights::infos)
		*weights.find(info.name) = onDevice(*weights.find(info.name));
	auto floats = mdl::runtime::loadSafetensorsAs<float>(path);
	auto ints = mdl::runtime::loadSafetensorsAs<int32_t>(path);
	std::printf("loaded %zu weights (%zu other tensors in the file)\n", Model::Weights::infos.size(), loaded.unused.size());
	const DLTensor* ids = onDevice(ints.at("input.ids"));
	const DLTensor* types = onDevice(ints.at("input.types"));
	const DLTensor* positions = onDevice(ints.at("input.positions"));
	const auto& reference = floats.at("reference.output");

	mdl::runtime::BasicTensor<Float> output(reference.shape());
	const size_t hidden = static_cast<size_t>(reference.shape().back());
#ifdef MDL_BERT_MASKED
	auto masks = mdl::runtime::loadSafetensorsAs<uint8_t>(path);
	auto& hostMask = masks.at("input.mask");
	const DLTensor* mask = onDevice(hostMask);
	intoHost(output, [&](DLTensor* out) { Model::forward(ids, types, positions, mask, weights, out); });
	auto valid = [&](size_t i) { return hostMask.data()[i / hidden] != 0; };
#else
	intoHost(output, [&](DLTensor* out) { Model::forward(ids, types, positions, weights, out); });
	auto valid = [](size_t) { return true; };
#endif

	double maxErr = 0, maxRef = 0, maxMasked = 0;
	size_t masked = 0;
	for (size_t i = 0; i < output.size(); ++i) {
		if (!valid(i)) {
			maxMasked = std::max(maxMasked, double(std::abs(float(output.data()[i]))));
			++masked;
			continue;
		}
		maxErr = std::max(maxErr, double(std::abs(float(output.data()[i]) - reference.data()[i])));
		maxRef = std::max(maxRef, double(std::abs(reference.data()[i])));
	}
	if (masked > 0)
		std::printf("masked positions: %zu (max |output| there: %.3g)\n", masked / hidden, maxMasked);
	std::printf("batch %lld, seq %lld: max abs error vs Hugging Face %.3g (max |output| %.3g)\n",
				(long long)ids->shape[0], (long long)ids->shape[1], maxErr, maxRef);
	bool ok = maxErr < kTolerance * std::max(1.0, maxRef) && maxMasked == 0.0;

	// The pooled output (the transformed first position of each sequence): [batch, hidden].
	const auto& referencePooled = floats.at("reference.pooled");
	mdl::runtime::BasicTensor<Float> pooled(referencePooled.shape());
#ifdef MDL_BERT_MASKED
	intoHost(pooled, [&](DLTensor* out) { Model::pooled(ids, types, positions, mask, weights, out); });
#else
	intoHost(pooled, [&](DLTensor* out) { Model::pooled(ids, types, positions, weights, out); });
#endif
	double maxPooledErr = 0;
	for (size_t i = 0; i < pooled.size(); ++i)
		maxPooledErr = std::max(maxPooledErr, double(std::abs(float(pooled.data()[i]) - referencePooled.data()[i])));
	std::printf("pooled output: max abs error vs Hugging Face %.3g\n", maxPooledErr);
	ok &= maxPooledErr < kTolerance;
	std::puts(ok ? "OK" : "FAILED");
#ifdef MDL_BERT_DEVICE
	deviceTensors.clear(); // before the device runtime is unloaded at exit
#endif
	return ok ? 0 : 1;
}
