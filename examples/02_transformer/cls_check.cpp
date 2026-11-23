// Compares the logits of the MDL model `BertForSequenceClassification` (compiled with `--mask logits`) with Hugging
// Face's BertForSequenceClassification on padded sequences (data/bert_tiny_cls.safetensors, see
// export_bert_reference.py --classes). With MDL_CLS_TRAIN, the objective `loss` is compiled as a masked gradient build
// (`--mask loss --grad loss`, labels [batch] via `--ellipsis-rank labels=1`): its value is compared with Hugging
// Face's loss, and its gradients with central finite differences. With MDL_QUANTIZED (a tolerance), the model is compiled
// with quantized weight matrices (`--quantize`), which are quantized from the float checkpoint while loading; the error
// relative to the largest logit must be below MDL_QUANTIZED.
#include "model.hpp"

#include <mdl/runtime.hpp>

#include <algorithm>
#include <array>
#include <tuple>
#include <vector>
#include <cmath>
#include <cstdio>
#include <string>

using Model = mdl::models::BertForSequenceClassification;

int main(int argc, char** argv) {
	const std::string path = argc > 1 ? argv[1] : MDL_BERT_REFERENCE;
#ifdef MDL_QUANTIZED
	auto loaded = mdl::runtime::loadWeights<Model>(
			path, {}, mdl::runtime::FloatConversion::none, mdl::runtime::Quantization::fromFloat
	);
	const double tolerance = MDL_QUANTIZED;
#else
	auto loaded = mdl::runtime::loadWeights<Model>(path);
	const double tolerance = 1e-4;
#endif
	auto floats = mdl::runtime::loadSafetensorsAs<float>(path);
	auto ints = mdl::runtime::loadSafetensorsAs<int32_t>(path);
	auto masks = mdl::runtime::loadSafetensorsAs<uint8_t>(path);
	const auto& reference = floats.at("reference.logits");

	auto& ids = ints.at("input.ids");
	auto& types = ints.at("input.types");
	auto& positions = ints.at("input.positions");
	auto& mask = masks.at("input.mask");
	mdl::runtime::Tensor logits(reference.shape());
	Model::logits(ids, types, positions, mask, loaded.weights, logits);
	double maxErr = 0;
	for (size_t i = 0; i < logits.size(); ++i)
		maxErr = std::max(maxErr, double(std::abs(logits.data()[i] - reference.data()[i])));
	std::printf("logits [%lld, %lld]: max abs error vs Hugging Face %.3g\n", (long long)reference.shape()[0],
				(long long)reference.shape()[1], maxErr);
#ifdef MDL_QUANTIZED
	double largest = 0;
	for (size_t i = 0; i < reference.size(); ++i)
		largest = std::max(largest, double(std::abs(reference.data()[i])));
	std::printf("error relative to the largest logit: %.3g\n", maxErr / largest);
	bool ok = maxErr < tolerance * largest;
#else
	bool ok = maxErr < tolerance;
#endif

#ifdef MDL_CLS_TRAIN
	// loss(ids, types, positions, labels, mask) -> loss, d_<weight>... (accumulated: start from zero)
	auto& labels = ints.at("input.labels");
	constexpr size_t numGrads = Model::loss_outputs.size() - 1;
	auto shapeOf = [](const auto& info) { return std::vector<int64_t>(info.shape.begin(), info.shape.begin() + info.ndim); };
	mdl::runtime::Tensor loss(std::vector<int64_t>{});
	std::vector<mdl::runtime::Tensor> grads, scratch;
	for (size_t i = 1; i <= numGrads; ++i) {
		grads.emplace_back(shapeOf(Model::loss_outputs[i]));
		scratch.emplace_back(shapeOf(Model::loss_outputs[i]));
	}
	std::array<DLTensor*, numGrads> gradPtrs, scratchPtrs;
	for (size_t i = 0; i < numGrads; ++i) {
		gradPtrs[i] = grads[i];
		scratchPtrs[i] = scratch[i];
	}
	std::apply([&](auto*... g) { Model::loss(ids, types, positions, labels, mask, loaded.weights, loss, g...); }, gradPtrs);
	const float referenceLoss = floats.at("reference.loss").data()[0];
	std::printf("loss %.6f, Hugging Face %.6f\n", loss.data()[0], referenceLoss);
	ok &= std::abs(loss.data()[0] - referenceLoss) < 1e-5;

	// Central finite differences for a few elements of every weight.
	auto objective = [&] {
		std::apply([&](auto*... g) { Model::loss(ids, types, positions, labels, mask, loaded.weights, loss, g...); }, scratchPtrs);
		return double(loss.data()[0]);
	};
	double maxRelErr = 0;
	for (size_t p = 0; p < numGrads; ++p) {
		auto& t = loaded.floats.at(Model::loss_outputs[p + 1].name[0] == 'd' ? std::string(Model::Weights::infos[p].name)
																			 : std::string());
		for (size_t i : {size_t{0}, t.size() / 2, t.size() - 1}) {
			const float original = t.data()[i];
			const float eps = 1e-2f * std::max(1.0f, std::abs(original));
			t.data()[i] = original + eps;
			const double plus = objective();
			t.data()[i] = original - eps;
			const double minus = objective();
			t.data()[i] = original;
			const double numeric = (plus - minus) / (2 * eps), analytic = grads[p].data()[i];
			const double relErr = std::abs(numeric - analytic) / std::max(1.0, std::abs(numeric));
			maxRelErr = std::max(maxRelErr, relErr);
			if (relErr > 2e-2)
				std::printf("mismatch %s[%zu]: analytic %g, numeric %g\n", Model::loss_outputs[p + 1].name, i, analytic, numeric);
		}
	}
	std::printf("max relative error of %zu gradients (finite differences): %.3g\n", numGrads, maxRelErr);
	ok &= maxRelErr < 2e-2;
#endif

	std::puts(ok ? "OK" : "FAILED");
	return ok ? 0 : 1;
}
