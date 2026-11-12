// Compares the MDL model `Bert` (token, segment, and position embeddings plus the encoder) with Hugging Face's
// BertModel: the weights, inputs, and reference output come from a safetensors file written by
// export_bert_reference.py (default: data/bert_tiny.safetensors next to this file). With MDL_BERT_MASKED, the model is
// compiled with `--mask forward` and the file contains padded sequences and their mask: only the valid positions are
// compared (they must not depend on the padding), and the masked positions of the output must be zero.
//
//     02_transformer_bert [file.safetensors]
#include "model.hpp"

#include <mdl/runtime.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

using Model = mdl::models::Bert;

int main(int argc, char** argv) {
	const std::string path = argc > 1 ? argv[1] : MDL_BERT_REFERENCE;
	// The weights are loaded directly into Model::Weights (with dtype and shape checks); the file also contains the
	// inputs and the reference output, which are not weights of the model.
	auto loaded = mdl::runtime::loadWeights<Model>(path);
	const auto& weights = loaded.weights;
	auto floats = mdl::runtime::loadSafetensorsAs<float>(path);
	auto ints = mdl::runtime::loadSafetensorsAs<int32_t>(path);
	std::printf("loaded %zu weights (%zu other tensors in the file)\n", Model::Weights::infos.size(), loaded.unused.size());
	auto& ids = ints.at("input.ids");
	auto& types = ints.at("input.types");
	auto& positions = ints.at("input.positions");
	const auto& reference = floats.at("reference.output");

	mdl::runtime::Tensor output(reference.shape());
	const size_t hidden = static_cast<size_t>(reference.shape().back());
#ifdef MDL_BERT_MASKED
	auto masks = mdl::runtime::loadSafetensorsAs<uint8_t>(path);
	auto& mask = masks.at("input.mask");
	Model::forward(ids, types, positions, mask, weights, output);
	auto valid = [&](size_t i) { return mask.data()[i / hidden] != 0; };
#else
	Model::forward(ids, types, positions, weights, output);
	auto valid = [](size_t) { return true; };
#endif

	double maxErr = 0, maxRef = 0, maxMasked = 0;
	size_t masked = 0;
	for (size_t i = 0; i < output.size(); ++i) {
		if (!valid(i)) {
			maxMasked = std::max(maxMasked, double(std::abs(output.data()[i])));
			++masked;
			continue;
		}
		maxErr = std::max(maxErr, double(std::abs(output.data()[i] - reference.data()[i])));
		maxRef = std::max(maxRef, double(std::abs(reference.data()[i])));
	}
	if (masked > 0)
		std::printf("masked positions: %zu (max |output| there: %.3g)\n", masked / hidden, maxMasked);
	std::printf("batch %lld, seq %lld: max abs error vs Hugging Face %.3g (max |output| %.3g)\n",
				(long long)ids.shape()[0], (long long)ids.shape()[1], maxErr, maxRef);
	const bool ok = maxErr < 1e-4 * std::max(1.0, maxRef) && maxMasked == 0.0;
	std::puts(ok ? "OK" : "FAILED");
	return ok ? 0 : 1;
}
