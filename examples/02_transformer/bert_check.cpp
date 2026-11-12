// Compares the MDL model `Bert` (token, segment, and position embeddings plus the encoder) with Hugging Face's
// BertModel: the weights, inputs, and reference output come from a safetensors file written by
// export_bert_reference.py (default: data/bert_tiny.safetensors next to this file).
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
	auto floats = mdl::runtime::loadSafetensorsAs<float>(path);
	auto ints = mdl::runtime::loadSafetensorsAs<int32_t>(path);

	Model::Weights weights;
	for (const auto& info : Model::Weights::infos) {
		auto it = floats.find(info.name);
		if (it == floats.end()) {
			std::printf("FAILED: missing weight %s in %s\n", info.name, path.c_str());
			return 1;
		}
		*weights.find(info.name) = it->second;
	}
	auto& ids = ints.at("input.ids");
	auto& types = ints.at("input.types");
	auto& positions = ints.at("input.positions");
	const auto& reference = floats.at("reference.output");

	mdl::runtime::Tensor output(reference.shape());
	Model::forward(ids, types, positions, weights, output);

	double maxErr = 0, maxRef = 0;
	for (size_t i = 0; i < output.size(); ++i) {
		maxErr = std::max(maxErr, double(std::abs(output.data()[i] - reference.data()[i])));
		maxRef = std::max(maxRef, double(std::abs(reference.data()[i])));
	}
	std::printf("batch %lld, seq %lld: max abs error vs Hugging Face %.3g (max |output| %.3g)\n",
				(long long)ids.shape()[0], (long long)ids.shape()[1], maxErr, maxRef);
	const bool ok = maxErr < 1e-4 * std::max(1.0, maxRef);
	std::puts(ok ? "OK" : "FAILED");
	return ok ? 0 : 1;
}
