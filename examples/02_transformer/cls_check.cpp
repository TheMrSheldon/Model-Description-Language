// Compares the logits of the MDL model `BertForSequenceClassification` (compiled with `--mask logits`) with Hugging
// Face's BertForSequenceClassification on padded sequences (data/bert_tiny_cls.safetensors, see
// export_bert_reference.py --classes).
#include "model.hpp"

#include <mdl/runtime.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

using Model = mdl::models::BertForSequenceClassification;

int main(int argc, char** argv) {
	const std::string path = argc > 1 ? argv[1] : MDL_BERT_REFERENCE;
	auto loaded = mdl::runtime::loadWeights<Model>(path);
	auto floats = mdl::runtime::loadSafetensorsAs<float>(path);
	auto ints = mdl::runtime::loadSafetensorsAs<int32_t>(path);
	auto masks = mdl::runtime::loadSafetensorsAs<uint8_t>(path);
	const auto& reference = floats.at("reference.logits");

	mdl::runtime::Tensor logits(reference.shape());
	Model::logits(ints.at("input.ids"), ints.at("input.types"), ints.at("input.positions"), masks.at("input.mask"),
				  loaded.weights, logits);
	double maxErr = 0;
	for (size_t i = 0; i < logits.size(); ++i)
		maxErr = std::max(maxErr, double(std::abs(logits.data()[i] - reference.data()[i])));
	std::printf("logits [%lld, %lld]: max abs error vs Hugging Face %.3g\n", (long long)reference.shape()[0],
				(long long)reference.shape()[1], maxErr);
	const bool ok = maxErr < 1e-4;
	std::puts(ok ? "OK" : "FAILED");
	return ok ? 0 : 1;
}
